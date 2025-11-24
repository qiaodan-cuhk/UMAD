"""异构多轮qmix，sentence level Q"""

import ray
import torch
import numpy as np
from collections import defaultdict
import os

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.metric import (
    reduce_metrics,
)

from verl.trainer.ppo.core_algos import agg_loss  # fit
from marl.utils.marl_utils import compute_advantage, apply_kl_penalty, _timer

# 用于embed model
from marl.modules.agents.ppo_agent import ResourcePoolManager
from verl.single_controller.ray import RayWorkerGroup
from marl.modules.workers.embed_worker import EmbedWorker
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray import create_colocated_worker_cls
from marl.utils.marl_utils import MARLRole

class RayQMIXLearner():
    def __init__(self,
                 config,
                 mac,
                 num_agents,  # cotrian LLMs
                 tokenizer_list,   # 异构tokenizer list
                 device_name="cuda",
                 **kwargs
                 ):

                         
        self.config = config
        self.num_agents = num_agents

        self.extra_params = kwargs
        self.use_critic = kwargs.get('use_critic', False)
        self.use_reference_policy = kwargs.get('use_reference_policy', False)
        # self.use_rm = kwargs.get('use_rm', False)
        self.ref_in_actor = kwargs.get('ref_in_actor', False)
        self.kl_ctrl_in_reward = kwargs.get('kl_ctrl_in_reward', False)
        # self.hybrid_engine = kwargs.get('hybrid_engine', False)


        self.mac = mac # 传入
        self.turns = config.marl.turns
        self.tokenizer_list = tokenizer_list

        # 这个参数没用
        self.device = device_name
        # assert self.device == "cuda", "qmix learner only support cuda device"



        from marl.modules.mixers.qmix import QMixer
        from marl.modules.mixers.vdn import VDNMixer

        # Choose mixer
        if config.marl.name == "qmix":
            self.mixer = QMixer(config)
        elif config.marl.name == "vdn":
            self.mixer = VDNMixer(config)


        self.gamma_turn = config.marl.gamma_turn

        if config.marl.name == "qmix":
            self.mixer_params = list(self.mixer.hyper_w_1.parameters()) + \
                    list(self.mixer.hyper_w_final.parameters()) + \
                    list(self.mixer.hyper_b_1.parameters()) + \
                    list(self.mixer.V.parameters())
        elif config.marl.name == "vdn":
            self.mixer_params = []  # vdn不需要参数


        # 2. Q_i 网络参数（self.critics_sentence 是 list）
        for critic in self.mixer.critics_sentence:
            self.mixer_params += list(critic.parameters())

        # 3. 创建 optimizer
        self.qmix_lr = config.marl.qmix_lr
        self.optimizer = torch.optim.Adam(self.mixer_params, lr=self.qmix_lr)


        self.qmix_batch = config.marl.qmix_batch_size
        self.qmix_update_count = 0 # 用于稳定target q更新
        self.qmix_update_steps = config.marl.qmix_update_steps
        self.qmix_encode_batch_size = config.marl.qmix_encode_batch_size

        self.embedding_micro_batch_size = config.marl.embedding_micro_batch_size


    

    """每个agent计算自己的old log prob存储到metric agent id"""
    def _compute_old_log_prob(self, agent_batchs, metrics, turn_idx):
        
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]

            # 额外计算confidence   # 5k + 2k 下 bs 64 turn 1 bs 320 显存接近 80G 占用
            old_log_prob = agent.actor_rollout_wg.compute_log_prob(agent_batch)
            entropys = old_log_prob.batch["entropys"]

            # confidence
            confidences = old_log_prob.batch["confidences"]


            response_masks = agent_batch.batch["response_mask"]
            loss_agg_mode = agent.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)

            # 假设confidence与entropy相同维度数据
            # confidence_loss = agg_loss(loss_mat=confidences, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            confidence_bs = (confidences * response_masks).sum(-1) / (response_masks.sum(-1) + 1e-8)
            confidence_bs_group = confidence_bs.reshape(-1, self.config.actor_rollout_ref.rollout.n)
            confidence_bs_group_mean = confidence_bs_group.mean(dim=-1).mean().detach().item()
            confidence_bs_group_std = confidence_bs_group.std(dim=-1).mean().detach().item()



            old_log_prob_metrics = {f"agent_{agent_id}_actor/entropy_loss_turn_{turn_idx}": entropy_loss.detach().item(),
                                    f"agent_{agent_id}_actor/confidence_mean_turn_{turn_idx}": confidence_bs_group_mean,
                                    f"agent_{agent_id}_actor/confidence_std_turn_{turn_idx}": confidence_bs_group_std}

            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            # 清理confidence
            old_log_prob.batch.pop("confidences")
            agent_batch = agent_batch.union(old_log_prob)
            agent_batchs[f"agent_{agent_id}"] = agent_batch

            agent_batch.batch["confidences"] = confidence_bs.detach()

            if "rollout_log_probs" in agent_batch.batch.keys():
                # TODO: we may want to add diff of probs too.
                rollout_old_log_probs = agent_batch.batch["rollout_log_probs"]
                actor_old_log_probs = agent_batch.batch["old_log_probs"]
                attention_mask = agent_batch.batch["attention_mask"]
                responses = agent_batch.batch["responses"]
                response_length = responses.size(1)
                response_mask = attention_mask[:, -response_length:]

                rollout_probs = torch.exp(rollout_old_log_probs)
                actor_probs = torch.exp(actor_old_log_probs)
                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                metrics.update(
                    {
                        f"agent_{agent_id}_training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                        f"agent_{agent_id}_training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                        f"agent_{agent_id}_training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                    }
                )



    def _compute_ref_log_prob(self, agent_batchs):
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]

            """这里要考虑用marl层还是agent层"""
            if not self.ref_in_actor:
                ref_log_prob = agent.ref_policy_wg.compute_ref_log_prob(agent_batch)
            else:
                ref_log_prob = agent.actor_rollout_wg.compute_ref_log_prob(agent_batch)

            # 获取ref_confidence
            ref_confidences = ref_log_prob.batch["ref_confidences"]  # 新增，但是暂时没用于记录
            # 清理不需要的字段
            ref_log_prob.batch.pop("ref_confidences")  # 新增：清理ref_confidences

            agent_batch = agent_batch.union(ref_log_prob)
            
            agent_batchs[f"agent_{agent_id}"] = agent_batch



    def _compute_values(self, agent_batchs):
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            values = agent.critic_wg.compute_values(agent_batch)
            # values维度与response length有关，无效token对应response mask为0，不计算critic
            agent_batch = agent_batch.union(values)
            agent_batchs[f"agent_{agent_id}"] = agent_batch

    # def _compute_values_preds(self, agent_batchs):
    #     vpreds = []
    #     for agent_id, agent in enumerate(self.mac.agents):
    #         agent_batch = agent_batchs[f"agent_{agent_id}"]
    #         # 先试试能不能用select出来的这部分直接计算critic，不能的话额外保留一个完整的dataproto数据切分
    #         values = agent.critic_wg.compute_values(agent_batch)
    #         values_tensor = values.batch['values']
    #         # 确保values_tensor需要梯度
    #         values_tensor.requires_grad_(True)  # 添加这行
    #         vpreds.append(values_tensor)
    #     return vpreds

    # def encode_in_batches(self, texts, max_batch_size=None):
    #     """分批编码，避免内存爆炸"""
    #     embeddings = []
    #     for i in range(0, len(texts), max_batch_size):
    #         batch_texts = texts[i:i+max_batch_size]
    #         with torch.no_grad():
    #             batch_embeddings = self.mixer.text_lm.encode(
    #                 batch_texts, 
    #                 batch_size=len(batch_texts), 
    #                 convert_to_tensor=True
    #             )
    #         embeddings.append(batch_embeddings.cpu())
    #     return torch.cat(embeddings, dim=0)
    

    """目前版本针对tree grpo设计，会按照grpo.n进行group收敛，trajectory grpo需要修改"""
    
    def transfer_embed_ids(self, agent_batchs, rewards_all, turns=2, mode="td0"):
        """
        将各agent的token ids转为文本，再用统一的embed_base_lm tokenizer编码为统一token ids，
        返回(state_ids, actions_ids, next_state_ids, rewards, dones)用于buffer存储。
        """

        # td0 是标准的1 step transition，mc是td2，用于对比第一轮的critic能不能学好
        if mode == "td0":
            num_agents = self.num_agents
            assert len(agent_batchs['turn_0']) == num_agents
            batch_size = agent_batchs['turn_0']['agent_0'].batch['input_ids'].shape[0]

            action_lengths_list = []  # 统计回复长度

            # 1. 获取全局state（假设用agent_0的input_ids代表全局state）
            # 也可以自定义全局state的拼接方式
            state_ids_list = []
            next_state_ids_list = []
            actions_ids_list = []
            rewards_list = []
            dones_list = []

            #### 这里还要考虑turn的问题，目前只考虑了单轮，reward也是自然支持按照turn添加
            for turn_idx in range(turns):
                agent_batchs_turn = agent_batchs[f'turn_{turn_idx}']

                terminate = (turn_idx == turns - 1)
                batch_size = agent_batchs_turn[f'agent_0'].batch['input_ids'].shape[0]


                # Batch decode states
                # 取当前step的state（用agent_0的input_ids），global state
                state_token_ids_batch = agent_batchs_turn['agent_0'].batch['prompts']   # 可能带有system，影响应该不大
                state_texts = self.tokenizer_list[0].batch_decode(state_token_ids_batch, skip_special_tokens=True)

                # 移除chat template
                state_texts = [self.remove_template(text) for text in state_texts]

                state_ids_batch = [self.mixer.text_tokenizer.encode(text, add_special_tokens=False) for text in state_texts]
                state_ids_list.extend(state_ids_batch)


                # Batch decode actions for each agent
                actions_ids_batch = []
                rewards_batch = []
                action_lengths_batch = []
                for agent_id in range(num_agents):
                    action_token_ids_batch = agent_batchs_turn[f'agent_{agent_id}'].batch['responses']
                    action_texts = self.tokenizer_list[agent_id].batch_decode(action_token_ids_batch, skip_special_tokens=True)
                    action_ids_batch = [self.mixer.text_tokenizer.encode(text, add_special_tokens=False) for text in action_texts]
                    actions_ids_batch.append(action_ids_batch)

                    # 统计回复长度
                    action_lengths = agent_batchs_turn[f'agent_{agent_id}'].batch['response_mask'].sum(dim=-1).tolist()
                    action_lengths_batch.append(action_lengths) #[ [bs], [bs], ...]
                    
                    # Batch rewards
                    reward_batch = rewards_all[f'turn_{turn_idx}'][f'agent_{agent_id}'].sum(dim=-1).tolist()
                    rewards_batch.append(reward_batch)

                # Verify rewards are equal across agents
                if self.config.marl.agg_mode != "ind":
                    for i in range(batch_size):
                        assert all(rewards_batch[agent_id][i] == rewards_batch[0][i] for agent_id in range(num_agents)), "rewards for all agents should be equal"



                # Transpose actions_ids_batch to get per-sample format
                for i in range(batch_size):
                    actions_ids_list.append([actions_ids_batch[agent_id][i] for agent_id in range(num_agents)])
                    action_lengths_list.append([action_lengths_batch[agent_id][i] for agent_id in range(num_agents)])
                    rewards_list.append(rewards_batch[0][i])
                    dones_list.append(terminate)

                # Next states
                if turn_idx < turns - 1:

                    # tree grpo需要按照n进行采样
                    # grpo_n = self.config.actor_rollout_ref.rollout.n 
                    # next_state_token_ids_batch = agent_batchs[f'turn_{turn_idx+1}']['agent_0'].batch['prompts'][::grpo_n]

                    # traj grpo不需要按照n进行采样
                    next_state_token_ids_batch = agent_batchs[f'turn_{turn_idx+1}']['agent_0'].batch['prompts']
                    assert next_state_token_ids_batch.shape[0] == batch_size
                    next_state_texts = self.tokenizer_list[0].batch_decode(next_state_token_ids_batch, skip_special_tokens=True)

                    # 移除chat template
                    next_state_texts = [self.remove_template(text) for text in next_state_texts]

                    next_state_ids_batch = [self.mixer.text_tokenizer.encode(text, add_special_tokens=False) for text in next_state_texts]
                    next_state_ids_list.extend(next_state_ids_batch)
                else:
                    # Terminal state for last turn
                    terminal_text = "The conversation is over, please respond <END>."
                    terminal_ids = self.mixer.text_tokenizer.encode(terminal_text, add_special_tokens=False)
                    next_state_ids_list.extend([terminal_ids] * batch_size)
        elif mode == "mc":
            num_agents = self.num_agents
            assert len(agent_batchs['turn_0']) == num_agents
            batch_size = agent_batchs['turn_0']['agent_0'].batch['input_ids'].shape[0]
            
            # 1. 获取全局state（假设用agent_0的input_ids代表全局state）
            # 也可以自定义全局state的拼接方式
            state_ids_list = []
            next_state_ids_list = []
            actions_ids_list = []
            rewards_list = []
            dones_list = []
            action_lengths_list = [] # 统计回复长度

            #### 这里还要考虑turn的问题，目前只考虑了单轮，reward也是自然支持按照turn添加
            for turn_idx in range(turns):
                agent_batchs_turn = agent_batchs[f'turn_{turn_idx}']

                # terminate = (turn_idx == turns - 1)
                terminate = True    # mc算法任何step都是一步done
                batch_size = agent_batchs_turn[f'agent_0'].batch['input_ids'].shape[0]


                # Batch decode states
                # 取当前step的state（用agent_0的input_ids），global state
                state_token_ids_batch = agent_batchs_turn['agent_0'].batch['prompts']
                state_texts = self.tokenizer_list[0].batch_decode(state_token_ids_batch, skip_special_tokens=True)
                state_ids_batch = [self.mixer.text_tokenizer.encode(text, add_special_tokens=False) for text in state_texts]
                state_ids_list.extend(state_ids_batch)


                # Batch decode actions for each agent
                actions_ids_batch = []
                rewards_batch = []
                action_lengths_batch = []
                for agent_id in range(num_agents):
                    action_token_ids_batch = agent_batchs_turn[f'agent_{agent_id}'].batch['responses']
                    action_texts = self.tokenizer_list[agent_id].batch_decode(action_token_ids_batch, skip_special_tokens=True)
                    action_ids_batch = [self.mixer.text_tokenizer.encode(text, add_special_tokens=False) for text in action_texts]
                    actions_ids_batch.append(action_ids_batch)

                    # 统计回复长度
                    action_lengths = agent_batchs_turn[f'agent_{agent_id}'].batch['response_mask'].sum(dim=-1).tolist()
                    action_lengths_batch.append(action_lengths) #[ [bs], [bs], ...]
                    
                    # Batch rewards
                    reward_batch = (self.gamma_turn ** (1-turn_idx) * rewards_all[f'turn_{turn_idx}'][f'agent_{agent_id}'].sum(dim=-1)).tolist()
                    rewards_batch.append(reward_batch)

                # Verify rewards are equal across agents
                for i in range(batch_size):
                    assert all(rewards_batch[agent_id][i] == rewards_batch[0][i] for agent_id in range(num_agents)), "rewards for all agents should be equal"



                # Transpose actions_ids_batch to get per-sample format
                for i in range(batch_size):
                    actions_ids_list.append([actions_ids_batch[agent_id][i] for agent_id in range(num_agents)])
                    action_lengths_list.append([action_lengths_batch[agent_id][i] for agent_id in range(num_agents)])
                    rewards_list.append(rewards_batch[0][i])
                    dones_list.append(terminate)

                # Next states
                # if turn_idx < turns - 1:
                #     next_state_token_ids_batch = agent_batchs[f'turn_{turn_idx+1}']['agent_0'].batch['prompts']
                #     next_state_texts = self.tokenizer_list[0].batch_decode(next_state_token_ids_batch, skip_special_tokens=True)
                #     next_state_ids_batch = [self.mixer.text_tokenizer.encode(text, add_special_tokens=True) for text in next_state_texts]
                #     next_state_ids_list.extend(next_state_ids_batch)
                # else:

                # Terminal state for 1 and 2 turn
                terminal_text = "The conversation is over, please respond <END>."
                terminal_ids = self.mixer.text_tokenizer.encode(terminal_text, add_special_tokens=False)
                next_state_ids_list.extend([terminal_ids] * batch_size)
        
        return state_ids_list, actions_ids_list, next_state_ids_list, rewards_list, dones_list, action_lengths_list



    def confidence_norm(self, confidence_advantage, group_size=5):
        """
        简单的组内标准化
        """
        bs = confidence_advantage.shape[0]
        normalized = confidence_advantage.clone()
        
        for i in range(0, bs, group_size):
            end_idx = min(i + group_size, bs)
            group = confidence_advantage[i:end_idx]
            
            # 组内标准化
            group_mean = group.mean()
            group_std = group.std()
            normalized[i:end_idx] = (group - group_mean) / (group_std + 1e-8)
        
        return normalized


    """根据相对其他agent的confidence来额外增益adv的gain"""
    def add_relative_confidence_ratio(self, agent_batchs):
        all_confidences = []
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            confidence = agent_batch.batch["confidences"]  # shape: [bs, 1]
            all_confidences.append(confidence)

        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            current_confidence = all_confidences[agent_id]  # shape: [bs, 1]
            
            # 计算其他agent的平均confidence（排除自己）
            other_confidences = [conf for i, conf in enumerate(all_confidences) if i != agent_id]
            other_confidence_mean = torch.stack(other_confidences, dim=0).mean(dim=0)  # shape: [bs, 1]
            
            # 计算confidence优势 (当前agent - 其他agent平均)
            confidence_advantage = current_confidence - other_confidence_mean  # shape: [bs, 1]
            confidence_advantage = self.confidence_norm(confidence_advantage, self.config.actor_rollout_ref.rollout.n)

            # confidence_advantage_metrics = {f"agent_{agent_id}_confidence_advantage_mean_turn_{turn_idx}": confidence_advantage.mean().item()}
            # metrics.update(confidence_advantage_metrics)

            original_adv = agent_batch.batch["advantages"]  # shape: [bs, 1]
            
            # 计算ratio = exp(sign(adv) * alpha * confidence_advantage)
            adv_sign = torch.sign(original_adv.mean(-1))  # shape: [bs, 1]
            
            # ratio = exp(sign(adv) * alpha * confidence_advantage)
            ratio = torch.exp(adv_sign * self.config.marl.confidence_ratio_alpha * confidence_advantage)  # shape: [bs, 1]
            
            # 计算新的advantage
            new_adv = original_adv * ratio.unsqueeze(-1)  # shape: [bs, 1]
            
            # 更新batch
            # agent_batch.batch["confidence_ratio"] = ratio
            agent_batch.batch["advantages"] = new_adv
            agent_batchs[f"agent_{agent_id}"] = agent_batch



    """根据confidence来额外增益adv的gain"""
    def add_confidence_ratio(self, agent_batchs):
        all_confidences = []
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            confidence = agent_batch.batch["confidences"]  # shape: [bs, 1]
            all_confidences.append(confidence)

        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            confidence = agent_batch.batch["confidences"]  # shape: [bs, 1]


            current_confidence = all_confidences[agent_id]  # shape: [bs, 1]
            confidence_advantage = self.confidence_norm(current_confidence, self.config.actor_rollout_ref.rollout.n)


            original_adv = agent_batch.batch["advantages"]  # shape: [bs, 1]
            
            # 计算ratio = exp(sign(adv) * alpha * confidence_advantage)
            adv_sign = torch.sign(original_adv.mean(-1))  # shape: [bs, 1]
            
            # ratio = exp(sign(adv) * alpha * confidence_advantage)
            ratio = torch.exp(adv_sign * self.config.marl.confidence_ratio_alpha * confidence_advantage)  # shape: [bs, 1]
            
            # 计算新的advantage
            new_adv = original_adv * ratio.unsqueeze(-1)  # shape: [bs, 1]
            
            # 更新batch
            agent_batch.batch["advantages"] = new_adv
            agent_batchs[f"agent_{agent_id}"] = agent_batch


    """根据log prob来额外增益adv的gain"""
    def add_log_prob_ratio(self, agent_batchs):
        all_log_probs = []
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            log_prob = agent_batch.batch["old_log_probs"]  # shape: [bs, response_len]
            log_prob_mean = log_prob.mean(dim=-1)  # shape: [bs, 1]

            log_prob_mean_normalized = self.confidence_norm(log_prob_mean, self.config.actor_rollout_ref.rollout.n)


            original_adv = agent_batch.batch["advantages"]  # shape: [bs, 1]
            
            # 计算ratio = exp(sign(adv) * alpha * confidence_advantage)
            adv_sign = torch.sign(original_adv.mean(-1))  # shape: [bs, 1]
            
            # ratio = exp(sign(adv) * alpha * confidence_advantage)
            ratio = torch.exp(adv_sign * self.config.marl.log_prob_ratio_alpha * log_prob_mean_normalized)  # shape: [bs, 1]
            
            # 计算新的advantage
            new_adv = original_adv * ratio.unsqueeze(-1)  # shape: [bs, 1]
            
            # 更新batch
            agent_batch.batch["advantages"] = new_adv
            agent_batchs[f"agent_{agent_id}"] = agent_batch


    # 考虑增加mutual info loss或者intrinsic reward或者gain？
    # def mutual_info_loss(self, agent_batchs):


    
    """新增根据log prob或者confidence来增益advantage"""
    def _compute_adv(self, agent_batchs, estimate_reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics):
        kl_metrics_all = {}


        # 在计算advantage之前，先分析reward分布
        self._analyze_reward_distribution(agent_batchs, estimate_reward_tensor_all, reward_extra_infos_dict_all, metrics)
    


        for agent_id, agent in enumerate(self.mac.agents):

            agent_batch = agent_batchs[f"agent_{agent_id}"]

            # we combine with rule-based rm
            reward_extra_infos_dict: dict[str, list]
            # 使用marl.config判断reward model模式
            if self.config.reward_model.launch_reward_fn_async:
                for future_reward in future_reward_all.values():
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            else:
            # 使用各自critic估计的turn reward
                reward_tensor = estimate_reward_tensor_all[f"agent_{agent_id}"]
                # reward_extra_infos_dict = reward_extra_infos_dict_all[f"agent_{agent_id}"]

            agent_batch.batch["token_level_scores"] = reward_tensor

            # print(f"{list(reward_extra_infos_dict.keys())=}")
            # if reward_extra_infos_dict:
            #     agent_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})


            # compute rewards. apply_kl_penalty if available
            # print(f" self.config.algorithm.use_kl_in_reward {self.config.algorithm.use_kl_in_reward}")
            if self.config.algorithm.use_kl_in_reward:
                agent_batch, kl_metrics = apply_kl_penalty(agent_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                kl_metrics_all[f"agent_{agent_id}"] = kl_metrics
            else:
                agent_batch.batch["token_level_rewards"] = agent_batch.batch["token_level_scores"]

            agent_batchs[f"agent_{agent_id}"] = agent_batch


        """处理完毕两个agent的kl计算，这里暂时考虑合并两个agent的kl_metrics，考虑分开存储"""
        if self.config.algorithm.use_kl_in_reward:
            for agent_key, agent_metrics in kl_metrics_all.items():
                metrics.update({f"{agent_key}_{k}": v for k, v in agent_metrics.items()})


        # compute advantages, executed on the driver process
        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            agent_batch = compute_advantage(
                agent_batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
            )
            agent_batchs[f"agent_{agent_id}"] = agent_batch

        if self.config.marl.use_confidence_gain:
            self.add_confidence_ratio(agent_batchs)
        elif self.config.marl.use_relative_confidence_gain:
            self.add_relative_confidence_ratio(agent_batchs)
        elif self.config.marl.use_log_prob_gain:
            self.add_log_prob_ratio(agent_batchs)

        # 分析GRPO计算后的最终分布
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            agent_key = f"agent_{agent_id}"
            
            # 分析GRPO计算后的最终分布
            if "advantages" in agent_batch.batch:
                advantages = agent_batch.batch["advantages"]
                response_mask = agent_batch.batch.get("response_mask", None)
                
                if response_mask is not None:
                    final_advantages = []
                    for i in range(advantages.shape[0]):
                        response_end = response_mask[i].sum().item() - 1
                        final_advantages.append(advantages[i, response_end].item())
                    final_advantages = torch.tensor(final_advantages)
                else:
                    final_advantages = advantages.sum(dim=-1)
                
                # 记录最终的advantage分布统计
                metrics.update({
                    f"Final_Adv_Stats/{agent_key}_mean": final_advantages.mean().item(),
                    f"Final_Adv_Stats/{agent_key}_std": final_advantages.std().item(),
                    f"Final_Adv_Stats/{agent_key}_min": final_advantages.min().item(),
                    f"Final_Adv_Stats/{agent_key}_max": final_advantages.max().item(),
                })

 

    # #### todo: 目前只处理qwen 2.5 template，要检查一下phi3等其他模型的template是否适用？？
    # def remove_template(self, text):
    #     """
    #     移除文本中的模板标记，输入的是不带special token的文本
    #     text: 
    #     You are Qwen, created by Alibaba Cloud. You are a helpful assistant. \n user \n
    #     Tom receives a $12 allowance per month. In the first week, he spends a third of it; 
    #     in the second week, he spends a quarter of what he has left. 
    #     How much money does he have left to finish the month?
    #     Let\'s think step by step and output the final answer after "####".\nassistant\n

    #     return:
    #     Tom receives a $12 allowance per month. In the first week, he spends a third of it; 
    #     in the second week, he spends a quarter of what he has left. 
    #     How much money does he have left to finish the month?
    #     """

    #     try:
    #         # 先用"user\n"切分，取后半部分
    #         _, after_user = text.split("user\n")
    #         # 再用"Let's"切分，取前半部分
    #         question = after_user.split("Let\'s")[0]
    #         # 去除首尾空白
    #         return question.strip()
    #     except ValueError:
    #         # 如果切分失败（找不到分隔符），返回原文本
    #         return text.strip()


    #### todo: 新版本，适配于直接从non tensor batch 提取原始prompt问题
    # 保持与multi turn ht相同的函数
    def remove_template(self, text):
        """
        移除文本中的模板标记，输入的是不带special token的文本
        text: 
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant. \n user \n
        Tom receives a $12 allowance per month. In the first week, he spends a third of it; 
        in the second week, he spends a quarter of what he has left. 
        How much money does he have left to finish the month?
        Let\'s think step by step and output the final answer after "####".\nassistant\n

        return:
        Tom receives a $12 allowance per month. In the first week, he spends a third of it; 
        in the second week, he spends a quarter of what he has left. 
        How much money does he have left to finish the month?
        """

        try:
            # 如果qwen2.5存在系统模版，进行额外切分
            if "user\n" in text:
                text = text.split("user\n", 1)[1]
            # 去除首尾空白
            if "\nassistant\n" in text:
                text = text.split("\nassistant\n", 1)[0]

            text = text.strip()

            instruction_suffixes = [
                "Let\'s think step by step and output the final answer after \"####\".",
                "Let\'s think step by step and output the final answer within \\boxed{}.",
            ]

            for suffix in instruction_suffixes:
                if text.endswith(suffix):
                    text = text[:-len(suffix)].rstrip()
                    break

            return text
            
        except ValueError:
            return text.strip()


    
    # soft update target qmix critics
    def _update_target_critics(self, tau=0.005):
        for i in range(self.num_agents):
            for target_param, param in zip(self.mixer.target_critics_sentence[i].parameters(),
                                        self.mixer.critics_sentence[i].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


    def generate_next_actions(self, text_next_states):
        """生成next actions"""
    
        """ 可以考虑移除template，目前都采用qwen template的state """
        # text_next_states_remove_template = []
        # for text in text_next_states:
        #     remove_text = self.remove_template(text)
        #     text_next_states_remove_template.append(remove_text)

        instruction_following = 'Let\'s think step by step and output the final answer within \\boxed{}.'
        # next state的rollout actions
        """todo: 注意需要copy一个mac的actor，作为old policy来采样动作？？？？FACMAC"""
        text_next_states_with_instruction = [text + instruction_following for text in text_next_states]
        text_next_actions = []
        for agent_id in range(self.num_agents):
            agent_tokenizer = self.tokenizer_list[agent_id]
            processed_input_ids = []
            processed_attention_masks = []
            processed_position_ids = []


            # 套用模版
            prompt_strs = []
            for text in text_next_states_with_instruction:
                new_prompt_message = [{"role": "user", "content": text}]
                if self.mac.agents[agent_id].unthinking_mode:
                    new_prompt = self.mac.agents[agent_id].tokenizer.apply_chat_template(new_prompt_message, add_generation_prompt=True, tokenize=False, enable_thinking=False)
                else:
                    new_prompt = self.mac.agents[agent_id].tokenizer.apply_chat_template(new_prompt_message, add_generation_prompt=True, tokenize=False)
                prompt_strs.append(new_prompt)


            batch_next_states = agent_tokenizer(prompt_strs, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False, max_length=self.config.data.max_prompt_length)
            input_ids = batch_next_states['input_ids']  # [batch, seq_len]
            attention_mask = batch_next_states['attention_mask']

            # 不支持batch处理，会导致不同长度的prompt被编码为第一个输入的长度
            for i in range(input_ids.size(0)):
                next_state_ids, next_state_mask = verl_F.postprocess_data(
                                input_ids=input_ids[i].unsqueeze(0),
                                attention_mask=attention_mask[i].unsqueeze(0),
                                max_length=self.config.data.max_prompt_length,
                                pad_token_id=agent_tokenizer.pad_token_id,
                                left_pad=True,
                                truncation=self.config.get("truncation", "error"),
                            )
                next_state_pos_ids = compute_position_id_with_mask(next_state_mask)
                processed_input_ids.append(next_state_ids[0])
                processed_attention_masks.append(next_state_mask[0])
                processed_position_ids.append(next_state_pos_ids[0]) 


            # 将处理后的结果堆叠成批次
            batch_input_ids = torch.stack(processed_input_ids)
            batch_attention_masks = torch.stack(processed_attention_masks)
            batch_position_ids = torch.stack(processed_position_ids)


            next_state_batch_dict = {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_masks,
                'position_ids': batch_position_ids,
            }

            next_state_batch = DataProto.from_single_dict(next_state_batch_dict)

            # 用meta info控制greedy采样tmp=0，rollout.n=1, do sample=False启动确定性采样
            next_state_batch.meta_info = {
                    'eos_token_id': self.tokenizer_list[agent_id].eos_token_id,
                    'pad_token_id': self.tokenizer_list[agent_id].pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,  # 如果使用False的greedy sample会得到非常奇怪的回复
                    'validate': True,
                }

            # 采样next action ids
            gen_batch_next_actions = self.mac.agents[agent_id].actor_rollout_wg.generate_sequences(next_state_batch)

            # 解码next action ids为text
            text_next_action = self.tokenizer_list[agent_id].batch_decode(gen_batch_next_actions.batch['responses'], skip_special_tokens=True)
            text_next_actions.append(text_next_action) # list shape: [n_agents, batch]
            
        return text_next_actions


    # 更新sentence level qmix
    def update_qmix(self, qmix_data, **kwargs):
        """先从buffer采样数据，更新sentence QMIX"""

        states = qmix_data['state']           # [batch, ...]
        actions = qmix_data['actions']        # [batch, n_agents, ...]
        rewards = qmix_data['rewards']        # [batch, n_agents]
        next_states = qmix_data['next_state'] # [batch, ...]
        dones = qmix_data['done']             # [batch]
        
        text_next_states = self.mixer.text_tokenizer.batch_decode(next_states, skip_special_tokens=True)
        text_next_actions = qmix_data['text_next_actions'] # list shape: [n_agents, batch]

        # 其实不用这个with torch.no_grad()，因为embed分布式推理本身移除了梯度
        with torch.no_grad():
            # 使用sentence_transformers的模型，从文本直接转换为embedding
            # self.mixer.critic 和 self.mixer各种网络的state输入都是embed

            next_actions_embeds_cpu = []
            for agent_id in range(self.num_agents):
                # next_actions_emb = self.mixer.text_lm.encode(text_next_actions[agent_id], batch_size=self.qmix_encode_batch_size, convert_to_tensor=True)
                # next_actions_emb = next_actions_emb.detach().to('cpu')
                next_actions_emb = self._embed_encode_fn(texts=text_next_actions[agent_id],  max_batch_size=self.embedding_micro_batch_size, per_gpu_micro_bs=self.qmix_encode_batch_size)
                next_actions_embeds_cpu.append(next_actions_emb)
            # del next_actions_emb


            text_state = self.mixer.text_tokenizer.batch_decode(states, skip_special_tokens=True)
            # state_embed = self.mixer.text_lm.encode(text_state, batch_size=self.qmix_encode_batch_size, convert_to_tensor=True)  # [self.qmix_batch, 1024]  # cuda, float32, 
            # state_embed_cpu = state_embed.detach().to('cpu')  # 卸载到cpu上
            # del state_embed
            state_embed_cpu = self._embed_encode_fn(texts=text_state,  max_batch_size=self.embedding_micro_batch_size, per_gpu_micro_bs=self.qmix_encode_batch_size)
            


            # 处理 actions
            # actions 的维度是 batch, n_agents 的 list,extend变为一个batch数据再chunk回agent wise
            temp_actions = []
            for agent_idx in range(self.num_agents):
                temp_actions.extend(actions[:,agent_idx])
            text_action = self.mixer.text_tokenizer.batch_decode(temp_actions, skip_special_tokens=True)
            # action_embed_all = self.mixer.text_lm.encode(text_action, batch_size=self.qmix_encode_batch_size, convert_to_tensor=True)  # cuda, float32, 
            # action_embed_all = action_embed_all.detach().to('cpu')  # 卸载到cpu上
            action_embed_all_cpu = self._embed_encode_fn(texts=text_action,  max_batch_size=self.embedding_micro_batch_size, per_gpu_micro_bs=self.qmix_encode_batch_size)
            actions_embed_all_cpu = torch.chunk(action_embed_all_cpu, self.num_agents, dim=0)
            # del action_embed_all

            # next_state_embed = self.mixer.text_lm.encode(text_next_states, batch_size=self.qmix_encode_batch_size, convert_to_tensor=True)  # [self.qmix_batch, 1024]
            # next_state_embed_cpu = next_state_embed.detach().to('cpu')  # 卸载到cpu上
            next_state_embed_cpu = self._embed_encode_fn(texts=text_next_states,  max_batch_size=self.embedding_micro_batch_size, per_gpu_micro_bs=self.qmix_encode_batch_size)
            # del next_state_embed


        # 计算q_target
        q_targets = []
        with torch.no_grad():
            q_targets = []
            # next_state_embed_gpu = next_state_embed_cpu.to('cuda')
            for agent_id in range(self.num_agents):
                # next_action_embed_gpu = next_actions_embeds_cpu[agent_id].to('cuda')
                next_action_embed_cpu = next_actions_embeds_cpu[agent_id]
                q_target = self.mixer.target_critics_sentence[agent_id](next_state_embed_cpu, next_action_embed_cpu)  # 输入critic网络的是句子级别embedding
                q_targets.append(q_target)
            
            q_targets = torch.stack(q_targets, dim=1)   # 输入需要 bs, num, 1  # 这个很小，不用卸载
            # mixed Q total
            qmix_targets = self.mixer(q_targets, next_state_embed_cpu)

            # del next_action_embed_gpu, next_state_embed_gpu
        

        # q_value = self.critic(states, actions)
        q_values = []
        # state_embed_gpu = state_embed_cpu.to('cuda')
        for agent_id in range(self.num_agents):
            # action_embed_gpu = actions_embed_all_cpu[agent_id].to('cuda')
            action_embed_cpu = actions_embed_all_cpu[agent_id]
            q_value = self.mixer.critics_sentence[agent_id](state_embed_cpu, action_embed_cpu)
            q_values.append(q_value)
        
        q_values = torch.stack(q_values, dim=1)
        qmix_value = self.mixer(q_values, state_embed_cpu)
        # del action_embed_gpu, state_embed_gpu


        rewards_torch = torch.tensor(rewards, dtype=torch.float32, device=qmix_value.device)  # cpu
        dones_torch = torch.tensor(dones, dtype=torch.int, device=qmix_value.device)

        qtotal_targets = rewards_torch + self.gamma_turn * qmix_targets.squeeze(1) * (1-dones_torch)
        qtotal_loss = ((qtotal_targets - qmix_value.squeeze(1))**2).mean()


        self.optimizer.zero_grad()
        qtotal_loss.backward()
        self.optimizer.step()

        return qtotal_loss.detach().item()


    # 更新sentence level vdn
    # 暂时未同步修改qmix的新逻辑 todo
    def update_vdn(self, qmix_data, **kwargs):
        """先从buffer采样数据，更新sentence QMIX"""
        # qmix_data = buffer.sample(qmix_batch_size)

        states = qmix_data['state']           # [batch, ...]
        actions = qmix_data['actions']        # [batch, n_agents, ...]
        rewards = qmix_data['rewards']        # [batch, n_agents]
        next_states = qmix_data['next_state'] # [batch, ...]
        dones = qmix_data['done']             # [batch]
        
        text_next_states = self.mixer.text_tokenizer.batch_decode(next_states, skip_special_tokens=True)
        # === 可以考虑移除template，目前都采用qwen template的state ===
        # text_next_states_remove_template = []
        # for text in text_next_states:
        #     remove_text = self.remove_template(text)
        #     text_next_states_remove_template.append(remove_text)

        # next state的rollout actions
        #### todo: 注意需要copy一个mac的actor，作为old policy来采样动作？？？？FACMAC
        gen_batch_next_states = {}
        for agent_id in range(self.num_agents):
            agent_tokenizer = self.tokenizer_list[agent_id]
            processed_input_ids = []
            processed_attention_masks = []
            processed_position_ids = []

            # 都会pad到最大长度？这合理吗
            batch_next_states = agent_tokenizer(text_next_states, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False, max_length=self.config.data.max_prompt_length)
            input_ids = batch_next_states['input_ids']  # [batch, seq_len]
            attention_mask = batch_next_states['attention_mask']

            for i in range(input_ids.size(0)):
                next_state_ids, next_state_mask = verl_F.postprocess_data(
                                input_ids=input_ids[i].unsqueeze(0),
                                attention_mask=attention_mask[i].unsqueeze(0),
                                max_length=self.config.data.max_prompt_length,
                                pad_token_id=agent_tokenizer.pad_token_id,
                                left_pad=True,
                                truncation=self.config.get("truncation", "error"),
                            )
                next_state_pos_ids = compute_position_id_with_mask(next_state_mask)
                processed_input_ids.append(next_state_ids[0])
                processed_attention_masks.append(next_state_mask[0])
                processed_position_ids.append(next_state_pos_ids[0]) 


            # for next_state in text_next_states:
            #     # 这里不可以用batch输入，会导致不同长度的prompt被编码为第一个输入的长度
            #     encode_next_states = agent_tokenizer(next_state, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True, max_length=self.config.data.max_prompt_length)
            #     next_state_input_ids = encode_next_states.pop('input_ids')
            #     next_state_attn_mask = encode_next_states.pop('attention_mask')

            #     processed_ids, processed_mask = verl_F.postprocess_data(
            #                     input_ids=next_state_input_ids,
            #                     attention_mask=next_state_attn_mask,
            #                     max_length=self.config.data.max_prompt_length,
            #                     pad_token_id=agent_tokenizer.pad_token_id,
            #                     left_pad=True,
            #                     truncation=self.config.get("truncation", "error"),
            #                 )
            #     processed_pos_ids = compute_position_id_with_mask(processed_mask)

            #     # 添加到列表
            #     processed_input_ids.append(processed_ids[0])
            #     processed_attention_masks.append(processed_mask[0])
            #     processed_position_ids.append(processed_pos_ids[0])
                
            # 将处理后的结果堆叠成批次
            batch_input_ids = torch.stack(processed_input_ids)
            batch_attention_masks = torch.stack(processed_attention_masks)
            batch_position_ids = torch.stack(processed_position_ids)

            batch_dict = {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_masks,
                'position_ids': batch_position_ids,
            }

            sample_action_batch = DataProto.from_single_dict(batch_dict)
            # 用meta info控制greedy采样tmp=0，而不需要rollout.n
            sample_action_batch.meta_info = {
                    'eos_token_id': self.tokenizer_list[agent_id].eos_token_id,
                    'pad_token_id': self.tokenizer_list[agent_id].pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    'validate': True,
                }
            
            gen_batch_next_states[f'agent_{agent_id}'] = sample_action_batch

            # gen_batch_next_states[f'agent_{agent_id}'] = DataProto.from_single_dict(batch_dict)

        # 采样next state的next action, 基于当前policy
        gen_batch_outputs_all = {}
        for agent, gen_batch_agent in zip(self.mac.agents, gen_batch_next_states.values()):
            gen_batch_output = agent.actor_rollout_wg.generate_sequences(gen_batch_agent)
            gen_batch_outputs_all[f"agent_{agent.agent_id}"] = gen_batch_output

        # print("Start Encoding")

        # 编码next action为标准token ids
        next_actions_embed = []
        # === text编码为embedding必须移除计算图，否则速度慢，累计梯度 ===
        with torch.no_grad():

            text_next_actions = []
            for agent_id in range(self.num_agents):
                text_next_action = self.tokenizer_list[agent_id].batch_decode(gen_batch_outputs_all[f"agent_{agent_id}"].batch['responses'])
                text_next_actions.extend(text_next_action)

            # 使用sentence_transformers的模型，从文本直接转换为embedding
            next_action_embedding = self.mixer.text_lm.encode(text_next_actions, batch_size=self.qmix_batch, convert_to_tensor=True)  # [self.qmix_batch*agent_num, 1024]
            next_actions_embed = torch.chunk(next_action_embedding, self.num_agents, dim=0)
            # next_actions_embed.append(next_action_embedding.detach())

            # === 基于当前策略采样的next q ===
            # self.mixer.critic 和 self.mixer各种网络的state输入都是embed
            text_state = self.mixer.text_tokenizer.batch_decode(states, skip_special_tokens=True)
            state_embed = self.mixer.text_lm.encode(text_state, batch_size=self.qmix_batch, convert_to_tensor=True)  # [self.qmix_batch, 1024]

            # actions 的维度是 batch, n_agents 的 list
            
            # 支持任意agent数量
            # action_embed = []

            temp_actions = []
            for agent_idx in range(self.num_agents):
                temp_actions.extend(actions[:,agent_idx])
            # temp_actions = []
            # temp_actions.extend(actions[:,0])
            # temp_actions.extend(actions[:,1])
            # for agent_idx in range(self.num_agents):
            text_action = self.mixer.text_tokenizer.batch_decode(temp_actions, skip_special_tokens=True)
            action_embed_all = self.mixer.text_lm.encode(text_action, batch_size=self.qmix_batch, convert_to_tensor=True)
            action_embed = torch.chunk(action_embed_all, self.num_agents, dim=0)

            next_state_embed = self.mixer.text_lm.encode(text_next_states, batch_size=self.qmix_batch, convert_to_tensor=True)  # [self.qmix_batch, 1024]


        # embed不训练，重新设置grad
        state_embed = state_embed.clone().detach().requires_grad_()
        next_state_embed = next_state_embed.clone().detach().requires_grad_()
        action_embed = [x.clone().detach().requires_grad_() for x in action_embed]
        next_actions_embed = [x.clone().detach().requires_grad_() for x in next_actions_embed]


        # Start VDN
        # 计算q_target
        q_targets = []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                next_action = next_actions_embed[agent_id]
                q_target = self.mixer.target_critics_sentence[agent_id](next_state_embed, next_action)  # 输入critic网络的是句子级别embedding
                q_targets.append(q_target)
            
            q_targets = torch.stack(q_targets, dim=1)   # 输入需要 bs, num, 1


            # mixed Q total
            vdn_targets = self.mixer(q_targets)
            # qmix_targets = self.mixer(q_targets, next_state_embed)

        # q_value = self.critic(states, actions)
        q_values = []
        for agent_id in range(self.num_agents):
            q_value = self.mixer.critics_sentence[agent_id](state_embed, action_embed[agent_id])
            q_values.append(q_value)

        q_values = torch.stack(q_values, dim=1)
        vdn_value = self.mixer(q_values)


        rewards_torch = torch.tensor(rewards, dtype=torch.float32, device=vdn_value.device)  # cpu
        dones_torch = torch.tensor(dones, dtype=torch.int, device=vdn_value.device)

        qtotal_targets = rewards_torch + self.gamma_turn * vdn_targets.squeeze(1) * (1-dones_torch)
        qtotal_loss = ((qtotal_targets - vdn_value.squeeze(1))**2).mean()


        self.optimizer.zero_grad()
        qtotal_loss.backward()
        self.optimizer.step()

        return qtotal_loss.detach().item()


    """
    改成multi turn的
    对于不需要计算sentence level Q的，对每个turn的reward做sentence level discount，依次独立处理每个turn的训练
    对于需要计算sentence level Q的，需要额外设置一个 sentence level state+action 的Critic网络，接收embedding，输出sentence Q value作为每一轮的turn reward，再依次处理每个turn的训练
    """
    def train(self, multi_turn_batchs, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, buffer, metrics, global_steps, timing_raw):
        
        # 1. 更新qmix
        if buffer.size > self.config.marl.train_start_size:
            with _timer("qmix_update", timing_raw):
                # 支持更大batch size qmix，进行多次更新
                # target_samples = 700  # 目标使用的样本数, 必须是 qmix_batch_size 的倍数
                target_samples = self.qmix_batch * self.qmix_update_steps
                actual_samples = min(target_samples, buffer.size)

                # 一次性采样大量数据
                large_qmix_data = buffer.sample(actual_samples)

                # 统计采样样本长度分布
                action_lengths = large_qmix_data['action_lengths'] # [qmix_bs, num_agents]
                short_th = 50
                for agent_id in range(self.num_agents):
                    agent_samples = action_lengths[:, agent_id].astype(np.float32)
                    metrics[f"response_length/qmix_sample_agent_{agent_id}_mean"] = float(np.mean(agent_samples))
                    metrics[f"response_length/qmix_sample_agent_{agent_id}_min"] = float(np.min(agent_samples))
                    metrics[f"response_length/qmix_sample_agent_{agent_id}_max"] = float(np.max(agent_samples))
                    metrics[f"response_length/qmix_sample_agent_{agent_id}_short_ratio"] = float(np.mean(agent_samples < short_th))


                # 分批处理以避免内存问题
                # num_mini_batches = (actual_samples + self.qmix_batch - 1) // self.qmix_batch

                total_qmix_loss = 0

                next_states = large_qmix_data['next_state'] # [batch, ...]
                text_next_states = self.mixer.text_tokenizer.batch_decode(next_states, skip_special_tokens=True)
                # 生成next action  # 1024 bs下基本上只有50G占用，可以开大
                text_next_actions = self.generate_next_actions(text_next_states)

                # 切分成n次更新的qmix数据
                assert actual_samples % self.qmix_batch == 0
                qmix_update_steps = actual_samples // self.qmix_batch
                for qmix_update_step in range(qmix_update_steps):
                    start_idx = qmix_update_step * self.qmix_batch
                    end_idx = min(start_idx + self.qmix_batch, actual_samples)
                    
                    qmix_batch_data = {
                        'state': large_qmix_data['state'][start_idx:end_idx],
                        'actions': large_qmix_data['actions'][start_idx:end_idx],
                        'rewards': large_qmix_data['rewards'][start_idx:end_idx],
                        'next_state': large_qmix_data['next_state'][start_idx:end_idx],
                        'done': large_qmix_data['done'][start_idx:end_idx],
                        'text_next_actions': [text_next_actions[agent_id][start_idx:end_idx] for agent_id in range(self.num_agents)],
                    }


                    if self.config.marl.name == "qmix":
                        mini_batch_loss = self.update_qmix(qmix_batch_data)
                    elif self.config.marl.name == "vdn":
                        mini_batch_loss = self.update_vdn(qmix_batch_data)
                    
                    total_qmix_loss += mini_batch_loss
                    self.qmix_update_count += 1

                    # self.config.trainer.qmix_update_interval
                    if self.qmix_update_count % 40 == 0:
                        self._update_target_critics()

                if self.config.marl.name == "qmix":
                    metrics["total/qmix_loss"] = total_qmix_loss 
                elif self.config.marl.name == "vdn":
                    metrics["total/vdn_loss"] = total_qmix_loss

                # self.embed_wg.unload_model()
                
            if self.qmix_update_count >= self.config.marl.qmix_critic_warmup:
                print(f"qmix critic warmup done, begin llm policy update, qmix update count: {self.qmix_update_count}")

                # 2. 利用更新后的critic，评估当前multi turn batchs每句话action的value作为reward，用于更新policy，标准PPO
                """每个turn、每个agent计算"""
                for turn_idx in range(self.turns):

                    # turn_idx = 1  # 用于测试

                    agent_batchs = multi_turn_batchs[f"turn_{turn_idx}"]

                    ## === 这个 bs 需要调整成适配grpo的动态bs,或者标准的balance后的bs ===
                    onpolicy_batch_size = len(agent_batchs['agent_0'].batch['prompts'])


                    states_text = self.tokenizer_list[0].batch_decode(agent_batchs['agent_0'].batch['prompts'], skip_special_tokens=True)
                    states_embed = self._embed_encode_fn(texts=states_text,  max_batch_size=self.embedding_micro_batch_size, per_gpu_micro_bs=self.qmix_encode_batch_size)
                    # with torch.no_grad():   
                    #     # states_embed = self.encode_in_batches(states_text, max_batch_size=state_embed_max_batch_size)
                    #     states_embed = self.mixer.text_lm.encode(states_text, batch_size=self.qmix_encode_batch_size, convert_to_tensor=True)  # [train_batch_size, 1024]
                    # states_embed = states_embed.clone().detach()

                    turn_credits = {}
                    all_qi_values = []
                    # 用critic计算turn score，更新token critic PPO adv
                    for agent_id in range(self.num_agents):
                        agent_batch = agent_batchs[f"agent_{agent_id}"]
                        actions_text = self.tokenizer_list[agent_id].batch_decode(agent_batch.batch['responses'], skip_special_tokens=True)
                        actions_embed = self._embed_encode_fn(texts=actions_text,  max_batch_size=self.embedding_micro_batch_size, per_gpu_micro_bs=self.qmix_encode_batch_size)
                        # with torch.no_grad():
                        #     # actions_embed = self.encode_in_batches(actions_text, max_batch_size=state_embed_max_batch_size)
                        #     actions_embed = self.mixer.text_lm.encode(actions_text, batch_size=self.qmix_encode_batch_size, convert_to_tensor=True)  # [train_batch_size, 1024]
                        # actions_embed = actions_embed.clone().detach()

                        # 计算q_value,小心有时候会oom
                        estimate_q_values = self.mixer.critics_sentence[agent_id](states_embed, actions_embed).detach()

                        # # 分批计算q_value
                        # estimate_q_values = []
                        # for i in range(0, len(states_embed), self.qmix_encode_batch_size):
                        #     batch_states = states_embed[i:i+self.qmix_encode_batch_size].to(self.device)
                        #     batch_actions = actions_embed[i:i+self.qmix_encode_batch_size].to(self.device)
                        #     with torch.no_grad():
                        #         batch_q_value = self.mixer.critics_sentence[agent_id](batch_states, batch_actions).detach()
                        #     estimate_q_values.append(batch_q_value)
                        # estimate_q_values = torch.cat(estimate_q_values, dim=0)

                        all_qi_values.append(estimate_q_values)

                        # todo === 这里要考虑一个问题，最后一轮的奖励用原始奖励还是critic估计奖励，虽然我觉得critic可能会快速收敛？ ===
                        turn_credit = estimate_q_values.squeeze(-1)  # .to('cpu')   # cuda embed model tensor转换回cpu

                        # 添加 response length 惩罚
                        response_mask = agent_batch.batch['response_mask']  # [bs, seq_len]
                        response_lengths = response_mask.sum(dim=-1).float()  # [bs]
                        min_length = 50  # 最小响应长度
                        max_length = self.config.data.max_response_length
                        length_penalty_coeff = self.config.marl.length_penalty_coeff  # 惩罚系数

                        # 归一到[0,1]后得到[1,0]的惩罚，再乘系数
                        length_penalty = (1 - ((response_lengths - min_length)/(max_length - min_length)).clamp(0.0, 1.0)) * length_penalty_coeff
                        # 将长度惩罚加到 turn_credit
                        turn_credit -= length_penalty

                        # 记录统计
                        metrics[f"rewards_length_penalty/agent_{agent_id}_turn_{turn_idx}_length_penalty_mean"] = length_penalty.mean().item()
                        metrics[f"rewards_length_penalty/agent_{agent_id}_turn_{turn_idx}_length_penalty_std"] = length_penalty.std(unbiased=False).item()
                        metrics[f"rewards_length_penalty/agent_{agent_id}_turn_{turn_idx}_length_penalty_max"] = length_penalty.max().item()
                        metrics[f"rewards_length_penalty/agent_{agent_id}_turn_{turn_idx}_length_penalty_min"] = length_penalty.min().item()

                        # 记录回复长度统计
                        metrics[f"response_length/rollout_sample_agent_{agent_id}_mean"] = response_lengths.mean().item()
                        metrics[f"response_length/rollout_sample_agent_{agent_id}_min"] = response_lengths.min().item()
                        metrics[f"response_length/rollout_sample_agent_{agent_id}_max"] = response_lengths.max().item()
                        metrics[f"response_length/rollout_sample_agent_{agent_id}_short_ratio"] = (response_lengths < min_length).float().mean().item()


                        # credit放在reward tensor中
                        turn_reward_tensor = reward_tensor_all[f"turn_{turn_idx}"][f"agent_{agent_id}"]
                        turn_reward_tensor = turn_reward_tensor.clone()
                        # 新版本reward按照最后一个mask有效token处理
                        last_valid_token_position = agent_batch.batch['response_mask'].sum(-1) - 1
                        turn_reward_tensor[torch.arange(onpolicy_batch_size), last_valid_token_position] = turn_credit  

                        # 旧版本reward按照最后一个token处理，但是有点不够精确
                        # seq_len = turn_reward_tensor.shape[1]
                        # turn_reward_tensor[torch.arange(onpolicy_batch_size), seq_len - 1] = turn_credit
                        # reward tensor 的维度是 bs, response_len，但是只有最后一个位置有reward，其他都是0

                        turn_credits[f"agent_{agent_id}"] = turn_reward_tensor
                        agent_batch.batch['estimated_rewards'] = turn_reward_tensor  # 记录估计的credits reward

                        # 记录 critic 估计的 rewards
                        turn_mean_reward = turn_reward_tensor.sum() / onpolicy_batch_size
                        metrics[f"rewards_estimated/agent_{agent_id}_turn_{turn_idx}"] = turn_mean_reward.item()

                    # 用于验证qmix估计
                    all_qi_values = torch.stack(all_qi_values, dim=1)   # 输入需要 bs, num, 1
                    with torch.no_grad():
                        if self.config.marl.name == "qmix":
                            qtotal_estimated = self.mixer(all_qi_values, states_embed)
                        elif self.config.marl.name == "vdn":
                            qtotal_estimated = self.mixer(all_qi_values)
                    # qtotal_estimated = self.mixer(all_qi_values, states_embed)
                    qtotal_estimated = qtotal_estimated.squeeze(1)  
                    metrics[f"rewards_estimated/q_total_turn_{turn_idx}_mean"] = qtotal_estimated.mean().item()
                    metrics[f"rewards_estimated/q_total_turn_{turn_idx}_std"] = qtotal_estimated.std(unbiased=False).item()
                    metrics[f"rewards_estimated/q_total_turn_{turn_idx}_max"] = qtotal_estimated.max().item()
                    metrics[f"rewards_estimated/q_total_turn_{turn_idx}_min"] = qtotal_estimated.min().item()

                    # del states_embed, actions_embed

                    # 卸载embedding model 显存
                    # self.embed_wg.unload_model()

                    print('Begin Old Log Prob: Turn', turn_idx)
                    """每个agent计算自己的old log prob存储到metric agent id"""
                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        self._compute_old_log_prob(agent_batchs, metrics, turn_idx)


                    print('Begin Ref Log Prob: Turn', turn_idx)
                    """计算reference log prob"""
                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            self._compute_ref_log_prob(agent_batchs)


                    # 为每个agent计算values，用于adv计算
                    if self.use_critic: 
                        print('Begin Critic: Turn', turn_idx)
                        with _timer("values", timing_raw):  
                            self._compute_values(agent_batchs)

                    # 传入 turn credits all
                    # reward_extra_infos_dict_all = reward_extra_infos_dict_all["turn_0"]
                    # future_reward_all = future_reward_all["turn_0"]

                    print('Begin Update Adv: Turn', turn_idx)
                    with _timer("adv", timing_raw):
                        # 增加两阶段训练，先用team reward，再换成turn credits
                        if self.config.marl.two_stage_train:
                            if global_steps < self.config.marl.two_stage_train_steps:
                                self._compute_adv(agent_batchs, reward_tensor_all[f"turn_{turn_idx}"], reward_extra_infos_dict_all, future_reward_all, metrics)
                            else:
                                self._compute_adv(agent_batchs, turn_credits, reward_extra_infos_dict_all, future_reward_all, metrics)
                        else:
                            self._compute_adv(agent_batchs, turn_credits, reward_extra_infos_dict_all, future_reward_all, metrics)


                    # 这里 reward tensor turn已经转化为critic sentence value
                    # update critic
                    if self.use_critic:
                        
                        with _timer('update_critic', timing_raw):
                            critic_output_metrics_all = {}
                            for agent_id, agent in enumerate(self.mac.agents):
                                agent_batch = agent_batchs[f"agent_{agent_id}"]
                                print('Begin Update Critic: Turn', turn_idx, 'Agent', agent_id)

                                critic_output = agent.critic_wg.update_critic(agent_batch)

                                critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                                critic_output_metrics_all[f"agent_{agent_id}"] = critic_output_metrics

                            for agent_key, agent_metrics in critic_output_metrics_all.items():
                                metrics.update({f"{agent_key}_{k}": v for k, v in agent_metrics.items()})

                        
                        
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            agent_output_metrics_all = {}
                            for agent_id, agent in enumerate(self.mac.agents):
                                agent_batch = agent_batchs[f"agent_{agent_id}"]

                                print('Begin Update Actor: Turn', turn_idx, 'Agent', agent_id)

                                agent_batch.meta_info["multi_turn"] = agent.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = agent.actor_rollout_wg.update_actor(agent_batch)

                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                agent_output_metrics_all[f"agent_{agent_id}"] = actor_output_metrics

                                # 更新了meta info，覆盖一下
                                agent_batchs[f"agent_{agent_id}"] = agent_batch

                        # 分别存储每个agent的actor metrics
                        for agent_key, agent_metrics in agent_output_metrics_all.items():
                            metrics.update({f"{agent_key}_{k}": v for k, v in agent_metrics.items()})

                    multi_turn_batchs[f"turn_{turn_idx}"] = agent_batchs

        

    # 用于创建learner使用的embed model，目前仅支持qwen3-embedding-0.6b
    def init_workers(self, resource_pool_manager: ResourcePoolManager, ray_worker_group_cls: RayWorkerGroup):


        pool = resource_pool_manager.get_global_resource_pool(MARLRole["Embed"])

        # 2) 封装 worker 类
        worker_dict = {"embed": RayClassWithInitArgs(cls=ray.remote(EmbedWorker), config=self.config)}
        embed_cls = create_colocated_worker_cls(class_dict=worker_dict)


        # 3) 创建并 spawn
        wg = ray_worker_group_cls(resource_pool=pool, ray_cls_with_init=embed_cls)
        spawned = wg.spawn(prefix_set=worker_dict.keys())
        self.embed_wg = spawned["embed"]

        # 4) 初始化模型
        self.embed_wg.init_model(model_name="/root/models/qwen3-embedding-0.6b", normalize=True)



    # 5) 分布式编码客户端（DP_COMPUTE_PROTO）
    # 需要在这里做mini batch split
    def _embed_encode_fn(self, texts: list[str], max_batch_size: int = 256, per_gpu_micro_bs: int = 8):
        if len(texts) == 0:
            return torch.empty(0, dtype=torch.float32)

        
        all_embeddings = []
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i+max_batch_size]
            
            # 创建当前批次的数据
            dummy_tensor = torch.zeros(len(batch_texts), 1)
            data = DataProto.from_dict(
                tensors={"dummy": dummy_tensor},
                non_tensors={"texts": np.array(batch_texts, dtype=object)},
                meta_info={"micro_batch_size": per_gpu_micro_bs}
            )
            
            # 调用embed workers处理当前批次
            out = self.embed_wg.encode_texts(data)
            batch_embeddings = out.batch["embeddings"]
            
            # 添加到结果列表，保持顺序
            all_embeddings.append(batch_embeddings)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)

        return all_embeddings
            
            # # 清理内存
            # del data, out, batch_embeddings
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
        
        # # 创建一个与 texts 长度相同的虚拟 tensor
        # dummy_tensor = torch.zeros(len(texts), 1)  # shape: [len(texts), 1]
        # data = DataProto.from_dict(
        #     tensors={"dummy": dummy_tensor},
        #     non_tensors={"texts": np.array(texts, dtype=object)},
        #     meta_info={"micro_batch_size": per_gpu_micro_bs}
        # )
        # out = self.embed_wg.encode_texts(data)


        # return out.batch["embeddings"]  # CPU Tensor




    # 测试
    # def _compute_embedding_diversity_metrics(self, states_embed, actions_embed_all, metrics, prefix="qmix"):
    #     """
    #     计算embedding特征的多样性指标，用于诊断state和action embedding的区分度
        
    #     Args:
    #         states_embed: state embeddings [batch_size, embed_dim]
    #         actions_embed_all: list of action embeddings for each agent [agent_embeds]
    #         metrics: 用于记录指标的字典
    #         prefix: 指标名称前缀
    #     """
    #     import torch.nn.functional as F
        
    #     batch_size = states_embed.size(0)
    #     embed_dim = states_embed.size(1)
        
    #     # 1. State embedding 内部相似度统计
    #     # 计算所有state pairs之间的cosine相似度
    #     states_norm = F.normalize(states_embed, p=2, dim=1)
    #     state_similarity_matrix = torch.mm(states_norm, states_norm.t())  # [batch_size, batch_size]
        
    #     # 排除对角线元素（自己与自己的相似度）
    #     mask = torch.eye(batch_size, device=state_similarity_matrix.device).bool()
    #     state_similarities = state_similarity_matrix[~mask]
        
    #     metrics[f"{prefix}/state_embed_similarity_mean"] = state_similarities.mean().item()
    #     metrics[f"{prefix}/state_embed_similarity_std"] = state_similarities.std().item()
    #     metrics[f"{prefix}/state_embed_similarity_max"] = state_similarities.max().item()
    #     metrics[f"{prefix}/state_embed_similarity_min"] = state_similarities.min().item()
        
    #     # 高相似度比例（>0.8认为是高相似）
    #     high_sim_ratio = (state_similarities > 0.8).float().mean().item()
    #     metrics[f"{prefix}/state_embed_high_similarity_ratio"] = high_sim_ratio
        
    #     # 2. Action embedding 内部相似度统计（每个agent分别统计）
    #     for agent_id, action_embed in enumerate(actions_embed_all):
    #         actions_norm = F.normalize(action_embed, p=2, dim=1)
    #         action_similarity_matrix = torch.mm(actions_norm, actions_norm.t())
            
    #         # 排除对角线元素
    #         mask = torch.eye(batch_size, device=action_similarity_matrix.device).bool()
    #         action_similarities = action_similarity_matrix[~mask]
            
    #         metrics[f"{prefix}/action_embed_agent_{agent_id}_similarity_mean"] = action_similarities.mean().item()
    #         metrics[f"{prefix}/action_embed_agent_{agent_id}_similarity_std"] = action_similarities.std().item()
    #         metrics[f"{prefix}/action_embed_agent_{agent_id}_similarity_max"] = action_similarities.max().item()
    #         metrics[f"{prefix}/action_embed_agent_{agent_id}_similarity_min"] = action_similarities.min().item()
            
    #         high_sim_ratio = (action_similarities > 0.8).float().mean().item()
    #         metrics[f"{prefix}/action_embed_agent_{agent_id}_high_similarity_ratio"] = high_sim_ratio
        
    #     # 3. State-Action embedding 跨模态相似度统计
    #     for agent_id, action_embed in enumerate(actions_embed_all):
    #         # 计算state和action之间的cosine相似度
    #         state_action_similarities = F.cosine_similarity(states_embed, action_embed, dim=1)
            
    #         metrics[f"{prefix}/state_action_agent_{agent_id}_similarity_mean"] = state_action_similarities.mean().item()
    #         metrics[f"{prefix}/state_action_agent_{agent_id}_similarity_std"] = state_action_similarities.std().item()
    #         metrics[f"{prefix}/state_action_agent_{agent_id}_similarity_max"] = state_action_similarities.max().item()
    #         metrics[f"{prefix}/state_action_agent_{agent_id}_similarity_min"] = state_action_similarities.min().item()
        
    #     # 4. 不同agent之间的action embedding相似度
    #     if len(actions_embed_all) > 1:
    #         for i in range(len(actions_embed_all)):
    #             for j in range(i + 1, len(actions_embed_all)):
    #                 agent_i_norm = F.normalize(actions_embed_all[i], p=2, dim=1)
    #                 agent_j_norm = F.normalize(actions_embed_all[j], p=2, dim=1)
    #                 cross_agent_similarities = F.cosine_similarity(agent_i_norm, agent_j_norm, dim=1)
                    
    #                 metrics[f"{prefix}/cross_agent_{i}_{j}_action_similarity_mean"] = cross_agent_similarities.mean().item()
    #                 metrics[f"{prefix}/cross_agent_{i}_{j}_action_similarity_std"] = cross_agent_similarities.std().item()
    #                 metrics[f"{prefix}/cross_agent_{i}_{j}_action_similarity_max"] = cross_agent_similarities.max().item()
    #                 metrics[f"{prefix}/cross_agent_{i}_{j}_action_similarity_min"] = cross_agent_similarities.min().item()
        
    #     # 5. Embedding 范数统计（检测是否存在梯度消失或爆炸）
    #     metrics[f"{prefix}/state_embed_norm_mean"] = torch.norm(states_embed, p=2, dim=1).mean().item()
    #     metrics[f"{prefix}/state_embed_norm_std"] = torch.norm(states_embed, p=2, dim=1).std().item()
        
    #     for agent_id, action_embed in enumerate(actions_embed_all):
    #         action_norms = torch.norm(action_embed, p=2, dim=1)
    #         metrics[f"{prefix}/action_embed_agent_{agent_id}_norm_mean"] = action_norms.mean().item()
    #         metrics[f"{prefix}/action_embed_agent_{agent_id}_norm_std"] = action_norms.std().item()
        
    #     # 6. 计算embedding的方差（检测特征是否过于集中）
    #     state_var = torch.var(states_embed, dim=0).mean().item()  # 对每个维度求方差，再求平均
    #     metrics[f"{prefix}/state_embed_variance"] = state_var
        
    #     for agent_id, action_embed in enumerate(actions_embed_all):
    #         action_var = torch.var(action_embed, dim=0).mean().item()
    #         metrics[f"{prefix}/action_embed_agent_{agent_id}_variance"] = action_var

    

    def _analyze_reward_distribution(self, agent_batchs, reward_tensor_all, reward_extra_infos_dict_all, metrics):
        """
        分析QMIX critic分配的reward与真实reward的分布差异，以及GRPO计算前后的分布变化
        
        Args:
            agent_batchs: 每个agent的batch数据
            reward_tensor_all: QMIX critic分配的reward
            reward_extra_infos_dict_all: 包含真实reward信息的字典
            metrics: 用于记录分析结果的metrics字典
        """
        
        
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            agent_key = f"agent_{agent_id}"
            
            # 获取QMIX critic分配的reward
            qmix_rewards = reward_tensor_all[agent_key]  # shape: (bs, seq_len)
            
            # 获取真实reward信息（如果可用）
            true_rewards = None
            if reward_extra_infos_dict_all and agent_key in reward_extra_infos_dict_all:
                reward_extra_info = reward_extra_infos_dict_all[agent_key]
                # 尝试从reward_extra_info中获取真实reward
                if "score" in reward_extra_info:
                    true_rewards = torch.tensor(reward_extra_info["score"], dtype=torch.float32)
                elif "reward" in reward_extra_info:
                    true_rewards = torch.tensor(reward_extra_info["reward"], dtype=torch.float32)
            
            # 获取uid信息用于分组分析
            uids = agent_batch.non_tensor_batch.get("uid", None)
            
            # 计算response级别的reward（每个样本的最终reward）
            response_mask = agent_batch.batch.get("response_mask", None)
            if response_mask is not None:
                # 使用response_mask来获取每个response的最终reward
                qmix_response_rewards = []
                for i in range(qmix_rewards.shape[0]):
                    # 找到response的最后一个token位置
                    response_end = response_mask[i].sum().item() - 1
                    qmix_response_rewards.append(qmix_rewards[i, response_end].item())
                qmix_response_rewards = torch.tensor(qmix_response_rewards)
            else:
                # 如果没有response_mask，使用最后一个非零位置
                qmix_response_rewards = qmix_rewards.sum(dim=-1)  # 简单求和作为response reward
            
            # 1. 分析QMIX critic reward分布
            qmix_mean = qmix_response_rewards.mean().item()
            qmix_std = qmix_response_rewards.std().item()
            qmix_min = qmix_response_rewards.min().item()
            qmix_max = qmix_response_rewards.max().item()
            
            metrics.update({
                f"qmix_stats/{agent_key}_critic_reward_mean": qmix_mean,
                f"qmix_stats/{agent_key}_critic_reward_std": qmix_std,
                f"qmix_stats/{agent_key}_critic_reward_min": qmix_min,
                f"qmix_stats/{agent_key}_critic_reward_max": qmix_max,
            })
            
            # 2. 分析真实reward分布（如果可用）
            # if true_rewards is not None:
            #     true_mean = true_rewards.mean().item()
            #     true_std = true_rewards.std().item()
            #     true_min = true_rewards.min().item()
            #     true_max = true_rewards.max().item()
                
            #     # 计算QMIX reward与真实reward的差异
            #     if len(true_rewards) == len(qmix_response_rewards):
            #         reward_diff = (qmix_response_rewards - true_rewards).abs().mean().item()
            #         reward_corr = torch.corrcoef(torch.stack([qmix_response_rewards, true_rewards]))[0, 1].item()
                    
            #         metrics.update({
            #             f"{agent_key}_true_reward_mean": true_mean,
            #             f"{agent_key}_true_reward_std": true_std,
            #             f"{agent_key}_true_reward_min": true_min,
            #             f"{agent_key}_true_reward_max": true_max,
            #             f"{agent_key}_reward_diff_mean": reward_diff,
            #             f"{agent_key}_reward_correlation": reward_corr,
            #         })
            
            # 3. 按uid分组分析QMIX reward分布
            if uids is not None:
                uid2qmix_rewards = defaultdict(list)
                for i, uid in enumerate(uids):
                    uid2qmix_rewards[uid].append(qmix_response_rewards[i].item())
                
                # 计算每个uid组内的统计信息
                uid_group_stats = {}
                for uid, rewards in uid2qmix_rewards.items():
                    if len(rewards) > 1:  # 只分析有多个样本的组
                        rewards_tensor = torch.tensor(rewards)
                        uid_group_stats[uid] = {
                            'mean': rewards_tensor.mean().item(),
                            'std': rewards_tensor.std().item(),
                            'count': len(rewards)
                        }
                
                # 计算所有组的平均统计信息
                if uid_group_stats:
                    avg_group_mean = np.mean([stats['mean'] for stats in uid_group_stats.values()])
                    avg_group_std = np.mean([stats['std'] for stats in uid_group_stats.values()])
                    avg_group_count = np.mean([stats['count'] for stats in uid_group_stats.values()])
                    
                    metrics.update({
                        f"qmix_groups_stats/{agent_key}_uid_groups_count": len(uid_group_stats),
                        f"qmix_groups_stats/{agent_key}_avg_group_mean": avg_group_mean,
                        f"qmix_groups_stats/{agent_key}_avg_group_std": avg_group_std,
                        f"qmix_groups_stats/{agent_key}_avg_group_size": avg_group_count,
                    })
            
            # 4. 分析GRPO计算前的reward分布（token_level_rewards）
            # if "token_level_rewards" in agent_batch.batch:
            #     grpo_input_rewards = agent_batch.batch["token_level_rewards"]
            #     if response_mask is not None:
            #         grpo_input_response_rewards = []
            #         for i in range(grpo_input_rewards.shape[0]):
            #             response_end = response_mask[i].sum().item() - 1
            #             grpo_input_response_rewards.append(grpo_input_rewards[i, response_end].item())
            #         grpo_input_response_rewards = torch.tensor(grpo_input_response_rewards)
            #     else:
            #         grpo_input_response_rewards = grpo_input_rewards.sum(dim=-1)
                
            #     grpo_input_mean = grpo_input_response_rewards.mean().item()
            #     grpo_input_std = grpo_input_response_rewards.std().item()
                
            #     metrics.update({
            #         f"{agent_key}_grpo_input_reward_mean": grpo_input_mean,
            #         f"{agent_key}_grpo_input_reward_std": grpo_input_std,
            #     })
            
            # 5. 分析GRPO计算后的advantage分布
            if "advantages" in agent_batch.batch:
                advantages = agent_batch.batch["advantages"]
                if response_mask is not None:
                    grpo_output_advantages = []
                    for i in range(advantages.shape[0]):
                        response_end = response_mask[i].sum().item() - 1
                        grpo_output_advantages.append(advantages[i, response_end].item())
                    grpo_output_advantages = torch.tensor(grpo_output_advantages)
                else:
                    grpo_output_advantages = advantages.sum(dim=-1)
                
                grpo_output_mean = grpo_output_advantages.mean().item()
                grpo_output_std = grpo_output_advantages.std().item()
                grpo_output_min = grpo_output_advantages.min().item()
                grpo_output_max = grpo_output_advantages.max().item()

                metrics.update({
                    f"grpo_stats/{agent_key}_grpo_adv_mean": grpo_output_mean,
                    f"grpo_stats/{agent_key}_grpo_adv_std": grpo_output_std,
                    f"grpo_stats/{agent_key}_grpo_adv_min": grpo_output_min,
                    f"grpo_stats/{agent_key}_grpo_adv_max": grpo_output_max,
                })
                
                # # 计算GRPO放大倍数（如果输入输出都有的话）
                # if "token_level_rewards" in agent_batch.batch:
                #     grpo_amplification = grpo_output_std / (grpo_input_std + 1e-8)
                #     metrics[f"{agent_key}_grpo_amplification"] = grpo_amplification
            
            # 6. 按uid分组分析GRPO后的advantage分布
            # if uids is not None and "advantages" in agent_batch.batch:
            #     uid2advantages = defaultdict(list)
            #     for i, uid in enumerate(uids):
            #         if response_mask is not None:
            #             response_end = response_mask[i].sum().item() - 1
            #             adv_value = advantages[i, response_end].item()
            #         else:
            #             adv_value = advantages[i].sum().item()
            #         uid2advantages[uid].append(adv_value)
                
            #     # 计算每个uid组内的advantage统计信息
            #     uid_adv_stats = {}
            #     for uid, advs in uid2advantages.items():
            #         if len(advs) > 1:
            #             advs_tensor = torch.tensor(advs)
            #             uid_adv_stats[uid] = {
            #                 'mean': advs_tensor.mean().item(),
            #                 'std': advs_tensor.std().item(),
            #                 'count': len(advs)
            #             }
                
            #     if uid_adv_stats:
            #         avg_adv_group_mean = np.mean([stats['mean'] for stats in uid_adv_stats.values()])
            #         avg_adv_group_std = np.mean([stats['std'] for stats in uid_adv_stats.values()])
                    
            #         metrics.update({
            #             f"{agent_key}_uid_adv_groups_count": len(uid_adv_stats),
            #             f"{agent_key}_avg_adv_group_mean": avg_adv_group_mean,
            #             f"{agent_key}_avg_adv_group_std": avg_adv_group_std,
            #         })



    def save_qmix_network(self, save_path: str):
        """
        保存 QMIX/VDN 相关网络权重，用于离线估值：
        - critics_sentence（每个agent一个）
        - target_critics_sentence（每个agent一个）
        - 当 config.marl.name == "qmix" 时，保存超网络 hyper_w_1/hyper_w_final/hyper_b_1/V
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 元信息，便于离线加载与一致性检查
        meta = {
            "marl_name": self.config.marl.name,
            "num_agents": self.num_agents,
            "text_lm_name": getattr(self.mixer, "text_lm_name", None),
            "text_lm_embedding_dim": getattr(self.mixer, "text_lm_embedding_dim", None),
            "gamma_turn": self.gamma_turn,
        }

        # 保存 mixer（仅在 QMIX 下有超网络；VDN 不需要）
        mixer_state = {}
        if self.config.marl.name == "qmix":
            mixer_state = {
                "hyper_w_1": self.mixer.hyper_w_1.state_dict(),
                "hyper_w_final": self.mixer.hyper_w_final.state_dict(),
                "hyper_b_1": self.mixer.hyper_b_1.state_dict(),
                "V": self.mixer.V.state_dict(),
            }

        # 每个 agent 的 critic/target_critic
        critics_state = [critic.state_dict() for critic in self.mixer.critics_sentence]
        target_critics_state = [critic.state_dict() for critic in self.mixer.target_critics_sentence]

        payload = {
            "meta": meta,
            "mixer": mixer_state,
            "critics_sentence": critics_state,
            "target_critics_sentence": target_critics_state,
        }

        torch.save(payload, save_path)


    # 这里需要注意，加权平均的距离在语义空间上可能未必合理，如果不好使，可以考虑使用阈值惩罚，也就是回复模式应当与第一轮相似保持一定范围
    # 试试 1-max similarity
    def cal_emb_intrinsic_reward(self, agent_batchs, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics):
        """
        根据t轮回复与t-1轮回复的embedding加权距离计算t轮回复的embedding差异度内在奖励，避免I agree的情况，以及避免直接抄答案的高相似度，鼓励对上一轮回复的有效利用
        最简单草稿：
        - 第一轮：仅缓存上一轮所有agent的action embedding，不改动奖励
        - 后续轮：对每个样本i，计算当前agent的action embedding与上一轮所有agent同一batch索引i的embedding的余弦相似度，取max，相应内在奖励=1-max
        - 将该内在奖励加到该样本最后一个有效token位置
        """
        import torch
        import torch.nn.functional as F

        num_agents = self.num_agents
        coeff = float(getattr(self.config.marl, "emb_intrinsic_coeff", 1.0))

        # 1) 计算当前轮各agent的action embedding，按 [agent] -> [bs, dim] 存储
        curr_turn_action_embeds = []
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            # 解码当前轮回复文本
            actions_text = self.tokenizer_list[agent_id].batch_decode(
                agent_batch.batch["responses"], skip_special_tokens=True
            )
            # 编码为embedding [bs, dim]
            actions_embed = self._embed_encode_fn(
                texts=actions_text, max_batch_size=self.embedding_micro_batch_size, per_gpu_micro_bs=self.qmix_encode_batch_size
            )
            curr_turn_action_embeds.append(actions_embed)

        # 2) 如果没有上一轮缓存（第一轮），只缓存然后返回，不做任何奖励修改
        if not hasattr(self, "_last_turn_action_embeds") or self._last_turn_action_embeds is None:
            self._last_turn_action_embeds = [e.detach() for e in curr_turn_action_embeds]
            # 记录度量为0，表示第一轮未使用内在奖励
            for agent_id, _ in enumerate(self.mac.agents):
                metrics[f"emb_intrinsic/agent_{agent_id}_used"] = 0.0
            return

        # 3) 准备上一轮 embedding: List[[bs, dim]] -> [num_agents, bs, dim]
        prev_stack = torch.stack(self._last_turn_action_embeds, dim=0)  # [A, B, D]
        # 归一化（防止除0）
        prev_norm = F.normalize(prev_stack, p=2, dim=-1)
        metrics["emb_intrinsic/num_agents"] = float(num_agents)

        # 4) 逐agent计算当前轮与上一轮所有agent（同一batch位点）的最大相似度，并转换为内在奖励 1 - max_sim
        for agent_id, agent in enumerate(self.mac.agents):
            agent_key = f"agent_{agent_id}"
            agent_batch = agent_batchs[agent_key]

            curr = curr_turn_action_embeds[agent_id]                  # [B, D]
            curr_norm = F.normalize(curr, p=2, dim=-1)                # [B, D]

            # 计算 [A, B, D] 与 [B, D] 的点积相似度 -> [A, B]
            # 通过广播：将 curr 扩到 [A, B, D]
            sims = (prev_norm * curr_norm.unsqueeze(0)).sum(dim=-1)   # [A, B]
            max_sim, _ = sims.max(dim=0)                               # [B]

            # 内在奖励：1 - max_sim
            intrinsic = (1.0 - max_sim).clamp(min=0.0, max=2.0) * coeff  # [B]

            # 写入到 qmix_rewards 的“最后一个有效token”位置
            qmix_rewards = reward_tensor_all[agent_key]                # [B, T]
            last_valid_pos = agent_batch.batch["response_mask"].sum(-1) - 1  # [B]
            idx = torch.arange(qmix_rewards.size(0), device=qmix_rewards.device)

            # 累加到该位置
            qmix_rewards[idx, last_valid_pos] = qmix_rewards[idx, last_valid_pos] + intrinsic.to(qmix_rewards.device)

            # 回写与记录
            reward_tensor_all[agent_key] = qmix_rewards
            metrics[f"emb_intrinsic/agent_{agent_id}_mean"] = intrinsic.mean().item()
            metrics[f"emb_intrinsic/agent_{agent_id}_std"] = intrinsic.std(unbiased=False).item() if intrinsic.numel() > 1 else 0.0
            metrics[f"emb_intrinsic/agent_{agent_id}_max"] = intrinsic.max().item()
            metrics[f"emb_intrinsic/agent_{agent_id}_min"] = intrinsic.min().item()
            metrics[f"emb_intrinsic/agent_{agent_id}_used"] = 1.0

        # 5) 更新缓存为当前轮，用于下一轮比较
        self._last_turn_action_embeds = [e.detach() for e in curr_turn_action_embeds]


            
    # 可以用绝对confidence也可以用相对confidence
    def cal_confidence_intrinsic_reward(self, agent_batchs, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics):
        """
        计算confidence的内在奖励，根据debate的鞅过程，实现debate turn的信念改进，主要通过 cov<confidence, belief advantage>计算贡献的新颖度，鼓励低自信高收益行为。
        """
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            agent_key = f"agent_{agent_id}"
            
            # 获取QMIX critic分配的reward
            qmix_rewards = reward_tensor_all[agent_key]  # shape: (bs, seq_len)


    # 类似RLPR，用标准答案替换生成文本中的错误答案，观察log prob的变化？
    # 参考答案序列我们也可以提取到，这个不是问题
    # 记录每句回复的confidence



    def debate_stable_loss(self, agent_batchs, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics):
        """
        计算debate的稳定性损失，要求Q学习的时候期望奖励改进不低于0，但是这个有点小问题，假设策略分布不变，但是Q值天然有衰减discount，这怎么办？min pi*R
        """
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            agent_key = f"agent_{agent_id}"
            
            # 获取QMIX critic分配的reward

            
    # 计算self certainty也有助于aggregation 2 agents


    # 过短回复和抄袭的惩罚
    # 在debate提供答案的情况下，agree是更高概率的回复，但是放到原始问题下几乎不可能有I agree，自然的放到原始prompt下计算mean log prob就行。