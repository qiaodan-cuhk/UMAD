"""新版本VDN，使用qmix的total q"""

import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, Optional
from copy import deepcopy
from tqdm import tqdm

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs

from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tracking import ValidationGenerationsLogger

# 下放到agent层
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.single_controller.ray.base import create_colocated_worker_cls



# 新版本新增
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.ppo.reward import compute_reward, compute_reward_async  # fit
from verl.trainer.ppo.core_algos import agg_loss  # fit

from verl.workers.rollout.async_server import AsyncLLMServerManager

from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.utils.metric import (
    reduce_metrics,
)

import torch
from verl.utils.torch_functional import masked_mean


# MARL层
WorkerType = Type[Worker]


from marl.utils.marl_utils import MARLRole, Role


from marl.modules.agents.ppo_agent import ResourcePoolManager
from marl.utils.marl_utils import AdvantageEstimator, compute_advantage, compute_response_mask, apply_kl_penalty



from marl.utils.marl_utils import _convert_marl_to_ppo_roles, _timer 

"""vdn版本loss计算，暂时不用"""
# from marl.modules.mixers.vdn import compute_vdn_value_loss
from verl.trainer.ppo.core_algos import compute_value_loss


from marl.learners.marl_learner import RayMARLLearner

class RayVDNLearner_MultiTurn(RayMARLLearner):
    def __init__(self,
                 config,
                 mac,
                 num_agents,  # cotrian LLMs
                 tokenizer,   # 单个，表示共用model
                 role_worker_mapping: list[dict[Role, WorkerType]],
                 resource_pool_manager: ResourcePoolManager, # 共用全局manager
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, # fsdp/Megatron
                 processor=None,  # 单个，表示共用model
                 reward_fn=None, # reward function公用相同的
                 val_reward_fn=None,
                 train_dataset: Optional[Dataset] = None,
                 val_dataset: Optional[Dataset] = None,
                 collate_fn=None,
                 train_sampler: Optional[Sampler] = None,
                 device_name="cuda",
                 **kwargs
                 ):
        super().__init__(config,
                         mac,
                         num_agents,
                         tokenizer,
                         role_worker_mapping,
                         resource_pool_manager,
                         ray_worker_group_cls,
                         processor,
                         reward_fn,
                         val_reward_fn,
                         train_dataset,
                         val_dataset,
                         collate_fn,
                         train_sampler,
                         device_name)
        
        # self.is_sum = kwargs.get('sum_rewards', False)  # 控制是否sum all agents rewards
        self.is_sum = config.marl.sum_reward
        self.cory_flip = config.marl.cory_flip
        self.tb_dir = config.marl.tensorboard_dir

        # self.turns = config.marl.turns
        self.turns = 2 # 默认2轮


    

    """每个agent计算自己的old log prob存储到metric agent id"""
    def _compute_old_log_prob(self, agent_batchs, metrics):
        
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]

            old_log_prob = agent.actor_rollout_wg.compute_log_prob(agent_batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = agent_batch.batch["response_mask"]
            loss_agg_mode = agent.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {f"agent_{agent_id}_actor/entropy_loss": entropy_loss.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            agent_batch = agent_batch.union(old_log_prob)

            agent_batchs[f"agent_{agent_id}"] = agent_batch

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
            agent_batch = agent_batch.union(ref_log_prob)
            
            agent_batchs[f"agent_{agent_id}"] = agent_batch



    def _compute_values(self, agent_batchs):
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            values = agent.critic_wg.compute_values(agent_batch)
            # values维度与response length有关，无效token对应response mask为0，不计算critic
            agent_batch = agent_batch.union(values)
            agent_batchs[f"agent_{agent_id}"] = agent_batch

    def _compute_values_preds(self, agent_batchs):
        vpreds = []
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            # 先试试能不能用select出来的这部分直接计算critic，不能的话额外保留一个完整的dataproto数据切分
            values = agent.critic_wg.compute_values(agent_batch)
            values_tensor = values.batch['values']
            # 确保values_tensor需要梯度
            values_tensor.requires_grad_(True)  # 添加这行
            vpreds.append(values_tensor)
        return vpreds


        

    def _compute_adv(self, agent_batchs, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics):
        kl_metrics_all = {}
        for agent_id, agent in enumerate(self.mac.agents):

            agent_batch = agent_batchs[f"agent_{agent_id}"]

            # we combine with rule-based rm
            reward_extra_infos_dict: dict[str, list]
            # 使用marl.config判断reward model模式
            if self.config.reward_model.launch_reward_fn_async:
                for future_reward in future_reward_all.values():
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            else:
            # 采用公共reward计算一次，分别传给每个agent
                reward_tensor = reward_tensor_all[f"agent_{agent_id}"]
                reward_extra_infos_dict = reward_extra_infos_dict_all[f"agent_{agent_id}"]

            agent_batch.batch["token_level_scores"] = reward_tensor

            # print(f"{list(reward_extra_infos_dict.keys())=}")
            if reward_extra_infos_dict:
                agent_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})


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



    """
    这里不能用total q替代个体q直接传入每个agent计算，梯度有问题
    在trainer层计算total q，完整的batch size
    计算total q的loss和梯度，用fsdp和dp critic的计算方式得到loss，确保loss可以保留梯度
    然后loss计算对于每个critic的梯度，添加到batch data，每个batch的data额外union了loss/qi的梯度用作系数
    然后data传回agent的worker进程层，实现dp，然后update critic中调用了dp critic
    这个dp critic需要修改他的update逻辑，不用计算前面的loss，只用q做梯度乘系数即可
    """

    
    """计算mini batch的q total loss，然后传入每个agent的critic"""
    def _compute_vdn_values(self, agent_batchs, q_total_clipbase, metrics):
        """agent batchs是dict形式的DataProto"""


        origin_values = []
        
        # 计算当前策略的q total vpreds，保留梯度
        vpreds = self._compute_values_preds(agent_batchs)  # [num_agents, minibatch_size, response_length]


        stacked_vpreds = torch.stack(vpreds, dim=0)
        # 用于计算loss的clip
        q_total = torch.sum(stacked_vpreds, dim=0)


        """修改用于计算q total，目前只用第一个agent的returns和mask"""
        # Support all devices
        from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available

        agent_batch_data = agent_batchs['agent_1']
        if isinstance(agent_batch_data, DataProto):
            agent_batch_data = {**agent_batch_data.batch.to(get_torch_device().current_device()), **agent_batch_data.non_tensor_batch}
        else:
            agent_batch_data = agent_batch_data.to(get_torch_device().current_device())  # critic device is cpu when using offload


        # 这里只用第一个agent的returns和mask，或者也可以考虑max
        responses = agent_batch_data["responses"]
        attention_mask = agent_batch_data["attention_mask"]
        # values = data["values"]
        # returns本身已经是sum team rewards
        team_returns = agent_batch_data["returns"]
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length - 1 : -1]

        
        # 计算q total loss
        """如果考虑每个agent单独计算response mask，那么需要改成for agent循环"""
        q_total_loss, q_total_clipfrac = compute_value_loss(
            vpreds=q_total,
            returns=team_returns,
            values=q_total_clipbase,   # 用于clip q total边界
            response_mask=response_mask,
            cliprange_value=self.config.critic.cliprange_value,
        )

        q_total_metrics = {
            "total/q_total_loss": q_total_loss.detach().mean().item(),
            "total/team_returns": team_returns.detach().mean().item(),
        }
        metrics.update(q_total_metrics)

        """计算q grad传入batch,这里的问题是不知道最终的loss是哪个max的，暂时先用vpred代替"""
        for agent_id, agent in enumerate(self.mac.agents):
            q_grad_i = torch.autograd.grad(q_total_loss, vpreds[agent_id], create_graph=True)[0]
            agent_batchs[f"agent_{agent_id}"].batch['q_grad'] = q_grad_i.detach()


        # 这样通过team reward 0-2 计算了q total loss，将q total收敛到0-2之间
        # 每个agent batchs仍然只保留自己的critic values，通过q grad传入每个agent的critic，实现critic的更新
        return agent_batchs, metrics




    def split_dataproto_to_mini_batches(self, agent_batch: DataProto, mini_batch_size: int):
        # 1. select batch数据，tensordict格式，切分多个minibatch
        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]
        batch = agent_batch.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in agent_batch.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = agent_batch.batch.batch_size[0] // self.config.critic.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            mini_dataloader = agent_batch.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            mini_dataloader = batch.split(self.config.critic.ppo_mini_batch_size)

        """暂时不切分，下一步可以继续切"""
        new_meta_info = agent_batch.meta_info.copy()

        # 2. 切分non_tensor_batch
        total_size = len(agent_batch.non_tensor_batch['uid'])  # 使用uid长度作为总长度
        num_splits = total_size // mini_batch_size
        
        mini_batches = []
        for i, mini_tensor_dict in enumerate(mini_dataloader):
            # 计算当前mini batch的索引范围
            start_idx = i * mini_batch_size
            end_idx = start_idx + mini_batch_size
            
            # 切分non_tensor_batch中的每个数组
            mini_non_tensor_batch = {}
            for key, value in agent_batch.non_tensor_batch.items():
                mini_non_tensor_batch[key] = value[start_idx:end_idx]
                
            # 创建新的DataProto
            mini_batch_proto = DataProto(
                batch=mini_tensor_dict,
                non_tensor_batch=mini_non_tensor_batch,
                meta_info=agent_batch.meta_info  # meta_info可以保持不变，因为它包含的是配置信息
            )
            mini_batches.append(mini_batch_proto)
        
        # 
        return mini_batches



    def _compute_sentence_level_reward(self, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics):
        """
        计算多轮对话中的句子级别折扣奖励
        
        Args:
            reward_tensor_all: 每一轮的原始奖励张量字典
            reward_extra_infos_dict_all: 每一轮的额外奖励信息字典
            future_reward_all: 每一轮的未来奖励字典
            metrics: 用于记录指标的字典
            
        Returns:
            更新后的reward相关字典
        """
        turns = len(reward_tensor_all)
        discount_sentence = 0.7  # 句子level的discount rate
        
        # 对每个智能体分别计算
        for agent_id, agent in enumerate(self.mac.agents):
            reward_tensor_turn = None  # 用于存储上一轮的累积奖励
            
            # 从最后一轮开始,向前计算折扣奖励
            for turn_idx in reversed(range(turns)):
                turn_key = f"turn_{turn_idx}"
                
                # 获取当前轮次的奖励
                current_reward_tensor = reward_tensor_all[turn_key][f"agent_{agent_id}"]
                current_reward_info = reward_extra_infos_dict_all[turn_key][f"agent_{agent_id}"]
                current_future_reward = future_reward_all[turn_key][f"agent_{agent_id}"]
                
                if turn_idx == turns - 1:
                    # 最后一轮,直接使用原始奖励
                    reward_tensor_turn = current_reward_tensor
                else:
                    # 非最后一轮,计算折扣累积奖励
                    reward_tensor_turn = current_reward_tensor + discount_sentence * reward_tensor_turn
                
                # 更新奖励字典
                reward_tensor_all[turn_key][f"agent_{agent_id}"] = reward_tensor_turn
                
                # 记录每轮的折扣累积奖励到metrics中
                metrics[f"agent_{agent_id}/turn_{turn_idx}_discounted_reward"] = reward_tensor_turn.mean().item()
                
                # 保持extra info和future reward的一致性
                reward_extra_infos_dict_all[turn_key][f"agent_{agent_id}"] = current_reward_info
                future_reward_all[turn_key][f"agent_{agent_id}"] = current_future_reward

        return reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics



    def _compute_sentence_level_q(self, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics):
        """
        对于需要计算sentence level Q的，需要额外设置一个 sentence level state+action 的Critic网络，接收embedding，输出sentence Q value作为每一轮的turn reward，再依次处理每个turn的训练
        """
        pass


    """
    改成multi turn的
    对于不需要计算sentence level Q的，对每个turn的reward做sentence level discount，依次独立处理每个turn的训练
    对于需要计算sentence level Q的，需要额外设置一个 sentence level state+action 的Critic网络，接收embedding，输出sentence Q value作为每一轮的turn reward，再依次处理每个turn的训练
    """
    def train(self, multi_turn_batchs, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics, global_steps, timing_raw):

        turns = len(multi_turn_batchs)
        # # 对于非sentence level Q，计算每个turn的discount reward
        # reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics = self._compute_sentence_level_reward(reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics)
        
        """需要考虑把discount的reward放到batch中，或者利用adv计算"""


        """把下面的都改成每个turn的计算"""
        for turn_idx in range(turns):
            agent_batchs = multi_turn_batchs[f"turn_{turn_idx}"]
            reward_tensor_turn = reward_tensor_all[f"turn_{turn_idx}"]
            reward_extra_infos_dict_turn = reward_extra_infos_dict_all[f"turn_{turn_idx}"]
            future_reward_turn = future_reward_all[f"turn_{turn_idx}"]
            
            

            """每个agent计算自己的old log prob存储到metric agent id"""
            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw):
                self._compute_old_log_prob(agent_batchs, metrics)


            """计算reference log prob"""
            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    self._compute_ref_log_prob(agent_batchs)
                    
            # if self.use_critic:
            #     with _timer("vdn_values", timing_raw):
            #         self._compute_vdn_values(agent_batchs)


            # 为每个agent计算values，用于adv计算
            self._compute_values(agent_batchs)

            with _timer("adv", timing_raw):
                self._compute_adv(agent_batchs, reward_tensor_turn, reward_extra_infos_dict_turn, future_reward_turn, metrics)


            # update critic
            with _timer('update_critic', timing_raw):
                critic_output_metrics_all = {f"agent_{agent_id}": [] for agent_id in range(self.num_agents)}

                # 计算q total clipbase，保留梯度
                # self._compute_values(agent_batchs)
                origin_values_clipbase = []

                for agent_id, agent in enumerate(self.mac.agents):
                    agent_batch = agent_batchs[f"agent_{agent_id}"]
                    origin_values_clipbase.append(agent_batch.batch['values'])

                """sum q value，后续可以改成self.mixer的形式"""
                stacked_q_clipbase = torch.stack(origin_values_clipbase, dim=0)
                # 用于计算loss的clip
                q_total_clipbase = torch.sum(stacked_q_clipbase, dim=0)
                q_total_clipbase_detach = q_total_clipbase.detach()


                # 这个clipbase可能需要考虑放到batch中union保证格式。

                """
                切分mini batch，计算划分出几个mini batch,不用考虑dataparallel，self.config.ppo_mini+bs就是所有worker的总bs
                mini_bs = self.config.batch_size // self.config.critic.ppo_mini_batch_size
                但是需要将切分的select data重新封装回dataproto，用于后续的critic_wg的update和value，dp通信格式
                """
                mini_dataloaders = []
                meta_info_all = []
                non_tensor_batch_all = []
                # num_mini_batches = agent_batch.batch.batch_size[0] // self.config.critic.ppo_mini_batch_size
                mini_bs_epochs = self.config.data.train_batch_size // self.config.critic.ppo_mini_batch_size
                for agent_id, agent in enumerate(self.mac.agents):
                    agent_batch = agent_batchs[f"agent_{agent_id}"]

                    agent_mini_batches = self.split_dataproto_to_mini_batches(agent_batch, self.config.critic.ppo_mini_batch_size)
                    mini_dataloaders.append(agent_mini_batches)
                    # 【agent1的多个mini batch dataproto， agent2的】


                # assert len(mini_dataloaders) == mini_bs_epochs, "mini batch num not match"
                
                """计算每个mini bs的clip q total loss，然后进行micro bs的critic更新"""
                for mini_batch_idx in range(mini_bs_epochs):
                    agents_mini_batchs = {}
                    for agent_id in range(self.num_agents):
                        agent_mini_batchs = mini_dataloaders[agent_id][mini_batch_idx]
                        # 一组agent的mini batchs数据
                        agents_mini_batchs[f"agent_{agent_id}"] = agent_mini_batchs

                    # 取出对应mini batch的q total clipbase
                    start_idx = mini_batch_idx * self.config.critic.ppo_mini_batch_size
                    end_idx = start_idx + self.config.critic.ppo_mini_batch_size
                    q_total_clipbase_mini = q_total_clipbase[start_idx:end_idx]
                    # 计算mini batch的q total loss，然后传入每个agent的critic
                    agents_mini_batchs, metrics = self._compute_vdn_values(agents_mini_batchs, q_total_clipbase_mini, metrics)


                    for agent_id, agent in enumerate(self.mac.agents):
                        agent_mini_batch_update = agents_mini_batchs[f"agent_{agent_id}"]

                        # 将当前算出来的mini batch传入每个agent去做critic update
                        if agent.use_critic:
                            critic_output = agent.critic_wg.update_critic(agent_mini_batch_update)

                        # 多个mini batch的metric需要append方式合并
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        critic_output_metrics_all[f"agent_{agent_id}"].append(critic_output_metrics)


                # """这里要处理多个mini batch的metric，统一成bs维度的"""
                # for agent_key, agent_metrics in critic_output_metrics_all.items():
                #     metrics.update({f"{agent_key}_{k}": v for k, v in agent_metrics.items()})

                """这里要处理多个mini batch的metric，统一成bs维度的"""
                for agent_key, agent_metrics_list in critic_output_metrics_all.items():
                    # agent_metrics_list是一个列表，每个元素是一个mini batch的metrics字典
                    # 需要先合并这些metrics
                    merged_metrics = {}
                    for metrics_dict in agent_metrics_list:
                        for k, v in metrics_dict.items():
                            if k not in merged_metrics:
                                merged_metrics[k] = []
                            merged_metrics[k].append(v)
                    
                    # 计算每个指标的平均值
                    averaged_metrics = {k: sum(v)/len(v) for k, v in merged_metrics.items()}
                    
                    # 更新到总的metrics中
                    metrics.update({f"{agent_key}_{k}": v for k, v in averaged_metrics.items()})


            """
            要考虑adv是否影响了critic计算，还是只影响actor；
            目前放到critic update之后，actor update之前，
            如果只影响actor，考虑使用更新后的critic计算还是更新前的critic计算adv
            """ 
            # with _timer("adv", timing_raw):
            #     self._compute_adv(agent_batchs, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics)


            # implement critic warmup
            if self.config.trainer.critic_warmup <= global_steps:
                # update actor
                with _timer("update_actor", timing_raw):
                    agent_output_metrics_all = {}
                    for agent_id, agent in enumerate(self.mac.agents):
                        agent_batch = agent_batchs[f"agent_{agent_id}"]
                        agent_batch.meta_info["multi_turn"] = agent.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = agent.actor_rollout_wg.update_actor(agent_batch)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        agent_output_metrics_all[f"agent_{agent_id}"] = actor_output_metrics

                        # 更新了meta info，覆盖一下
                        agent_batchs[f"agent_{agent_id}"] = agent_batch

                # 分别存储每个agent的actor metrics
                for agent_key, agent_metrics in agent_output_metrics_all.items():
                    metrics.update({f"{agent_key}_{k}": v for k, v in agent_metrics.items()})




