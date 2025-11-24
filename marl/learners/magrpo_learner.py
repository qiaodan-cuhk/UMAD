"""异构多轮qmix，sentence level Q"""

import ray
import torch



from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.metric import (
    reduce_metrics,
)

from verl.trainer.ppo.core_algos import agg_loss  # fit

from marl.utils.marl_utils import compute_advantage, apply_kl_penalty, _timer



# magrpo采取mappo类型的criticfree更新
class RayMAGRPOLearner():
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
        self.device = device_name
        assert self.device == "cuda", "qmix learner only support cuda device"


        self.mixer = None
        self.gamma_turn = config.marl.gamma_turn


    

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

    def encode_in_batches(self, texts, max_batch_size=None):
        """分批编码，避免内存爆炸"""
        embeddings = []
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i+max_batch_size]
            with torch.no_grad():
                batch_embeddings = self.mixer.text_lm.encode(
                    batch_texts, 
                    batch_size=len(batch_texts), 
                    convert_to_tensor=True
                )
            embeddings.append(batch_embeddings.cpu())
        return torch.cat(embeddings, dim=0)
    


    def transfer_embed_ids(self, agent_batchs, rewards_all, turns=2):
        """
        将各agent的token ids转为文本，再用统一的embed_base_lm tokenizer编码为统一token ids，
        返回(state_ids, actions_ids, next_state_ids, rewards, dones)用于buffer存储。
        """
        # num_agents = self.num_agents
        num_agents = len(agent_batchs)
        batch_size = agent_batchs['turn_0']['agent_0'].batch['input_ids'].shape[0]

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
            state_token_ids_batch = agent_batchs_turn['agent_0'].batch['input_ids']
            state_texts = self.tokenizer_list[0].batch_decode(state_token_ids_batch, skip_special_tokens=True)
            state_ids_batch = [self.mixer.text_tokenizer.encode(text, add_special_tokens=True) for text in state_texts]
            state_ids_list.extend(state_ids_batch)


            # Batch decode actions for each agent
            actions_ids_batch = []
            rewards_batch = []
            for agent_id in range(num_agents):
                action_token_ids_batch = agent_batchs_turn[f'agent_{agent_id}'].batch['responses']
                action_texts = self.tokenizer_list[agent_id].batch_decode(action_token_ids_batch, skip_special_tokens=True)
                action_ids_batch = [self.mixer.text_tokenizer.encode(text, add_special_tokens=True) for text in action_texts]
                actions_ids_batch.append(action_ids_batch)
                
                # Batch rewards
                reward_batch = rewards_all[f'turn_{turn_idx}'][f'agent_{agent_id}'].sum(dim=-1).tolist()
                rewards_batch.append(reward_batch)

            # Verify rewards are equal across agents
            for i in range(batch_size):
                assert all(rewards_batch[agent_id][i] == rewards_batch[0][i] for agent_id in range(num_agents)), "rewards for all agents should be equal"



            # Transpose actions_ids_batch to get per-sample format
            for i in range(batch_size):
                actions_ids_list.append([actions_ids_batch[agent_id][i] for agent_id in range(num_agents)])
                rewards_list.append(rewards_batch[0][i])
                dones_list.append(terminate)

            # Next states
            if turn_idx < turns - 1:
                next_state_token_ids_batch = agent_batchs[f'turn_{turn_idx+1}']['agent_0'].batch['prompts']
                next_state_texts = self.tokenizer_list[0].batch_decode(next_state_token_ids_batch, skip_special_tokens=True)
                next_state_ids_batch = [self.mixer.text_tokenizer.encode(text, add_special_tokens=True) for text in next_state_texts]
                next_state_ids_list.extend(next_state_ids_batch)
            else:
                # Terminal state for last turn
                terminal_text = "The conversation is over"
                terminal_ids = self.mixer.text_tokenizer.encode(terminal_text, add_special_tokens=True)
                next_state_ids_list.extend([terminal_ids] * batch_size)

        return state_ids_list, actions_ids_list, next_state_ids_list, rewards_list, dones_list
        


        
    # 处理的是单轮，多个agent的数据
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
            # 使用各自critic估计的turn reward
                reward_tensor = reward_tensor_all[f"agent_{agent_id}"]
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


    #### todo: 目前只处理qwen 2.5 template，要检查一下phi3等其他模型的template是否适用？？
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
            # 先用"user\n"切分，取后半部分
            _, after_user = text.split("user\n")
            # 再用"Let's"切分，取前半部分
            question = after_user.split("Let\'s")[0]
            # 去除首尾空白
            return question.strip()
        except ValueError:
            # 如果切分失败（找不到分隔符），返回原文本
            return text.strip()
    
    # soft update target qmix critics
    def _update_target_critics(self, tau=0.005):
        for i in range(self.num_agents):
            for target_param, param in zip(self.mixer.target_critics_sentence[i].parameters(),
                                        self.mixer.critics_sentence[i].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


    # 更新sentence level qmix
    def update_qmix(self, buffer, qmix_batch_size, **kwargs):
        """先从buffer采样数据，更新sentence QMIX"""
        qmix_data = buffer.sample(qmix_batch_size)

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
            batch_next_states = agent_tokenizer(text_next_states, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True, max_length=self.config.data.max_prompt_length)
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
            
            
            # action_embed = []
            temp_actions = []
            temp_actions.extend(actions[:,0])
            temp_actions.extend(actions[:,1])
            # for agent_idx in range(self.num_agents):
            text_action = self.mixer.text_tokenizer.batch_decode(temp_actions, skip_special_tokens=True)
            action_embed_all = self.mixer.text_lm.encode(text_action, batch_size=self.qmix_batch, convert_to_tensor=True)
            action_embed = torch.chunk(action_embed_all, self.num_agents, dim=0)

            next_state_embed = self.mixer.text_lm.encode(text_next_states, batch_size=self.qmix_batch, convert_to_tensor=True)  # [self.qmix_batch, 1024]



        # === 用于debug，直接加载debug tensor ===
        # loaded_data = torch.load('debug_tensors.pt')


        # embed不训练，重新设置grad
        state_embed = state_embed.clone().detach().requires_grad_()
        next_state_embed = next_state_embed.clone().detach().requires_grad_()
        action_embed = [x.clone().detach().requires_grad_() for x in action_embed]
        next_actions_embed = [x.clone().detach().requires_grad_() for x in next_actions_embed]

        # Start QMIX
        # 计算q_target
        q_targets = []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                next_action = next_actions_embed[agent_id]
                q_target = self.mixer.target_critics_sentence[agent_id](next_state_embed, next_action)  # 输入critic网络的是句子级别embedding
                q_targets.append(q_target)
            
            q_targets = torch.stack(q_targets, dim=1)   # 输入需要 bs, num, 1
            # mixed Q total
            qmix_targets = self.mixer(q_targets, next_state_embed)

        # q_value = self.critic(states, actions)
        q_values = []
        for agent_id in range(self.num_agents):
            q_value = self.mixer.critics_sentence[agent_id](state_embed, action_embed[agent_id])
            q_values.append(q_value)

        q_values = torch.stack(q_values, dim=1)
        qmix_value = self.mixer(q_values, state_embed)


        rewards_torch = torch.tensor(rewards, dtype=torch.float32, device=qmix_value.device)  # cpu
        dones_torch = torch.tensor(dones, dtype=torch.int, device=qmix_value.device)

        qtotal_targets = rewards_torch + self.gamma_turn * qmix_targets.squeeze(1) * (1-dones_torch)
        qtotal_loss = ((qtotal_targets - qmix_value.squeeze(1))**2).mean()


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

        """每个turn、每个agent计算"""
        for turn_idx in range(self.turns):
            agent_batchs = multi_turn_batchs[f"turn_{turn_idx}"]

            turn_reward_tensor = reward_tensor_all[f"turn_{turn_idx}"]
            onpolicy_batch_size = len(agent_batchs['agent_0'].batch['prompts'])


            # 这里是team rewards，记录没有意义
            # for agent_id in range(self.num_agents):
            #     turn_mean_reward = turn_reward_tensor[f"agent_{agent_id}"].sum() / onpolicy_batch_size
            #     metrics[f"original_rewards/agent_{agent_id}_turn_{turn_idx}"] = turn_mean_reward.item()
                # turn_reward_tensor[f"agent_{agent_id}"] = turn_reward_tensor[f"agent_{agent_id}"].sum() / onpolicy_batch_size
                # metrics[f"original_rewards/agent_{agent_id}_turn_{turn_idx}"] = turn_reward_tensor[f"agent_{agent_id}"].item()


            # magrpo需要sum计算后续turn的reward
            turn_acc_reward_tensor = {}
            for agent_id in range(self.num_agents):
                # 计算累计奖励
                acc_reward = torch.zeros_like(turn_reward_tensor[f"agent_{agent_id}"])
                for t in range(turn_idx, self.turns):
                    acc_reward += reward_tensor_all[f"turn_{t}"][f"agent_{agent_id}"]
                # 你可以根据需要对 acc_reward 做 sum/mean 或直接保存
                acc_reward_mean = acc_reward.sum() / onpolicy_batch_size
                metrics[f"rewards_accumulate/agent_{agent_id}_turn_{turn_idx}"] = acc_reward_mean.item()

                turn_acc_reward_tensor[f"agent_{agent_id}"] = acc_reward


            
            """每个agent计算自己的old log prob存储到metric agent id"""
            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw):
                self._compute_old_log_prob(agent_batchs, metrics)

            """计算reference log prob"""
            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    self._compute_ref_log_prob(agent_batchs)

            # 为每个agent计算values，用于adv计算
            if self.use_critic: 
                with _timer("values", timing_raw):  
                    self._compute_values(agent_batchs)

            # 传入 turn credits all
            # reward_extra_infos_dict_all = reward_extra_infos_dict_all["turn_0"]
            # future_reward_all = future_reward_all["turn_0"]

            # 检查一下后续轮次保持uid一致是否可以计算center critic value，使用accumulated reward
            with _timer("adv", timing_raw):
                self._compute_adv(agent_batchs, turn_acc_reward_tensor, reward_extra_infos_dict_all, future_reward_all, metrics)


            # 这里 reward tensor turn已经转化为critic sentence value
            # update critic
            if self.use_critic:
                with _timer('update_critic', timing_raw):
                    critic_output_metrics_all = {}
                    for agent_id, agent in enumerate(self.mac.agents):
                        agent_batch = agent_batchs[f"agent_{agent_id}"]

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
                        agent_batch.meta_info["multi_turn"] = agent.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = agent.actor_rollout_wg.update_actor(agent_batch)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        agent_output_metrics_all[f"agent_{agent_id}"] = actor_output_metrics

                        # 更新了meta info，覆盖一下
                        agent_batchs[f"agent_{agent_id}"] = agent_batch

                # 分别存储每个agent的actor metrics
                for agent_key, agent_metrics in agent_output_metrics_all.items():
                    metrics.update({f"{agent_key}_{k}": v for k, v in agent_metrics.items()})

        

    def init_workers(self):
        pass
