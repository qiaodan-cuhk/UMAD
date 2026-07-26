"""UMAD/IPPO learner for heterogeneous multi-agent debate"""

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

# Optional embedding-model resource helpers.
from marl.modules.agents.ppo_agent import ResourcePoolManager
from verl.single_controller.ray import RayWorkerGroup

class RayIPPOLearner():
    def __init__(self,
                 config,
                 mac,
                 num_agents,  # cotrian LLMs
                 tokenizer_list,   # Per-agent tokenizer list.
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


        self.mac = mac
        self.turns = config.marl.turns
        self.tokenizer_list = tokenizer_list

        self.device = device_name


        self.gamma_turn = config.marl.gamma_turn


    

    """Compute each agent's old log-probs and entropy metrics."""
    def _compute_old_log_prob(self, agent_batchs, metrics, turn_idx):
        
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]

            old_log_prob = agent.actor_rollout_wg.compute_log_prob(agent_batch)
            entropys = old_log_prob.batch["entropys"]


            response_masks = agent_batch.batch["response_mask"]
            loss_agg_mode = agent.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)

            old_log_prob_metrics = {f"agent_{agent_id}_actor/entropy_loss_turn_{turn_idx}": entropy_loss.detach().item()}

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
            # Values follow response length; invalid response tokens are masked later.
            agent_batch = agent_batch.union(values)
            agent_batchs[f"agent_{agent_id}"] = agent_batch

    def group_normalize(self, values, group_size=5):
        """Normalize scalar trajectory values inside each GRPO group."""
        bs = values.shape[0]
        normalized = values.clone()
        
        for i in range(0, bs, group_size):
            end_idx = min(i + group_size, bs)
            group = values[i:end_idx]
            
            group_mean = group.mean()
            group_std = group.std(unbiased=False)
            normalized[i:end_idx] = (group - group_mean) / (group_std + 1e-8)
        
        return normalized


    """Scale advantages by masked NLL as the aleatoric-cost signal."""
    def add_log_prob_ratio(self, agent_batchs, metrics=None):
        # all_log_probs = []
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            log_prob = agent_batch.batch["old_log_probs"]  # shape: [bs, response_len]

            mask = agent_batch.batch["response_mask"]  # shape: [bs, response_len]
            log_prob_sum = (log_prob * mask).sum(dim=-1, keepdim=True)  # shape: [bs, 1]
            nll_mean = -log_prob_sum / (mask.sum(dim=-1, keepdim=True) + 1e-8)
            nll_mean_normalized = self.group_normalize(nll_mean, self.config.actor_rollout_ref.rollout.n)


            original_adv = agent_batch.batch["advantages"]  # shape: [bs, 1]

            # Paper formula: W(U_bar)=exp(-alpha * U_bar). Lower NLL gets
            # ratio > 1, while higher NLL gets ratio < 1.
            ratio = torch.exp(-self.config.marl.log_prob_ratio_alpha * nll_mean_normalized)  # shape: [bs, 1]

            # Log the NLL gain statistics for debugging and ablation analysis.
            if metrics is not None:
                agent_key = f"agent_{agent_id}"
                metrics.update({
                    f"Log_Prob_Gain/{agent_key}_nll_raw_mean": nll_mean.mean().item(),
                    f"Log_Prob_Gain/{agent_key}_normed_val_mean": nll_mean_normalized.mean().item(),
                    f"Log_Prob_Gain/{agent_key}_ratio_mean": ratio.mean().item(),
                    f"Log_Prob_Gain/{agent_key}_ratio_max": ratio.max().item(),
                    f"Log_Prob_Gain/{agent_key}_ratio_min": ratio.min().item(),
                    f"Log_Prob_Gain/{agent_key}_ratio_std": ratio.std().item(),
                })
            
            new_adv = original_adv * ratio # shape: [bs, len]
            
            agent_batch.batch["advantages"] = new_adv
            agent_batchs[f"agent_{agent_id}"] = agent_batch


    def cal_influence_intrinsic_reward(self, agent_batchs, reward_tensor_all, current_turn_idx, metrics):
        """
        Add an intrinsic reward based on the next-turn effect on other agents.
        
        Args:
            agent_batchs: {"agent_0": batch, "agent_1": batch, ...}
            reward_tensor_all: dict, {"turn_0": {"agent_0": tensor, "agent_1": tensor}, ...}
            current_turn_idx: Current turn index.
            metrics: Mutable metrics dictionary.
        """
        
        if current_turn_idx >= self.turns - 1:
            # The final turn has no next-turn effect to measure.
            return
        
        for current_agent_id in range(self.num_agents):
            current_agent_key = f"agent_{current_agent_id}"
            
            current_reward_tensor = reward_tensor_all[f"turn_{current_turn_idx}"][current_agent_key].clone()
            bs, seq_len = current_reward_tensor.shape
            device = current_reward_tensor.device
            
            other_agents_delta_rewards = []
            
            for other_agent_id in range(self.num_agents):
                if other_agent_id == current_agent_id:
                    continue
                
                other_agent_key = f"agent_{other_agent_id}"
                
                other_current_reward = reward_tensor_all[f"turn_{current_turn_idx}"][other_agent_key]  # [bs, seq_len]
                other_next_reward = reward_tensor_all[f"turn_{current_turn_idx+1}"][other_agent_key]  # [bs, seq_len]
                
                other_agent_batch = agent_batchs[other_agent_key]
                response_mask = other_agent_batch.batch.get("response_mask")
                current_total = other_current_reward.sum(-1)
                next_total = other_next_reward.sum(-1)
                # if response_mask is not None:
                #     # Use the mask to compute total reward.
                #     current_total = (other_current_reward * response_mask).sum(-1)  # [bs*rollout.n]
                #     next_total = (other_next_reward * response_mask).sum(-1)
                # else:
                #     # Direct average fallback.
                #     current_avg = other_current_reward.mean(-1)
                #     next_avg = other_next_reward.mean(-1)
                
                delta = next_total - current_total  # [bs*rollout.n]
                other_agents_delta_rewards.append(delta)
            
            if len(other_agents_delta_rewards) == 0:
                # Single-agent runs have no peer influence term.
                continue
            
            # [num_other_agents, bs] -> [bs]
            avg_delta_per_sample = torch.stack(other_agents_delta_rewards, dim=0).mean(dim=0)  # [bs]
            
            # Compute one intrinsic reward per sample.
            intrinsic_reward_strength = self.config.marl.influence_intrinsic_reward_strength
            intrinsic_reward = torch.sign(avg_delta_per_sample) * intrinsic_reward_strength  # [bs]
            
            # Optional finer-grained linear relation:
            # intrinsic_reward = torch.clamp(avg_delta_per_sample * 0.5, -0.5, 0.5)
            
            # Add the intrinsic reward to the final valid token of this turn.
            current_agent_batch = agent_batchs[current_agent_key]
            response_mask = current_agent_batch.batch.get("response_mask")
            
            if response_mask is not None:
                last_valid_positions = response_mask.sum(-1) - 1  # [bs]
                last_valid_positions = last_valid_positions.clamp(min=0, max=seq_len-1)
                
                batch_indices = torch.arange(bs, device=device)
                current_reward_tensor[batch_indices, last_valid_positions] += intrinsic_reward
            else:
                current_reward_tensor[:, -1] += intrinsic_reward
            
            reward_tensor_all[f"turn_{current_turn_idx}"][current_agent_key] = current_reward_tensor
            
            metrics[f"influence_intrinsic/turn_{current_turn_idx}_agent_{current_agent_id}_mean"] = intrinsic_reward.mean().item()
            metrics[f"influence_intrinsic/turn_{current_turn_idx}_agent_{current_agent_id}_positive_ratio"] = (intrinsic_reward > 0).float().mean().item()
            metrics[f"influence_intrinsic/turn_{current_turn_idx}_agent_{current_agent_id}_std"] = intrinsic_reward.std().item()
            metrics[f"influence_intrinsic/turn_{current_turn_idx}_other_agents_delta"] = avg_delta_per_sample.mean().item()





    
    """Compute rewards/advantages and apply optional UMAD gain terms."""
    def _compute_adv(self, agent_batchs, estimate_reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics):
        kl_metrics_all = {}


        # Analyze reward distribution before advantage computation.
        self._analyze_reward_distribution(agent_batchs, estimate_reward_tensor_all, reward_extra_infos_dict_all, metrics)
    


        for agent_id, agent in enumerate(self.mac.agents):

            agent_batch = agent_batchs[f"agent_{agent_id}"]

            # we combine with rule-based rm
            reward_extra_infos_dict: dict[str, list]
            if self.config.reward_model.launch_reward_fn_async:
                for future_reward in future_reward_all.values():
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            else:
                reward_tensor = estimate_reward_tensor_all[f"agent_{agent_id}"]
                # reward_extra_infos_dict = reward_extra_infos_dict_all[f"agent_{agent_id}"]


            # if self.config.marl.use_influence_intrinsic_reward:
            #     reward_tensor = self.cal_influence_intrinsic_reward(agent_batchs, estimate_reward_tensor_all, metrics)

            

            agent_batch.batch["token_level_scores"] = reward_tensor

            # compute rewards. apply_kl_penalty if available
            # print(f" self.config.algorithm.use_kl_in_reward {self.config.algorithm.use_kl_in_reward}")
            if self.config.algorithm.use_kl_in_reward:
                agent_batch, kl_metrics = apply_kl_penalty(agent_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                kl_metrics_all[f"agent_{agent_id}"] = kl_metrics
            else:
                agent_batch.batch["token_level_rewards"] = agent_batch.batch["token_level_scores"]

            agent_batchs[f"agent_{agent_id}"] = agent_batch


        """Merge KL metrics after per-agent reward processing."""
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

        if self.config.marl.use_log_prob_gain:
            self.add_log_prob_ratio(agent_batchs, metrics=metrics)

        # Analyze the final advantage distribution after GRPO and UMAD gains.
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            agent_key = f"agent_{agent_id}"
            
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
                
                metrics.update({
                    f"Final_Adv_Stats/{agent_key}_mean": final_advantages.mean().item(),
                    f"Final_Adv_Stats/{agent_key}_std": final_advantages.std().item(),
                    f"Final_Adv_Stats/{agent_key}_min": final_advantages.min().item(),
                    f"Final_Adv_Stats/{agent_key}_max": final_advantages.max().item(),
                })

 

    def remove_template(self, text):
        """
        Remove chat-template markers from decoded prompt text.

        The input should already have special tokens removed. This helper is
        kept aligned with the multi-turn runner's prompt cleaning path.

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
            # Qwen-style chat templates may include a system/user wrapper.
            if "user\n" in text:
                text = text.split("user\n", 1)[1]
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


    


    """Train each turn as an independent IPPO/UMAD update stage."""
    def train(self, multi_turn_batchs, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics, global_steps, timing_raw):
        
        """Compute reward/advantage and update every agent for each turn."""
        for turn_idx in range(self.turns):

            # turn_idx = 1  # Debug single turn if needed.

            agent_batchs = multi_turn_batchs[f"turn_{turn_idx}"]

            # onpolicy_batch_size = len(agent_batchs['agent_0'].batch['prompts'])


            # states_text = self.tokenizer_list[0].batch_decode(agent_batchs['agent_0'].batch['prompts'], skip_special_tokens=True)
            print('Begin Old Log Prob: Turn', turn_idx)
            """Compute old log-probs and entropy metrics per agent."""
            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw):
                self._compute_old_log_prob(agent_batchs, metrics, turn_idx)


            print('Begin Ref Log Prob: Turn', turn_idx)
            """Compute reference log-probs when KL regularization is enabled."""
            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    self._compute_ref_log_prob(agent_batchs)


            # Compute values for advantage estimation when using a critic.
            if self.use_critic: 
                print('Begin Critic: Turn', turn_idx)
                with _timer("values", timing_raw):  
                    self._compute_values(agent_batchs)

            # reward_extra_infos_dict_all = reward_extra_infos_dict_all["turn_0"]
            # future_reward_all = future_reward_all["turn_0"]

            """Add optional influence-based intrinsic reward."""
            if self.config.marl.use_influence_intrinsic_reward and turn_idx < self.turns - 1:
                print('Begin Influence Intrinsic Reward: Turn', turn_idx)
                with _timer("influence_intrinsic", timing_raw):
                    self.cal_influence_intrinsic_reward(agent_batchs, reward_tensor_all, turn_idx, metrics)


            print('Begin Update Adv: Turn', turn_idx)
            with _timer("adv", timing_raw):
                self._compute_adv(agent_batchs, reward_tensor_all[f"turn_{turn_idx}"], reward_extra_infos_dict_all, future_reward_all, metrics)

                # Historical experiments used critic-estimated rewards here.
                # self._compute_adv(agent_batchs, turn_credits, reward_extra_infos_dict_all, future_reward_all, metrics)


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

                        # Preserve the updated meta info for downstream metrics.
                        agent_batchs[f"agent_{agent_id}"] = agent_batch

                for agent_key, agent_metrics in agent_output_metrics_all.items():
                    metrics.update({f"{agent_key}_{k}": v for k, v in agent_metrics.items()})

            multi_turn_batchs[f"turn_{turn_idx}"] = agent_batchs

        

    

    def _analyze_reward_distribution(self, agent_batchs, reward_tensor_all, reward_extra_infos_dict_all, metrics):
        """
        Analyze reward and advantage distributions for each agent.
        
        Args:
            agent_batchs: Batch data per agent.
            reward_tensor_all: Current reward assignment.
            reward_extra_infos_dict_all: Optional ground-truth reward metadata.
            metrics: Mutable metrics dictionary.
        """
        
        
        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            agent_key = f"agent_{agent_id}"
            
            reward_tensor = reward_tensor_all[agent_key]  # shape: (bs, seq_len)
            
            true_rewards = None
            if reward_extra_infos_dict_all and agent_key in reward_extra_infos_dict_all:
                reward_extra_info = reward_extra_infos_dict_all[agent_key]
                if "score" in reward_extra_info:
                    true_rewards = torch.tensor(reward_extra_info["score"], dtype=torch.float32)
                elif "reward" in reward_extra_info:
                    true_rewards = torch.tensor(reward_extra_info["reward"], dtype=torch.float32)
            
            uids = agent_batch.non_tensor_batch.get("uid", None)
            
            response_mask = agent_batch.batch.get("response_mask", None)
            if response_mask is not None:
                response_rewards = []
                for i in range(reward_tensor.shape[0]):
                    response_end = response_mask[i].sum().item() - 1
                    response_rewards.append(reward_tensor[i, response_end].item())
                response_rewards = torch.tensor(response_rewards)
            else:
                response_rewards = reward_tensor.sum(dim=-1)

            # Reward distribution.
            reward_mean = response_rewards.mean().item()
            reward_std = response_rewards.std().item()
            reward_min = response_rewards.min().item()
            reward_max = response_rewards.max().item()

            metrics.update({
                f"reward_stats/{agent_key}_reward_mean": reward_mean,
                f"reward_stats/{agent_key}_reward_std": reward_std,
                f"reward_stats/{agent_key}_reward_min": reward_min,
                f"reward_stats/{agent_key}_reward_max": reward_max,
            })
            
            # Reward distribution grouped by GRPO uid.
            if uids is not None:
                uid2rewards = defaultdict(list)
                for i, uid in enumerate(uids):
                    uid2rewards[uid].append(response_rewards[i].item())
                
                uid_group_stats = {}
                for uid, rewards in uid2rewards.items():
                    if len(rewards) > 1:
                        rewards_tensor = torch.tensor(rewards)
                        uid_group_stats[uid] = {
                            'mean': rewards_tensor.mean().item(),
                            'std': rewards_tensor.std().item(),
                            'count': len(rewards)
                        }
                
                if uid_group_stats:
                    avg_group_mean = np.mean([stats['mean'] for stats in uid_group_stats.values()])
                    avg_group_std = np.mean([stats['std'] for stats in uid_group_stats.values()])
                    avg_group_count = np.mean([stats['count'] for stats in uid_group_stats.values()])
                    
                    metrics.update({
                        f"reward_groups_stats/{agent_key}_uid_groups_count": len(uid_group_stats),
                        f"reward_groups_stats/{agent_key}_avg_group_mean": avg_group_mean,
                        f"reward_groups_stats/{agent_key}_avg_group_std": avg_group_std,
                        f"reward_groups_stats/{agent_key}_avg_group_size": avg_group_count,
                    })
            
            # Optional pre-GRPO reward distribution analysis.
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
            
            # Post-GRPO advantage distribution.
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
                
                # # Compute GRPO amplification if both input/output stats exist.
                # if "token_level_rewards" in agent_batch.batch:
                #     grpo_amplification = grpo_output_std / (grpo_input_std + 1e-8)
                #     metrics[f"{agent_key}_grpo_amplification"] = grpo_amplification
            
            # Optional post-GRPO advantage distribution grouped by uid.
            # if uids is not None and "advantages" in agent_batch.batch:
            #     uid2advantages = defaultdict(list)
            #     for i, uid in enumerate(uids):
            #         if response_mask is not None:
            #             response_end = response_mask[i].sum().item() - 1
            #             adv_value = advantages[i, response_end].item()
            #         else:
            #             adv_value = advantages[i].sum().item()
            #         uid2advantages[uid].append(adv_value)
                
            #     # Compute advantage statistics per uid group.
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
