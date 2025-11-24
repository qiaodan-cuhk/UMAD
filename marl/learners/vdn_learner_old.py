

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




from marl.learners.marl_learner import RayMARLLearner

class RayVDNLearner(RayMARLLearner):
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


        # ippo 不需要 mixer和单独设置optimiser
    

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
            agent_batch = agent_batch.union(values)
            agent_batchs[f"agent_{agent_id}"] = agent_batch


        

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


    def train(self, agent_batchs, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics, global_steps, timing_raw):

        """每个agent计算自己的old log prob存储到metric agent id"""
        # recompute old_log_probs
        with _timer("old_log_prob", timing_raw):
            self._compute_old_log_prob(agent_batchs, metrics)


        """计算reference log prob"""
        if self.use_reference_policy:
            # compute reference log_prob
            with _timer("ref", timing_raw):
                self._compute_ref_log_prob(agent_batchs)
                

        # compute values
        """这里应该改成marl.use_critic，因为learning算法会统一指定是否使用critic/Remax"""
        if self.use_critic:
            origin_values = []
            
            with _timer("values", timing_raw):
                self._compute_values(agent_batchs)

                for agent_id, agent in enumerate(self.mac.agents):
                    agent_batch = agent_batchs[f"agent_{agent_id}"]
                    origin_values.append(agent_batch.batch['values'])

                """sum q value，后续可以改成self.mixer的形式"""
                stacked_q_i = torch.stack(origin_values, dim=0)
                q_total = torch.sum(stacked_q_i, dim=0)

                # 这里需要考虑是否需要和reward对齐
                for agent_id, agent in enumerate(self.mac.agents):
                    agent_batch = agent_batchs[f"agent_{agent_id}"]
                    agent_batch.batch['values'] = q_total
                    # agent_batch = agent_batch.union(q_total) 
                    # 使用vdn sum critic
                    agent_batchs[f"agent_{agent_id}"] = agent_batch
                

        """
        这里需要慎重考虑,有些marl算法需要把两个人的不同actor的adv,结合common sum reward进行处理
        """ 
        with _timer("adv", timing_raw):
            self._compute_adv(agent_batchs, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics)


        # 处理完毕rollout和数据组织，开始更新actor和critic
        """目前是独立ippo更新，考虑qmix需要total td loss"""
        # update critic
        with _timer('update_critic', timing_raw):
            critic_output_metrics_all = {}
            for agent_id, agent in enumerate(self.mac.agents):
                agent_batch = agent_batchs[f"agent_{agent_id}"]
            
                if agent.use_critic:
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






# class RayVDNLearner(RayMARLLearner):
#     def __init__(self,
#                  config,
#                  mac, 
#                  num_agents,  # cotrian LLMs
#                  tokenizer,   # 单个，表示共用model
#                  role_worker_mapping: list[dict[Role, WorkerType]],
#                  resource_pool_manager: ResourcePoolManager, # 共用全局manager
#                  ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, # fsdp/Megatron
#                  processor=None,  # 单个，表示共用model
#                  reward_fn=None, # reward function公用相同的
#                  val_reward_fn=None,
#                  train_dataset: Optional[Dataset] = None,
#                  val_dataset: Optional[Dataset] = None,
#                  collate_fn=None,
#                  train_sampler: Optional[Sampler] = None,
#                  device_name="cuda",
#                  **kwargs
#                  ):
#         super().__init__(config,
#                          mac,
#                          num_agents,
#                          tokenizer,
#                          role_worker_mapping,
#                          resource_pool_manager,
#                          ray_worker_group_cls,
#                          processor,
#                          reward_fn,
#                          val_reward_fn,
#                          train_dataset,
#                          val_dataset,
#                          collate_fn,
#                          train_sampler,
#                          device_name)
        
#         # self.is_sum = kwargs.get('sum_rewards', False)  # 控制是否sum all agents rewards
#         self.is_sum = config.marl.sum_reward
#         self.cory_flip = config.marl.cory_flip
#         self.tb_dir = config.marl.tensorboard_dir


#         """
#         定义优化器和优化参数,这里的参数args和config需要再确认
#         """

#         # from marl.modules.mixers.vdn import VDNMixer
#         # from marl.modules.mixers.qmix import QMixer

#         # if config.marl.mixer is not None:
#         #     if config.marl.mixer == "vdn":
#         #         self.mixer = VDNMixer()
#         #     elif config.marl.mixer == "qmix":
#         #         self.mixer = QMixer(config)
#         #     else:
#         #         raise ValueError("Mixer {} is not vdn and qmix.".format(config.marl.mixer))
#         #     # self.critic_params += list(self.mixer.parameters())
#         #     # self.target_mixer = copy.deepcopy(self.mixer)
        
#         # from torch.optim import Adam
#         # self.actor_optimiser = Adam(params=self.actor_params, lr=config.lr, alpha=config.optim_alpha, eps=config.optim_eps)
#         # self.critic_optimiser = Adam(params=self.critic_params, lr=config.lr, alpha=config.optim_alpha, eps=config.optim_eps)

#         # # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
#         # self.target_mac = copy.deepcopy(self.mac)
    
#     # vdn
#     def fit(self):
#         """
#         The training loop of PPO.
#         The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
#         The light-weight advantage computation is done on the driver process.
#         """
#         from verl.utils.tracking import Tracking
#         from omegaconf import OmegaConf

#         logger = Tracking(project_name=self.config.trainer.project_name,
#                           experiment_name=self.config.trainer.experiment_name,
#                           default_backend=self.config.trainer.logger,
#                           config=OmegaConf.to_container(self.config, resolve=True))

#         self.global_steps = 0

#         """debug时候修改self.config.trainer['val_before_train']=False跳过validate和load ckpt"""
#         # # load checkpoint before doing anything
#         # self._load_checkpoint()
        
#         # perform validation before training
#         # currently, we only support validation using the reward_function.
#         # if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
#         #     val_metrics = self._validate()
#         #     pprint(f'Initial validation metrics: {val_metrics}')
#         #     logger.log(data=val_metrics, step=self.global_steps)
#         #     if self.config.trainer.get('val_only', False):
#         #         return

#         # add tqdm
#         progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

#         # we start from step 1
#         self.global_steps += 1
#         last_val_metrics = None

#         for epoch in range(self.config.trainer.total_epochs):
#             for batch_dict in self.train_dataloader:
#                 metrics = {}
#                 timing_raw = {}
                

#                 """这里需要考虑batch_dict的组织方式，需要考虑两个agent的batch_dict如何组织"""
#                 # 共享marl.train_dataloader控制统一的input/init state，然后每个agent处理自己的batch_dict用于模型训练
#                 agent_batchs: Dict[str, DataProto] = {}
#                 for agent_id, agent in enumerate(self.mac.agents):
#                     agent_batchs[f"agent_{agent_id}"] = DataProto.from_single_dict(deepcopy(batch_dict))

#                 # batch: DataProto = DataProto.from_single_dict(batch_dict)

#                 """这里不需要每个agent单独处理，因为batch data统一输入，不是每个agent单独设置"""
#                 # pop those keys for generation
#                 batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
#                 non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
#                 if "multi_modal_data" in agent_batchs[f"agent_0"].non_tensor_batch:
#                     non_tensor_batch_keys_to_pop.append("multi_modal_data")
#                 if "raw_prompt" in agent_batchs[f"agent_0"].non_tensor_batch:
#                     non_tensor_batch_keys_to_pop.append("raw_prompt")
#                 if "tools_kwargs" in agent_batchs[f"agent_0"].non_tensor_batch:
#                     non_tensor_batch_keys_to_pop.append("tools_kwargs")

#                 gen_batch_all = {}
#                 for batch_key, batchs in agent_batchs.items():
#                     gen_batch = batchs.pop(
#                         batch_keys=batch_keys_to_pop,
#                         non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
#                     )
#                     gen_batch_all[batch_key] = gen_batch

#                 # gen_batch_all: {batch_key: DataProto},每个agent单独靠key存储

#                 is_last_step = self.global_steps >= self.total_training_steps

#                 """每个agent内部处理各自的gen_batch_output，但是需要考虑数据如何在marl层面组织
#                 以及调用CORY的sequential_rollout需要修改input concat方式"""
#                 with _timer('step', timing_raw):
#                     # generate a batch
#                     with _timer('gen', timing_raw):
#                         # IPPO
#                         gen_batch_outputs_all = {}
#                         for agent, gen_batch_agent in zip(self.mac.agents, gen_batch_all.values()):
#                             gen_batch_output = agent.actor_rollout_wg.generate_sequences(gen_batch_agent)
#                             gen_batch_outputs_all[f"agent_{agent.agent_id}"] = gen_batch_output

                        


#                     """不用remax所以先注释掉"""
#                     # if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
#                     #     with _timer('gen_max', timing_raw):
#                     #         gen_baseline_batch = deepcopy(gen_batch)
#                     #         gen_baseline_batch.meta_info['do_sample'] = False
#                     #         gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

#                     #         batch = batch.union(gen_baseline_output)
#                     #         reward_baseline_tensor = self.reward_fn(batch)
#                     #         reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

#                     #         batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

#                     #         batch.batch['reward_baselines'] = reward_baseline_tensor

#                     #         del gen_baseline_batch, gen_baseline_output



#                     """完成了每个agent的rollout过程，后续复杂pipeline考虑函数包装，目前用于IPPO"""
#                     for agent_key, agent_batch  in agent_batchs.items():
#                         agent_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(agent_batch.batch))],
#                                                              dtype=object)
#                         # repeat to align with repeated responses in rollout
#                         agent_batch = agent_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
#                         agent_batch = agent_batch.union(gen_batch_outputs_all[agent_key])

#                         agent_batch.batch['response_mask'] = compute_response_mask(agent_batch)
#                         # balance the number of valid tokens on each dp rank.
#                         # Note that this breaks the order of data inside the batch.
#                         # Please take care when you implement group based adv computation such as GRPO and rloo
#                         if self.config.trainer.balance_batch:
#                             self._balance_batch(agent_batch, metrics=metrics)
#                         """小心检查这个metrics是否需要改成marl对齐"""
                        
#                         # compute global_valid tokens
#                         agent_batch.meta_info['global_token_num'] = torch.sum(agent_batch.batch['attention_mask'], dim=-1).tolist()

#                         agent_batchs[agent_key] = agent_batch

#                     """分别计算每个agent的reward，还没存储"""
#                     with _timer("reward", timing_raw):
#                         # compute reward model score
#                         reward_tensor_all = {}
#                         reward_extra_infos_dict_all = {}
#                         future_reward_all = {}
#                         # 这三个会用于后续的adv的计算
#                         for agent_id, agent in enumerate(self.mac.agents):
#                             agent_batch = agent_batchs[f"agent_{agent_id}"]
#                             if agent.use_rm:
#                                 reward_tensor = agent.rm_wg.compute_rm_score(agent_batch)
#                                 agent_batch = agent_batch.union(reward_tensor)

#                             if agent.config.reward_model.launch_reward_fn_async:
#                                 future_reward = compute_reward_async.remote(agent_batch, agent.config, agent.tokenizer)
#                                 future_reward_all[f"agent_{agent_id}"] = future_reward
#                             else:
#                                 reward_tensor, reward_extra_infos_dict = compute_reward(agent_batch, self.reward_fn)
#                                 reward_tensor_all[f"agent_{agent_id}"] = reward_tensor
#                                 reward_extra_infos_dict_all[f"agent_{agent_id}"] = reward_extra_infos_dict

#                             # 记录原始的reward
#                                 agent_reward_metrics = {
#                                     f"agent_{agent_id}_origin_reward/mean": torch.mean(reward_tensor.sum(-1)).detach().item(),
#                                     f"agent_{agent_id}_origin_reward/max": torch.max(reward_tensor.sum(-1)).detach().item(),
#                                     f"agent_{agent_id}_origin_reward/min": torch.min(reward_tensor.sum(-1)).detach().item(),
#                                 }
#                                 metrics.update(agent_reward_metrics)


#                         # 用于测试vdn的reward sum结果
#                         if self.is_sum:
#                             sum_reward = None
#                             for agent_key, agent_reward_tensor in reward_tensor_all.items():
#                                 if sum_reward is None:
#                                     sum_reward = agent_reward_tensor
#                                 else:
#                                     sum_reward += agent_reward_tensor
#                             # 将总和奖励赋值给每个agent
#                             for agent_key in reward_tensor_all.keys():
#                                 reward_tensor_all[agent_key] = sum_reward

                            


#                     """每个agent计算自己的old log prob存储到metric agent id"""
#                     # recompute old_log_probs
#                     with _timer("old_log_prob", timing_raw):
#                         for agent_id, agent in enumerate(self.mac.agents):
#                             agent_batch = agent_batchs[f"agent_{agent_id}"]


#                             old_log_prob = agent.actor_rollout_wg.compute_log_prob(agent_batch)
#                             entropys = old_log_prob.batch["entropys"]
#                             response_masks = agent_batch.batch["response_mask"]
#                             loss_agg_mode = agent.config.actor_rollout_ref.actor.loss_agg_mode
#                             entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
#                             old_log_prob_metrics = {f"agent_{agent_id}_actor/entropy_loss": entropy_loss.detach().item()}
#                             metrics.update(old_log_prob_metrics)
#                             old_log_prob.batch.pop("entropys")
#                             agent_batch = agent_batch.union(old_log_prob)

#                             agent_batchs[f"agent_{agent_id}"] = agent_batch

#                             if "rollout_log_probs" in agent_batch.batch.keys():
#                                 # TODO: we may want to add diff of probs too.
#                                 rollout_old_log_probs = agent_batch.batch["rollout_log_probs"]
#                                 actor_old_log_probs = agent_batch.batch["old_log_probs"]
#                                 attention_mask = agent_batch.batch["attention_mask"]
#                                 responses = agent_batch.batch["responses"]
#                                 response_length = responses.size(1)
#                                 response_mask = attention_mask[:, -response_length:]

#                                 rollout_probs = torch.exp(rollout_old_log_probs)
#                                 actor_probs = torch.exp(actor_old_log_probs)
#                                 rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
#                                 rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
#                                 rollout_probs_diff_max = torch.max(rollout_probs_diff)
#                                 rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
#                                 rollout_probs_diff_std = torch.std(rollout_probs_diff)
#                                 metrics.update(
#                                     {
#                                         f"agent_{agent_id}_training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
#                                         f"agent_{agent_id}_training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
#                                         f"agent_{agent_id}_training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
#                                     }
#                                 )


#                     """计算reference log prob"""
#                     if self.use_reference_policy:
#                         # compute reference log_prob
#                         with _timer("ref", timing_raw):
#                             for agent_id, agent in enumerate(self.mac.agents):
#                                 agent_batch = agent_batchs[f"agent_{agent_id}"]

#                                 """这里要考虑用marl层还是agent层"""
#                                 if not self.ref_in_actor:
#                                     ref_log_prob = agent.ref_policy_wg.compute_ref_log_prob(agent_batch)
#                                 else:
#                                     ref_log_prob = agent.actor_rollout_wg.compute_ref_log_prob(agent_batch)
#                                 agent_batch = agent_batch.union(ref_log_prob)
#                                 agent_batchs[f"agent_{agent_id}"] = agent_batch


#                     # compute values
#                     """这里应该改成marl.use_critic，因为learning算法会统一指定是否使用critic/Remax"""
#                     if self.use_critic:
#                         origin_values = []
                        
#                         with _timer("values", timing_raw):
#                             for agent_id, agent in enumerate(self.mac.agents):
#                                 agent_batch = agent_batchs[f"agent_{agent_id}"]
#                                 values = agent.critic_wg.compute_values(agent_batch)
#                                 agent_batch = agent_batch.union(values)

#                                 origin_values.append(agent_batch.batch['values'])
                            
#                             """后续可以改成self.mixer的形式"""
#                             stacked_q_i = torch.stack(origin_values, dim=0)
#                             q_total = torch.sum(stacked_q_i, dim=0)


#                             # 这里需要考虑是否需要和reward对齐
#                             for agent_id, agent in enumerate(self.mac.agents):
#                                 agent_batch = agent_batchs[f"agent_{agent_id}"]
#                                 agent_batch.batch['values'] = q_total
#                                 # agent_batch = agent_batch.union(q_total) 
#                                 # 使用vdn sum critic
#                                 agent_batchs[f"agent_{agent_id}"] = agent_batch

                            
#                     """这里需要慎重考虑，有些marl算法需要把两个人的不同actor的adv，结合common sum reward进行处理""" 
#                     with _timer("adv", timing_raw):

#                         kl_metrics_all = {}

#                         for agent_id, agent in enumerate(self.mac.agents):

#                             agent_batch = agent_batchs[f"agent_{agent_id}"]

#                             # we combine with rule-based rm
#                             reward_extra_infos_dict: dict[str, list]
#                             # 使用marl.config判断reward model模式
#                             if self.config.reward_model.launch_reward_fn_async:
#                                 for future_reward in future_reward_all.values():
#                                     reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
#                             else:
#                             # 采用公共reward计算一次，分别传给每个agent
#                                 reward_tensor = reward_tensor_all[f"agent_{agent_id}"]
#                                 reward_extra_infos_dict = reward_extra_infos_dict_all[f"agent_{agent_id}"]

#                             agent_batch.batch["token_level_scores"] = reward_tensor

#                             # print(f"{list(reward_extra_infos_dict.keys())=}")
#                             if reward_extra_infos_dict:
#                                 agent_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})


#                             # compute rewards. apply_kl_penalty if available
#                             # print(f" self.config.algorithm.use_kl_in_reward {self.config.algorithm.use_kl_in_reward}")
#                             if self.config.algorithm.use_kl_in_reward:
                                
#                                 agent_batch, kl_metrics = apply_kl_penalty(agent_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
#                                 kl_metrics_all[f"agent_{agent_id}"] = kl_metrics
#                             else:
#                                 agent_batch.batch["token_level_rewards"] = agent_batch.batch["token_level_scores"]

#                             agent_batchs[f"agent_{agent_id}"] = agent_batch

#                         """处理完毕两个agent的kl计算，这里暂时考虑合并两个agent的kl_metrics，考虑分开存储"""
#                         if self.config.algorithm.use_kl_in_reward:
#                             kl_metrics = {k: sum(v for v in kl_metrics_all.values()) for k in kl_metrics_all[0].keys()}
#                             metrics.update(kl_metrics)

#                         # compute advantages, executed on the driver process
#                         norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

#                         for agent_id, agent in enumerate(self.mac.agents):
#                             agent_batch = agent_batchs[f"agent_{agent_id}"]
#                             agent_batch = compute_advantage(
#                                 agent_batch,
#                                 adv_estimator=self.config.algorithm.adv_estimator,
#                                 gamma=self.config.algorithm.gamma,
#                                 lam=self.config.algorithm.lam,
#                                 num_repeat=self.config.actor_rollout_ref.rollout.n,
#                                 norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
#                                 multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
#                                 use_pf_ppo=self.config.algorithm.use_pf_ppo,
#                                 pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
#                                 pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
#                             )
#                             agent_batchs[f"agent_{agent_id}"] = agent_batch


#                     # 处理完毕rollout和数据组织，开始更新actor和critic
#                     """目前是独立ippo更新"""
#                     # update critic
#                     with _timer('update_critic', timing_raw):
#                         critic_output_metrics_all = {}
#                         for agent_id, agent in enumerate(self.mac.agents):
#                             agent_batch = agent_batchs[f"agent_{agent_id}"]
                        
#                             if agent.use_critic:
#                                 critic_output = agent.critic_wg.update_critic(agent_batch)

#                             """metrics需要合并"""
#                             critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])

#                             critic_output_metrics_all[f"agent_{agent_id}"] = critic_output_metrics

#                         for agent_key, agent_metrics in critic_output_metrics_all.items():
#                             metrics.update({f"{agent_key}_{k}": v for k, v in agent_metrics.items()})
#                         """目前是sum的，后续考虑分别记录"""
#                         # critic_output_metrics = {k: sum(v[k] for v in critic_output_metrics_all.values()) for k in next(iter(critic_output_metrics_all.values())).keys()}
#                         # metrics.update(critic_output_metrics)

#                     # implement critic warmup
#                     if self.config.trainer.critic_warmup <= self.global_steps:
#                         # update actor
#                         with _timer("update_actor", timing_raw):
#                             agent_output_metrics_all = {}
#                             for agent_id, agent in enumerate(self.mac.agents):
#                                 agent_batch = agent_batchs[f"agent_{agent_id}"]
#                                 agent_batch.meta_info["multi_turn"] = agent.config.actor_rollout_ref.rollout.multi_turn.enable
#                                 actor_output = agent.actor_rollout_wg.update_actor(agent_batch)

#                                 actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
#                                 agent_output_metrics_all[f"agent_{agent_id}"] = actor_output_metrics

#                                 # 更新了meta info，覆盖一下
#                                 agent_batchs[f"agent_{agent_id}"] = agent_batch

#                         # 分别存储每个agent的actor metrics
#                         for agent_key, agent_metrics in agent_output_metrics_all.items():
#                             metrics.update({f"{agent_key}_{k}": v for k, v in agent_metrics.items()})
#                         """目前是sum的"""
#                         # actor_output_metrics = {k: sum(v for v in agent_output_metrics_all.values()) for k in agent_output_metrics_all[0].keys()}
#                         # metrics.update(actor_output_metrics)

                    
#                     """暂时不debug，跳过这部分的rollout data，self dump generations"""
#                     # # Log rollout generations if enabled
#                     # rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
#                     # if rollout_data_dir:
#                     #     with _timer("dump_rollout_generations", timing_raw):

#                     #         print(batch.batch.keys())
#                     #         inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
#                     #         outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
#                     #         scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
#                     #         self._dump_generations(
#                     #             inputs=inputs,
#                     #             outputs=outputs,
#                     #             scores=scores,
#                     #             reward_extra_infos_dict=reward_extra_infos_dict,
#                     #             dump_path=rollout_data_dir,
#                     #         )


#                     """当前只validate了一个model，需要改成eval两个，metric记录两个"""
#                     # validate
#                     if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
#                         print(f"testing {self.global_steps}")

#                         """暂时不debug"""
#                         # with _timer("testing", timing_raw):
#                         #     val_metrics: dict = self._validate()
#                         #     if is_last_step:
#                         #         last_val_metrics = val_metrics
#                         # metrics.update(val_metrics)

#                     """这里通过marl的wrapper依次save了两个model，暂时不debug"""
#                     if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
#                         print(f"save_checkpoint {self.global_steps}")
#                         # with _timer("save_checkpoint", timing_raw):
#                         #     self._save_checkpoint()


#                 """这里需要合并两个agent的metrics，变成.agent1/.agent2形式，或者只用一个agent？"""
#                 metrics.update(
#                     {
#                         "training/global_step": self.global_steps,
#                         "training/epoch": epoch,
#                     }
#                 )
#                 # collect metrics
#                 for agent_key, agent_batch in agent_batchs.items():
#                     # data metrics
#                     data_metrics = compute_data_metrics(batch=agent_batch, use_critic=self.use_critic)
#                     metrics.update({f"{agent_key}_{k}": v for k, v in data_metrics.items()})

#                     # timing metrics
#                     timing_metrics = compute_timing_metrics(batch=agent_batch, timing_raw=timing_raw)
#                     metrics.update({f"{agent_key}_{k}": v for k, v in timing_metrics.items()})

#                     # throughput metrics
#                     n_gpus = self.resource_pool_manager.get_n_gpus()
#                     throughput_metrics = compute_throughout_metrics(batch=agent_batch, timing_raw=timing_raw, n_gpus=n_gpus)
#                     metrics.update({f"{agent_key}_{k}": v for k, v in throughput_metrics.items()})

#                 # for agent_batch in agent_batchs.values():
#                 #     metrics.update(compute_data_metrics(batch=agent_batch, use_critic=self.use_critic))
#                 #     metrics.update(compute_timing_metrics(batch=agent_batch, timing_raw=timing_raw))
#                 #     # TODO: implement actual tflpo and theoretical tflpo
#                 #     n_gpus = self.resource_pool_manager.get_n_gpus()
#                 #     metrics.update(compute_throughout_metrics(batch=agent_batch, timing_raw=timing_raw, n_gpus=n_gpus))

#                 # TODO: make a canonical logger that supports various backend
#                 logger.log(data=metrics, step=self.global_steps)

#                 progress_bar.update(1)
#                 self.global_steps += 1
#                 if is_last_step:
#                     pprint(f"Final validation metrics: {last_val_metrics}")
#                     progress_bar.close()
#                     return
