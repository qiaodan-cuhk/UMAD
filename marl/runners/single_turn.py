
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

# Worker/checkpoint management is delegated to each agent.
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.single_controller.ray.base import create_colocated_worker_cls


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




WorkerType = Type[Worker]


from marl.utils.marl_utils import MARLRole, Role


from marl.modules.agents.ppo_agent import ResourcePoolManager
from marl.utils.marl_utils import AdvantageEstimator, compute_advantage, compute_response_mask, apply_kl_penalty


from marl.utils.marl_utils import _convert_marl_to_ppo_roles, _timer 


class Single_Turn_Runner:
    def __init__(self,
                 config,
                 mac,
                 num_agents,  # cotrian LLMs
                 tokenizer,   # Shared-model tokenizer.
                 role_worker_mapping: list[dict[Role, WorkerType]],
                 resource_pool_manager: ResourcePoolManager,  # Shared global manager.
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, # fsdp/Megatron
                 processor=None,  # Shared-model processor.
                 reward_fn=None,  # Shared reward function.
                 val_reward_fn=None,
                 train_dataset: Optional[Dataset] = None,
                 val_dataset: Optional[Dataset] = None,
                 collate_fn=None,
                 train_sampler: Optional[Sampler] = None,
                 device_name="cuda"):
        
        self.config = config
        self.reward_fn = reward_fn

        self.is_sum = config.marl.sum_reward
        self.cory_flip = config.marl.cory_flip
        self.tb_dir = config.marl.tensorboard_dir

        self.mac = mac
        # self.mac.agents = [PPO PPO]



    # Sequence balancing is kept as a utility, but rollout disables it because
    # independent reordering can desynchronize heterogeneous agent responses.
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        # world_size = self.actor_rollout_wg.world_size
        world_size = self.mac.agents[0].actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)

        metrics.update(global_balance_stats)



    def rollout(self, gen_batch_all, agent_batchs, metrics):

        gen_batch_outputs_all = {}
        for agent, gen_batch_agent in zip(self.mac.agents, gen_batch_all.values()):
            gen_batch_output = agent.actor_rollout_wg.generate_sequences(gen_batch_agent)
            gen_batch_outputs_all[f"agent_{agent.agent_id}"] = gen_batch_output



        """Attach each agent's rollout output and IPPO trajectory metadata."""
        for agent_key, agent_batch  in agent_batchs.items():
            agent_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(agent_batch.batch))],
                                                    dtype=object)
            # repeat to align with repeated responses in rollout
            agent_batch = agent_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            agent_batch = agent_batch.union(gen_batch_outputs_all[agent_key])

            agent_batch.batch['response_mask'] = compute_response_mask(agent_batch)
            # balance the number of valid tokens on each dp rank.
            # Note that this breaks the order of data inside the batch.
            # Please take care when you implement group based adv computation such as GRPO and rloo

            # Disabled for multi-agent trajectories because reordering one
            # agent's batch can desynchronize response pairs across agents.
            # if self.config.trainer.balance_batch:
            #     self._balance_batch(agent_batch, metrics=metrics)
            
            # compute global_valid tokens
            agent_batch.meta_info['global_token_num'] = torch.sum(agent_batch.batch['attention_mask'], dim=-1).tolist()

            agent_batchs[agent_key] = agent_batch



    """Compute per-agent rewards and optional summed team rewards."""
    def cal_reward(self, agent_batchs, metrics):
        # compute reward model score
        reward_tensor_all = {}
        reward_extra_infos_dict_all = {}
        future_reward_all = {}

        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            if agent.use_rm:
                reward_tensor = agent.rm_wg.compute_rm_score(agent_batch)
                agent_batch = agent_batch.union(reward_tensor)

            if agent.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(agent_batch, agent.config, agent.tokenizer)
                future_reward_all[f"agent_{agent_id}"] = future_reward
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(agent_batch, self.reward_fn)
                reward_tensor_all[f"agent_{agent_id}"] = reward_tensor
                reward_extra_infos_dict_all[f"agent_{agent_id}"] = reward_extra_infos_dict

            # Keep the original per-agent reward before team aggregation.
            agent_reward_metrics = {
                f"agent_{agent_id}_origin_reward/mean": torch.mean(reward_tensor.sum(-1)).detach().item(),
                f"agent_{agent_id}_origin_reward/max": torch.max(reward_tensor.sum(-1)).detach().item(),
                f"agent_{agent_id}_origin_reward/min": torch.min(reward_tensor.sum(-1)).detach().item(),
            }
            metrics.update(agent_reward_metrics)

        
        if self.is_sum:
            sum_reward = None
            for agent_key, agent_reward_tensor in reward_tensor_all.items():
                if sum_reward is None:
                    sum_reward = agent_reward_tensor
                else:
                    sum_reward += agent_reward_tensor
            for agent_key in reward_tensor_all.keys():
                reward_tensor_all[agent_key] = sum_reward

            team_reward_metrics = {f"total/reward_mean": torch.mean(sum_reward.sum(-1)).detach().item(),
                                   f"total/reward_max": torch.max(sum_reward.sum(-1)).detach().item(),
                                   f"total/reward_min": torch.min(sum_reward.sum(-1)).detach().item(),
                                   }
            metrics.update(team_reward_metrics)


        return reward_tensor_all, reward_extra_infos_dict_all, future_reward_all



    """Compute the deterministic rollout baseline used by REMAX."""
    def rollout_remax(self, gen_batch_all, ):

        gen_baseline_batch = deepcopy(gen_batch_all)
        gen_baseline_batch.meta_info['do_sample'] = False
        gen_baseline_output = self.mac.agents.actor_rollout_wg.generate_sequences(gen_baseline_batch)

        batch = batch.union(gen_baseline_output)
        reward_baseline_tensor = self.reward_fn(batch)
        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

        batch.batch['reward_baselines'] = reward_baseline_tensor

        del gen_baseline_batch, gen_baseline_output


    # Reserved for optional LLM-based rollout extensions.
    def init_workers(self):
        pass
