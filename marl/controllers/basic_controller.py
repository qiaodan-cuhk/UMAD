

"""
保留功能


外部调用


"""



"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

# Standard MARL Traning Base Cls

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

from marl.modules.agents import REGISTRY as AGENT_REGISTRY


# mac目前返回的只是一个 self.mac.agents = [PPO PPO PPO]

class BasicMAC:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
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
                 device_name="cuda"):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        # 奖励函数相关
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # 新增
        self.num_agents = num_agents

        """考虑放在marl层还是PPO层"""
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            # 检查每个agent是否都有ACTOR角色
            for agent_id in range(self.num_agents):
                assert MARLRole[f"agent_{agent_id}_ActorRollout"] in role_worker_mapping[agent_id], \
                    f'agent_{agent_id}_ActorRollout not found in {role_worker_mapping[agent_id].keys()=}'
            # assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        # Ray相关配置（需要传给每个agent）
        self.role_worker_mapping = role_worker_mapping   # list
        self.resource_pool_manager = resource_pool_manager  # global
        self.ray_worker_group_cls = ray_worker_group_cls  # global
        self.validation_generations_logger = ValidationGenerationsLogger()

        """改成MARL的，any ref"""
        # self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        # self.use_rm = Role.RewardModel in role_worker_mapping
        """这里后续需要考虑优化，目前参数统一传给所有homo agents，有些agent可能不需要ref model"""
        self.use_reference_policy = any(MARLRole[f"agent_{i}_RefPolicy"] in role_worker_mapping[i] 
                              for i in range(self.num_agents))
        self.use_rm = any(MARLRole[f"agent_{i}_RewardModel"] in role_worker_mapping[i] 
                  for i in range(self.num_agents))  # 默认false

        self.device_name = device_name



        """新版本verl新增"""
        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0


        # new version
        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError


        self._build_agents()


    def _build_agents(self):

        # 为每个agent创建独立的PPO trainer
        self.agents = []

        """
        要注意这些self.config的参数，因为是load到marl层
        对于独立PPO版本（目前），把self.use_critic下放到每个agent中，调用的公共config
        """
        shared_kwargs = {
            'hybrid_engine': self.hybrid_engine,
            'use_reference_policy': self.use_reference_policy,
            'use_rm': self.use_rm,
            # 'use_critic': self.use_critic,
        }

        
        for agent_id in range(self.num_agents):

            # 把MARLRole的mapping转换为每个PPO的Role mapping
            ppo_role_mapping = _convert_marl_to_ppo_roles(
                self.role_worker_mapping[agent_id], 
                agent_id
            )
            
            # agent_cls = AGENT_REGISTRY[self.config.agent_type]
            agent_cls = AGENT_REGISTRY['ppo']

            # 新增功能，为每个不同的agent rollout设置seed
            agent_config = deepcopy(self.config)
            agent_config.actor_rollout_ref.rollout.seed = 1000*agent_id + 9

            self.agents.append(agent_cls(agent_id=agent_id, 
                                            config=agent_config,  # 支持单独设置config
                                            tokenizer=self.tokenizer,
                                            processor=self.processor,
                                            role_worker_mapping=ppo_role_mapping,
                                            resource_pool_manager=self.resource_pool_manager,
                                            ray_worker_group_cls=self.ray_worker_group_cls,
                                            **shared_kwargs
                                        )
                                        )


