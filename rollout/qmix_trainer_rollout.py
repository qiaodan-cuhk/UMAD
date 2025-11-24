# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

# Standard MARL Traning Base Cls
import csv
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



class RayQMIXRolloutTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 num_agents,  # cotrian LLMs
                 tokenizer_list,   # 支持异构tokenizer list
                 role_worker_mapping: list[dict[Role, WorkerType]],
                 resource_pool_manager: ResourcePoolManager, # 共用全局manager
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, # fsdp/Megatron
                 processor_list=None,  # 支持异构processor list
                 reward_fn_list=None, # reward function公用相同的
                 val_reward_fn_list=None,
                 train_dataset: Optional[Dataset] = None,
                 val_dataset: Optional[Dataset] = None,
                 collate_fn=None,
                 train_sampler: Optional[Sampler] = None,
                 device_name="cuda"):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.config = config
        self.tokenizer_list = tokenizer_list
        self.processor_list = processor_list

        # 奖励函数相关
        self.reward_fn_list = reward_fn_list
        self.val_reward_fn_list = val_reward_fn_list

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
        else:
            self.kl_ctrl_in_reward = None

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


        extra_params = {
            'use_critic': self.use_critic,
            'use_reference_policy': self.use_reference_policy,
            'use_rm': self.use_rm,
            'ref_in_actor': self.ref_in_actor,
            'kl_ctrl_in_reward': self.kl_ctrl_in_reward,
            'hybrid_engine': self.hybrid_engine,
        }


        """
        全局层抽象类，主要功能
            mac: 控制 agents, self.agents = [PPO, PPO]
            learner: 负责训练，CTDE 算法，控制optimizer和update逻，继承verl的fit部分，以及core algorithm
            runner: 负责推理，组织多个agent rollout以及summary/major voting每一轮
            dataloader: 全局dataloader，所有agent共享一个，负责采样train dataloader bs作为prompts
        """

        from marl.controllers import REGISTRY as MAC_REGISTRY
        from marl.learners import REGISTRY as LEARNER_REGISTRY
        from marl.runners import REGISTRY as RUNNER_REGISTRY

        mac_cls = MAC_REGISTRY[self.config.marl.mac_cls]
        learner_cls = LEARNER_REGISTRY[self.config.marl.learner_cls]
        runner_cls = RUNNER_REGISTRY[self.config.marl.runner_cls]

        
        self.mac = mac_cls(config,
                 num_agents,  # cotrian LLMs
                 tokenizer_list,   # 异构list
                 role_worker_mapping,
                 resource_pool_manager, # 共用全局manager
                 ray_worker_group_cls, # fsdp/Megatron
                 processor_list,  # 异构list
                 **extra_params)

        
        self.learner = learner_cls(config,
                                    self.mac,
                                    num_agents,  # cotrian LLMs
                                    tokenizer_list,   # 异构tokenizer list
                                    device_name,
                                    **extra_params)
        

        self.runner = runner_cls(config,
                                    self.mac,
                                    num_agents,  # cotrian LLMs
                                    reward_fn_list, # reward function公用相同的
                                    device_name)

        
        # 创建全局dataloader，这个应该放在 fit 层
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        
        self.turns = self.config.marl.turns # 默认2轮
        self.sentence_level_reward = self.config.marl.sentence_level_reward    # 'mean', discount, counterfactual, learn
        # self.learn_sentence_q = self.config.marl.learn_sentence_q # 默认不学习sentence level的Q

        from marl.utils.buffer import MultiAgentTurnReplayBuffer
        self.buffer = MultiAgentTurnReplayBuffer(num_agents=self.num_agents, capacity=300000)   # 后期改成根据batch size和rollout n调整 buffer size

        self.rollout_n = self.config.actor_rollout_ref.rollout.n   # GRPO 的组样本数量


    # 根据每个agent的tokenizer处理对应的dataset
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        # from marl.utils.marl_utils import create_rl_dataset, create_rl_sampler
        from marl.utils.marl_utils import create_marl_dataset, create_marl_sampler

        if train_dataset is None:
            train_dataset = create_marl_dataset(self.config.data.train_files, self.config.data, self.tokenizer_list, self.processor_list)
        if val_dataset is None:
            val_dataset = create_marl_dataset(self.config.data.val_files, self.config.data, self.tokenizer_list, self.processor_list)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_marl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
            # from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")


    """计算原始reward"""
    def cal_origin_reward(self, agent_batchs, reward_fn_list):
        # compute reward model score
        reward_tensor_all = {}
        reward_extra_infos_dict_all = {}
        future_reward_all = {}
        # 这三个会用于后续的adv的计算

        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            if agent.use_rm:
                reward_tensor = agent.rm_wg.compute_rm_score(agent_batch)
                agent_batch = agent_batch.union(reward_tensor)

            if agent.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(agent_batch, agent.config, agent.tokenizer)
                future_reward_all[f"agent_{agent_id}"] = future_reward
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(agent_batch, reward_fn_list[agent_id])
                reward_tensor_all[f"agent_{agent_id}"] = reward_tensor
                reward_extra_infos_dict_all[f"agent_{agent_id}"] = reward_extra_infos_dict

        return reward_tensor_all




    """支持异构agent多轮推理"""

    # rollout 完了要计算confidence？之后再加吧
    def _validate(self):
        """
        为每个agent进行验证，返回带有agent前缀的metrics
        """
        agent_metrics = {}
        world_size = self.mac.agents[0].actor_rollout_wg.world_size
        sample_indexs = []  # 用于记录val集的索引
        all_team_rewards = [] # 记录所有agent的reward

          
        for test_data in self.val_dataloader:

            multi_turn_test_batchs: Dict[str, Dict[str, DataProto]] = {}

            # test_bs = self.config.data.val_batch_size  # 目前是null，即完整val一次bs
            test_bs = len(test_data["agent_0"]['index'])

            # 
            for turn_idx in range(self.turns):
                multi_turn_test_batchs[f"turn_{turn_idx}"] = {}
                for agent_id in range(self.num_agents):
                    test_batch = DataProto.from_single_dict(deepcopy(test_data[f"agent_{agent_id}"]))

                    test_batch_padded, test_batch_pad_size = pad_dataproto_to_divisor(test_batch, world_size)

                    # repeat test batch
                    test_batch_padded = test_batch_padded.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                                interleave=True)

                    multi_turn_test_batchs[f"turn_{turn_idx}"][f"agent_{agent_id}"] = test_batch_padded

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and multi_turn_test_batchs["turn_0"]["agent_0"][0].non_tensor_batch['reward_model']['style'] == 'model':
                continue

            test_metrics = {}
            reward_tensor_lst = []
            data_source_lst = []


            # 新增：收集索引
            print(f"test index shape: {len(test_batch.non_tensor_batch['index'])}")
            indices = test_batch.non_tensor_batch.get("index", [0] * test_bs)
            sample_indexs.extend(indices)


            #### 存储dialogues
            test_dialogues_dump = [{} for _ in range(test_bs)]
            test_team_rewards_dump = [{} for _ in range(test_bs)]

            # Lists to collect samples for the table
            sample_inputs = []
            sample_outputs = []
            sample_scores = []

            
            # Store original inputs
            input_ids = multi_turn_test_batchs["turn_0"]["agent_0"].batch['input_ids']
            input_texts = [self.tokenizer_list[0].decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)


            ##### heterogenenous LLM agents，each agent has different tokenized gen_batch
            test_batch_all = {}
            for agent_id in range(self.num_agents):
                test_batch_agent = multi_turn_test_batchs["turn_0"][f"agent_{agent_id}"]

                if 'multi_modal_inputs' in test_batch_agent.non_tensor_batch.keys():
                    test_gen_batch = test_batch_agent.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    test_gen_batch = test_batch_agent.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer_list[agent_id].eos_token_id,
                    'pad_token_id': self.tokenizer_list[agent_id].pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    'validate': True,
                }
                print(f'agent_{agent_id} test_gen_batch meta info: {test_gen_batch.meta_info}')

                # pad to be divisible by dp_size for all agents
                # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.mac.agents[agent_id].actor_rollout_wg.world_size)

                test_batch_all[f"agent_{agent_id}"] = test_gen_batch


            
            # test_output_gen_batch_padded = agent.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            
            #### 这里要考虑mamt版本的test rollout，不需要存储metric和turn test
            # 这是grpo版本rollout
            """测试early stop"""
            if self.config.algorithm.adv_estimator == "grpo":
                if self.config.marl.rollout_mode == "earlystop":
                    test_output_gen_batch_padded = self.runner.rollout_multi_turn_grpo_earlystop(test_batch_all, multi_turn_test_batchs, test_metrics, turn=self.turns, validate=True)
                else:
                    test_output_gen_batch_padded = self.runner.rollout_multi_turn_grpo(test_batch_all, multi_turn_test_batchs, test_metrics, turn=self.turns, validate=True)
            else:
                test_output_gen_batch_padded = self.runner.rollout_multi_turn(test_batch_all, multi_turn_test_batchs, test_metrics, turn=self.turns)

            test_output_gen_batch = {}
            for turn_id in range(self.turns):
                test_output_gen_batch[f"turn_{turn_id}"] = {} 
            # unpad
            for turn_id in range(self.turns):
                for agent_id in range(self.num_agents):
                    test_output_gen_batch_unpadded = unpad_dataproto(test_output_gen_batch_padded[f"turn_{turn_id}"][f"agent_{agent_id}"], pad_size=test_batch_pad_size)
                    test_output_gen_batch[f"turn_{turn_id}"][f"agent_{agent_id}"] = test_output_gen_batch_unpadded


            print(f'All agents validation generation end')

            # Store generated outputs
            # output_ids = test_output_gen_batch.batch['responses']
            # output_texts = [agent.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            # sample_outputs.extend(output_texts)

            #### log dialogues
            test_dialogues_dump, test_team_rewards_dump = self._get_dialogue_data(test_output_gen_batch, self.num_agents, self.turns, test_bs, test=True)
            # 收集团队奖励数据
            all_team_rewards.extend(test_team_rewards_dump)


            #### cal reward
            # for test_batch in 
            # reward_batchs = {}
            # score_batchs = {}
            # reward_tensor_batchs = {}
            # data_source_batchs = {}
            # for agent_id in range(self.num_agents):
                # reward_batch = multi_turn_test_batchs[f'turn_{self.turns-1}'][f'agent_{agent_id}'].union(test_output_gen_batch[f"turn_{self.turns-1}"][f"agent_{agent_id}"])
                # reward_batchs[f'agent_{agent_id}'] = reward_batch
                # test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # reward_tensor = self.val_reward_fn(test_batch)
                # Store scores
                # scores = reward_tensor.sum(-1).cpu().tolist()
                # sample_scores.extend(scores)
                # reward_tensor_lst.append(reward_tensor)
                # data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

                #### todo: 目前重复计算，而且response包含agent0和agent1的结果（很奇怪）
                # agent_reward_tensor = self.val_reward_fn_list[agent_id](reward_batch)
                # agent_scores = agent_reward_tensor.sum(-1).cpu()  # .tolist()
                # score_batchs[f'agent_{agent_id}'] = agent_scores
                # reward_tensor_batchs[f'agent_{agent_id}'] = agent_reward_tensor
                # data_source_batchs[f'agent_{agent_id}'] = reward_batch.non_tensor_batch.get('data_source', ['unknown'] * agent_reward_tensor.shape[0])


            # Log generations for this agent
            self._maybe_log_val_generations(
                dialogues = test_dialogues_dump,
                rewards_list = test_team_rewards_dump
            )

            # 这段是为了支持gsm8k/math混合validate的，根据data source区分reward
            # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
            # data_sources = np.concatenate(data_source_lst, axis=0)

            # # evaluate test_score based on data source
            # data_source_reward = {}
            # for i in range(reward_tensor.shape[0]):
            #     data_source = data_sources[i]
            #     if data_source not in data_source_reward:
            #         data_source_reward[data_source] = []
            #     data_source_reward[data_source].append(reward_tensor[i].item())

            # 为每个agent的metrics添加前缀
            agent_metric_dict = {}

            reward_keys = test_team_rewards_dump[0].keys()
            reward_len = len(test_team_rewards_dump)

            for key in reward_keys:
                reward_mean = sum(r[key] for r in test_team_rewards_dump) / reward_len
                agent_metric_dict[f'val/{key}_test_score'] = round(reward_mean, 3)
            
            # for agent_idx, (agent_key, agent_rewards) in enumerate(score_batchs.items()):
            #     agent_metric_dict[f'val/{agent_key}_test_score'] = agent_rewards.mean()
            agent_metrics.update(agent_metric_dict)

        # 如果需要计算sum reward metrics
        # if self.config.use_sum_reward:
        #     sum_metrics = {}
        #     for data_source in next(iter(agent_metrics.values())).keys():
        #         base_key = data_source.split('/')[-1]  # 获取数据源名称
        #         agent_rewards = [
        #             metrics[f'agent_{i}/val/test_score/{base_key}'] 
        #             for i, metrics in enumerate(agent_metrics.values())
        #         ]
        #         sum_metrics[f'sum/val/test_score/{base_key}'] = sum(agent_rewards)
        #     agent_metrics.update(sum_metrics)


        


        print("Calculate All Turn Reward Tensor")

        # 记录每一轮的reward
        all_turn_reward_tensor = []
        for turn_id in range(self.turns):
            turn_key = f"turn_{turn_id}"
            turn_batchs = test_output_gen_batch[turn_key]

            agents_turn_reward_tensor = self.cal_origin_reward(turn_batchs, self.val_reward_fn_list)

            # 为每个样本记录这一轮的奖励
            turn_rewards = []
            for bs_idx in range(len(sample_indexs)):
                sample_reward = {}
                for agent_idx in range(self.num_agents):
                    agent_key = f"agent_{agent_idx}"
                    agent_reward = agents_turn_reward_tensor[agent_key][bs_idx].sum().item()
                    sample_reward[agent_key] = agent_reward
                turn_rewards.append(sample_reward)
            
            all_turn_reward_tensor.append(turn_rewards)


        print("Save All Turn Reward Tensor to CSV")
        # 在所有batch处理完成后，记录index和各agent的reward
        val_data_index_score = []

        for idx in range(len(sample_indexs)):
            row_data = {"index": sample_indexs[idx]}
            ## 为每个agent每个turn添加奖励列
            for turn_id in range(self.turns):
                for agent_idx in range(self.num_agents):
                    agent_key = f"agent_{agent_idx}"
                    turn_key = f"turn_{turn_id}"
                    column_key = f"{agent_key}_{turn_key}"
                    row_data[column_key] = all_turn_reward_tensor[turn_id][idx][agent_key]
            
            val_data_index_score.append(row_data)


        validation_data_dir = os.path.join(self.config.trainer.get("rollout_data_dir", None), self.config.trainer.get("experiment_name", "latest"))
        if validation_data_dir:
            os.makedirs(validation_data_dir, exist_ok=True)
            file_path = os.path.join(validation_data_dir, f"val_index_score_{self.global_steps}.csv")
            print(f"save val index score csv in {file_path}")
            
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # 修改头部，包含每个agent每个turn的列
                header = ["index"]
                for turn_id in range(self.turns):
                    for agent_idx in range(self.num_agents):
                        header.append(f"agent_{agent_idx}_turn_{turn_id}")
                writer.writerow(header)
                
                # 写入数据
                for row in val_data_index_score:
                    row_values = [row["index"]]
                    for turn_id in range(self.turns):
                        for agent_idx in range(self.num_agents):
                            column_key = f"agent_{agent_idx}_turn_{turn_id}"
                            row_values.append(row[column_key])
                    writer.writerow(row_values)
                    
                
                # header = ["index"] + [f"agent_{i}" for i in range(self.num_agents)]
                # writer.writerow(header)
                
                # # 写入数据
                # for row in val_data_index_score:
                #     row_values = [row["index"]] + [row[f"agent_{i}"] for i in range(self.num_agents)]
                #     writer.writerow(row_values)

        return agent_metrics



    def _maybe_log_val_generations(self, dialogues, rewards_list):
        """为每个agent记录生成结果"""
        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        # 随机选择要记录的对话索引
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(dialogues), min(generations_to_log, len(dialogues)), replace=False)

        # 获取采样的对话和reward
        sampled_dialogues = [dialogues[i] for i in sample_indices]
        sampled_rewards = [rewards_list[i] for i in sample_indices]


        # sampled_entries = []
        # for dialogue, reward in zip(sampled_dialogues, sampled_rewards):
        #     entry = {}
            
        #     # 处理每个turn的对话
        #     for turn_key, messages in dialogue.items():
        #         entry[turn_key] = messages
            
        #     # 添加rewards
        #     entry["rewards"] = reward
        #     entry["step"] = self.global_steps
            
        #     sampled_entries.append(entry)

        # 记录到logger，仅支持wandb，改成jsonl
        # self.validation_generations_logger.log(
        #     self.config.trainer.logger,
        #     sampled_entries,
        #     self.global_steps,
        #     table_name="validation_samples"
        # )
        # rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        rollout_data_dir = os.path.join(self.config.trainer.get("rollout_data_dir", None), self.config.trainer.get("experiment_name", "latest"))
        self._dump_generations(dialogues=sampled_dialogues,
                                team_rewards=sampled_rewards,
                                reward_extra_infos_dict=None,
                                dump_path=rollout_data_dir,
                                validate=True
                            )

        # # Create tuples of (input, output, score) and sort by input text
        # samples = list(zip(inputs, outputs, scores))
        # samples.sort(key=lambda x: x[0])  # Sort by input text

        # # Use fixed random seed for deterministic shuffling
        # rng = np.random.RandomState(42)
        # rng.shuffle(samples)

        # # Take first N samples after shuffling
        # samples = samples[:generations_to_log]

        # # Log to each configured logger with agent prefix
        # self.validation_generations_logger.log(
        #     self.config.trainer.logger, 
        #     samples, 
        #     self.global_steps,
        #     table_name=f"agent_{agent_id}_generations"  # 区分不同agent的生成结果
        # )


    """
    这个版本是每个agent注册自己的ray address, 保持config和对应功能的相对独立
    需要调用mac.init_workers创建每个agent的RayWorkerGroup
    以及根据config创建runner和mixer的RayWorkerGroup
    """
    def init_workers(self):
        # shared resource pool
        self.resource_pool_manager.create_resource_pool()

        # create mac agents
        # for agent_id, agent in enumerate(self.mac.agents):
        #     agent.init_workers()
        for agent in self.mac.agents:
            agent.init_workers()
        print(f"All Agents init_workers Done")

        # create learner embed model
        if self.config.marl.learner_cls in ["qmix", "vdn"]:
            self.learner.init_workers(self.resource_pool_manager, self.ray_worker_group_cls)
        else:
            self.learner.init_workers()
        print(f"Learner init_workers Done")

        self.runner.init_workers()
        print(f"All Agents init_workers Done")



    """
    这两个目前先不管，还需要考虑ppo层的存储路径问题
    但是需要放在trainer层完成，根据train steps决定
    """
    def _save_checkpoint(self, path):
        """直接调用每个agent的save"""
        for agent_id, agent in enumerate(self.mac.agents):
            agent_path = f"{path}/{agent_id}"
            agent._save_checkpoint(agent_path)

    def _load_checkpoint(self, path):
        """直接调用每个agent的load"""
        for agent_id, agent in enumerate(self.mac.agents):
            agent_path = f"{path}/{agent_id}"
            agent._load_checkpoint(agent_path)


    # def _write_dialogues_to_json(dialogue_data_list, rewards_data_list, filename):
    #     all_dialogues = []
        
    #     # 遍历每组对话数据
    #     for dialogue_data, rewards_data in zip(dialogue_data_list, rewards_data_list):
    #         # 构建单个对话的完整数据
    #         dialogue_entry = dialogue_data[0].copy()  # 复制所有turns
    #         dialogue_entry["rewards"] = rewards_data[0]  # 添加rewards
    #         all_dialogues.append(dialogue_entry)
        
    #     # 写入JSON文件
    #     with open(filename, "w", encoding='utf-8') as f:
    #         json.dump(all_dialogues, f, ensure_ascii=False, indent=2)


    """dump每个step下bs的文本数据"""
    def _dump_generations(self, dialogues, team_rewards, reward_extra_infos_dict, dump_path, validate=False):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        
        if validate:
            filename = os.path.join(dump_path, f"test_{self.global_steps}.jsonl")
        else:
            filename = os.path.join(dump_path, f"train_{self.global_steps}.jsonl")

        n = len(dialogues)
        base_data = {
            "dialogues": dialogues,
            "team_rewards": team_rewards,
            "step": [self.global_steps] * n,
        }

        # all turns and agents have the same reward_extra_infos_dict
        if not validate:
            for k, v in reward_extra_infos_dict['turn_0']['agent_0'].items():
                if len(v) == n:
                    base_data[k] = v

        # self._write_dialogues_to_json()
        with open(filename, "w", encoding='utf-8') as f:
            for i in range(n):
                entry = {}
                
                # 处理每个turn的对话
                for turn_key, messages in dialogues[i].items():
                    entry[turn_key] = messages
                
                # 添加rewards
                entry["rewards"] = team_rewards[i]
                entry["step"] = self.global_steps
                
                # 添加其他额外的reward信息
                for k, v in base_data.items():
                    if k not in ["dialogues", "team_rewards", "step"]:
                        entry[k] = v[i]
                
                # 使用indent参数来格式化JSON输出
                f.write(json.dumps(entry, ensure_ascii=False, indent=2) + "\n\n")

        print(f"Dumped generations to {filename}")

    

    def _get_dialogue_data(self, multi_turn_batchs, num_agents, turns, bs, test=False):
        """
        args:
            multi_turn_batchs: [{'turn_0': {"agent_0": DataProto, "agent_1": DataProto}, 'turn_1': {"agent_0": DataProto, "agent_1": DataProto}}]
            num_agents: int
            turns: int
            bs: int
            test: use val_reward_fn or reward_fn to calculate response rewards
        return:
            dialogues_dump: [{"turn_0": [{"role": "user", "content": prompt}, {"role": "agent_0", "content": response 0}, {"role": "agent_1", "content": response 1}]
                            "turn_1": [{"role": "user", "content": prompt_1}, {"role": "agent_0", "content": response 0}, {"role": "agent_1", "content": response 1}]}]
            team_rewards_dump: [{"agent_0": origin_reward, "agent_1": origin_reward, "team_reward": team_sum}]
            scores_dump: 暂时不存，太长了
        """

        dialogues_dump = [{} for _ in range(bs)]
        team_rewards_dump = [{} for _ in range(bs)]

        for turn_idx in range(turns):
            turn_key = f"turn_{turn_idx}"
            turn_batchs = multi_turn_batchs[turn_key]

            turn_prompts = self.tokenizer_list[0].batch_decode(turn_batchs['agent_0'].batch["prompts"], skip_special_tokens=True)

            agent_responses = {}
            for agent_idx in range(num_agents):
                agent_key = f"agent_{agent_idx}"
                agent_responses[agent_key] = self.tokenizer_list[agent_idx].batch_decode(turn_batchs[agent_key].batch["responses"], skip_special_tokens=True)

            for bs_idx in range(bs):
                turn_dialogue = [{"role": "user", "content": turn_prompts[bs_idx]}]
                for agent_idx in range(num_agents):
                    turn_dialogue.append({
                        "role": f"agent_{agent_idx}",
                        "content": agent_responses[f"agent_{agent_idx}"][bs_idx]
                    })

                dialogues_dump[bs_idx][turn_key] = turn_dialogue


        # 计算每个智能体的奖励（使用最后一轮的奖励）
        last_turn_key = f"turn_{turns-1}"
        if test:
            agents_origin_reward_tensor = self.cal_origin_reward(multi_turn_batchs[last_turn_key], self.val_reward_fn_list)
        else:
            agents_origin_reward_tensor = self.cal_origin_reward(multi_turn_batchs[last_turn_key], self.reward_fn_list)

        for bs_idx in range(bs):
            team_reward_dict = {}
            team_total = 0
            
            for agent_idx in range(num_agents):
                agent_key = f"agent_{agent_idx}"
                # 获取该智能体在最后一轮的奖励

                agent_reward = agents_origin_reward_tensor[agent_key][bs_idx].sum().item()
                team_reward_dict[agent_key] = agent_reward
                team_total += agent_reward
            
            # 添加团队总奖励
            team_reward_dict["team_reward"] = team_total
            team_rewards_dump[bs_idx] = team_reward_dict


        return dialogues_dump, team_rewards_dump





    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0



        # # load checkpoint before doing anything
        # self._load_checkpoint()
        
        # === debug 先关掉 ===
        # perform validation before training
        # currently, we only support validation using the reward_function.


        if self.val_reward_fn_list is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)




    def _get_dialogue_data_grpo(self, multi_turn_batchs, num_agents, turns, reward_tensor_all, log_nums=5):
        """
        为GRPO获取对话数据，包含QMIX奖励信息、优势(advantages)，以及样本元信息(type/index/level)
        返回：
        - dialogues_dump: 每个样本的多轮对话
        - team_rewards_dump: 每个样本最终团队奖励聚合
        - qmix_rewards_dump: 每个样本、每轮、每agent的estimated rewards(句级/词级)
        - adv_values_dump: 每个样本、每轮、每agent的advantages(句级/词级)
        - problem_type / problem_index / problem_level: 每个样本的元信息
        """


        all_log_samples = []

        for log_idx in range(log_nums):
            
            group_data = {
                "problem_type": multi_turn_batchs["turn_0"]["agent_0"].non_tensor_batch["type"][self.rollout_n*log_idx],
                "problem_index": multi_turn_batchs["turn_0"]["agent_0"].non_tensor_batch["index"][self.rollout_n*log_idx],
                "problem_level": multi_turn_batchs["turn_0"]["agent_0"].non_tensor_batch["level"][self.rollout_n*log_idx],
                "dialogues": {}
            }

                      
            for turn_idx in range(turns):
                all_turns_dialogues = []  
                turn_key = f"turn_{turn_idx}"
                turn_sample_nums = int(self.rollout_n ** (turn_idx+1))  # 该turn的样本数量
                start_idx = turn_sample_nums*log_idx
                end_idx = turn_sample_nums*(log_idx+1)

                turn_batchs = multi_turn_batchs[turn_key]


                turn_prompts = self.tokenizer_list[0].batch_decode(
                    turn_batchs['agent_0'].batch["prompts"][start_idx:end_idx], skip_special_tokens=True
                )

                agent_responses = {}
                for agent_idx in range(num_agents):
                    agent_key = f"agent_{agent_idx}"
                    agent_responses[agent_key] = self.tokenizer_list[agent_idx].batch_decode(
                        turn_batchs[agent_key].batch["responses"][start_idx:end_idx], skip_special_tokens=True
                    )

                if self.global_steps >= 10:  
                    estimated_rewards = [turn_batchs[f'agent_{agent_idx}'].batch["estimated_rewards"][start_idx:end_idx] for agent_idx in range(num_agents)]
                    adv_values = [turn_batchs[f'agent_{agent_idx}'].batch["advantages"][start_idx:end_idx] for agent_idx in range(num_agents)]
                else:
                    estimated_rewards = [torch.zeros_like(turn_batchs[f'agent_{agent_idx}'].batch["responses"][start_idx:end_idx], dtype=torch.float32) for agent_idx in range(num_agents)]
                    adv_values = [torch.zeros_like(turn_batchs[f'agent_{agent_idx}'].batch["responses"][start_idx:end_idx], dtype=torch.float32) for agent_idx in range(num_agents)]

                uids = [turn_batchs[f'agent_{agent_idx}'].non_tensor_batch["uid"][start_idx:end_idx] for agent_idx in range(num_agents)]

                if turn_idx == turns - 1:

                    
                    rewards_tensor_current = [reward_tensor_all[f"turn_{turn_idx}"][f"agent_{agent_idx}"][start_idx:end_idx] for agent_idx in range(num_agents)]

                    for bs_idx in range(turn_sample_nums):
                        turn_dialogue = [{"role": "user", "content": turn_prompts[bs_idx]}]
                        for agent_idx in range(num_agents):
                            agent_key = f"agent_{agent_idx}"
                            turn_dialogue.append({
                                "role": agent_key,
                                "content": agent_responses[agent_key][bs_idx],
                                'estimated_rewards': estimated_rewards[agent_idx][bs_idx].sum().item(),
                                'GRPO_Adv': adv_values[agent_idx][bs_idx].mean().item(),
                                'Real Rewards': rewards_tensor_current[agent_idx][bs_idx].sum().item(),
                                'uid': uids[agent_idx][bs_idx],
                            })

                        all_turns_dialogues.append(turn_dialogue)
                else:
                    for bs_idx in range(turn_sample_nums):
                        turn_dialogue = [{"role": "user", "content": turn_prompts[bs_idx]}]
                        for agent_idx in range(num_agents):
                            agent_key = f"agent_{agent_idx}"
                            turn_dialogue.append({
                                "role": agent_key,
                                "content": agent_responses[agent_key][bs_idx],
                                'estimated_rewards': estimated_rewards[agent_idx][bs_idx].sum().item(),
                                'GRPO_Adv': adv_values[agent_idx][bs_idx].mean().item(),
                                'uid': uids[agent_idx][bs_idx],
                                # 'Real Rewards': reward_tensor_all[agent_key][bs_idx].sum().item(),
                            })

                        all_turns_dialogues.append(turn_dialogue)

                group_data["dialogues"][turn_key] = all_turns_dialogues

            all_log_samples.append(group_data)

        return all_log_samples


    
    def dump_dialogues_grpo(self, data_list, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        for i, item in enumerate(data_list):
            problem_index = item.get('problem_index')
            file_name = f'problem_{problem_index}.json'
            file_path = os.path.join(out_dir, f'{file_name}')

            # 如有不可序列化对象（如 numpy/torch/tensor），用 default=str 兜底
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False, indent=2, default=str)





    def _extract_problem_data(self, multi_turn_batchs, reward_tensor_all, problem_idx, step):
        """提取单个问题的完整数据"""
        
        problem_data = {
            "problem_id": f"problem_{problem_idx}",
            "step": step,
            "trajectories": [],
            "groups": {}
        }
        
        # 构建所有轨迹数据
        for trajectory_id in range(25):  # GRPO生成的轨迹数量
            trajectory_data = self._build_trajectory_data(
                multi_turn_batchs, reward_tensor_all,
                problem_idx, trajectory_id
            )
            problem_data["trajectories"].append(trajectory_data)
        
        # 分析每个turn的组内奖励分布
        problem_data["groups"] = self._analyze_turn_groups(problem_data["trajectories"])
        
        return problem_data



    def _build_trajectory_data(self, multi_turn_batchs, qmix_rewards, 
                            reward_tensor_all, problem_idx, trajectory_id):
        """构建单条轨迹的数据"""
        
        trajectory_data = {
            "trajectory_id": trajectory_id,
            "group_id": 0,  # 简化处理，所有轨迹为一组
            "dialogues": {},
            "rewards": {}
        }
        
        # 构建每轮的对话数据
        for turn_idx in range(self.turns):
            turn_key = f"turn_{turn_idx}"
            turn_dialogue = []
            
            # 添加用户prompt
            turn_dialogue.append({
                "role": "user",
                "content": self._get_turn_prompt(multi_turn_batchs, problem_idx, turn_idx)
            })
            
            # 添加每个agent的回复
            turn_rewards = {}
            for agent_id in range(self.num_agents):
                agent_key = f"agent_{agent_id}"
                agent_data = self._extract_agent_data(
                    multi_turn_batchs, qmix_rewards, reward_tensor_all,
                    problem_idx, trajectory_id, turn_idx, agent_id
                )
                
                turn_dialogue.append({
                    "role": agent_key,
                    "content": agent_data["response"],
                    "rewards": agent_data["rewards"]
                })
                
                turn_rewards[agent_key] = agent_data["rewards"]["final_reward"]
            
            trajectory_data["dialogues"][turn_key] = turn_dialogue
            trajectory_data["rewards"][turn_key] = turn_rewards
        
        # 计算总奖励
        total_rewards = {}
        for turn_key, turn_rewards in trajectory_data["rewards"].items():
            for agent_key, reward in turn_rewards.items():
                if agent_key not in total_rewards:
                    total_rewards[agent_key] = 0
                total_rewards[agent_key] += reward
        
        trajectory_data["rewards"]["total"] = total_rewards
        trajectory_data["rewards"]["team_total"] = sum(total_rewards.values())
        
        return trajectory_data

    def _analyze_turn_groups(self, trajectories):
        """分析每个turn的组内奖励分布"""
        
        groups = {}
        
        for turn_idx in range(self.turns):
            turn_key = f"turn_{turn_idx}"
            turn_groups = {
                "group_0": {  # 简化处理，所有轨迹为一组
                    "trajectory_ids": list(range(len(trajectories))),
                    "reward_statistics": {},
                    "ranking": []
                }
            }
            
            # 统计每个agent的奖励分布
            for agent_id in range(self.num_agents):
                agent_key = f"agent_{agent_id}"
                agent_rewards = [
                    traj["rewards"][turn_key][agent_key]
                    for traj in trajectories
                ]
                
                turn_groups["group_0"]["reward_statistics"][agent_key] = {
                    "mean": np.mean(agent_rewards),
                    "std": np.std(agent_rewards),
                    "min": np.min(agent_rewards),
                    "max": np.max(agent_rewards)
                }
            
            # 轨迹排名
            turn_total_rewards = [
                sum(traj["rewards"][turn_key].values())
                for traj in trajectories
            ]
            
            ranking = sorted(enumerate(turn_total_rewards), key=lambda x: x[1], reverse=True)
            turn_groups["group_0"]["ranking"] = [
                {"trajectory_id": idx, "total_reward": reward, "rank": rank+1}
                for rank, (idx, reward) in enumerate(ranking)
            ]
            
            groups[turn_key] = turn_groups
        
        return groups

