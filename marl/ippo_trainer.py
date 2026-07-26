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



class RayIPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 num_agents,  # cotrian LLMs
                 tokenizer_list,   # Per-agent tokenizer list.
                 role_worker_mapping: list[dict[Role, WorkerType]],
                 resource_pool_manager: ResourcePoolManager,  # Shared global manager.
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, # fsdp/Megatron
                 processor_list=None,  # Per-agent processor list.
                 reward_fn_list=None,  # Per-agent reward functions.
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

        # Reward functions are kept per agent because tokenizers may differ.
        self.reward_fn_list = reward_fn_list
        self.val_reward_fn_list = val_reward_fn_list

        self.num_agents = num_agents

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            # Every agent must own an ActorRollout role under the hybrid engine.
            for agent_id in range(self.num_agents):
                assert MARLRole[f"agent_{agent_id}_ActorRollout"] in role_worker_mapping[agent_id], \
                    f'agent_{agent_id}_ActorRollout not found in {role_worker_mapping[agent_id].keys()=}'
            # assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        # Ray runtime config is forwarded to each per-agent PPO wrapper.
        self.role_worker_mapping = role_worker_mapping   # list
        self.resource_pool_manager = resource_pool_manager  # global
        self.ray_worker_group_cls = ray_worker_group_cls  # global
        self.validation_generations_logger = ValidationGenerationsLogger()

        """Reference/RM usage is enabled if any MARL agent provides that role."""
        # self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        # self.use_rm = Role.RewardModel in role_worker_mapping
        self.use_reference_policy = any(MARLRole[f"agent_{i}_RefPolicy"] in role_worker_mapping[i] 
                              for i in range(self.num_agents))
        self.use_rm = any(MARLRole[f"agent_{i}_RewardModel"] in role_worker_mapping[i] 
                  for i in range(self.num_agents))

        self.device_name = device_name



        """LoRA actors use the actor weights as the reference policy."""
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
        Top-level MARL components:
            mac: owns per-agent PPO wrappers.
            learner: updates policies and advantages.
            runner: coordinates multi-agent rollout and prompt construction.
            dataloader: provides shared prompt batches for all agents.
        """

        from marl.controllers import REGISTRY as MAC_REGISTRY
        from marl.learners import REGISTRY as LEARNER_REGISTRY
        from marl.runners import REGISTRY as RUNNER_REGISTRY

        mac_cls = MAC_REGISTRY[self.config.marl.mac_cls]
        learner_cls = LEARNER_REGISTRY[self.config.marl.learner_cls]
        runner_cls = RUNNER_REGISTRY[self.config.marl.runner_cls]

        
        self.mac = mac_cls(config,
                 num_agents,  # cotrian LLMs
                 tokenizer_list,
                 role_worker_mapping,
                 resource_pool_manager,
                 ray_worker_group_cls, # fsdp/Megatron
                 processor_list,
                 **extra_params)

        
        self.learner = learner_cls(config,
                                    self.mac,
                                    num_agents,  # cotrian LLMs
                                    tokenizer_list,
                                    device_name,
                                    **extra_params)
        

        self.runner = runner_cls(config,
                                    self.mac,
                                    num_agents,  # cotrian LLMs
                                    reward_fn_list,
                                    device_name)

        
        # Build global dataloaders shared by all agents.
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        
        self.turns = self.config.marl.turns
        # self.sentence_level_reward = self.config.marl.sentence_level_reward    # 'mean', discount, counterfactual, learn
        # self.learn_sentence_q = self.config.marl.learn_sentence_q

        # from marl.utils.buffer import MultiAgentTurnReplayBuffer
        # self.buffer = MultiAgentTurnReplayBuffer(num_agents=self.num_agents, capacity=300000)

        self.rollout_n = self.config.actor_rollout_ref.rollout.n


    # Build datasets with each agent's tokenizer/processor.
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


    """Compute original per-agent rewards for logging, before team aggregation."""
    def cal_origin_reward(self, agent_batchs, reward_fn_list):
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
                reward_tensor, reward_extra_infos_dict = compute_reward(agent_batch, reward_fn_list[agent_id])
                reward_tensor_all[f"agent_{agent_id}"] = reward_tensor
                reward_extra_infos_dict_all[f"agent_{agent_id}"] = reward_extra_infos_dict

        return reward_tensor_all




    """Run multi-turn validation for heterogeneous agents."""
    def _validate(self):
        """
        Validate each agent and return metrics with agent/team prefixes.
        """
        agent_metrics = {}
        world_size = self.mac.agents[0].actor_rollout_wg.world_size
        sample_indexs = []
        all_team_rewards = []

          
        for test_data in self.val_dataloader:

            multi_turn_test_batchs: Dict[str, Dict[str, DataProto]] = {}

            # Validation may run over the full validation set in one batch.
            test_bs = len(test_data["agent_0"]['index'])

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


            indices = test_batch.non_tensor_batch.get("index", [0] * test_bs)
            sample_indexs.extend(indices)


            # Store dialogues and team rewards for JSONL/CSV exports.
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


            # Heterogeneous agents each have their own tokenized generation batch.
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
                # pad to be divisible by dp_size for all agents
                # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.mac.agents[agent_id].actor_rollout_wg.world_size)

                test_batch_all[f"agent_{agent_id}"] = test_gen_batch


            
            # test_output_gen_batch_padded = agent.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            
            # Validation uses trajectory GRPO rollout for the current UMAD path.
            if self.config.algorithm.adv_estimator == "grpo":
                # test_output_gen_batch_padded = self.runner.rollout_multi_turn_grpo(test_batch_all, multi_turn_test_batchs, test_metrics, turn=self.turns, validate=True)  # tree grpo
                test_output_gen_batch_padded = self.runner.rollout_traj_grpo(test_batch_all, multi_turn_test_batchs, test_metrics, turn=self.turns, validate=True)  # trajectory grpo
            else:
                # test_output_gen_batch_padded = self.runner.rollout_multi_turn(test_batch_all, multi_turn_test_batchs, test_metrics, turn=self.turns)
                raise ValueError(f"Invalid algorithm.adv_estimator: {self.config.algorithm.adv_estimator}")

            test_output_gen_batch = {}
            for turn_id in range(self.turns):
                test_output_gen_batch[f"turn_{turn_id}"] = {} 
            # unpad
            for turn_id in range(self.turns):
                for agent_id in range(self.num_agents):
                    test_output_gen_batch_unpadded = unpad_dataproto(test_output_gen_batch_padded[f"turn_{turn_id}"][f"agent_{agent_id}"], pad_size=test_batch_pad_size)
                    test_output_gen_batch[f"turn_{turn_id}"][f"agent_{agent_id}"] = test_output_gen_batch_unpadded


            # Store generated outputs
            # output_ids = test_output_gen_batch.batch['responses']
            # output_texts = [agent.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            # sample_outputs.extend(output_texts)

            test_dialogues_dump, test_team_rewards_dump = self._get_dialogue_data(test_output_gen_batch, self.num_agents, self.turns, test_bs, test=True)
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

                # Debug path for per-agent validation reward inspection.
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

            # Optional data-source split metrics for mixed GSM8K/MATH validation.
            # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
            # data_sources = np.concatenate(data_source_lst, axis=0)

            # # evaluate test_score based on data source
            # data_source_reward = {}
            # for i in range(reward_tensor.shape[0]):
            #     data_source = data_sources[i]
            #     if data_source not in data_source_reward:
            #         data_source_reward[data_source] = []
            #     data_source_reward[data_source].append(reward_tensor[i].item())

            # Add prefixed validation metrics.
            agent_metric_dict = {}

            reward_keys = test_team_rewards_dump[0].keys()
            reward_len = len(test_team_rewards_dump)

            for key in reward_keys:
                reward_mean = sum(r[key] for r in test_team_rewards_dump) / reward_len
                agent_metric_dict[f'val/{key}_test_score'] = round(reward_mean, 3)
            
            # for agent_idx, (agent_key, agent_rewards) in enumerate(score_batchs.items()):
            #     agent_metric_dict[f'val/{agent_key}_test_score'] = agent_rewards.mean()
            agent_metrics.update(agent_metric_dict)

        # Optional summed-reward validation metrics.
        # if self.config.use_sum_reward:
        #     sum_metrics = {}
        #     for data_source in next(iter(agent_metrics.values())).keys():
        #         base_key = data_source.split('/')[-1]  # Extract the dataset name.
        #         agent_rewards = [
        #             metrics[f'agent_{i}/val/test_score/{base_key}'] 
        #             for i, metrics in enumerate(agent_metrics.values())
        #         ]
        #         sum_metrics[f'sum/val/test_score/{base_key}'] = sum(agent_rewards)
        #     agent_metrics.update(sum_metrics)

        # After validation, export sample indices with each agent reward.
        val_data_index_score = []
        for idx, team_reward in zip(sample_indexs, all_team_rewards):
            row_data = {"index": idx}
            for agent_idx in range(self.num_agents):
                agent_key = f"agent_{agent_idx}"
                row_data[agent_key] = team_reward[agent_key]
            val_data_index_score.append(row_data)


        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            validation_data_dir = os.path.join(rollout_data_dir, self.config.trainer.get("experiment_name", "latest"))
            os.makedirs(validation_data_dir, exist_ok=True)
            file_path = os.path.join(validation_data_dir, f"val_index_score_{self.global_steps}.csv")
            print(f"Saved validation index scores to {file_path}")
            
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                header = ["index"] + [f"agent_{i}" for i in range(self.num_agents)]
                writer.writerow(header)
                
                for row in val_data_index_score:
                    row_values = [row["index"]] + [row[f"agent_{i}"] for i in range(self.num_agents)]
                    writer.writerow(row_values)

        return agent_metrics



    def _maybe_log_val_generations(self, dialogues, rewards_list):
        """Dump a deterministic sample of validation generations."""
        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        # Use a fixed seed so the exported validation samples are reproducible.
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(dialogues), min(generations_to_log, len(dialogues)), replace=False)

        sampled_dialogues = [dialogues[i] for i in sample_indices]
        sampled_rewards = [rewards_list[i] for i in sample_indices]


        # sampled_entries = []
        # for dialogue, reward in zip(sampled_dialogues, sampled_rewards):
        #     entry = {}
            
        #     # Add each turn's dialogue.
        #     for turn_key, messages in dialogue.items():
        #         entry[turn_key] = messages
            
        #     # Add rewards.
        #     entry["rewards"] = reward
        #     entry["step"] = self.global_steps
            
        #     sampled_entries.append(entry)

        # JSONL export is the release-friendly logging path.
        # self.validation_generations_logger.log(
        #     self.config.trainer.logger,
        #     sampled_entries,
        #     self.global_steps,
        #     table_name="validation_samples"
        # )
        base_rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if not base_rollout_data_dir:
            return

        rollout_data_dir = os.path.join(base_rollout_data_dir, self.config.trainer.get("experiment_name", "latest"))
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
        #     table_name=f"agent_{agent_id}_generations"  # Separate generations per agent.
        # )


    """Initialize resource pools and each agent's Ray worker groups."""
    def init_workers(self):
        # shared resource pool
        self.resource_pool_manager.create_resource_pool()

        # create mac agents
        # for agent_id, agent in enumerate(self.mac.agents):
        #     agent.init_workers()
        for agent in self.mac.agents:
            agent.init_workers()
        print(f"All Agents init_workers Done")

        print(f"Learner init_workers Done")

        self.runner.init_workers()
        print(f"Runner init_workers Done")



    def _save_checkpoint(self):
        """Save checkpoints for all agents and dataloader state."""
        ckpt_folder = os.path.join(
            self.config.trainer.default_local_dir, 
            f"global_step_{self.global_steps}"
        )
        os.makedirs(ckpt_folder, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Saving checkpoint at step {self.global_steps}")
        print(f"Location: {ckpt_folder}")
        print(f"{'='*60}")
        
        # Apply configured checkpoint retention limits.
        max_actor_ckpt = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        max_critic_ckpt = self.config.trainer.get("max_critic_ckpt_to_keep", None)
        
        for agent_id, agent in enumerate(self.mac.agents):
            agent_folder = os.path.join(ckpt_folder, f"agent_{agent_id}")
            agent._save_checkpoint(
                local_path=agent_folder,
                global_steps=self.global_steps,
                max_actor_ckpt_to_keep=max_actor_ckpt,
                max_critic_ckpt_to_keep=max_critic_ckpt
            )
        
        torch.save(
            self.train_dataloader.state_dict(),
            os.path.join(ckpt_folder, "data.pt")
        )
        
        with open(os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"), "w") as f:
            f.write(str(self.global_steps))
        
        print(f"{'='*60}")
        print("Checkpoint saved successfully.")
        print(f"{'='*60}\n")


    def _load_checkpoint(self):
        """Load agent checkpoints and dataloader state."""
        if self.config.trainer.resume_mode == "disable":
            return 0
        
        ckpt_dir = self.config.trainer.default_local_dir
        if not os.path.isabs(ckpt_dir):
            ckpt_dir = os.path.join(os.getcwd(), ckpt_dir)
        
        ckpt_folder = find_latest_ckpt_path(ckpt_dir)
        
        if self.config.trainer.resume_mode == "auto" and ckpt_folder is None:
            print("No checkpoint found. Training from scratch.")
            return 0
        
        if self.config.trainer.resume_mode == "resume_path":
            ckpt_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(ckpt_folder):
                ckpt_folder = os.path.join(os.getcwd(), ckpt_folder)
        
        print(f"\n{'='*60}")
        print(f"Loading checkpoint from: {ckpt_folder}")
        print(f"{'='*60}")
        
        self.global_steps = int(ckpt_folder.split("global_step_")[-1])
        
        for agent_id, agent in enumerate(self.mac.agents):
            agent_folder = os.path.join(ckpt_folder, f"agent_{agent_id}")
            agent._load_checkpoint(local_path=agent_folder)
        
        dataloader_path = os.path.join(ckpt_folder, "data.pt")
        if os.path.exists(dataloader_path):
            self.train_dataloader.load_state_dict(
                torch.load(dataloader_path, weights_only=False)
            )
        
        print(f"{'='*60}")
        print(f"Checkpoint loaded. Resuming from step {self.global_steps}")
        print(f"{'='*60}\n")
        
        return self.global_steps



    """Dump per-step batch text data."""
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
                
                # Serialize each turn's dialogue.
                for turn_key, messages in dialogues[i].items():
                    entry[turn_key] = messages
                
                # Add rewards.
                entry["rewards"] = team_rewards[i]
                entry["step"] = self.global_steps
                
                # Add auxiliary reward metadata.
                for k, v in base_data.items():
                    if k not in ["dialogues", "team_rewards", "step"]:
                        entry[k] = v[i]
                
                # Format JSON output for readability.
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
            scores_dump: Not stored yet because it is too verbose.
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


        # Compute each agent's reward from the final turn only.
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
                # Read the final-turn reward for this agent.

                agent_reward = agents_origin_reward_tensor[agent_key][bs_idx].sum().item()
                team_reward_dict[agent_key] = agent_reward
                team_total += agent_reward
            
            # Add the team-level total reward.
            team_reward_dict["team_reward"] = team_total
            team_rewards_dump[bs_idx] = team_reward_dict


        return dialogues_dump, team_rewards_dump



    """
    Discussion note:
    turn-level Q updates affect token-level rewards. We keep each turn as an
    independent batch branch:
    {turn_idx: agent_batchs = {agent_id: agent_batch: DataProto}}
    """


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
        

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn_list is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            # if self.config.trainer.get('val_only', False):
            #     return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                
                """
                batch_dict: {
                    "agent_0": {
                        "input_ids": [...],
                        "attention_mask": [...],
                        "position_ids": [...],
                    },
                    "agent_1": {
                        "input_ids": [...],
                        "attention_mask": [...],
                        "position_ids": [...],
                    },
                }
                """

            
                multi_turn_batchs: Dict[str, Dict[str, DataProto]] = {}
                for turn_idx in range(self.turns):
                    multi_turn_batchs[f"turn_{turn_idx}"] = {}
                    for agent_id in range(self.num_agents):
                        multi_turn_batchs[f"turn_{turn_idx}"][f"agent_{agent_id}"] = DataProto.from_single_dict(deepcopy(batch_dict[f"agent_{agent_id}"]))


                #### assume common input format for all agents
                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in multi_turn_batchs["turn_0"]["agent_0"].non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in multi_turn_batchs["turn_0"]["agent_0"].non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in multi_turn_batchs["turn_0"]["agent_0"].non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")


                # Heterogeneous LLM agents each have a different tokenized generation batch.
                gen_batch_all = {}
                for batch_key, batchs in multi_turn_batchs["turn_0"].items():
                    gen_batch = batchs.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )
                    gen_batch_all[batch_key] = gen_batch

                # gen_batch_all: {"agent_0": DataProto{input_ids, attention_mask, position_ids}, 
                #                 "agent_1": DataProto{input_ids, attention_mask, position_ids}}
                
                is_last_step = self.global_steps >= self.total_training_steps

                # print("Start Generation")
                with _timer('step', timing_raw):
                    # generate a batch with multi-agent multi-turn debate
                    with _timer('gen', timing_raw):
                        # Trajectory GRPO rollout path.
                        if self.config.algorithm.adv_estimator == "grpo":
                            # self.runner.rollout_multi_turn_grpo(gen_batch_all, multi_turn_batchs, metrics, turn=self.turns)  # tree grpo
                            self.runner.rollout_traj_grpo(gen_batch_all, multi_turn_batchs, metrics, turn=self.turns)  # trajectory grpo
                        else:
                            # self.runner.rollout_multi_turn(gen_batch_all, multi_turn_batchs, metrics, turn=self.turns)
                            raise ValueError(f"Invalid algorithm.adv_estimator: {self.config.algorithm.adv_estimator}")


                    """Compute the selected multi-turn reward shape."""
                    if self.config.marl.team_reward_type == "dense":
                        reward_tensor_all, reward_extra_infos_dict_all, future_reward_all = self.runner.cal_reward_dense(multi_turn_batchs, metrics)
                    elif self.config.marl.team_reward_type == "sparse":
                        reward_tensor_all, reward_extra_infos_dict_all, future_reward_all = self.runner.cal_reward_sparse(multi_turn_batchs, metrics)
                    elif self.config.marl.team_reward_type == "accumulative":
                        reward_tensor_all, reward_extra_infos_dict_all, future_reward_all = self.runner.cal_reward_accumulative(multi_turn_batchs, metrics)
                    else:
                        raise ValueError(f"Invalid team_reward_type: {self.config.marl.team_reward_type}")

                    # reward_tensor_all, reward_extra_infos_dict_all, future_reward_all = self.runner.cal_reward_multi_turn_final(multi_turn_batchs, metrics)
                    

                    # Optional off-policy buffer path was kept out of release scope.
                    # print("Start Insert Buffer")
                    # s, a, s_, r, done, action_lengths = self.learner.transfer_embed_ids(multi_turn_batchs, reward_tensor_all, turns=self.turns, mode=self.config.marl.td_mode)
                    # self.buffer.insert_batch(s, a, r, s_, done, action_lengths)

                    # Buffer length statistics were part of the offline debugging path.
                    # response_length_summary = self.buffer.length_summary()
                    # for agent_id in range(self.num_agents):
                        # metrics.update({f"response_length/buffer_agent_{agent_id}_mean": response_length_summary[f"agent_{agent_id}"]["buffer_response_len_mean"]})
                        # metrics.update({f"response_length/buffer_agent_{agent_id}_min": response_length_summary[f"agent_{agent_id}"]["buffer_response_len_min"]})
                        # metrics.update({f"response_length/buffer_agent_{agent_id}_max": response_length_summary[f"agent_{agent_id}"]["buffer_response_len_max"]})
                        # metrics.update({f"response_length/buffer_agent_{agent_id}_short_ratio": response_length_summary[f"agent_{agent_id}"]["buffer_response_len_short_ratio"]})


                    # Main training path: the learner consumes the current multi-turn batch directly.
                    with _timer("update", timing_raw):
                        self.learner.train(multi_turn_batchs, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics, self.global_steps, timing_raw)


                    

                    # Log rollout generations of Multi-agent Multi-turn if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if self.config.algorithm.adv_estimator == "grpo":
                        if rollout_data_dir:
                            # Log a small deterministic subset of problems.
                            train_log_nums = self.config.marl.get("training_log_nums", 5)
                            train_log_nums = max(1, int(train_log_nums))

                            rollout_data_dir = os.path.join(self.config.trainer.get("rollout_data_dir", None), self.config.trainer.get("experiment_name", "latest"), f"train_step_{self.global_steps}")
                            os.makedirs(rollout_data_dir, exist_ok=True)

                            # Export the full batch so rewards, advantages, and metadata stay aligned.
                            dialogues_dumps = self._get_dialogue_data_grpo(multi_turn_batchs, self.num_agents, self.turns, reward_tensor_all, log_nums=train_log_nums)

                            self.dump_dialogues_grpo(dialogues_dumps, rollout_data_dir)
                    else:
                        if rollout_data_dir:
                            with _timer("dump_rollout_generations", timing_raw):
                                dialogues_dump, team_rewards_dump = self._get_dialogue_data(multi_turn_batchs, self.num_agents, self.turns, self.config.data.train_batch_size, test=False)
                                # GRPO does not use the generic generation dump path.
                                rollout_data_dir = os.path.join(self.config.trainer.get("rollout_data_dir", None), self.config.trainer.get("experiment_name", "latest"))
                                self._dump_generations(
                                    dialogues=dialogues_dump,
                                    team_rewards=team_rewards_dump,
                                    reward_extra_infos_dict=reward_extra_infos_dict_all,
                                    dump_path=rollout_data_dir,
                                )


                    # validate
                    if self.val_reward_fn_list is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        print(f"testing {self.global_steps}")


                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    """Persist both agent models through the MARL wrapper."""
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        print(f"save_checkpoint {self.global_steps}")

                        with _timer("save_checkpoint", timing_raw):
                            # self.learner._save_checkpoint()
                            self._save_checkpoint()



                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )


                for turn_key, turn_batchs in multi_turn_batchs.items():
                        for agent_key, agent_batch in turn_batchs.items():
                            # data metrics
                            data_metrics = compute_data_metrics(batch=agent_batch, use_critic=self.use_critic)
                            metrics.update({f"{agent_key}_turn{turn_key}_{k}": v for k, v in data_metrics.items()})

                            # timing metrics
                            timing_metrics = compute_timing_metrics(batch=agent_batch, timing_raw=timing_raw)
                            metrics.update({f"{agent_key}_turn{turn_key}_{k}": v for k, v in timing_metrics.items()})

                            # throughput metrics
                            n_gpus = self.resource_pool_manager.get_n_gpus()
                            throughput_metrics = compute_throughout_metrics(batch=agent_batch, timing_raw=timing_raw, n_gpus=n_gpus)
                            metrics.update({f"{agent_key}_turn{turn_key}_{k}": v for k, v in throughput_metrics.items()})
                    

                # for agent_batch in agent_batchs.values():
                #     metrics.update(compute_data_metrics(batch=agent_batch, use_critic=self.use_critic))
                #     metrics.update(compute_timing_metrics(batch=agent_batch, timing_raw=timing_raw))
                #     # TODO: implement actual tflpo and theoretical tflpo
                #     n_gpus = self.resource_pool_manager.get_n_gpus()
                #     metrics.update(compute_throughout_metrics(batch=agent_batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return


    def _get_dialogue_data_grpo(self, multi_turn_batchs, num_agents, turns, reward_tensor_all, log_nums=5):
        """
        Build GRPO validation/training samples with rewards, advantages, and metadata.

        Returns:
        - dialogues_dump: Multi-turn dialogue per sample.
        - team_rewards_dump: Final aggregated team reward per sample.
        - reward_dump: Estimated rewards per sample, turn, and agent.
        - adv_values_dump: Advantages per sample, turn, and agent.
        - problem_type / problem_index / problem_level: Sample metadata.
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


                # Trajectory GRPO keeps the same sample count per turn.
                turn_sample_nums = int(self.rollout_n)


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



                # if self.global_steps >= 10:  
                #     estimated_rewards = [turn_batchs[f'agent_{agent_idx}'].batch["estimated_rewards"][start_idx:end_idx] for agent_idx in range(num_agents)]
                #     adv_values = [turn_batchs[f'agent_{agent_idx}'].batch["advantages"][start_idx:end_idx] for agent_idx in range(num_agents)]
                # else:
                #     estimated_rewards = [torch.zeros_like(turn_batchs[f'agent_{agent_idx}'].batch["responses"][start_idx:end_idx], dtype=torch.float32) for agent_idx in range(num_agents)]
                #     adv_values = [torch.zeros_like(turn_batchs[f'agent_{agent_idx}'].batch["responses"][start_idx:end_idx], dtype=torch.float32) for agent_idx in range(num_agents)]

                real_rewards = self.cal_origin_reward(multi_turn_batchs[turn_key], self.reward_fn_list)
                adv_values = [turn_batchs[f'agent_{agent_idx}'].batch["advantages"][start_idx:end_idx] for agent_idx in range(num_agents)]


                uids = [turn_batchs[f'agent_{agent_idx}'].non_tensor_batch["uid"][start_idx:end_idx] for agent_idx in range(num_agents)]

                # if turn_idx == turns - 1:

                    
                #     rewards_tensor_current = [reward_tensor_all[f"turn_{turn_idx}"][f"agent_{agent_idx}"][start_idx:end_idx] for agent_idx in range(num_agents)]

                #     for bs_idx in range(turn_sample_nums):
                #         turn_dialogue = [{"role": "user", "content": turn_prompts[bs_idx]}]
                #         for agent_idx in range(num_agents):
                #             agent_key = f"agent_{agent_idx}"
                #             turn_dialogue.append({
                #                 "role": agent_key,
                #                 "content": agent_responses[agent_key][bs_idx],
                #                 'estimated_rewards': real_rewards[agent_key][bs_idx].sum().item(),
                #                 'GRPO_Adv': adv_values[agent_idx][bs_idx].mean().item(),
                #                 'Real Rewards': rewards_tensor_current[agent_idx][bs_idx].sum().item(),
                #                 'uid': uids[agent_idx][bs_idx],
                #             })

                #         all_turns_dialogues.append(turn_dialogue)
                # else:
                for bs_idx in range(turn_sample_nums):
                    turn_dialogue = [{"role": "user", "content": turn_prompts[bs_idx]}]
                    for agent_idx in range(num_agents):
                        agent_key = f"agent_{agent_idx}"
                        turn_dialogue.append({
                            "role": agent_key,
                            "content": agent_responses[agent_key][bs_idx],
                            'estimated_rewards': real_rewards[agent_key][bs_idx].sum().item(),
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

            # Fallback to default=str for non-serializable objects.
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False, indent=2, default=str)





    # def _extract_problem_data(self, multi_turn_batchs, reward_tensor_all, problem_idx, step):
    #     """Extract complete data for a single problem."""
        
    #     problem_data = {
    #         "problem_id": f"problem_{problem_idx}",
    #         "step": step,
    #         "trajectories": [],
    #         "groups": {}
    #     }
        
    #     # Build all trajectory data.
    #     for trajectory_id in range(25):  # Number of GRPO trajectories.
    #         trajectory_data = self._build_trajectory_data(
    #             multi_turn_batchs, reward_tensor_all,
    #             problem_idx, trajectory_id
    #         )
    #         problem_data["trajectories"].append(trajectory_data)
        
    #     # Analyze reward statistics within each turn group.
    #     problem_data["groups"] = self._analyze_turn_groups(problem_data["trajectories"])
        
    #     return problem_data



    def _analyze_turn_groups(self, trajectories):
        """Analyze reward distributions within each turn group."""
        
        groups = {}
        
        for turn_idx in range(self.turns):
            turn_key = f"turn_{turn_idx}"
            turn_groups = {
                "group_0": {  # Simplified case: all trajectories share one group.
                    "trajectory_ids": list(range(len(trajectories))),
                    "reward_statistics": {},
                    "ranking": []
                }
            }
            
            # Compute reward statistics for each agent.
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
            
            # Rank trajectories by total reward.
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
