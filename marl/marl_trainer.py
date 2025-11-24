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



class RayMARLTrainer:
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


        """
        全局层抽象类，主要功能
            mac: 控制 agents, self.agents = [PPO, PPO]
            learner: 负责训练，CTDE 算法，控制optimizer和update逻，继承verl的fit部分，以及core algorithm
            runner: 负责推理，组织多个agent rollout以及summary/major voting每一轮
            dataloader: 全局dataloader，所有agent共享一个，负责采样train dataloader bs作为prompts
        """
        # 这个功能下放到mac中

        from marl.controllers import REGISTRY as MAC_REGISTRY
        # self.mac = MAC_REGISTRY[self.config.mac.name](args=self.config)
        mac = MAC_REGISTRY['basic_mac']
        self.mac = mac(config,
                 num_agents,  # cotrian LLMs
                 tokenizer,   # 单个，表示共用model
                 role_worker_mapping,
                 resource_pool_manager, # 共用全局manager
                 ray_worker_group_cls, # fsdp/Megatron
                 processor,  # 单个，表示共用model
                 reward_fn, # reward function公用相同的
                 val_reward_fn,
                 train_dataset,
                 val_dataset,
                 collate_fn,
                 train_sampler,
                 device_name)
        # self._build_agents()

        from marl.learners import REGISTRY as LEARNER_REGISTRY
        from marl.runners import REGISTRY as RUNNER_REGISTRY

        # self.learner = LEARNER_REGISTRY[self.config.learner.name](self.config)
        # self.runner = RUNNER_REGISTRY[self.config.runner.name](self.config)

        learner = LEARNER_REGISTRY[config.marl.name]
        self.learner = learner(config,
                               self.mac,
                               num_agents,  # cotrian LLMs
                               tokenizer,   # 单个，表示共用model
                 role_worker_mapping,
                 resource_pool_manager, # 共用全局manager
                 ray_worker_group_cls, # fsdp/Megatron
                 processor,  # 单个，表示共用model
                 reward_fn, # reward function公用相同的
                 val_reward_fn,
                 train_dataset,
                 val_dataset,
                 collate_fn,
                 train_sampler,
                 device_name)
        
        """mac learner从原始的ippo中拆分，先保持class输入不变，runner需要额外重新指定input，先都一股脑传进去s"""

        runner = RUNNER_REGISTRY['single']
        self.runner = runner(config,
                             self.mac,
                 num_agents,  # cotrian LLMs
                 tokenizer,   # 单个，表示共用model
                 role_worker_mapping,
                 resource_pool_manager, # 共用全局manager
                 ray_worker_group_cls, # fsdp/Megatron
                 processor,  # 单个，表示共用model
                 reward_fn, # reward function公用相同的
                 val_reward_fn,
                 train_dataset,
                 val_dataset,
                 collate_fn,
                 train_sampler,
                 device_name)

        
        # 创建全局dataloader，这个应该放在 fit 层
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        
    # create dataloader的逻辑还是要留在marl这一层，两个agent共享一个dataloader就行了？
    # 需要注意存储计算变量的时候用没用self.train_dataloader
    # save/load checkpoint里也调用了self.train_dataloader.state_dict()



    """每个agent内部init的时候会调用内部的create dataloader，创建train dataset 属性"""
    # 新版本verl更新增加功能，微调，记得更新
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        # from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from marl.utils.marl_utils import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

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



    """这个保留在trainer层，统一validate多个agent的共同推理结果"""
    # 目前也保留了ppo层的validate，这个是初步的marl版本
    def _validate(self):
        """
        为每个agent进行验证，返回带有agent前缀的metrics
        """
        agent_metrics = {}

        # 为每个agent分别validate
        for agent_id, agent in enumerate(self.mac.agents):
            reward_tensor_lst = []
            data_source_lst = []

            # Lists to collect samples for the table
            sample_inputs = []
            sample_outputs = []
            sample_scores = []

            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # repeat test batch
                test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                            interleave=True)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    continue

                # Store original inputs
                input_ids = test_batch.batch['input_ids']
                input_texts = [agent.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)

                if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                    test_gen_batch = test_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    test_gen_batch = test_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                test_gen_batch.meta_info = {
                    'eos_token_id': agent.tokenizer.eos_token_id,
                    'pad_token_id': agent.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    'validate': True,
                }
                print(f'agent_{agent_id} test_gen_batch meta info: {test_gen_batch.meta_info}')

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, agent.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = agent.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print(f'agent_{agent_id} validation generation end')

                # Store generated outputs
                output_ids = test_output_gen_batch.batch['responses']
                output_texts = [agent.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                reward_tensor = self.val_reward_fn(test_batch)

                # Store scores
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

            # Log generations for this agent
            self._maybe_log_val_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                agent_id=agent_id
            )

            # 计算该agent的metrics
            reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
            data_sources = np.concatenate(data_source_lst, axis=0)

            # evaluate test_score based on data source
            data_source_reward = {}
            for i in range(reward_tensor.shape[0]):
                data_source = data_sources[i]
                if data_source not in data_source_reward:
                    data_source_reward[data_source] = []
                data_source_reward[data_source].append(reward_tensor[i].item())

            # 为每个agent的metrics添加前缀
            agent_metric_dict = {}
            for data_source, rewards in data_source_reward.items():
                agent_metric_dict[f'agent_{agent_id}/val/test_score/{data_source}'] = np.mean(rewards)
            
            agent_metrics.update(agent_metric_dict)

        # 如果需要计算sum reward metrics
        if self.config.use_sum_reward:
            sum_metrics = {}
            for data_source in next(iter(agent_metrics.values())).keys():
                base_key = data_source.split('/')[-1]  # 获取数据源名称
                agent_rewards = [
                    metrics[f'agent_{i}/val/test_score/{base_key}'] 
                    for i, metrics in enumerate(agent_metrics.values())
                ]
                sum_metrics[f'sum/val/test_score/{base_key}'] = sum(agent_rewards)
            agent_metrics.update(sum_metrics)

        return agent_metrics

    def _maybe_log_val_generations(self, inputs, outputs, scores, agent_id):
        """为每个agent记录生成结果"""
        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger with agent prefix
        self.validation_generations_logger.log(
            self.config.trainer.logger, 
            samples, 
            self.global_steps,
            table_name=f"agent_{agent_id}_generations"  # 区分不同agent的生成结果
        )


    """
    这个版本是每个agent注册自己的ray address, 保持config和对应功能的相对独立
    需要调用mac.init_workers创建每个agent的RayWorkerGroup
    以及根据config创建runner和mixer的RayWorkerGroup
    """
    def init_workers(self):
        # shared resource pool
        self.resource_pool_manager.create_resource_pool()

        # create mac agents
        for agent_id, agent in enumerate(self.mac.agents):
            agent.init_workers()

        # create learner runner and mixer
        self.learner.init_workers()
        self.runner.init_workers()
        # self.mixer.init_workers()  # mixer init 放在learner内部实现
        
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


    


    """目前还没测试，新版verl"""
    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")



    """
    fit 是核心的rollout和update流程，改成for agent循环做
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


        """debug时候修改self.config.trainer['val_before_train']=False跳过validate和load ckpt"""
        # # load checkpoint before doing anything
        # self._load_checkpoint()
        
        # perform validation before training
        # currently, we only support validation using the reward_function.
        # if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
        #     val_metrics = self._validate()
        #     pprint(f'Initial validation metrics: {val_metrics}')
        #     logger.log(data=val_metrics, step=self.global_steps)
        #     if self.config.trainer.get('val_only', False):
        #         return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None


        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                


                
                """这里需要考虑batch_dict的组织方式，需要考虑两个agent的batch_dict如何组织"""
                # 共享marl.train_dataloader控制统一的input/init state，然后每个agent处理自己的batch_dict用于模型训练
                agent_batchs: Dict[str, DataProto] = {}
                for agent_id, agent in enumerate(self.mac.agents):
                    agent_batchs[f"agent_{agent_id}"] = DataProto.from_single_dict(deepcopy(batch_dict))

                # batch: DataProto = DataProto.from_single_dict(batch_dict)


                """这里不需要每个agent单独处理，因为batch data统一输入，不是每个agent单独设置"""

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in agent_batchs[f"agent_0"].non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in agent_batchs[f"agent_0"].non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in agent_batchs[f"agent_0"].non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")

                gen_batch_all = {}
                for batch_key, batchs in agent_batchs.items():
                    gen_batch = batchs.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )
                    gen_batch_all[batch_key] = gen_batch

                # gen_batch_all: {batch_key: DataProto},每个agent单独靠key存储
                
                is_last_step = self.global_steps >= self.total_training_steps

                """每个agent内部处理各自的gen_batch_output，但是需要考虑数据如何在marl层面组织
                以及调用CORY的sequential_rollout需要修改input concat方式"""
                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        self.runner.rollout(gen_batch_all, agent_batchs, metrics)
                        
                    """不用remax所以先注释掉，而且在multi turn中remax grpo如何去计算baseline还需要考虑，目前ppo版本不用"""
                    # if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    #     with _timer('gen_max', timing_raw):
                            # self.runner.rollout_remax()

                    """rewards计算放到了runner rollout中,分别计算每个agent的reward，还没存储"""
                    with _timer("reward", timing_raw):
                        reward_tensor_all, reward_extra_infos_dict_all, future_reward_all = self.runner.cal_reward(agent_batchs, metrics)
                    

                    """把所有的ref log prob reward update都放到了learner中"""
                    with _timer("update", timing_raw):
                        self.learner.train(agent_batchs, reward_tensor_all, reward_extra_infos_dict_all, future_reward_all, metrics, self.global_steps, timing_raw)



                    # 放到 trainer 层，用于存储rollout log
                    """暂时不debug，跳过这部分的rollout data，self dump generations"""
                    # # Log rollout generations if enabled
                    # rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    # if rollout_data_dir:
                    #     with _timer("dump_rollout_generations", timing_raw):

                    #         print(batch.batch.keys())
                    #         inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    #         outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    #         scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    #         self._dump_generations(
                    #             inputs=inputs,
                    #             outputs=outputs,
                    #             scores=scores,
                    #             reward_extra_infos_dict=reward_extra_infos_dict,
                    #             dump_path=rollout_data_dir,
                    #         )


                    """当前只validate了一个model，需要改成eval两个，metric记录两个"""
                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        print(f"testing {self.global_steps}")

                        """暂时不debug"""
                        # with _timer("testing", timing_raw):
                        #     val_metrics: dict = self._validate()
                        #     if is_last_step:
                        #         last_val_metrics = val_metrics
                        # metrics.update(val_metrics)

                    """这里通过marl的wrapper依次save了两个model，暂时不debug"""
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        print(f"save_checkpoint {self.global_steps}")
                        # with _timer("save_checkpoint", timing_raw):
                        #     self.learner._save_checkpoint()
                        #     self._save_checkpoint()


                """这里需要合并两个agent的metrics，变成.agent1/.agent2形式，或者只用一个agent？"""
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                for agent_key, agent_batch in agent_batchs.items():
                    # data metrics
                    data_metrics = compute_data_metrics(batch=agent_batch, use_critic=self.use_critic)
                    metrics.update({f"{agent_key}_{k}": v for k, v in data_metrics.items()})

                    # timing metrics
                    timing_metrics = compute_timing_metrics(batch=agent_batch, timing_raw=timing_raw)
                    metrics.update({f"{agent_key}_{k}": v for k, v in timing_metrics.items()})

                    # throughput metrics
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    throughput_metrics = compute_throughout_metrics(batch=agent_batch, timing_raw=timing_raw, n_gpus=n_gpus)
                    metrics.update({f"{agent_key}_{k}": v for k, v in throughput_metrics.items()})

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


