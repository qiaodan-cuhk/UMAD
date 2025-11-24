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
完全异构agent，使用QMIX sentence critic，多轮对话

有几种实现方式：
1. QMIX 学习 Qi，作为return用，这样Q（s,a)还可以当作GRPO的verifier使用
2. 额外学习一个关于policy的Vi，利用baseline稳定训练，policy的更新loss是 logp*A
3. 用Qi直接梯度更新policy，loss为 - Qi，类似MADDPG或者LICA
"""

import os
import ray
import hydra

# marl dataset and sampler
from marl.utils.marl_utils import create_marl_dataset, create_marl_sampler
from marl.utils.marl_dataset import collate_fn_marl

from marl.modules.agents.ppo_agent import ResourcePoolManager
from marl.utils.marl_utils import MARLRole, Role

def get_custom_reward_fn(config):
    import importlib.util

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    return getattr(module, function_name)


# 读取bash脚本环境变量
algorithm_name = os.getenv("ALGORITHM_NAME", "marl_naive")
config_name = os.getenv("CONFIG_NAME", "ippo_trainer")
print(f"Algorithm: {algorithm_name}, Config: {config_name}")


@hydra.main(config_path='config', config_name=config_name, version_base=None)
def main(config):
    run_ippo(config)


def run_ippo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })

    runner = TaskRunner.remote()
    ray.get(runner.run_marl.remote(config))



@ray.remote(num_cpus=1, num_gpus=0)   
# qmix trainer need 1 gpu to deal with embeddings and qmix network
# please make sure main_task is not scheduled on head
class TaskRunner:

    def run_marl(self, config):
        from verl.utils.fs import copy_to_local
        from pprint import pprint
        from omegaconf import OmegaConf

        # print initial config
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)


        # breakpoint()


        # 确定agent数量,支持每个agent单独指定model，异构agent
        agent_nums = config.marl.num_agents  
        # download the checkpoint from hdfs
        local_path_list = []
        for agent_id in range(agent_nums):
            local_path = copy_to_local(config.marl.agent_configs[f"agent_{agent_id}"].model.path)
            local_path_list.append(local_path)
        

        # instantiate tokenizer list
        from verl.utils import hf_tokenizer, hf_processor
        tokenizer_list, processor_list = [], []
        for local_path in local_path_list:
            tokenizer = hf_tokenizer(local_path)
            processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none
            tokenizer_list.append(tokenizer)
            processor_list.append(processor)


        # define worker classes
        # 目前仅支持fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            
            # Token level VDN 需要，sentence level不需要这个criticworker
            # from marl.modules.workers.vdn_workers import VDNCriticWorker

            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        #### todo: (qd) 目前仅支持2 agents，后续考虑多个agent的动态扩展MARLRole
        # 为每个agent创建role mapping, role_worker_mappings = [{agent1 mapping}, {agent2 mapping}]
        role_worker_mappings = []
        for agent_id in range(agent_nums):
            agent_mapping = {
                MARLRole[f"agent_{agent_id}_ActorRollout"]: ray.remote(ActorRolloutRefWorker),
                MARLRole[f"agent_{agent_id}_Critic"]: ray.remote(CriticWorker),
                MARLRole[f"agent_{agent_id}_RefPolicy"]: ray.remote(ActorRolloutRefWorker)
            }
            role_worker_mappings.append(agent_mapping)
        

        # 单节点多卡训练，全局资源池
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # resource_pool_spec = {'global_pool': [8]}
        mapping = {}
        for agent_id in range(agent_nums):
            mapping[MARLRole[f"agent_{agent_id}_ActorRollout"]] = global_pool_id
            mapping[MARLRole[f"agent_{agent_id}_Critic"]] = global_pool_id
            mapping[MARLRole[f"agent_{agent_id}_RefPolicy"]] = global_pool_id
        # 用于 QMIX/VDN 的 Embedding Model
        mapping[MARLRole["Embed"]] = global_pool_id


        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            
            # 为每个agent创建reward model
            for agent_id in range(agent_nums):
                role_worker_mappings[agent_id][Role.RewardModel] = ray.remote(RewardModelWorker)
                mapping[MARLRole[f"agent_{agent_id}_RewardModel"]] = global_pool_id


        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager
            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == 'prime':
            from verl.workers.reward_manager import PrimeRewardManager
            reward_manager_cls = PrimeRewardManager
        else:
            raise NotImplementedError


        compute_score = get_custom_reward_fn(config)

        # 异构agents使用不同的tokenizer和processor，会影响dataset和dataloader以及compute_fn的变化
        # Note that we always use function-based RM for validation
        reward_fn_list = []
        val_reward_fn_list = []
        for agent_id in range(agent_nums):
            reward_fn = reward_manager_cls(tokenizer=tokenizer_list[agent_id], num_examine=0, compute_score=compute_score)
            val_reward_fn = reward_manager_cls(tokenizer=tokenizer_list[agent_id], num_examine=1, compute_score=compute_score)

            reward_fn_list.append(reward_fn)
            val_reward_fn_list.append(val_reward_fn)


        # 多个agent的分组mapping被flatten了，全局resource pool manager
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


        train_dataset = create_marl_dataset(config.data.train_files, config.data, tokenizer_list, processor_list, full_config=config)
        val_dataset = create_marl_dataset(config.data.val_files, config.data, tokenizer_list, processor_list, full_config=config)
        train_sampler = create_marl_sampler(config.data, train_dataset)

        # todo: 拓展为支持config对应的trainer，需要增加IPPO、MAGRPO，考虑CORY和REMA，MARFT
        # from marl.qmix_trainer import RayQMIXTrainer
        # from marl.magrpo_trainer import RayMAGRPOTrainer
        # from marl.ippo_trainer import RayIPPOTrainer

        

        # if algorithm_name == "qmix":
        #     trainer_cls = RayQMIXTrainer
        # elif algorithm_name == "magrpo":
        #     trainer_cls = RayMAGRPOTrainer
        # elif algorithm_name == "vdn":
        #     trainer_cls = RayQMIXTrainer  # 暂时共用qmix trainer
        # elif algorithm_name == "ippo":
        #     trainer_cls = RayIPPOTrainer


        from confidence.conf_trainer import ConfidenceTrainer
        trainer_cls = ConfidenceTrainer
        


        trainer = trainer_cls(config=config,  # 保持不变
                                num_agents=agent_nums,
                                tokenizer_list=tokenizer_list, # 支持异构tokenizer list
                                processor_list=processor_list, # 支持异构processor list
                                role_worker_mapping=role_worker_mappings,  # list
                                resource_pool_manager=resource_pool_manager, # flatten cls
                                ray_worker_group_cls=ray_worker_group_cls,  # fsdp/Megatron RayWorkerGroup
                                reward_fn_list=reward_fn_list, # 异构tokenizer reward fn
                                val_reward_fn_list=val_reward_fn_list,
                                train_dataset=train_dataset,
                                val_dataset=val_dataset,
                                collate_fn=collate_fn_marl,   # 换成marl异构collate fn
                                train_sampler=train_sampler,
                                device_name=config.trainer.device
                                )
        
    
        trainer.init_workers()
        trainer.fit()



if __name__ == '__main__':
    main()
