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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
# from verl.trainer.ppo.ray_trainer import RayPPOTrainer
# from marl.utils.marl_trainer_naive import RayMARLTrainer_naive


"""
多轮对话版本，目前只支持同构agent
"""

import os
import ray
import hydra
# from marl.utils import Registry
from verl.utils.dataset.rl_dataset import collate_fn
from marl.utils.marl_utils import create_rl_dataset, create_rl_sampler

from marl.modules.agents.ppo_agent import ResourcePoolManager
from marl.utils.marl_utils import MARLRole, Role

def get_custom_reward_fn(config):
    import importlib.util, os

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

print(f"Algorithm is set as {algorithm_name}")
print(f"Config is set as {config_name}")


# 加载 ippo trainer config，多了个num_agents属性
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



# 2 agent LLM IPPO/CORY
@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run_marl(self, config):
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # 本版本采用相同的base model
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer 
        # 本版本采用相同的base model，所以tokenizer和processor公用,而且兼容reward function
        from verl.utils import hf_tokenizer, hf_processor
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            
            # 新版本VDN，使用qmix的total q，性能表现好
            from marl.modules.workers.vdn_workers import VDNCriticWorker

            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError


        # 基于Role class动态定义MARLRole，可行key为Role class的成员变量
        # 为每个agent创建role mapping
        role_worker_mappings = []
        for agent_id in range(config.num_agents):
            agent_mapping = {
                MARLRole[f"agent_{agent_id}_ActorRollout"]: ray.remote(ActorRolloutRefWorker),
                MARLRole[f"agent_{agent_id}_Critic"]: ray.remote(VDNCriticWorker),
                MARLRole[f"agent_{agent_id}_RefPolicy"]: ray.remote(ActorRolloutRefWorker)
            }
            role_worker_mappings.append(agent_mapping)
        # agent_role_mappings = [{agent1 mapping}, {agent2 mapping}]

    
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # resource_pool_spec = {'global_pool': [8]}
        mapping = {}
        for agent_id in range(config.num_agents):
            mapping[MARLRole[f"agent_{agent_id}_ActorRollout"]] = global_pool_id
            mapping[MARLRole[f"agent_{agent_id}_Critic"]] = global_pool_id
            mapping[MARLRole[f"agent_{agent_id}_RefPolicy"]] = global_pool_id


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
            
            """这里逻辑要检查一下，Role和MARLRole统一"""
            # 为每个agent创建reward model
            for agent_id in range(config.num_agents):
                role_worker_mappings[agent_id][Role.RewardModel] = ray.remote(RewardModelWorker)
                mapping[MARLRole[f"agent_{agent_id}_RewardModel"]] = global_pool_id


        """新版本verl，暂时不需要"""
        # use reference model
        # if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        #     role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        #     mapping[Role.RefPolicy] = global_pool_id


        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager
            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == 'prime':
            from verl.workers.reward_manager import PrimeRewardManager
            reward_manager_cls = PrimeRewardManager
        else:
            raise NotImplementedError


        # 新版本不使用这个函数
        compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

        """新版本verl，待验证"""
        # reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        # val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))


        # resource_pool_spec = {'global_pool': [8]}
        # mapping = {AGENT0_ACTOR: global_pool_id,AGENT0_CRITIC: global_pool_id, AGENT0_REF: global_pool_id， AGENT0_REWARD_MODEL: global_pool_id}
        # 多个agent的分组mapping被flatten了，全局resource pool manager
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # 资源分配和config，以及tokenizer等保持不变，用同一套参数传入MARLTrainer进行deepcopy

        
        
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

    
        
        


        # from marl.marl_trainer import RayMARLTrainer
        from marl.marl_trainer_multiturn import RayMARLTrainer_MultiTurn
        trainer_cls = RayMARLTrainer_MultiTurn

        # breakpoint()

        trainer = trainer_cls(config=config,  # 保持不变
                                num_agents=config.num_agents,
                                tokenizer=tokenizer, # 单个tokenizer，表示共用model
                                processor=processor, # 单个processor，表示共用model
                                role_worker_mapping=role_worker_mappings,  # list
                                resource_pool_manager=resource_pool_manager, # flatten cls
                                ray_worker_group_cls=ray_worker_group_cls,  # fsdp/Megatron RayWorkerGroup
                                reward_fn=reward_fn, # 共用
                                val_reward_fn=val_reward_fn,
                                train_dataset=train_dataset,
                                val_dataset=val_dataset,
                                collate_fn=collate_fn,
                                train_sampler=train_sampler,
                                device_name=config.trainer.device
                                )
        
        
    
        trainer.init_workers()
        trainer.fit()

if __name__ == '__main__':
    main()
