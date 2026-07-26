import os
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from marl.ippo_main import get_custom_reward_fn
from marl.modules.agents.ppo_agent import ResourcePoolManager
from marl.utils.marl_dataset import collate_fn_marl
from marl.utils.marl_utils import MARLRole, create_marl_dataset

from eval.debate_evaluator import DebateEvaluator


CONFIG_NAME = os.getenv("CONFIG_NAME", "ippo_trainer")
print(f"Debate eval config: {CONFIG_NAME}")


@hydra.main(config_path="../marl/config", config_name=CONFIG_NAME, version_base=None)
def main(config):
    run_debate_eval(config)


def run_debate_eval(config):
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            }
        )

    runner = EvalTaskRunner.remote()
    summary = ray.get(runner.run.remote(config))
    pprint(summary)


@ray.remote(num_cpus=1, num_gpus=0)
class EvalTaskRunner:
    def run(self, config):
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        num_agents = int(config.marl.num_agents)
        local_path_list = []
        for agent_id in range(num_agents):
            model_path = config.marl.agent_configs[f"agent_{agent_id}"].model.path
            local_path_list.append(copy_to_local(model_path))

        tokenizer_list, processor_list = [], []
        for local_path in local_path_list:
            tokenizer_list.append(hf_tokenizer(local_path))
            processor_list.append(hf_processor(local_path, use_fast=True))

        if config.actor_rollout_ref.actor.strategy == "fsdp":
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker

            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError(
                f"Unsupported actor strategy: {config.actor_rollout_ref.actor.strategy}"
            )

        role_worker_mappings = []
        for agent_id in range(num_agents):
            role_worker_mappings.append(
                {
                    MARLRole[f"agent_{agent_id}_ActorRollout"]: ray.remote(
                        ActorRolloutRefWorker
                    ),
                }
            )

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {}
        for agent_id in range(num_agents):
            mapping[MARLRole[f"agent_{agent_id}_ActorRollout"]] = global_pool_id
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == "naive":
            from verl.workers.reward_manager import NaiveRewardManager

            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == "prime":
            from verl.workers.reward_manager import PrimeRewardManager

            reward_manager_cls = PrimeRewardManager
        else:
            raise NotImplementedError(f"Unsupported reward manager: {reward_manager_name}")

        compute_score = get_custom_reward_fn(config)
        val_reward_fn_list = [
            reward_manager_cls(
                tokenizer=tokenizer_list[agent_id],
                num_examine=1,
                compute_score=compute_score,
            )
            for agent_id in range(num_agents)
        ]

        val_dataset = create_marl_dataset(
            config.data.val_files,
            config.data,
            tokenizer_list,
            processor_list,
            full_config=config,
        )

        evaluator = DebateEvaluator(
            config=config,
            num_agents=num_agents,
            tokenizer_list=tokenizer_list,
            processor_list=processor_list,
            role_worker_mapping=role_worker_mappings,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            val_reward_fn_list=val_reward_fn_list,
            val_dataset=val_dataset,
            collate_fn=collate_fn_marl,
            device_name=config.trainer.device,
        )
        evaluator.init_workers()
        return evaluator.evaluate()


if __name__ == "__main__":
    main()
