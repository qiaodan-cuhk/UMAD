"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

# Standard MARL Traning Base Cls


from typing import Type, Dict, Optional, List
from copy import deepcopy

from verl.single_controller.base import Worker
from verl.single_controller.ray import RayWorkerGroup

WorkerType = Type[Worker]

from marl.utils.marl_utils import MARLRole, Role
from marl.modules.agents.ppo_agent import ResourcePoolManager
# from marl.utils.marl_utils import AdvantageEstimator, compute_advantage, compute_response_mask, apply_kl_penalty
from marl.utils.marl_utils import _convert_marl_to_ppo_roles
from marl.modules.agents import REGISTRY as AGENT_REGISTRY


# mac目前返回的只是一个 self.mac.agents = [PPO PPO PPO]
class BasicMAC_Hetero:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 num_agents,  # cotrian LLMs
                 tokenizer_list,   # 单个，表示共用model
                 role_worker_mapping: list[dict[Role, WorkerType]],
                 resource_pool_manager: ResourcePoolManager, # 共用全局manager
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, # fsdp/Megatron
                 processor_list: Optional[List] = None,  # 异构list
                 **kwargs):


        # 新增
        self.num_agents = num_agents
        self.config = config
        self.tokenizer_list = tokenizer_list
        self.processor_list = processor_list
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls



        self.extra_params = kwargs 
        self.use_critic = kwargs.get('use_critic', False)
        self.use_reference_policy = kwargs.get('use_reference_policy', False)
        self.use_rm = kwargs.get('use_rm', False)
        self.ref_in_actor = kwargs.get('ref_in_actor', False)
        self.kl_ctrl_in_reward = kwargs.get('kl_ctrl_in_reward', False)
        self.hybrid_engine = kwargs.get('hybrid_engine', False)


        self._build_agents()


    def _build_agent_config(self, agent_id):
        agent_config = deepcopy(self.config)
        agent_specific_config = agent_config.marl.agent_configs[f"agent_{agent_id}"]
        model_path = agent_specific_config.model.path
        model_seed = agent_specific_config.model.seed
        
        # 同步更新所有相关路径
        agent_config.actor_rollout_ref.model.path = model_path
        agent_config.critic.model.path = model_path
        agent_config.critic.model.tokenizer_path = model_path
        agent_config.reward_model.model.input_tokenizer = model_path
        agent_config.actor_rollout_ref.rollout.seed = model_seed

        return agent_config



    def _build_agents(self):

        # 为每个agent创建独立的PPO trainer
        self.agents = []


        shared_kwargs = {
            'hybrid_engine': self.hybrid_engine,
            'use_reference_policy': self.use_reference_policy,
            'use_rm': self.use_rm,
            'use_critic': self.use_critic,
            'ref_in_actor': self.ref_in_actor,
        }

        # 获取agent的unthinking状态
        unthinking_mode_list = []
        for agent_id in range(self.num_agents):
            unthinking_mode_list.append(self.config.marl.agent_configs[f"agent_{agent_id}"].model.unthinking_mode)

        print(f"unthinking_mode_list: {unthinking_mode_list}")
        
        for agent_id in range(self.num_agents):

            # 把MARLRole的mapping转换为每个PPO的Role mapping
            ppo_role_mapping = _convert_marl_to_ppo_roles(
                self.role_worker_mapping[agent_id], 
                agent_id
            )
            
            agent_cls = AGENT_REGISTRY[self.config.marl.agent_cls]
            # agent_cls = AGENT_REGISTRY['ppo']

            agent_config = self._build_agent_config(agent_id)
            agent_tokenizer = self.tokenizer_list[agent_id]
            agent_processor = self.processor_list[agent_id]


            self.agents.append(agent_cls(agent_id=agent_id, 
                                            config=agent_config,  # 支持异构agent config
                                            tokenizer=agent_tokenizer,
                                            processor=agent_processor,
                                            role_worker_mapping=ppo_role_mapping,
                                            resource_pool_manager=self.resource_pool_manager,
                                            ray_worker_group_cls=self.ray_worker_group_cls,
                                            unthinking_mode=unthinking_mode_list[agent_id],
                                            **shared_kwargs
                                        )
                                        )


