# Standard MARL Learner Base Class

class RayMARLLearner:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 mac,
                 num_agents,  # cotrian LLMs
                 **kwargs):

        self.config = config
        self.num_agents = num_agents

        self.extra_params = kwargs
        self.use_critic = kwargs.get('use_critic', False)
        self.use_reference_policy = kwargs.get('use_reference_policy', False)
        self.use_rm = kwargs.get('use_rm', False)
        self.ref_in_actor = kwargs.get('ref_in_actor', False)
        self.kl_ctrl_in_reward = kwargs.get('kl_ctrl_in_reward', False)
        self.hybrid_engine = kwargs.get('hybrid_engine', False)


        self.mac = mac # 传入
        self.mixer = None



    def train(self, timing_raw):
        pass

    def init_workers(self):
        pass


    """这两个目前先不管，还需要考虑ppo层的存储路径问题"""
    def _save_checkpoint(self, path):
        """直接调用每个agent的save"""
        for agent_id, agent in enumerate(self.agents):
            agent_path = f"{path}/{agent_id}"
            agent._save_checkpoint(agent_path)



    def _load_checkpoint(self, path):
        """直接调用每个agent的load"""
        for agent_id, agent in enumerate(self.agents):
            agent_path = f"{path}/{agent_id}"
            agent._load_checkpoint(agent_path)