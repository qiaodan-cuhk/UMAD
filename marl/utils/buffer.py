import numpy as np

class MultiAgentTurnReplayBuffer:
    def __init__(self, num_agents, capacity=10000):
        self.max_size = capacity
        self.size = 0
        self.num_agents = num_agents
        self.states = np.array(['']*self.max_size, dtype='object')
        self.actions = np.empty((self.max_size, num_agents), dtype='object')
        self.rewards = np.empty((self.max_size,), dtype=np.float32)
        self.next_states = np.array(['']*self.max_size, dtype='object')
        self.dones = np.empty((self.max_size,), dtype=bool)
        # self.mc_returns = np.empty((self.max_size, num_agents), dtype=np.float32)
        self.action_lengths = np.empty((self.max_size, num_agents), dtype=np.int32)


    def insert(self, state, actions, rewards, next_state, done, action_lengths):
        idx = self.size % self.max_size
        self.states[idx] = state
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_states[idx] = next_state
        self.dones[idx] = done
        # self.mc_returns[idx] = mc_returns
        self.action_lengths[idx] = np.array(
            action_lengths,
            dtype=np.int32
        )

        self.size += 1


    def insert_batch(self, states, actions, rewards, next_states, dones, action_lengths):
        batch_size = len(dones)
        for i in range(batch_size):
            self.insert(states[i], actions[i], rewards[i], next_states[i], dones[i], action_lengths[i])


    def sample(self, batch_size):
        max_idx = min(self.size, self.max_size)
        indices = np.random.choice(max_idx, batch_size, replace=False)
        return {
            "state": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_state": self.next_states[indices],
            "done": self.dones[indices],
            "action_lengths": self.action_lengths[indices]
            # "mc_returns": self.mc_returns[indices],
        }

    def __len__(self):
        return min(self.size, self.max_size)
    
    def length_summary(self, short_th=50):
        """
        统计当前buffer中所有样本（按agent分别）的长度均值、最小、最大、短回复占比。
        返回: dict[agent_i] -> {count, mean, min, max, short_ratio}
        """
        max_idx = min(self.size, self.max_size)
        if max_idx == 0:
            return {}

        L = self.action_lengths[:max_idx]  # [N, num_agents]
        out = {}
        for agent_id in range(self.num_agents):
            x = L[:, agent_id].astype(np.float32)
            out[f"agent_{agent_id}"] = {
                "buffer_response_len_mean": float(np.mean(x)),
                "buffer_response_len_min": float(np.min(x)),
                "buffer_response_len_max": float(np.max(x)),
                "buffer_response_len_short_ratio": float(np.mean(x < short_th)),
            }
        return out