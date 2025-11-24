# 用于sentence q估计
import torch
import torch.nn as nn

# Archer Critic
# class ArcherCritic(torch.nn.Module):
#     def __init__(self, device, accelerator, critic_lm, cache_dir, in_dim, out_dim):
#         # 使用统一的LLM编码器处理文本
#         self.base_lm = AutoModel.from_pretrained(critic_lm, cache_dir=cache_dir)
#         self.base_tokenizer = AutoTokenizer.from_pretrained(critic_lm, cache_dir=cache_dir)
        
#         # 关键设计：两个独立的critic网络
#         # Q(s,a) = critic1/2(concat(encode(s), encode(a)))
#         self.critic1 = nn.Sequential(nn.Linear(in_dim*2, in_dim), ...)  # Q网络
#         self.critic2 = nn.Sequential(nn.Linear(in_dim*2, in_dim), ...)  # Double Q
        
#         # V(s) = v_critic1/2(encode(s))
#         self.v_critic1 = nn.Sequential(nn.Linear(in_dim, in_dim), ...)  # 状态价值
#         self.v_critic2 = nn.Sequential(nn.Linear(in_dim, in_dim), ...)




# FACMAC like critic, only maintain Q network
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        # Q(s,a) = critic(concat(encode(s), encode(a)))
        """这里的in dim*2需要重新考虑下，给mixer设置state和action不同的embed size"""
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, state_embed, action_embed):
        # state_embed, action_embed: [batch, embed_dim]
        x = torch.cat([state_embed, action_embed], dim=-1)
        q = self.critic(x)
        return q  # [batch, out_dim]
        
