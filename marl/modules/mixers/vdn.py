import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from marl.modules.critics.qmix_critic import Critic
from sentence_transformers import SentenceTransformer



class VDNMixer(nn.Module):
    def __init__(self, config):
        super(VDNMixer, self).__init__()

        self.config = config
        self.n_agents = config.marl.num_agents
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # === 新增: 加载PLM用于文本编码（如BERT、Roberta） ===
        # self.text_lm_name = getattr(args, "text_lm_name", "roberta-base")
        self.text_lm_name = '/root/models/qwen3-embedding-0.6b'
        # 使用sentence_transformers的模型，从文本直接转换为embedding
        self.text_lm = SentenceTransformer(self.text_lm_name, device=self.device)    # 目前verl的resourcepool占用了所有卡，不可见
        self.text_tokenizer = self.text_lm.tokenizer

        # self.text_lm = AutoModel.from_pretrained(self.text_lm_name)
        # self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_lm_name)

        self.text_lm.eval()  # 推荐推理时不更新参数，只用于特征抽取
        self.text_lm_embedding_dim = self.text_lm.get_sentence_embedding_dimension()   # 1024

        self.state_dim = self.text_lm_embedding_dim
        self.action_dim = self.text_lm_embedding_dim
        self.embed_dim = 256
        
        # sentence Q network
        
        self.critic_embed = 1024
        self.critics_sentence = [Critic(self.state_dim+self.action_dim, self.critic_embed, 1) for _ in range(self.n_agents)]

        ### 这里要考虑是否deepcopy，以及polyak更新
        self.target_critics_sentence = [Critic(self.state_dim+self.action_dim, self.critic_embed, 1) for _ in range(self.n_agents)]

        # 将所有网络移到GPU
        # self.hyper_w_1 = self.hyper_w_1.to(self.device)
        # self.hyper_w_final = self.hyper_w_final.to(self.device)
        # self.hyper_b_1 = self.hyper_b_1.to(self.device)
        # self.V = self.V.to(self.device)
        
        # 将critic networks移到GPU
        self.critics_sentence = [critic.to(self.device) for critic in self.critics_sentence]
        self.target_critics_sentence = [critic.to(self.device) for critic in self.target_critics_sentence]



    def forward(self, agent_qs):
        """
        agent_qs: [batch, n_agents, 1]
        """

        # 确保输入数据在正确的设备上
        agent_qs = agent_qs.to(self.device)
        bs = agent_qs.size(0)
        q_tot = agent_qs.sum(dim=1, keepdim=True).squeeze(-1)  # [bs, 1]
        assert q_tot.shape[0] == bs, "q_tot shape is not (bs, 1)"

        return q_tot



