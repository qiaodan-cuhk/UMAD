import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import glob


from transformers import AutoTokenizer, AutoModel
from marl.modules.critics.qmix_critic import Critic


class QMixer(nn.Module):
    def __init__(self, config):
        super(QMixer, self).__init__()

        self.config = config
        self.n_agents = config.marl.num_agents
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')   # 固定在cpu上


        # === 新增: 加载PLM用于文本编码（如BERT、Roberta） ===
        # self.text_lm_name = getattr(args, "text_lm_name", "roberta-base")
        self.text_lm_name = '/root/models/qwen3-embedding-0.6b'


        # 使用sentence_transformers的模型，从文本直接转换为embedding
        # self.text_lm = SentenceTransformer(self.text_lm_name, device=self.device).to(torch.bfloat16)    # 固定在cpu上
        # self.text_tokenizer = self.text_lm.tokenizer # 只使用tokenizer，不用model
        # self.text_lm.eval()  # 推荐推理时不更新参数，只用于特征抽取
        # self.text_lm_embedding_dim = self.text_lm.get_sentence_embedding_dimension()   # 1024

        # 使用transformers的tokenizer，避免加载模型
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_lm_name)
        # 自动匹配embed model维度
        self.text_lm_embedding_dim = self.get_embedding_dim(self.text_lm_name)
        

        self.state_dim = self.text_lm_embedding_dim
        self.action_dim = self.text_lm_embedding_dim
        self.embed_dim = 256
        

        # if getattr(args, "hypernet_layers", 1) == 1:
        #     self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        #     self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        # elif getattr(args, "hypernet_layers", 1) == 2:
        # hypernet_embed = self.args.hypernet_embed
        hypernet_embed = 2048
        # 768-2048-2*embed_dim
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim))
        # elif getattr(args, "hypernet_layers", 1) > 2:
        #     raise Exception("Sorry >2 hypernet layers is not implemented!")
        # else:
        #     raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))


        # sentence Q network
        
        self.critic_embed = 1024
        self.critics_sentence = [Critic(self.state_dim+self.action_dim, self.critic_embed, 1) for _ in range(self.n_agents)]

        ### 这里要考虑是否deepcopy，以及polyak更新
        self.target_critics_sentence = [Critic(self.state_dim+self.action_dim, self.critic_embed, 1) for _ in range(self.n_agents)]

        # 将所有网络移到GPU
        self.hyper_w_1 = self.hyper_w_1.to(self.device)
        self.hyper_w_final = self.hyper_w_final.to(self.device)
        self.hyper_b_1 = self.hyper_b_1.to(self.device)
        self.V = self.V.to(self.device)
        
        # 将critic networks移到GPU
        self.critics_sentence = [critic.to(self.device) for critic in self.critics_sentence]
        self.target_critics_sentence = [critic.to(self.device) for critic in self.target_critics_sentence]



    """统一接受base llm tokenizer ids,转换为embedding再输入self.critic得到每个agent的q values，再经过mixer network"""
    def forward(self, agent_qs, state_embed):
        """
        agent_qs: [batch, n_agents, 1]
        state_embed: [batch, state_dim]
        """

        # 确保输入数据在正确的设备上
        device = next(self.parameters()).device  # 获取模型所在设备
        agent_qs = agent_qs.to(device)
        state_embed = state_embed.to(device)

        
        bs = agent_qs.size(0)
        # First layer
        w1 = torch.abs(self.hyper_w_1(state_embed))  # [batch, n_agents * embed_dim]
        b1 = self.hyper_b_1(state_embed)             # [batch, embed_dim]
        w1 = w1.view(-1, self.n_agents, self.embed_dim)  # [batch, n_agents, embed_dim]
        b1 = b1.view(-1, 1, self.embed_dim)              # [batch, 1, embed_dim]
        hidden = F.elu(torch.bmm(agent_qs.transpose(1, 2), w1) + b1)  # [batch, 1, embed_dim]
        # Second layer
        w_final = torch.abs(self.hyper_w_final(state_embed))  # [batch, embed_dim]
        w_final = w_final.view(-1, self.embed_dim, 1)         # [batch, embed_dim, 1]
        v = self.V(state_embed).view(-1, 1, 1)                # [batch, 1, 1]
        y = torch.bmm(hidden, w_final) + v                    # [batch, 1, 1]
        q_tot = y.view(bs, 1)
        return q_tot


    @staticmethod
    def get_embedding_dim(model_dir: str, fallback: int | None = None) -> int:
        """
        在不加载模型权重的情况下，从本地嵌入模型目录推断句向量维度（embedding dim）。
        优先解析 sentence-transformers 的 Pooling 配置；否则回退到 transformers 的 config.json。

        参数:
        - model_dir: 本地模型目录（如 '/root/models/qwen3-embedding-0.6b'）
        - fallback: 解析失败时的后备维度（可选）

        返回:
        - int: 句向量维度

        异常:
        - ValueError: 当无法解析且未提供 fallback 时抛出
        """
        if not os.path.isdir(model_dir):
            if isinstance(fallback, int) and fallback > 0:
                return fallback
            raise ValueError(f"模型目录不存在: {model_dir}")

        # 1) 优先解析 sentence-transformers 目录结构中的 Pooling 配置
        pooling_patterns = [
            os.path.join(model_dir, "*Pooling*/config.json"),
            os.path.join(model_dir, "1_Pooling/config.json"),
            os.path.join(model_dir, "modules.json"),  # 旧版结构
        ]
        for pat in pooling_patterns:
            for path in glob.glob(pat):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    if isinstance(cfg, dict):
                        for k in [
                            "pooling_output_dimension",
                            "sentence_embedding_dimension",
                            "word_embedding_dimension",
                            "embedding_size",
                        ]:
                            if isinstance(cfg.get(k), int):
                                return int(cfg[k])
                    elif isinstance(cfg, list):
                        # modules.json 旧结构
                        for m in cfg:
                            if isinstance(m, dict) and m.get("type", "").lower() == "pooling":
                                dim = m.get("pooling_output_dimension") or m.get("word_embedding_dimension")
                                if isinstance(dim, int):
                                    return int(dim)
                except Exception:
                    pass

        # 2) 回退 transformers 的 config.json
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                for k in [
                    "sentence_embedding_dimension",
                    "hidden_size",
                    "d_model",
                    "word_embedding_dimension",
                    "embedding_size",
                ]:
                    if isinstance(cfg.get(k), int):
                        return int(cfg[k])
            except Exception:
                pass

        # 3) 失败时使用 fallback
        if isinstance(fallback, int) and fallback > 0:
            return fallback

        raise ValueError(
            f"无法从 {model_dir} 自动解析 embedding 维度；请提供 fallback，或确保目录包含可解析的 Pooling/config.json 或 config.json。"
    )




