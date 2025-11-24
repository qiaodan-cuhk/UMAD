import torch
import torch.distributed as dist
from sentence_transformers import SentenceTransformer
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl import DataProto

class EmbedWorker(Worker):
    def __init__(self, config=None, role: str="embed"):
        super().__init__()
        self.config = config
        self.role = role
        # 初始化分布式（RayWorkerGroup 会设置 RANK/WORLD_SIZE/CUDA_VISIBLE_DEVICES）
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        # print(f"EmbedWorker rank: {dist.get_rank()}")
        # print(f"EmbedWorker world_size: {dist.get_world_size()}")
        

        torch.backends.cuda.enable_mem_efficient_sdp(True)
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self._normalize = True
        self.default_bs = 8

        # 新增：控制显存占用的micro batch大小
        # self.embedding_micro_batch_size = self.config.marl.embedding_micro_batch_size if hasattr(self.config.marl, "embedding_micro_batch_size") else 64  # 每个micro batch最多处理的文本数量


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, model_name="/root/models/qwen3-embedding-0.6b", dtype=None, normalize=True):
        current_device = get_torch_device().current_device()
        self.model = SentenceTransformer(model_name, device=current_device)
        if current_device != "cpu":
            self.model = self.model.to(torch.bfloat16 if dtype=="bf16" else torch.float32)
        self.model.eval()
        self._normalize = normalize


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def unload_model(self):
        # clear kv cache
        get_torch_device().empty_cache()


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @torch.inference_mode()
    def encode_texts(self, data: DataProto):

        data = data.to(get_torch_device().current_device())
        
        texts = data.non_tensor_batch["texts"]  # 已是当前 rank 的子批
        micro_bs = data.meta_info.get("micro_batch_size", self.default_bs)

        # print(f"embedding_micro_batch_size: {self.embedding_micro_batch_size}")
        # print(f"micro_bs: {micro_bs}")
        # print(f"texts length: {len(texts)}")

        # 打印每个worker处理的数据量
        # print(f"Worker on {get_torch_device().current_device()} processing {len(texts)} texts")

        if len(texts) > 0:

            # # 使用micro batch处理
            # results = []
            # for i in range(0, len(texts), self.embedding_micro_batch_size):
            #     micro_batch_texts = texts[i:i+self.embedding_micro_batch_size]
                
            #     # 处理当前micro batch
            #     micro_batch_result = self.model.encode(
            #         micro_batch_texts, batch_size=micro_bs, convert_to_tensor=True,
            #         normalize_embeddings=self._normalize, show_progress_bar=False
            #     )
            #     if micro_batch_result.dim() == 1:
            #         micro_batch_result = micro_batch_result.unsqueeze(0)
                
            #     # 立即移动到CPU并清理GPU内存
            #     micro_batch_result_cpu = micro_batch_result.cpu()
            #     results.append(micro_batch_result_cpu)
            #     # del micro_batch_result
            #     get_torch_device().empty_cache()
            
            # # 在CPU上拼接所有结果，然后移回GPU
            # local = torch.cat(results, dim=0)


            local = self.model.encode(
                texts, batch_size=micro_bs, convert_to_tensor=True,
                normalize_embeddings=self._normalize, show_progress_bar=False
            )

            if local.dim() == 1:
                local = local.unsqueeze(0)
        else:
            current_device = get_torch_device().current_device()
            dim = self.model.get_sentence_embedding_dimension()
            local = torch.empty(0, dim, dtype=torch.float32, device=current_device)

        get_torch_device().empty_cache()
        
        return DataProto.from_dict(tensors={"embeddings": local.to("cpu")})