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
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings
from typing import Union

import psutil
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from torch.distributed.device_mesh import init_device_mesh

import verl.utils.torch_functional as verl_F
from verl.utils.py_functional import convert_to_regular_types
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    layered_summon_lora_params,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available


from peft import LoraConfig, TaskType, get_peft_model
from codetiming import Timer

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from peft import PeftModel
from safetensors.torch import save_file
from dataclasses import asdict
import json


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"])
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


from verl.workers.fsdp_workers import CriticWorker

class VDNCriticWorker(CriticWorker):
    def __init__(self, config):
        super().__init__(config)
        # super().__init__()
        # import torch.distributed

        # if not torch.distributed.is_initialized():
        #     torch.distributed.init_process_group(backend="nccl" if is_cuda_available else "hccl")
        # self.config = config

        # # build device mesh for Ulysses Sequence Parallel
        # world_size = torch.distributed.get_world_size()
        # from torch.distributed.device_mesh import init_device_mesh

        # fsdp_size = self.config.model.fsdp_config.fsdp_size
        # self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        # self.ulysses_device_mesh = None
        # self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        # dp = world_size // self.ulysses_sequence_parallel_size
        # if self.ulysses_sequence_parallel_size > 1:
        #     self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        # self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # # set FSDP offload params
        # self._is_offload_param = self.config.model.fsdp_config.param_offload
        # self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # # normalize config
        # self.config.ppo_mini_batch_size *= self.config.rollout_n
        # self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        # if self.config.ppo_micro_batch_size is not None:
        #     self.config.ppo_micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        #     self.config.forward_micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        #     self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
        #     self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        # if self.config.ppo_micro_batch_size_per_gpu is not None:
        #     assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
        #     assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
        # self._is_lora = self.config.model.get('lora_rank', 0) > 0



    # def _build_critic_model_optimizer(self, config):
    #     # the following line is necessary
    #     from torch import optim
    #     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    #     from torch.distributed.fsdp import MixedPrecision

    #     from verl.utils.model import print_model_size
    #     from verl.utils.torch_dtypes import PrecisionType

    #     use_shm = config.model.get('use_shm', False)
    #     local_path = copy_to_local(config.model.path, use_shm=use_shm)
    #     # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
    #     # using random initialized model from any architecture. May not be the same as Actor.

    #     tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
    #     self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
    #     self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

    #     from omegaconf import OmegaConf

    #     override_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
    #     override_config_kwargs = {
    #         "bos_token_id": self.tokenizer.bos_token_id,
    #         "eos_token_id": self.tokenizer.eos_token_id,
    #         "pad_token_id": self.tokenizer.pad_token_id,
    #     }
    #     override_config_kwargs.update(override_config)
    #     if self.rank == 0:
    #         print(f"Critic overriding config {override_config_kwargs}")

    #     torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
    #     torch_dtype = PrecisionType.to_dtype(torch_dtype)

    #     from transformers import AutoConfig, AutoModelForTokenClassification

    #     critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=config.model.get("trust_remote_code", False))
    #     critic_model_config.num_labels = 1

    #     init_context = get_init_weight_context_manager(use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh)

    #     with init_context(), warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         critic_model_config.classifier_dropout = 0.0
    #         critic_model_config.hidden_dropout = "0"
    #         critic_module = AutoModelForTokenClassification.from_pretrained(
    #             pretrained_model_name_or_path=local_path,
    #             torch_dtype=torch_dtype,
    #             config=critic_model_config,
    #             attn_implementation="flash_attention_2",
    #             trust_remote_code=config.model.get("trust_remote_code", False),
    #         )

    #         use_remove_padding = config.model.get("use_remove_padding", False)

    #         apply_monkey_patch(
    #             model=critic_module,
    #             use_remove_padding=use_remove_padding,
    #             ulysses_sp_size=self.ulysses_sequence_parallel_size,
    #         )

    #         # some parameters may not in torch_dtype
    #         critic_module.to(torch_dtype)

    #         if config.model.get("enable_gradient_checkpointing", False):
    #             critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
    #     if self._is_lora:
    #         print("Applying LoRA to critic module")
    #         critic_module.enable_input_require_grads()
    #         # Convert config to regular Python types before creating PEFT model
    #         lora_config = {
    #             'task_type': TaskType.CAUSAL_LM,
    #             'r': self.config.model.lora_rank,
    #             'lora_alpha': self.config.model.lora_alpha,
    #             'target_modules': convert_to_regular_types(self.config.model.target_modules),
    #             'bias': "none",
    #         }
    #         critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

    #     if self.rank == 0:
    #         print_model_size(critic_module)

    #     self.critic_model_config = critic_model_config

    #     fsdp_config = self.config.model.fsdp_config
    #     mixed_precision_config = fsdp_config.get("mixed_precision", None)
    #     if mixed_precision_config is not None:
    #         param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
    #         reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
    #         buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
    #     else:
    #         param_dtype = torch.bfloat16
    #         reduce_dtype = torch.float32
    #         buffer_dtype = torch.float32

    #     mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

    #     auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy, is_lora=self.config.model.get('lora_rank', 0) > 0)

    #     log_gpu_memory_usage("Before critic FSDP", logger=None)

    #     fsdp_mesh = self.device_mesh
    #     sharding_strategy = get_sharding_strategy(fsdp_mesh)

    #     # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
    #     if config.strategy == "fsdp":
    #         critic_module = FSDP(
    #             critic_module,
    #             param_init_fn=init_fn,
    #             use_orig_params=False,
    #             auto_wrap_policy=auto_wrap_policy,
    #             device_id=get_torch_device().current_device(),
    #             sharding_strategy=sharding_strategy,
    #             mixed_precision=mixed_precision,
    #             sync_module_states=True,
    #             forward_prefetch=False,
    #             device_mesh=self.device_mesh,
    #             cpu_offload=None,
    #         )
    #     elif config.strategy == "fsdp2":
    #         assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
    #         mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
    #         offload_policy = None
    #         if fsdp_config.offload_policy:
    #             self._is_offload_param = False
    #             self._is_offload_optimizer = False
    #             offload_policy = CPUOffloadPolicy(pin_memory=True)

    #         fsdp_kwargs = {
    #             "mesh": fsdp_mesh,
    #             "mp_policy": mp_policy,
    #             "offload_policy": offload_policy,
    #             "reshard_after_forward": fsdp_config.reshard_after_forward,
    #         }
    #         full_state = critic_module.state_dict()
    #         apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
    #         fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
    #     else:
    #         raise NotImplementedError(f"Unknown strategy {config.strategy}")

    #     if config.model.get("enable_activation_offload", False):
    #         enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
    #         enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

    #     log_gpu_memory_usage("After critic FSDP", logger=None)

    #     critic_optimizer = optim.AdamW(
    #         critic_module.parameters(),
    #         lr=config.optim.lr,
    #         betas=config.optim.get("betas", (0.9, 0.999)),
    #         weight_decay=config.optim.get("weight_decay", 1e-2),
    #     )

    #     total_steps = config.optim.get("total_training_steps", 0)
    #     num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
    #     warmup_style = config.optim.get("warmup_style", "constant")
    #     if num_warmup_steps < 0:
    #         num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
    #         num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

    #     print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

    #     from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

    #     if warmup_style == "constant":
    #         critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps)
    #     elif warmup_style == "cosine":
    #         critic_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    #     else:
    #         raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

    #     return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)


        """这里替换成vdn的DPCritic,修改了update的逻辑"""
        from marl.modules.critics.vdn_critic import DataParallelVDNCritic
        # self.critic = DataParallelPPOCritic(config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer)
        self.critic = DataParallelVDNCritic(config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer)


        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_contents=self.config.checkpoint.contents,
        )

    # 目前不使用use_dynamic_bsz，所以不修改这个接口
    # @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # def compute_values(self, data: DataProto):
    #     # Support all hardwares
    #     data = data.to(get_torch_device().current_device())

    #     if self._is_offload_param:
    #         load_fsdp_model_to_gpu(self.critic_module)
    #     micro_batch_size = self.config.forward_micro_batch_size_per_gpu
    #     data.meta_info["micro_batch_size"] = micro_batch_size
    #     data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
    #     data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
    #     # perform forward computation
    #     with self.ulysses_sharding_manager:
    #         data = self.ulysses_sharding_manager.preprocess_data(data=data)
    #         values = self.critic.compute_values(data=data)
    #         output = DataProto.from_dict(tensors={"values": values})
    #         output = self.ulysses_sharding_manager.postprocess_data(data=output)

    #     output = output.to("cpu")
    #     if self._is_offload_param:
    #         offload_fsdp_model_to_cpu(self.critic_module)
    #     return output




    """这里要修改"""
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_torch_device().current_device())
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_torch_device().current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            # debug
            # print(f"GPU {self.rank} data.batch: {data.batch}")

            
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

