# Multi Agent RLHF Dataset to support different tokenizers and processors for different agents


import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

# data_list 从 row_dict 变成了 {agent0: row_dict, agent1: row_dict, ...}
# 对应的也要修改collate逻辑
def collate_fn_marl(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """

    # 获取agent数量
    num_agents = len(data_list[0])
    stack_datas = {f"agent_{i}": {} for i in range(num_agents)}

    for agent_id in range(num_agents):
        agent_key = f"agent_{agent_id}"
        tensors = defaultdict(list)
        non_tensors = defaultdict(list)

        agent_data = [item[agent_key] for item in data_list]

        for data in agent_data:
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    tensors[key].append(val)
                else:
                    non_tensors[key].append(val)

        for key, val in tensors.items():
            tensors[key] = torch.stack(val, dim=0)

        for key, val in non_tensors.items():
            non_tensors[key] = np.array(val, dtype=object)

        stack_datas[agent_key] = {**tensors, **non_tensors}

    return stack_datas



class MARLRLHFDataset(Dataset):
    """
    支持多个异构agent的RLHF数据集。
    每个agent可以有自己的tokenizer和processor，但处理相同的原始数据。
    """
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer_list: List[PreTrainedTokenizer],
        config: DictConfig,
        processor_list: Optional[List[ProcessorMixin]] = None,
        unthinking_mode: Optional[List[bool]] = None,   # 指定每个agent是否使用unthinking mode，专门用于qwen3系列关闭推理
        **kwargs
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # 用于恢复

        # 异构tokenizer
        self.tokenizer_list = tokenizer_list
        self.processor_list = processor_list if processor_list else [None] * len(tokenizer_list)
        self.num_agents = len(self.tokenizer_list)
        assert len(self.tokenizer_list) == len(self.processor_list), \
            f"Number of tokenizers ({len(self.tokenizer_list)}) must match number of agents ({len(self.processor_list)})"

        # 数据处理配置
        self.config = config
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get('use_shm', False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False

        self.sample_ratio = config.get("sample_ratio", 1.0)  # 降采样到 1/10 dataset

        # self.use_qwen3_unthinking = config.get("use_qwen3_unthinking", False)
        self.unthinking_mode_list = unthinking_mode
        assert len(self.unthinking_mode_list) == self.num_agents, \
            f"Number of unthinking mode ({len(self.unthinking_mode_list)}) must match number of agents ({self.num_agents})"

        # 初始化数据集
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)


    def check_length_for_all_agents(self, doc):
        """检查prompt是否对所有agent的tokenizer都满足长度要求，这里几乎可以不考虑unthinking mode的影响"""
        messages = doc[self.prompt_key]
        for i, tokenizer in enumerate(self.tokenizer_list):
            # if self.use_qwen3_unthinking:
            #     raw_prompt = tokenizer.apply_chat_template(
            #         messages, 
            #         add_generation_prompt=True,
            #         enable_thinking=False
            #     )
            # else:
            raw_prompt = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            
            if len(raw_prompt) > self.max_prompt_length:
                return False
        return True


    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe = datasets.concatenate_datasets(dataframes)
        
        print(f"Initial dataset size: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:

            # 过滤数据
            self.dataframe = self.dataframe.filter(
                self.check_length_for_all_agents,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens for all agents"
            )
            
            print(f"filter dataset len: {len(self.dataframe)}")

        # 降采样到 1/10 dataset
        if self.sample_ratio < 1.0:
            target = max(1, int(len(self.dataframe) * self.sample_ratio))
            self.dataframe = self.dataframe.shuffle().select(range(target))
            print(f"sample dataset len: {len(self.dataframe)}")


    # resume dataset state for checkpointing and resume training
    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages = example[self.prompt_key]

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list
        
        return messages


    """
    处理单个样本,生成每个agent的输入数据
    采样的是相同的文本text
    输出的是一组agent_outputs:{'agent_0': row_dict, 'agent_1': row_dict, ...}
    row_dict: {
        'input_ids': tensor,
        'attention_mask': tensor,
        'position_ids': tensor,
        'raw_prompt_ids': tensor,
        'raw_prompt': str,
        'full_prompts': str,
        'index': int,
        'tools_kwargs': dict,
        'multi_modal_data': dict,
        'multi_modal_inputs': dict,
    }
    """
    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        
        # 为每个agent处理数据
        agent_outputs = {}
        
        for agent_id in range(self.num_agents):
            agent_row_dict = copy.deepcopy(row_dict)
            agent_messages = copy.deepcopy(messages)
            
            tokenizer = self.tokenizer_list[agent_id]
            processor = self.processor_list[agent_id]
            
            model_inputs = {}
            agent_dict = {}

            agent_unthinking_mode = self.unthinking_mode_list[agent_id]
            # print(f"agent_{agent_id} unthinking mode: {agent_unthinking_mode}")   # 测试通过
            
            # 处理多模态数据
            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                raw_prompt = processor.apply_chat_template(agent_messages, add_generation_prompt=True, tokenize=False)
                multi_modal_data = {}

                images = None
                if self.image_key in agent_row_dict:
                    images = [process_image(image) for image in agent_row_dict.pop(self.image_key)]
                    multi_modal_data["image"] = images

                videos = None
                if self.video_key in agent_row_dict:
                    videos = [process_video(video) for video in agent_row_dict.pop(self.video_key)]
                    multi_modal_data["video"] = [video.numpy() for video in videos]

                model_inputs = processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

                input_ids = model_inputs.pop("input_ids")
                attention_mask = model_inputs.pop("attention_mask")

                if "second_per_grid_ts" in model_inputs:
                    model_inputs.pop("second_per_grid_ts")

                # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
                agent_row_dict["multi_modal_data"] = multi_modal_data
                agent_row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                agent_row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)
            else:
                if agent_unthinking_mode:  # 专门用于qwen3系列关闭推理
                    raw_prompt = tokenizer.apply_chat_template(agent_messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
                else:
                    raw_prompt = tokenizer.apply_chat_template(agent_messages, add_generation_prompt=True, tokenize=False)
                    
                # raw_prompt = tokenizer.apply_chat_template(agent_messages, add_generation_prompt=True, tokenize=False)
                model_inputs = tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
                input_ids = model_inputs.pop("input_ids")
                attention_mask = model_inputs.pop("attention_mask")
            
            # 后处理
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
            
            if processor is not None and processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
                from verl.models.transformers.qwen2_vl import get_rope_index

                position_ids = [
                    get_rope_index(
                        processor,
                        input_ids=input_ids[0],
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                        attention_mask=attention_mask[0],
                    )
                ]  # (1, 3, seq_len)

            else:
                position_ids = compute_position_id_with_mask(attention_mask)

            agent_row_dict["input_ids"] = input_ids[0]
            agent_row_dict["attention_mask"] = attention_mask[0]
            agent_row_dict["position_ids"] = position_ids[0]

            raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > self.max_prompt_length:
                if self.truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
                elif self.truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
                elif self.truncation == "middle":
                    left_half = self.max_prompt_length // 2
                    right_half = self.max_prompt_length - left_half
                    raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                elif self.truncation == "error":
                    raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

            agent_row_dict["raw_prompt_ids"] = raw_prompt_ids
            # encode prompts without chat template
            if self.return_raw_chat:
                agent_row_dict["raw_prompt"] = agent_messages

            # get prompts with chat template
            if self.return_full_prompt:
                agent_row_dict["full_prompts"] = raw_prompt  # array of strings

            # add index for each prompt
            index = agent_row_dict.get("extra_info", {}).get("index", 0)
            tools_kwargs = agent_row_dict.get("extra_info", {}).get("tools_kwargs", {})
            need_tools_kwargs = agent_row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
            if need_tools_kwargs and not tools_kwargs:
                logger.warning("tools_kwargs is empty for index {}, data source: {}", index, agent_row_dict["data_source"])
            agent_row_dict["index"] = index
            agent_row_dict["tools_kwargs"] = tools_kwargs
            
            # 构建输出
            agent_dict.update(agent_row_dict)
            agent_outputs[f"agent_{agent_id}"] = agent_dict
        
        return agent_outputs


    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()