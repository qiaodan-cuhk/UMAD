
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, Optional
from copy import deepcopy
from tqdm import tqdm

# 处理prompt ids
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

from tensordict import TensorDict

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs

from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tracking import ValidationGenerationsLogger

# 下放到agent层
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.single_controller.ray.base import create_colocated_worker_cls


from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.ppo.reward import compute_reward, compute_reward_async  # fit
from verl.trainer.ppo.core_algos import agg_loss  # fit

from verl.workers.rollout.async_server import AsyncLLMServerManager

from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.utils.metric import (
    reduce_metrics,
)

import torch
from verl.utils.torch_functional import masked_mean




WorkerType = Type[Worker]


from marl.utils.marl_utils import MARLRole, Role


from marl.modules.agents.ppo_agent import ResourcePoolManager
from marl.utils.marl_utils import AdvantageEstimator, compute_advantage, compute_response_mask, apply_kl_penalty


from marl.utils.marl_utils import _convert_marl_to_ppo_roles, _timer 


class Multi_Turn_Runner:
    def __init__(self,
                 config,
                 mac,
                 num_agents,  # cotrian LLMs
                 tokenizer,   # 单个，表示共用model
                 role_worker_mapping: list[dict[Role, WorkerType]],
                 resource_pool_manager: ResourcePoolManager, # 共用全局manager
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, # fsdp/Megatron
                 processor=None,  # 单个，表示共用model
                 reward_fn=None, # reward function公用相同的
                 val_reward_fn=None,
                 train_dataset: Optional[Dataset] = None,
                 val_dataset: Optional[Dataset] = None,
                 collate_fn=None,
                 train_sampler: Optional[Sampler] = None,
                 device_name="cuda"):
        
        self.config = config
        self.reward_fn = reward_fn

        self.is_sum = config.marl.sum_reward
        self.cory_flip = config.marl.cory_flip
        self.tb_dir = config.marl.tensorboard_dir

        self.mac = mac
        # self.mac.agents = [PPO PPO]



    """
    这个应该只影响dataloader，不用改
    需要放到runner层，与data生成一起处理
    """
    # 后续考虑优化计算效率的时候，把这个data parallel可以进行优化，也就是batch 数据只进行一次dp分发到所有gpu
    # 同时保证所有gpu上都有每个agent，这样只需要batch数据分发一次
    # balance发生在response之后，update之前，所以这个考虑放在marl层吧，但是可以写成单独的data而不是所有agent batch的形式
    # 因为即使agent的response会concate到一起作为下一个prompt，也不影响每个agent内部的策略更新
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        # world_size = self.actor_rollout_wg.world_size
        world_size = self.mac.agents[0].actor_rollout_wg.world_size   # 暂时用第一个agent的world size，后面考虑换成每个agent内部balance
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)

        # 这里要改成multi agent metric
        metrics.update(global_balance_stats)


    # 每一个turn存一个dataproto，这样方便处理每个turn的balance batch和mask等等功能，当成单个句子处理
    # 但是需要注意reward的存储方式？只有最后一个turn计算reward，其他环节不计算。

    def _copy_agent_batchs(self, agent_batchs):
        """
        安全地复制agent_batchs字典，避免直接使用deepcopy
        """
        copied_batchs = {}
        for agent_key, agent_batch in agent_batchs.items():
            # 获取原始batch对象
            original_batch = agent_batch.batch
            
            # 从verl.protocol导入TensorDict来创建新的batch
            
            ###### 其实这里没有成功复制，返回的batch部分是空的，是不是因为pop以后把数据都清出去了？
            # 创建新的TensorDict对象
            new_batch = TensorDict({
                k: v.clone() if isinstance(v, torch.Tensor) else v 
                for k, v in original_batch.items()
            }, batch_size=original_batch.batch_size)
            
            # 复制non_tensor_batch和meta_info
            new_non_tensor_batch = agent_batch.non_tensor_batch.copy() if agent_batch.non_tensor_batch else {}
            new_meta_info = agent_batch.meta_info.copy() if agent_batch.meta_info else {}
            
            # 创建新的DataProto对象
            copied_batch = DataProto(
                batch=new_batch,
                non_tensor_batch=new_non_tensor_batch,
                meta_info=new_meta_info
            )
            
            copied_batchs[agent_key] = copied_batch
        
        return copied_batchs


    def rollout_multi_turn(self, gen_batch_all, multi_turn_batchs, metrics, turn=2):
        """
        gen_batch_all: {batch_key: DataProto}, 与single turn一致，只是存储dataloader的prompt
        multi_turn_batchs: {turn_idx: {agent_id: DataProto}}，只有turn_0的数据
        metrics: Dict[str, float]
        turn: int  代表交互几轮
        """
        

        for turn_idx in range(turn):

            # 取出每一轮的作为agent_batchs，这里需要已经pop掉其他数据了
            agent_batchs = multi_turn_batchs[f"turn_{turn_idx}"]
            # turn_gen_batch_all = gen_batch_all

            # 复制下一轮的batchs，用于concate prompt
            """检查这里copy的是否DataProto格式"""
            # next_turn_batchs = self._copy_agent_batchs(agent_batchs)

            # 每一轮的rollout，保证输入的gen_batch_all是当前轮的形式
            gen_batch_outputs_all = {}
            for agent, gen_batch_agent in zip(self.mac.agents, gen_batch_all.values()):
                gen_batch_output = agent.actor_rollout_wg.generate_sequences(gen_batch_agent)
                gen_batch_outputs_all[f"agent_{agent.agent_id}"] = gen_batch_output


            """完成了每个agent的rollout过程，后续复杂pipeline考虑函数包装，目前用于IPPO，增加轨迹uid以及balance gpu分配"""
            for agent_key, agent_batch  in agent_batchs.items():
                #### todo 应该只有第一轮需要添加uid
                agent_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(agent_batch.batch))],
                                                        dtype=object)
                # repeat to align with repeated responses in rollout
                agent_batch = agent_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                agent_batch = agent_batch.union(gen_batch_outputs_all[agent_key])

                agent_batch.batch['response_mask'] = compute_response_mask(agent_batch)
                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                if self.config.trainer.balance_batch:
                    self._balance_batch(agent_batch, metrics=metrics)
                
                # compute global_valid tokens
                agent_batch.meta_info['global_token_num'] = torch.sum(agent_batch.batch['attention_mask'], dim=-1).tolist()
                agent_batchs[agent_key] = agent_batch

            # 处理完额外属性，放回
            multi_turn_batchs[f"turn_{turn_idx}"] = agent_batchs
            
            # 处理所有agent的response，concate成下一轮的prompt
            if turn_idx < turn - 1:  # 最后一轮不处理
                # 1. 收集所有agent的响应文本
                all_agent_responses_text = {}
                for agent_idx, agent in enumerate(self.mac.agents):
                    agent_key = f"agent_{agent_idx}"
                    agent_batch = agent_batchs[agent_key]
                    
                    # 获取响应的token IDs和mask
                    response_ids = agent_batch.batch['responses']
                    response_mask = agent_batch.batch['response_mask']

                    
                    # 使用tokenizer解码为文本
                    tokenizer = agent.tokenizer
                    responses_text = []
                    
                    # 对每个样本的响应进行解码
                    for i in range(response_ids.shape[0]):
                        valid_ids = response_ids[i][response_mask[i].bool()]
                        text = tokenizer.decode(valid_ids, skip_special_tokens=True)   # 如果 true 就没有最后一个151645 <|im_end|>
                        responses_text.append(text)
                    
                    all_agent_responses_text[agent_key] = responses_text

                
                # 2. 获取下一轮的batchs并更新
                next_turn_batchs = multi_turn_batchs[f"turn_{turn_idx + 1}"]


                # 3. 为每个agent更新下一轮的输入，利用每个agent的tokenizer分别处理
                for agent_idx, agent in enumerate(self.mac.agents):
                    agent_key = f"agent_{agent_idx}"
                    next_batch = next_turn_batchs[agent_key]
                    
                    # 获取原始prompt文本
                    original_prompts = []
                    if 'raw_prompt' in next_batch.non_tensor_batch:
                        original_prompts = next_batch.non_tensor_batch['raw_prompt']
                    elif 'raw_prompt_ids' in next_batch.non_tensor_batch:
                        # 如果是字符串列表，直接使用
                        if isinstance(next_batch.non_tensor_batch['raw_prompt_ids'][0], str):
                            original_prompts = next_batch.non_tensor_batch['raw_prompt_ids']
                        # 如果是token ID列表，需要解码
                        else:
                            tokenizer = agent.tokenizer
                            for ids in next_batch.non_tensor_batch['raw_prompt_ids']:
                                ##### todo: 这里暂时考虑skip特殊字符，raw prompt是有<|im_start|>和<|im_end|>的
                                text = tokenizer.decode(ids, skip_special_tokens=True)
                                original_prompts.append(text)
                    
                    # 构建新的prompt，包含所有agent的响应
                    new_prompts = []
                    for i, orig_prompt in enumerate(original_prompts):
                        # 创建包含所有agent响应的新prompt
                        new_prompt = orig_prompt

                        # 按照agent_0, agent_1, ...的顺序拼接响应
                        for agent_idx_concat in range(len(self.mac.agents)):
                            resp_agent_key = f"agent_{agent_idx_concat}"
                            if resp_agent_key in all_agent_responses_text and i < len(all_agent_responses_text[resp_agent_key]):
                                # 直接拼接响应，不添加额外标识
                                new_prompt += all_agent_responses_text[resp_agent_key][i]
                                
                        
                        #### todo: 这里考虑是否对齐特殊token，转换成content并应用self.template，采样每个agent的tokenizer
                        # new_raw_prompt = agent.tokenizer.apply_chat_template(new_prompt, add_generation_prompt=True, tokenize=False)

                        #### 目前简单点处理，直接添加句首的 <|im_start|> 和 句尾的 <|im_end|>\n<|im_start|>assistant\n 
                        new_raw_prompt = f"<|im_start|>\n{new_prompt}\n<|im_end|>\n<|im_start|>assistant\n"
                        # new_prompt += f"\nAgent {agent_key}: "
                        new_prompts.append(new_raw_prompt)


                    # 逐条处理每个prompt，避免批处理长度不一致问题
                    processed_input_ids = []
                    processed_attention_masks = []
                    processed_position_ids = []
                    processed_raw_prompt_ids = []  # 存储处理后的raw_prompt_ids

                    for i, new_prompt in enumerate(new_prompts):
                        # 单独处理每个prompt，添加最大长度truncate
                        single_model_inputs = agent.tokenizer(
                            new_prompt, 
                            return_tensors="pt", 
                            add_special_tokens=False,
                            truncation=True,
                            max_length=self.config.data.max_prompt_length
                        )
                        
                        single_input_ids = single_model_inputs.pop("input_ids")
                        single_attention_mask = single_model_inputs.pop("attention_mask")
                        
                        # 使用postprocess_data处理单个样本
                        processed_ids, processed_mask = verl_F.postprocess_data(
                            input_ids=single_input_ids,
                            attention_mask=single_attention_mask,
                            max_length=self.config.data.max_prompt_length,
                            pad_token_id=agent.tokenizer.pad_token_id,
                            left_pad=True,
                            truncation=self.config.get("truncation", "error"),
                        )
                        
                        # 计算position_ids
                        processed_pos_ids = compute_position_id_with_mask(processed_mask)
                        
                        # 添加到列表
                        processed_input_ids.append(processed_ids[0])
                        processed_attention_masks.append(processed_mask[0])
                        processed_position_ids.append(processed_pos_ids[0])
                        
                        # 处理raw_prompt_ids
                        agent_max_prompt_length = self.config.data.max_prompt_length
                        raw_prompt_ids = agent.tokenizer.encode(new_prompt, add_special_tokens=False)
                        
                        # 截断过长的raw_prompt_ids
                        if len(raw_prompt_ids) > agent_max_prompt_length:

                            # 暂时简单处理，直接从左侧截断
                            raw_prompt_ids = raw_prompt_ids[:agent_max_prompt_length]
                            

                            # agent_truncation = self.config.get("truncation", "error")
                            # if self.config.data.truncation == "left":
                            #     raw_prompt_ids = raw_prompt_ids[-agent_max_prompt_length:]
                            # elif self.config.data.truncation == "right":
                            #     raw_prompt_ids = raw_prompt_ids[:agent_max_prompt_length]
                            # elif self.config.data.truncation == "middle":
                            #     left_half = agent_max_prompt_length // 2
                            #     right_half = agent_max_prompt_length - left_half
                            #     raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                            # elif agent_truncation == "error":
                            #     raise RuntimeError(f"Turn {turn_idx} Prompt length {len(raw_prompt_ids)} is longer than {agent_max_prompt_length}.")
                        
                        processed_raw_prompt_ids.append(raw_prompt_ids)

                    # 将处理后的结果堆叠成批次
                    batch_input_ids = torch.stack(processed_input_ids)
                    batch_attention_masks = torch.stack(processed_attention_masks)
                    batch_position_ids = torch.stack(processed_position_ids)

                    # 更新next_batch
                    next_batch.batch['input_ids'] = batch_input_ids
                    next_batch.batch['attention_mask'] = batch_attention_masks
                    next_batch.batch['position_ids'] = batch_position_ids
                    next_batch.non_tensor_batch['raw_prompt_ids'] = processed_raw_prompt_ids


                    # # 更新raw_prompt_ids
                    # agent_max_prompt_length = self.config.data.max_prompt_length
                    # new_raw_prompt_ids = agent.tokenizer.encode(new_raw_prompt, add_special_tokens=False)
                    # if len(new_raw_prompt_ids) > agent_max_prompt_length:
                    #     if self.config.data.truncation == "left":
                    #         new_raw_prompt_ids = new_raw_prompt_ids[-self.config.data.max_prompt_length :]
                    #     elif self.config.data.truncation == "right":
                    #         new_raw_prompt_ids = new_raw_prompt_ids[: self.config.data.max_prompt_length]
                    #     elif self.config.data.truncation == "middle":
                    #         left_half = self.config.data.max_prompt_length // 2
                    #         right_half = agent_max_prompt_length - left_half
                    #         new_raw_prompt_ids = new_raw_prompt_ids[:left_half] + new_raw_prompt_ids[-right_half:]
                    #     elif agent_truncation == "error":
                    #         raise RuntimeError(f"Turn {turn_idx} Prompt length {len(new_raw_prompt_ids)} is longer than {agent_max_prompt_length}.")
                    
                    # # 更新raw_prompt_ids
                    # next_batch.non_tensor_batch['raw_prompt_ids'] = new_raw_prompt_ids

                    #### tools 和 index 不用更新
                
                # 4. 更新gen_batch_all用于下一轮生成
                # 重新pop出需要的字段
                new_gen_batch_all = {}                

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

                for agent_key, next_batch in next_turn_batchs.items():
                    next_gen_batch = next_batch.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )
                    new_gen_batch_all[agent_key] = next_gen_batch

                # 更新gen_batch_all
                gen_batch_all = new_gen_batch_all   
                
                # 更新multi_turn_batchs
                multi_turn_batchs[f"turn_{turn_idx + 1}"] = next_turn_batchs

        # {
        # turn_0: {agent_0: batch DataProto, agent_1: batch DataProto},  prompt=prompt0
        # turn_1: {agent_0: batch DataProto, agent_1: batch DataProto},  prompt=prompt0+ai_0+aj_0
        # }
        return multi_turn_batchs
                
                


    """处理每个turn的reward"""
    def cal_reward(self, agent_batchs, turn_idx, metrics):
        # compute reward model score
        reward_tensor_all = {}
        reward_extra_infos_dict_all = {}
        future_reward_all = {}
        # 这三个会用于后续的adv的计算

        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            if agent.use_rm:
                reward_tensor = agent.rm_wg.compute_rm_score(agent_batch)
                agent_batch = agent_batch.union(reward_tensor)

            if agent.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(agent_batch, agent.config, agent.tokenizer)
                future_reward_all[f"agent_{agent_id}"] = future_reward
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(agent_batch, self.reward_fn)
                reward_tensor_all[f"agent_{agent_id}"] = reward_tensor
                reward_extra_infos_dict_all[f"agent_{agent_id}"] = reward_extra_infos_dict

            # 记录原始的reward
            agent_reward_metrics = {
                f"agent_{agent_id}_origin_reward/turn_{turn_idx}_mean": torch.mean(reward_tensor.sum(-1)).detach().item(),
                f"agent_{agent_id}_origin_reward/turn_{turn_idx}_max": torch.max(reward_tensor.sum(-1)).detach().item(),
                f"agent_{agent_id}_origin_reward/turn_{turn_idx}_min": torch.min(reward_tensor.sum(-1)).detach().item(),
            }
            metrics.update(agent_reward_metrics)

        
        # 用于测试ippo的reward sum结果
        if self.is_sum:
            sum_reward = None
            for agent_key, agent_reward_tensor in reward_tensor_all.items():
                if sum_reward is None:
                    sum_reward = agent_reward_tensor
                else:
                    sum_reward += agent_reward_tensor
            # 将总和奖励赋值给每个agent
            for agent_key in reward_tensor_all.keys():
                reward_tensor_all[agent_key] = sum_reward

            team_reward_metrics = {f"total/turn_{turn_idx}_reward_mean": torch.mean(sum_reward.sum(-1)).detach().item(),
                                   f"total/turn_{turn_idx}_reward_max": torch.max(sum_reward.sum(-1)).detach().item(),
                                   f"total/turn_{turn_idx}_reward_min": torch.min(sum_reward.sum(-1)).detach().item(),
                                   }
            metrics.update(team_reward_metrics)


        return reward_tensor_all, reward_extra_infos_dict_all, future_reward_all



    """
    计算reward
    传入的是整个multi turn batchs，需要根据agent_id进行拆分
    输入数据格式：
    {
        turn_0: {agent_0: batch DataProto, agent_1: batch DataProto},  prompt=prompt0
        turn_1: {agent_0: batch DataProto, agent_1: batch DataProto},  prompt=prompt0+ai_0+aj_0
    }
    返回数据格式：
    {
        turn_0: {agent_0: reward_tensor, agent_1: reward_tensor},
        turn_1: {agent_0: reward_tensor, agent_1: reward_tensor},
    }
    """
    def cal_reward_multi_turn(self, multi_turn_batchs, metrics):

        reward_tensor_all_turns = {}
        reward_extra_infos_dict_all_turns = {}
        future_reward_all_turns = {}

        for turn_idx, turn_batchs in multi_turn_batchs.items():
            reward_tensor_all, reward_extra_infos_dict_all, future_reward_all = self.cal_reward(turn_batchs, turn_idx, metrics)

            reward_tensor_all_turns[f"turn_{turn_idx}"] = reward_tensor_all
            reward_extra_infos_dict_all_turns[f"turn_{turn_idx}"] = reward_extra_infos_dict_all
            future_reward_all_turns[f"turn_{turn_idx}"] = future_reward_all
        
        return reward_tensor_all_turns, reward_extra_infos_dict_all_turns, future_reward_all_turns

    """用于learn sentence level Q，只计算最后一轮的奖励"""
    def cal_reward_multi_turn_final(self, multi_turn_batchs, metrics):

        reward_tensor_all_turns = {}
        reward_extra_infos_dict_all_turns = {}
        future_reward_all_turns = {}

        for turn_idx, turn_batchs in multi_turn_batchs.items():
            if turn_idx == len(multi_turn_batchs) - 1:
                reward_tensor_all, reward_extra_infos_dict_all, future_reward_all = self.cal_reward(turn_batchs, turn_idx, metrics)
            else:
                reward_tensor_all = []
                reward_extra_infos_dict_all = []
                future_reward_all = []
                # 其他轮奖励都是0

            reward_tensor_all_turns[f"turn_{turn_idx}"] = reward_tensor_all
            reward_extra_infos_dict_all_turns[f"turn_{turn_idx}"] = reward_extra_infos_dict_all
            future_reward_all_turns[f"turn_{turn_idx}"] = future_reward_all
        
        return reward_tensor_all_turns, reward_extra_infos_dict_all_turns, future_reward_all_turns


    def cal_reward_multi_turn_mean(self, multi_turn_batchs, metrics):
        """
        计算多轮对话的奖励，将最后一轮的奖励平均分配给所有轮次
        
        Args:
            multi_turn_batchs: 多轮对话的数据，格式为 {turn_idx: {agent_id: DataProto}}
            metrics: 用于记录指标的字典
            
        Returns:
            reward_tensor_all_turns: 每轮的奖励张量，格式为 {turn_idx: {agent_id: reward_tensor}}
            reward_extra_infos_dict_all_turns: 每轮的额外奖励信息
            future_reward_all_turns: 每轮的未来奖励
        """
        reward_tensor_all_turns = {}
        reward_extra_infos_dict_all_turns = {}
        future_reward_all_turns = {}
        
        # 获取轮次数量
        num_turns = len(multi_turn_batchs)
        last_turn_idx = f"turn_{num_turns - 1}"
        
        # 只计算最后一轮的奖励
        last_turn_batchs = multi_turn_batchs[last_turn_idx]
        last_reward_tensor_all, last_reward_extra_infos_dict_all, last_future_reward_all = self.cal_reward(
            last_turn_batchs, last_turn_idx, metrics
        )
        
        # 记录最后一轮的奖励
        reward_tensor_all_turns[last_turn_idx] = last_reward_tensor_all
        reward_extra_infos_dict_all_turns[last_turn_idx] = last_reward_extra_infos_dict_all
        future_reward_all_turns[last_turn_idx] = last_future_reward_all
        
        # 计算平均奖励（将最后一轮的奖励除以轮次数）
        mean_rewards = {}
        for agent_key, reward_tensor in last_reward_tensor_all.items():
            # 计算平均奖励
            mean_reward = reward_tensor / num_turns
            mean_rewards[agent_key] = mean_reward
            
            # # 记录平均奖励的指标
            # agent_mean_reward_metrics = {
            #     f"{agent_key}_mean_reward/mean": torch.mean(mean_reward.sum(-1)).detach().item(),
            #     f"{agent_key}_mean_reward/max": torch.max(mean_reward.sum(-1)).detach().item(),
            #     f"{agent_key}_mean_reward/min": torch.min(mean_reward.sum(-1)).detach().item(),
            # }
            # metrics.update(agent_mean_reward_metrics)
        
        # 将平均奖励分配给前面的所有轮次
        for turn_idx in range(num_turns - 1):
            turn_key = f"turn_{turn_idx}"
            reward_tensor_all_turns[turn_key] = mean_rewards
            reward_extra_infos_dict_all_turns[turn_key] = last_reward_extra_infos_dict_all  # 空字典，因为前面轮次没有计算额外信息
            future_reward_all_turns[turn_key] = last_future_reward_all  # 空字典，因为前面轮次没有计算未来奖励
        
        
        return reward_tensor_all_turns, reward_extra_infos_dict_all_turns, future_reward_all_turns

    
    def cal_reward_multi_turn_discount(self, multi_turn_batchs, metrics):
        pass
    
    def cal_reward_multi_turn_counterfactual(self, multi_turn_batchs, metrics):
        pass


    # 暂时不用实现支持llm rollout,后续可以用于支持每轮concate变成summary
    def init_workers(self):
        pass