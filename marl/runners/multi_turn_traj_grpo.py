
import uuid
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
import numpy as np
from verl import DataProto
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.trainer.ppo.reward import compute_reward, compute_reward_async  
import torch
from marl.utils.marl_utils import compute_response_mask
import copy

from verl.utils.reward_score.math import compute_score, last_boxed_only_string, remove_boxed, is_equiv


_FROZEN_VERL_FOLLOWUP_SINGLE_SAMPLE_KEY = "mag" + "rpo"


class Multi_Turn_Runner_Traj_GRPO:
    def __init__(self,
                 config,
                 mac,
                 num_agents,  # cotrian LLMs
                 reward_fn_list=None,  # Per-agent reward functions/tokenizers.
                 device_name="cuda",
                 **kwargs):
        
        self.config = config
        self.reward_fn_list = reward_fn_list
        self.agg_mode = config.marl.agg_mode  # sum/max/dictator
        self.mac = mac


    # Sequence balancing is kept as a utility but disabled in the rollout path:
    # heterogeneous agents can produce different response lengths, and balancing
    # independently would break the cross-agent sample alignment.
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        # world_size = self.actor_rollout_wg.world_size
        world_size = self.mac.agents[0].actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)

        metrics.update(global_balance_stats)


    def remove_template(self, text):
        """
        Remove chat-template markers from decoded prompt text.

        The input should already have special tokens removed. This strips the
        system/user/assistant wrapper and the dataset instruction suffix so the
        next debate turn can reuse the original problem statement only.

        text: 
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant. \n user \n
        Tom receives a $12 allowance per month. In the first week, he spends a third of it; 
        in the second week, he spends a quarter of what he has left. 
        How much money does he have left to finish the month?
        Let\'s think step by step and output the final answer after "####".\nassistant\n

        return:
        Tom receives a $12 allowance per month. In the first week, he spends a third of it; 
        in the second week, he spends a quarter of what he has left. 
        How much money does he have left to finish the month?
        """

        try:
            # Qwen-style chat templates may include a system preamble.
            if "user\n" in text:
                text = text.split("user\n", 1)[1]
            if "\nassistant\n" in text:
                text = text.split("\nassistant\n", 1)[0]

            text = text.strip()

            instruction_suffixes = [
                "Let\'s think step by step and output the final answer after \"####\".",
                "Let\'s think step by step and output the final answer within \\boxed{}.",
            ]

            for suffix in instruction_suffixes:
                if text.endswith(suffix):
                    text = text[:-len(suffix)].rstrip()
                    break

            return text
            
        except ValueError:
            return text.strip()



    def _replace_boxed_with_answer(self, string):
        """
        Replace the final boxed answer span with a neutral placeholder.

        This follows the boxed-answer parsing style used by verl's math reward
        helpers while preserving the surrounding reasoning text.
        """
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return string
        
        # Handle "\boxed 123" without braces.
        if "\\boxed " in string and idx == string.rfind("\\boxed "):
            boxed_content = "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
            return string.replace(boxed_content, "<answer>")
        
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        
        # Find the matching right brace for "\boxed{...}".
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        
        if right_brace_idx is not None:
            boxed_content = string[idx : right_brace_idx + 1]
            return string.replace(boxed_content, "<answer>")
        
        return string

    
    def clean_response_for_next_turn(self, response_text, dataset_type="math"):
        """
        Remove the final-answer surface form before feeding responses forward.

        Keeping only reasoning context reduces degenerate follow-up behavior
        where later agents simply agree with a visible final answer.
        """
        import re
        
        if dataset_type == "math":
            cleaned = self._replace_boxed_with_answer(response_text)
        elif dataset_type == "gsm8k":
            if "####" in response_text:
                cleaned = response_text.split("####")[0].strip()
                if not cleaned.endswith(('.', '!', '?')):
                    cleaned += " <to be calculated>."
            else:
                cleaned = response_text
        else:
            cleaned = response_text
        
        return cleaned



    def merge_prompt_response(self, original_prompts, all_agent_responses_text, template=None, ego_id=None, turn_idx=None):
        """
        Build the next-turn prompt from the original question and agent outputs.

        Final answer spans are masked before merging so the next turn focuses on
        checking and integrating reasoning instead of copying a visible answer.

        Args:
            original_prompts (list[str]): Original problem statements.
            all_agent_responses_text (dict): Mapping like
                {"agent_0": [responses], "agent_1": [responses], ...}.
            template (str, optional): Custom prompt template for response merge.
        
        Returns:
            list[str]: Merged next-turn prompts.
        """
        instruction_following = 'Let\'s think step by step and output the final answer within \\boxed{}.'   

        if template is None:
            template = "Given the following problem: {original_prompt} \n We have two answers: \n {responses} \n Please carefully review these answers and recognize which one is right. If one or all of them are right, please summarize the reasoning process of right ones and give the final answer. If both of them are wrong, please correct their mistakes and provide a novel and complete solution to the problem and give the final answer. {instruction_following}"

        merged_texts = []
        for i, original_prompt in enumerate(original_prompts):
            clean_responses = {}
            for agent_key in all_agent_responses_text.keys():
                if i < len(all_agent_responses_text[agent_key]):
                    response = all_agent_responses_text[agent_key][i]
                    cleaned_response = self.clean_response_for_next_turn(
                        response, 
                        dataset_type=self.config.data.get("dataset_type", "math")
                    )
                    clean_responses[agent_key] = cleaned_response
            
            if ego_id:
                ego_responses = clean_responses[f"agent_{ego_id}"]
                other_responses = [clean_responses[k] for k in clean_responses.keys() if k != f"agent_{ego_id}"]
                other_responses_text = "\n".join(other_responses)
                responses_text = f"agent {ego_id} response is: {ego_responses}\n Other agents' responses are: \n{other_responses_text}"
            else:
                responses_text = "\n".join(clean_responses.values())
            
            
            try:
                merged_text = template.format(
                    # turn_idx=turn_idx,
                    original_prompt=original_prompt,
                    responses=responses_text,
                    instruction_following=instruction_following
                )
                merged_texts.append(merged_text)
            except Exception as e:
                fallback_text = f"{original_prompt}\n{responses_text}{instruction_following}"
                merged_texts.append(fallback_text)
        
        return merged_texts
                
    
    
    # Traj-GRPO rolls out n trajectories only on the first turn; later turns
    # preserve the first-turn uid grouping and extend each trajectory once.
    def rollout_traj_grpo(self, gen_batch_all, multi_turn_batchs, metrics, turn=2, validate=False):
        """
        gen_batch_all: {batch_key: DataProto}, prompt-only generation batches.
        multi_turn_batchs: {turn_idx: {agent_id: DataProto}}, with turn_0 initialized.
        metrics: Dict[str, float]
        turn: Number of interaction turns.
        """
        
        for turn_idx in range(turn):

            agent_batchs = multi_turn_batchs[f"turn_{turn_idx}"]


            # Generate one response batch per agent for the current turn.
            gen_batch_outputs_all = {}
            for agent, gen_batch_agent in zip(self.mac.agents, gen_batch_all.values()):
                # Frozen verl compatibility: follow-up turns extend each
                # trajectory once instead of branching by rollout.n again.
                gen_batch_agent.meta_info.update({
                    _FROZEN_VERL_FOLLOWUP_SINGLE_SAMPLE_KEY: turn_idx > 0
                })

                gen_batch_output = agent.actor_rollout_wg.generate_sequences(gen_batch_agent)
                gen_batch_outputs_all[f"agent_{agent.agent_id}"] = gen_batch_output


            for agent_key, agent_batch  in agent_batchs.items():
                if turn_idx == 0:
                    agent_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(agent_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    if not validate:
                        agent_batch = agent_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)                                        
                elif turn_idx > 0:
                    # Keep follow-up turns in the same GRPO group as turn 0.
                    agent_batch.non_tensor_batch['uid'] = multi_turn_batchs["turn_0"][f"{agent_key}"].non_tensor_batch['uid']
                
                
                agent_batch = agent_batch.union(gen_batch_outputs_all[agent_key])
                agent_batch.batch['response_mask'] = compute_response_mask(agent_batch)


                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo

                # Disabled for multi-agent trajectories because reordering one
                # agent's batch can desynchronize response pairs across agents.
                # if self.config.trainer.balance_batch:
                #     self._balance_batch(agent_batch, metrics=metrics)
                
                # compute global_valid tokens
                if not isinstance(agent_batch.meta_info, dict):
                    agent_batch.meta_info = {}
                agent_batch.meta_info['global_token_num'] = torch.sum(agent_batch.batch['attention_mask'], dim=-1).tolist()
                agent_batchs[agent_key] = agent_batch

            # Updated agent batches stay in-place inside multi_turn_batchs.
            # multi_turn_batchs[f"turn_{turn_idx}"] = agent_batchs

            # Build prompts for the next turn from all current agent responses.
            if turn_idx < turn - 1:
                all_agent_responses_text = {}
                for agent_idx, agent in enumerate(self.mac.agents):
                    agent_key = f"agent_{agent_idx}"
                    agent_batch = agent_batchs[agent_key]
                    
                    response_ids = agent_batch.batch['responses']
                    response_mask = agent_batch.batch['response_mask']
                    
                    valid_ids_list = [response_ids[i][response_mask[i].bool()].tolist() for i in range(response_ids.shape[0])]
                    texts = agent.tokenizer.batch_decode(valid_ids_list, skip_special_tokens=True)
                    responses_text = [f"agent {agent_idx} response is: {t}" for t in texts]
                    
                    all_agent_responses_text[agent_key] = responses_text
                
                next_turn_batchs = multi_turn_batchs[f"turn_{turn_idx + 1}"]
                # next_turn_batchs intentionally starts from the dataloader
                # structure, then receives the updated prompt tensors below.


                # Recover original problem statements from the repeated prompt metadata.
                texts = [agent_batchs['agent_0'].non_tensor_batch['prompt'][i][0]['content'] for i in range(len(agent_batchs['agent_0'].non_tensor_batch['prompt']))]
                original_prompts = [self.remove_template(t) for t in texts]
                new_prompts = self.merge_prompt_response(original_prompts, all_agent_responses_text, template=None, turn_idx=turn_idx)
                new_prompt_messages = [[{"role": "user", "content": p}] for p in new_prompts]


                # Tokenize the shared next-turn messages with each agent's own tokenizer.
                for agent_idx, agent in enumerate(self.mac.agents):
                    agent_key = f"agent_{agent_idx}"


                    processed_input_ids = []
                    processed_attention_masks = []
                    processed_position_ids = []
                    processed_raw_prompt_ids = []

                    if agent.unthinking_mode:
                        new_prompt_strs = [agent.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False) for msg in new_prompt_messages]
                    else:
                        new_prompt_strs = [agent.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in new_prompt_messages]


                    model_inputs = agent.tokenizer(new_prompt_strs,
                                                    return_tensors="pt",
                                                    add_special_tokens=False,
                                                    truncation=True,
                                                    max_length=self.config.data.max_prompt_length,
                                                    padding=True
                                                )
                    turn_input_ids = model_inputs["input_ids"]
                    turn_attention_mask = model_inputs["attention_mask"]


                    processed_ids = []
                    processed_mask = []
                    for i in range(turn_input_ids.shape[0]):
                        processed_ids, processed_mask = verl_F.postprocess_data(
                            input_ids=turn_input_ids[i].unsqueeze(0),
                            attention_mask=turn_attention_mask[i].unsqueeze(0),
                            max_length=self.config.data.max_prompt_length,
                            pad_token_id=agent.tokenizer.pad_token_id,
                            left_pad=True,
                            truncation=self.config.get("truncation", "error"),
                        )

                        processed_pos_ids = compute_position_id_with_mask(processed_mask)
                        processed_input_ids.append(processed_ids[0])
                        processed_attention_masks.append(processed_mask[0])
                        processed_position_ids.append(processed_pos_ids[0])

                    agent_max_prompt_length = self.config.data.max_prompt_length
                    raw_prompt_ids = agent.tokenizer(new_prompt_strs, add_special_tokens=False)
                    processed_raw_prompt_ids = [ids[:agent_max_prompt_length] for ids in raw_prompt_ids['input_ids']]


                    batch_input_ids = torch.stack(processed_input_ids)
                    batch_attention_masks = torch.stack(processed_attention_masks)
                    batch_position_ids = torch.stack(processed_position_ids)

                    new_batch_data = {
                        'input_ids': batch_input_ids,
                        'attention_mask': batch_attention_masks,
                        'position_ids': batch_position_ids,
                    }

                    # new_next_meta_info = torch.sum(new_batch_data['attention_mask'], dim=-1).tolist()

                    new_next_batch = DataProto.from_dict(
                        tensors=new_batch_data,
                        non_tensors={'raw_prompt_ids': processed_raw_prompt_ids},
                    )
                    # Preserve dataset metadata such as prompt, index, and reward fields.
                    last_turn_non_tensor_batch = copy.deepcopy(agent_batchs[agent_key].non_tensor_batch)
                    new_next_batch.non_tensor_batch.update(last_turn_non_tensor_batch)
                    next_turn_batchs[agent_key] = new_next_batch

                
                # Prepare prompt-only generation batches for the next turn.
                new_gen_batch_all = {}                

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

                for agent_key, next_batch in next_turn_batchs.items():
                    next_gen_batch = next_batch.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )

                    if validate:
                        agent_id = int(agent_key.split('_')[-1])
                        next_gen_batch.meta_info = {
                            'eos_token_id': self.mac.agents[agent_id].tokenizer.eos_token_id,
                            'pad_token_id': self.mac.agents[agent_id].tokenizer.pad_token_id,
                            'recompute_log_prob': False,
                            'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                            'validate': True,
                        }

                    new_gen_batch_all[agent_key] = next_gen_batch

                gen_batch_all = new_gen_batch_all   


        # {
        # turn_0: {agent_0: batch DataProto, agent_1: batch DataProto},  prompt=prompt0
        # turn_1: {agent_0: batch DataProto, agent_1: batch DataProto},  prompt=prompt0+ai_0+aj_0
        # }
        return multi_turn_batchs



    """Compute per-turn rewards and aggregate them across agents."""
    def cal_reward(self, agent_batchs, turn_idx, metrics):
        # compute reward model score
        reward_tensor_all = {}
        reward_extra_infos_dict_all = {}
        future_reward_all = {}

        for agent_id, agent in enumerate(self.mac.agents):
            agent_batch = agent_batchs[f"agent_{agent_id}"]
            if agent.use_rm:
                reward_tensor = agent.rm_wg.compute_rm_score(agent_batch)
                agent_batch = agent_batch.union(reward_tensor)

            if agent.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(agent_batch, agent.config, agent.tokenizer)
                future_reward_all[f"agent_{agent_id}"] = future_reward
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(agent_batch, self.reward_fn_list[agent_id])
                reward_tensor_all[f"agent_{agent_id}"] = reward_tensor
                reward_extra_infos_dict_all[f"agent_{agent_id}"] = reward_extra_infos_dict

            # Keep the original per-agent reward before team aggregation.
            agent_reward_metrics = {
                f"rewards_origin/agent_{agent_id}_turn_{turn_idx}_mean": torch.mean(reward_tensor.sum(-1)).detach().item(),
                # f"origin_reward/agent_{agent_id}_turn_{turn_idx}_max": torch.max(reward_tensor.sum(-1)).detach().item(),
                # f"origin_reward/agent_{agent_id}_turn_{turn_idx}_min": torch.min(reward_tensor.sum(-1)).detach().item(),
            }
            metrics.update(agent_reward_metrics)

        # Write the team reward to the final valid response token for each agent.
        if self.agg_mode == "sum":
            sum_reward = None
            for agent_key, agent_reward_tensor in reward_tensor_all.items():
                if sum_reward is None:
                    sum_reward = agent_reward_tensor.clone()
                else:
                    sum_reward += agent_reward_tensor.clone()
            team_reward_scalar = sum_reward.sum(-1)
            batch_indices = torch.arange(sum_reward.size(0)) 
            for agent_key in reward_tensor_all.keys():
                agent_reward_position = agent_batchs[agent_key].batch['response_mask'].sum(-1) - 1
                reward_tensor_all[agent_key][batch_indices, agent_reward_position] = team_reward_scalar.clone()
            
            team_reward_metrics = {f"total/turn_{turn_idx}_reward_mean": torch.mean(sum_reward.sum(-1)).detach().item(),
                                   f"total/turn_{turn_idx}_reward_max": torch.max(sum_reward.sum(-1)).detach().item(),
                                   f"total/turn_{turn_idx}_reward_min": torch.min(sum_reward.sum(-1)).detach().item(),
                                   }
            metrics.update(team_reward_metrics)
        elif self.agg_mode == "max":
            max_reward = None
            for agent_key, agent_reward_tensor in reward_tensor_all.items():
                if max_reward is None:
                    max_reward = agent_reward_tensor.clone()
                else:
                    max_reward = torch.max(max_reward, agent_reward_tensor.clone())
            team_reward_scalar = max_reward.sum(-1)
            batch_indices = torch.arange(max_reward.size(0))

            for agent_key in reward_tensor_all.keys():
                agent_reward_position = agent_batchs[agent_key].batch['response_mask'].sum(-1) - 1
                reward_tensor_all[agent_key][batch_indices, agent_reward_position] = team_reward_scalar.clone()

            team_reward_metrics = {f"total/turn_{turn_idx}_reward_mean": torch.mean(max_reward.sum(-1)).detach().item(),
                                   f"total/turn_{turn_idx}_reward_max": torch.max(max_reward.sum(-1)).detach().item(),
                                   f"total/turn_{turn_idx}_reward_min": torch.min(max_reward.sum(-1)).detach().item(),
                                   }
            metrics.update(team_reward_metrics)

        elif self.agg_mode == "dictator":
            # Use agent_0 as the team-level reward source.
            dictator_reward = reward_tensor_all['agent_0'].clone()
            team_reward_scalar = dictator_reward.sum(-1)
            batch_indices = torch.arange(dictator_reward.size(0))

            for agent_key in reward_tensor_all.keys():
                agent_reward_position = agent_batchs[agent_key].batch['response_mask'].sum(-1) - 1
                reward_tensor_all[agent_key][batch_indices, agent_reward_position] = team_reward_scalar.clone()

            team_reward_metrics = {f"total/turn_{turn_idx}_reward_mean": torch.mean(team_reward_scalar.sum(-1)).detach().item(),
                                   f"total/turn_{turn_idx}_reward_max": torch.max(team_reward_scalar.sum(-1)).detach().item(),
                                   f"total/turn_{turn_idx}_reward_min": torch.min(team_reward_scalar.sum(-1)).detach().item(),
                                   }
            metrics.update(team_reward_metrics)


        return reward_tensor_all, reward_extra_infos_dict_all, future_reward_all




    """Dense traj-GRPO reward: evaluate each turn independently."""
    def cal_reward_dense(self, multi_turn_batchs, metrics):
        reward_tensor_all_turns = {}
        reward_extra_infos_dict_all_turns = {}
        future_reward_all_turns = {}
        
        num_turns = len(multi_turn_batchs)
        # last_turn_idx = f"turn_{num_turns - 1}"

        for turn_idx in range(num_turns):
            turn_key = f"turn_{turn_idx}"
            turn_batchs = multi_turn_batchs[turn_key]
            # Reward computation is independent of GRPO group uid handling.
            turn_reward_tensor_all, turn_reward_extra_infos_dict_all, turn_future_reward_all = self.cal_reward(turn_batchs, turn_idx, metrics)
        
            reward_tensor_all_turns[turn_key] = turn_reward_tensor_all
            reward_extra_infos_dict_all_turns[turn_key] = turn_reward_extra_infos_dict_all
            future_reward_all_turns[turn_key] = turn_future_reward_all
        

        return reward_tensor_all_turns, reward_extra_infos_dict_all_turns, future_reward_all_turns



    """Sparse traj-GRPO reward: only the final turn receives reward."""
    def cal_reward_sparse(self, multi_turn_batchs, metrics):
        """
        Compute rewards for a multi-turn trajectory with zeroed earlier turns.
        
        Args:
            multi_turn_batchs: {turn_idx: {agent_id: DataProto}}.
            metrics: Mutable metrics dictionary.
            
        Returns:
            reward tensors, reward extra info, and async reward futures by turn.
        """
        reward_tensor_all_turns = {}
        reward_extra_infos_dict_all_turns = {}
        future_reward_all_turns = {}
        
        num_turns = len(multi_turn_batchs)
        last_turn_idx = f"turn_{num_turns - 1}"
        
        last_turn_batchs = multi_turn_batchs[last_turn_idx]
        last_reward_tensor_all, last_reward_extra_infos_dict_all, last_future_reward_all = self.cal_reward(
            last_turn_batchs, last_turn_idx, metrics
        )
        
        reward_tensor_all_turns[last_turn_idx] = last_reward_tensor_all
        reward_extra_infos_dict_all_turns[last_turn_idx] = last_reward_extra_infos_dict_all
        future_reward_all_turns[last_turn_idx] = last_future_reward_all
        
        # Earlier turns keep the same reward width but receive zero reward.
        for turn_idx in range(num_turns - 1):
            turn_key = f"turn_{turn_idx}"
            zero_rewards = {}
            turn_bs = multi_turn_batchs[turn_key]['agent_0'].batch.batch_size[0]
            turn_len = last_reward_tensor_all['agent_0'].shape[1]
            for agent_key, reward_tensor in last_reward_tensor_all.items():
                zero_rewards[agent_key] = torch.zeros(turn_bs, turn_len, dtype=reward_tensor.dtype, device=reward_tensor.device)
            reward_tensor_all_turns[turn_key] = zero_rewards
            reward_extra_infos_dict_all_turns[turn_key] = last_reward_extra_infos_dict_all
            future_reward_all_turns[turn_key] = last_future_reward_all
        
        
        return reward_tensor_all_turns, reward_extra_infos_dict_all_turns, future_reward_all_turns



    """Accumulative traj-GRPO reward: propagate later rewards backward."""
    def cal_reward_accumulative(self, multi_turn_batchs, metrics):
        """
        Compute per-turn rewards and add future rewards to earlier turns.
        
        Args:
            multi_turn_batchs: {turn_idx: {agent_id: DataProto}}.
            metrics: Mutable metrics dictionary.
            
        Returns:
            reward tensors, reward extra info, and async reward futures by turn.
        """
        reward_tensor_all_turns = {}
        reward_extra_infos_dict_all_turns = {}
        future_reward_all_turns = {}
        
        num_turns = len(multi_turn_batchs)
        # last_turn_idx = f"turn_{num_turns - 1}"

        accumulative_reward_tensor_all = {}
        for turn_idx in reversed(range(num_turns)):
            turn_key = f"turn_{turn_idx}"
            turn_batchs = multi_turn_batchs[turn_key]
            turn_reward_tensor_all, turn_reward_extra_infos_dict_all, turn_future_reward_all = self.cal_reward(turn_batchs, turn_idx, metrics)

            if turn_idx == num_turns-1:
                for agent_key, reward_tensor in turn_reward_tensor_all.items():
                    accumulative_reward_tensor_all[agent_key] = reward_tensor.clone()
            else: 
                # Align accumulated reward to each turn's final response token.
                for agent_key, reward_tensor in turn_reward_tensor_all.items():
                    acc_reward_scalar = accumulative_reward_tensor_all[agent_key].sum(-1)  # bs, 1
                    non_zero_mask = reward_tensor != 0  # (batch_size, seq_len)
                    agent_reward_position = turn_batchs[agent_key].batch['response_mask'].sum(-1) - 1
                    assert agent_reward_position.shape == acc_reward_scalar.shape, f"agent_reward_position.shape: {agent_reward_position.shape}, acc_reward_scalar.shape: {acc_reward_scalar.shape}"

                    new_reward_tensor = reward_tensor.clone()
                    batch_size = reward_tensor.size(0)
                    batch_indices = torch.arange(batch_size, device=reward_tensor.device)

                    new_reward_tensor[batch_indices, agent_reward_position] += acc_reward_scalar.clone()
                    accumulative_reward_tensor_all[agent_key] = new_reward_tensor

            reward_tensor_all_turns[turn_key] = {
                agent_key: reward_tensor.clone() 
                for agent_key, reward_tensor in accumulative_reward_tensor_all.items()
            }
            reward_extra_infos_dict_all_turns[turn_key] = turn_reward_extra_infos_dict_all
            future_reward_all_turns[turn_key] = turn_future_reward_all
            

        return reward_tensor_all_turns, reward_extra_infos_dict_all_turns, future_reward_all_turns



    # Reserved for optional LLM-based turn summarization rollouts.
    def init_workers(self):
        pass



    # Optional rollout variant with early stopping after all agents solve a sample.
    def rollout_multi_turn_grpo_earlystop(self, gen_batch_all, multi_turn_batchs, metrics, turn=2, validate=False):
        """
        gen_batch_all: {batch_key: DataProto}, prompt-only generation batches.
        multi_turn_batchs: {turn_idx: {agent_id: DataProto}}, with turn_0 initialized.
        metrics: Dict[str, float]
        turn: Number of interaction turns.
        """
        
        # Termination state is tracked across turns by sample index.
        terminated_samples = set()

        for turn_idx in range(turn):

            agent_batchs = multi_turn_batchs[f"turn_{turn_idx}"]     #[bs]

            # Generate one response batch per agent for the current turn.
            gen_batch_outputs_all = {}
            for agent, gen_batch_agent in zip(self.mac.agents, gen_batch_all.values()):    
                gen_batch_output = agent.actor_rollout_wg.generate_sequences(gen_batch_agent)
                gen_batch_outputs_all[f"agent_{agent.agent_id}"] = gen_batch_output


            # last_batch_response = self.mac.agents[0].tokenizer.batch_decode(multi_turn_batchs['turn_0']['agent_0'].batch['responses'], skip_special_tokens=True)
            # new_batch_response = self.mac.agents[1].tokenizer.batch_decode(gen_batch_output.batch['responses'], skip_special_tokens=True)
            """Attach generated responses and trajectory metadata to each agent batch."""
            for agent_key, agent_batch  in agent_batchs.items():
                agent_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(agent_batch.batch))], dtype=object)
                # repeat to align with repeated responses in rollout
                if not validate:
                    agent_batch = agent_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)


                agent_batch = agent_batch.union(gen_batch_outputs_all[agent_key])

                agent_batch.batch['response_mask'] = compute_response_mask(agent_batch)

                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                # if self.config.trainer.balance_batch:
                #     self._balance_batch(agent_batch, metrics=metrics)
                
                # compute global_valid tokens
                if not isinstance(agent_batch.meta_info, dict):
                    agent_batch.meta_info = {}
                agent_batch.meta_info['global_token_num'] = torch.sum(agent_batch.batch['attention_mask'], dim=-1).tolist()
                agent_batchs[agent_key] = agent_batch


            

            # Build prompts for the next turn from all current agent responses.
            if turn_idx < turn - 1:
                all_agent_responses_text = {}
                for agent_idx, agent in enumerate(self.mac.agents):
                    agent_key = f"agent_{agent_idx}"
                    agent_batch = agent_batchs[agent_key]
                    
                    response_ids = agent_batch.batch['responses']
                    response_mask = agent_batch.batch['response_mask']

                    valid_ids_list = [response_ids[i][response_mask[i].bool()].tolist() for i in range(response_ids.shape[0])]
                    texts = agent.tokenizer.batch_decode(valid_ids_list, skip_special_tokens=True)
                    responses_text = [f"agent {agent_idx} response is: {t}" for t in texts]
                    all_agent_responses_text[agent_key] = responses_text

                
                """Debug hook for leakage checks; keep disabled in release runs."""
                # all_agent_responses_text["agent_1"] = all_agent_responses_text["agent_0"].copy()
                
                next_turn_batchs = multi_turn_batchs[f"turn_{turn_idx + 1}"]
                

                # Recover original problem statements from the repeated prompt metadata.
                texts = [agent_batchs['agent_0'].non_tensor_batch['prompt'][i][0]['content'] for i in range(len(agent_batchs['agent_0'].non_tensor_batch['prompt']))]
                original_prompts = [self.remove_template(t) for t in texts]


                new_prompts = self.merge_prompt_response(original_prompts, all_agent_responses_text, template=None, turn_idx=turn_idx)


                """Apply early-stop decisions before creating next-turn batches."""
                if turn_idx>0:
                    reward_tensor_all_tmp, _, _ = self.cal_reward(agent_batchs, turn_idx, metrics)
                    first_agent_key = list(reward_tensor_all_tmp.keys())[0]
                    current_bs = reward_tensor_all_tmp[first_agent_key].shape[0]
                    # terminate_list = [False] * current_bs

                    total_rewards = torch.zeros(current_bs)
                    for agent_key, reward_tensor in reward_tensor_all_tmp.items():
                        sample_total_rewards = reward_tensor.sum(dim=-1)  # shape: [batch_size]
                        total_rewards += sample_total_rewards

                    terminate_list = (total_rewards == self.mac.num_agents).tolist()

                    for i, terminated in enumerate(terminate_list):
                        if terminated:
                            terminated_samples.add(i)

                    print(f"Turn {turn_idx}: Terminated samples count = {len(terminated_samples)}, ratio = {len(terminated_samples)}/{current_bs}")


                for i in range(len(new_prompts)):
                    if i in terminated_samples:
                        new_prompts[i] = "The conversation is over, just output <End>"



                new_prompt_messages = [[{"role": "user", "content": p}] for p in new_prompts]


                for agent_idx, agent in enumerate(self.mac.agents):
                    agent_key = f"agent_{agent_idx}"
                    processed_input_ids = []
                    processed_attention_masks = []
                    processed_position_ids = []
                    processed_raw_prompt_ids = []

                    if agent.unthinking_mode:
                        new_prompt_strs = [agent.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False) for msg in new_prompt_messages]
                    else:
                        new_prompt_strs = [agent.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in new_prompt_messages]


                    model_inputs = agent.tokenizer(new_prompt_strs,
                                                    return_tensors="pt",
                                                    add_special_tokens=False,
                                                    truncation=True,
                                                    max_length=self.config.data.max_prompt_length,
                                                    padding=True
                                                )
                    turn_input_ids = model_inputs["input_ids"]
                    turn_attention_mask = model_inputs["attention_mask"]


                    processed_ids = []
                    processed_mask = []
                    for i in range(turn_input_ids.shape[0]):
                        processed_ids, processed_mask = verl_F.postprocess_data(
                            input_ids=turn_input_ids[i].unsqueeze(0),
                            attention_mask=turn_attention_mask[i].unsqueeze(0),
                            max_length=self.config.data.max_prompt_length,
                            pad_token_id=agent.tokenizer.pad_token_id,
                            left_pad=True,
                            truncation=self.config.get("truncation", "error"),
                        )

                        processed_pos_ids = compute_position_id_with_mask(processed_mask)
                        processed_input_ids.append(processed_ids[0])
                        processed_attention_masks.append(processed_mask[0])
                        processed_position_ids.append(processed_pos_ids[0])

                    agent_max_prompt_length = self.config.data.max_prompt_length
                    raw_prompt_ids = agent.tokenizer(new_prompt_strs, add_special_tokens=False)
                    processed_raw_prompt_ids = [ids[:agent_max_prompt_length] for ids in raw_prompt_ids['input_ids']]


                    batch_input_ids = torch.stack(processed_input_ids)
                    batch_attention_masks = torch.stack(processed_attention_masks)
                    batch_position_ids = torch.stack(processed_position_ids)

                    new_batch_data = {
                        'input_ids': batch_input_ids,
                        'attention_mask': batch_attention_masks,
                        'position_ids': batch_position_ids,
                    }

                    new_next_batch = DataProto.from_dict(
                        tensors=new_batch_data,
                        non_tensors={'raw_prompt_ids': processed_raw_prompt_ids},
                    )
                    # Preserve dataset metadata such as prompt, index, and reward fields.
                    last_turn_non_tensor_batch = copy.deepcopy(agent_batchs[agent_key].non_tensor_batch)
                    new_next_batch.non_tensor_batch.update(last_turn_non_tensor_batch)

                    next_turn_batchs[agent_key] = new_next_batch


                # active_indices = [i for i, val in enumerate(terminate_list) if not val]

                # Filtering terminated samples would require dynamic padding
                # across agents, so early-stopped samples are replaced with a
                # terminal prompt instead of being physically removed.
                # for agent_key, next_batch in next_turn_batchs.items():
                #     new_tensors = {}
                #     for k, v in next_batch.batch.items():
                #         if isinstance(v, torch.Tensor):
                #             new_tensors[k] = v[active_indices]
                #         else:
                #             new_tensors[k] = v
                    
                #     new_non_tensors = {}
                #     for k, v in next_batch.non_tensor_batch.items():
                #         if isinstance(v, np.ndarray):
                #             new_non_tensors[k] = v[active_indices]
                #         elif isinstance(v, list):
                #             new_non_tensors[k] = [v[i] for i in active_indices]
                #         else:
                #             new_non_tensors[k] = v

                #     next_batch = DataProto.from_dict(tensors=new_tensors,non_tensors=new_non_tensors,meta_info=next_batch.meta_info)
                #     next_turn_batchs[agent_key] = next_batch
                

                



                # Prepare prompt-only generation batches for the next turn.
                new_gen_batch_all = {}                
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

                for agent_key, next_batch in next_turn_batchs.items():
                    next_gen_batch = next_batch.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )

                    # Validation avoids multi-sample GRPO branching.
                    if validate:
                        agent_id = int(agent_key.split('_')[-1])
                        next_gen_batch.meta_info = {
                            'eos_token_id': self.mac.agents[agent_id].tokenizer.eos_token_id,
                            'pad_token_id': self.mac.agents[agent_id].tokenizer.pad_token_id,
                            'recompute_log_prob': False,
                            'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                            'validate': True,
                        }

                    new_gen_batch_all[agent_key] = next_gen_batch

                gen_batch_all = new_gen_batch_all   


        return multi_turn_batchs
