
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

class Multi_Turn_Runner_Traj_GRPO:
    def __init__(self,
                 config,
                 mac,
                 num_agents,  # cotrian LLMs
                 reward_fn_list=None, # 异构 reward fn tokenizer
                 device_name="cuda",
                 **kwargs):
        
        self.config = config
        self.reward_fn_list = reward_fn_list
        self.agg_mode = config.marl.agg_mode  # sum/max/dictator
        self.mac = mac


    # 后续考虑优化计算效率的时候，把这个data parallel可以进行优化，也就是batch 数据只进行一次dp分发到所有gpu
    # 同时保证所有gpu上都有每个agent，这样只需要batch数据分发一次,
    # 考虑到异构agent的response长度不一致，无法匹配balance后的pair，取消balance功能
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


    # 更严格的remove，避免Let's evaluate 的截断
    def remove_template(self, text):
        """
        移除文本中的模板标记，输入的是不带special token的文本
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
            # 如果qwen2.5存在系统模版，进行额外切分
            if "user\n" in text:
                text = text.split("user\n", 1)[1]
            # 去除首尾空白
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
        参考 math.py 中的 last_boxed_only_string 函数
        将 \boxed{content} 替换为 \boxed{<answer>}
        """
        # 找到最后一个 \boxed 的位置
        idx = string.rfind("\\boxed")
        if idx < 0:
            # 如果没有找到 \boxed，尝试 \fbox
            idx = string.rfind("\\fbox")
            if idx < 0:
                return string
        
        # 处理 \boxed 后面没有花括号的情况（如 \boxed 123）
        if "\\boxed " in string and idx == string.rfind("\\boxed "):
            # 找到 \boxed 后面的内容直到 $ 或字符串结束
            boxed_content = "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
            return string.replace(boxed_content, "<answer>")
        
        # 处理 \boxed{content} 的情况
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        
        # 找到匹配的右花括号
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
            # 提取完整的 \boxed{content} 部分
            boxed_content = string[idx : right_brace_idx + 1]
            # 替换为 \boxed{<answer>}
            return string.replace(boxed_content, "<answer>")
        
        return string

    
    def clean_response_for_next_turn(self, response_text, dataset_type="math"):
        """
        清理response文本，移除final answer部分
        参考 math.py 中的 last_boxed_only_string 和 remove_boxed 函数
        """
        import re
        
        if dataset_type == "math":
            # 参考 math.py 中的实现，精准处理 \boxed{} 格式
            cleaned = self._replace_boxed_with_answer(response_text)
        elif dataset_type == "gsm8k":
            # 处理####格式
            if "####" in response_text:
                cleaned = response_text.split("####")[0].strip()
                # 如果句子不完整，添加提示
                if not cleaned.endswith(('.', '!', '?')):
                    cleaned += " <to be calculated>."
            else:
                cleaned = response_text
        else:
            cleaned = response_text
        
        return cleaned



    def merge_prompt_response(self, original_prompts, all_agent_responses_text, template=None, ego_id=None, turn_idx=None):
        """
        合并原始prompt和多个agent的响应
        新增：移除每一轮的最终 \\boxed{}等形式的答案，避免陷入I agree情况

        Args:
            original_prompts (list[str]): 原始prompt
            all_agent_responses_text (dict): 包含所有agent响应的字典，格式为 {"agent_0": [responses], "agent_1": [responses], ...}
            template (str, optional): 自定义模板，用于定义如何组织responses。
                如果为None，则使用默认模板"agent {idx} response is: {response}"
        
        Returns:
            str: 合并后的文本
        """
        # agent_num = len(all_agent_responses_text)
        # gsm8k 的额外instruction
        # instruction_following = 'Let\'s think step by step and output the final answer after "####".'

        # math 的 instruction
        instruction_following = 'Let\'s think step by step and output the final answer within \\boxed{}.'   

        #### todo:简单的concate，后续可以考虑使用template
        if template is None:
            # 1. 不区分agent的concate模版，容易陷入 base other opinion 模式
            # template = "{original_prompt}\n These are the solutions to the problem from all agents: {responses} Based off the opinion of other agents, can you give an updated response? {instruction_following}"  
            #  
            # 2. 提示review then answer，但是容易陷入 review 模式胡说
            # template = "{original_prompt}\n The following are reasoning attempts from multiple agents:\n {responses} \n Your task: 1. Carefully review the above reasoning attempts. Identify useful intermediate steps or insights, and notice potential mistakes or contradictions.2. Based on these insights, answer the problem in a self-contained, rigorous reasoning chain from the beginning. Do not simply copy or pick one agent's final answer. {instruction_following}" 
 
            # 3. mask掉答案，让agent自己整合思路
            # template = "{original_prompt}\n The following are reasoning attempts from multiple agents:\n {responses} \n Analyze these approaches and provide your own complete solution. {instruction_following}" 
            # template = "{original_prompt}\n The following are reasoning attempts from multiple agents:\n {responses} \n Use these different perspectives, and develop an updated comprehensive solution. {instruction_following}" 
            # template = "Given the following problem: {original_prompt}\n and these solution attempts: \n {responses} \n Carefully review the provided solutions, recognize the useful parts, using them as starting points, correcting mistakes, filling in gaps, and combining useful ideas—to produce an updated solution to the problem. If you believe the attempts are incorrect, feel free to revise it. You must provide a novel andcomplete solution to the problem and you can't repeat the same answers in the provided solution attempts. {instruction_following}" 
            # template = "In last turn we have collected some solution attempts from multiple agents, now you need to summarize all agents' core ideas, correct errors, integrate the useful parts to get a new solution to completely and comprehensively solve the original problem from the first step. Do not repeat or plagiarize solution attempts. These are the solution attempts from multiple agents: \n {responses} \n {instruction_following}" 
            # template = "These are the solution attempts from multiple agents: {responses} \n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response. Do not copy or repeat the same answer in the provided solution attempts. \n {instruction_following}"
            template = "Given the following problem: {original_prompt} \n We have two answers: \n {responses} \n Please carefully review these answers and recognize which one is right. If one or all of them are right, please summarize the reasoning process of right ones and give the final answer. If both of them are wrong, please correct their mistakes and provide a novel and complete solution to the problem and give the final answer. {instruction_following}"

            # Identify the specific point of conflict or error (e.g., probability calculation, formula application) and provide an explicit verification or counter-argument before presenting the final solution.
            
            # template = "{original_prompt}\n The following are reasoning attempts from multiple agents:\n {responses} \n You can combine insights from these approaches. You need to develop a comprehensive solution from the beginning. {instruction_following}" 
        
             
            # 待测试的新模版
            # template = "{original_prompt}\n The following are reasoning attempts from multiple agents:\n {responses} \n You need to solve the problem again by extracting useful steps from the above attempts and rebuilding a full step-by-step reasoning chain. {instruction_following}"
            # think role的模版
            # template = "{original_prompt}\n These are the solutions to the problem from all agents in the last turn: {responses} You are cooperatively solving this math problem step-by-step. You need to analysis the other agents' responses and choose your <policy> from [verify, new_proposition, merge, deny, accept] and then follow the <policy> to solve the problem. {instruction_following}"


        """考虑多种不同模版：支持debate、concate、sequential、ego、major voting、instruction、role playing"""
        # template_sequential = "{original_prompt}\n These are the solutions to the problem from all agents: {responses} Based off the opinion of other agents, can you give an updated response? {instruction_following}"
        # template_summary = "{original_prompt}\n These are the summarize of other agents' solutions:{responses} {instruction_following}"
        # template_ego = "{original_prompt}\n These are the solutions to the problem from all agents: {responses} Based off the opinion of other agents, can you give an updated response? {instruction_following}"
        # template_majority = "{original_prompt}\n These are the solutions to the problem from all agents: {responses} Based off the opinion of other agents, can you give an updated response? {instruction_following}"
        # template_instruction = "{original_prompt}\n These are the solutions to the problem from all agents: {responses} Based off the opinion of other agents, can you give an updated response? Please think the problem from scratch."
        # template_role = "{original_prompt}\n These are the solutions to the problem from all agents: {responses} Based off the opinion of other agents, select your role in this cooperative reasoning task and output it in <Role>, then base your role and the discussion, give an updated response? {instruction_following}"



        merged_texts = []
        for i, original_prompt in enumerate(original_prompts):
            # 收集并格式化所有agent的响应
            # agent_responses = []
            clean_responses = {}
            for agent_key in all_agent_responses_text.keys():
                if i < len(all_agent_responses_text[agent_key]):
                    response = all_agent_responses_text[agent_key][i]
                    # # 为每个response添加agent标识
                    # formatted_response = f"agent {agent_key.split('_')[1]} response is: {response}"

                    # 清理response用于下一轮输入
                    cleaned_response = self.clean_response_for_next_turn(
                        response, 
                        dataset_type=self.config.data.get("dataset_type", "math")
                    )
                    clean_responses[agent_key] = cleaned_response
                    # agent_responses.append(cleaned_response)
            
            # 将所有响应合并成一个字符串
            if ego_id:
                ego_responses = clean_responses[f"agent_{ego_id}"]
                other_responses = [clean_responses[k] for k in clean_responses.keys() if k != f"agent_{ego_id}"]
                other_responses_text = "\n".join(other_responses)
                responses_text = f"agent {ego_id} response is: {ego_responses}\n Other agents' responses are: \n{other_responses_text}"
            else:
                responses_text = "\n".join(clean_responses.values())
            
            
            # 使用模板格式化文本
            try:
                merged_text = template.format(
                    # turn_idx=turn_idx,
                    original_prompt=original_prompt,
                    responses=responses_text,
                    instruction_following=instruction_following
                )
                merged_texts.append(merged_text)
            except Exception as e:
                # 如果模板格式化失败，使用简单的拼接方式
                fallback_text = f"{original_prompt}\n{responses_text}{instruction_following}"
                merged_texts.append(fallback_text)
        
        return merged_texts
                
    
    
    # traj_grpo 只 roll n 条 traj，按照第一次rollout的uid进行后续turn的分组
    def rollout_traj_grpo(self, gen_batch_all, multi_turn_batchs, metrics, turn=2, validate=False):
        """
        gen_batch_all: {batch_key: DataProto}, 与single turn一致，只是存储dataloader的prompt
        multi_turn_batchs: {turn_idx: {agent_id: DataProto}}，只有turn_0的数据
        metrics: Dict[str, float]
        turn: int  代表交互几轮
        """
        
        for turn_idx in range(turn):

            # 取出每一轮的作为agent_batchs，这里需要已经pop掉其他数据了
            agent_batchs = multi_turn_batchs[f"turn_{turn_idx}"]


            # 每一轮的rollout，保证输入的gen_batch_all是当前轮的形式
            gen_batch_outputs_all = {}
            for agent, gen_batch_agent in zip(self.mac.agents, gen_batch_all.values()):
                if turn_idx > 0:
                    # 用meta info控制greedy采样tmp=0，而不需要rollout.n
                    gen_batch_agent.meta_info.update({'magrpo': True})
                else:
                    gen_batch_agent.meta_info.update({'magrpo': False})

                gen_batch_output = agent.actor_rollout_wg.generate_sequences(gen_batch_agent)
                gen_batch_outputs_all[f"agent_{agent.agent_id}"] = gen_batch_output


            #### 这里的balance也需要考虑根据index对齐，不然两个不同的response会进入不同的pair，目前移除balance功能
            for agent_key, agent_batch  in agent_batchs.items():
                #### 只有第一轮需要添加uid
                if turn_idx == 0:
                    agent_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(agent_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    if not validate:
                        agent_batch = agent_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)                                        
                elif turn_idx > 0:
                    # 保证后续轮的uid与第一轮一致，属于同一组
                    agent_batch.non_tensor_batch['uid'] = multi_turn_batchs["turn_0"][f"{agent_key}"].non_tensor_batch['uid']
                
                
                agent_batch = agent_batch.union(gen_batch_outputs_all[agent_key])
                agent_batch.batch['response_mask'] = compute_response_mask(agent_batch)


                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo

                #### remark: (qd) 因为balance batch会打乱顺序，不同agent得response长度不一致，这里暂时移除balance功能
                # if self.config.trainer.balance_batch:
                #     self._balance_batch(agent_batch, metrics=metrics)
                
                # compute global_valid tokens
                if not isinstance(agent_batch.meta_info, dict):
                    agent_batch.meta_info = {}
                agent_batch.meta_info['global_token_num'] = torch.sum(agent_batch.batch['attention_mask'], dim=-1).tolist()
                agent_batchs[agent_key] = agent_batch

            # 处理完额外属性，放回
            # multi_turn_batchs[f"turn_{turn_idx}"] = agent_batchs

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
                    
                    # 对每个样本的响应进行解码，移除特殊token
                    valid_ids_list = [response_ids[i][response_mask[i].bool()].tolist() for i in range(response_ids.shape[0])]
                    texts = agent.tokenizer.batch_decode(valid_ids_list, skip_special_tokens=True)
                    responses_text = [f"agent {agent_idx} response is: {t}" for t in texts]
                    
                    all_agent_responses_text[agent_key] = responses_text
                
                # 2. 获取下一轮的batchs并更新
                next_turn_batchs = multi_turn_batchs[f"turn_{turn_idx + 1}"]
                # 这里 next turn batchs与经过balance和gen union的agent batchs不一致，non tensor的reward model、extra info、index打乱顺序，少了raw prompt ids，多了uid
                # 而且meta info少了sum tokens


                # 对公共文本进行合并处理
                # 利用上一轮的agent batchs repeated数据，提取对应的原始prompt文本
                texts = [agent_batchs['agent_0'].non_tensor_batch['prompt'][i][0]['content'] for i in range(len(agent_batchs['agent_0'].non_tensor_batch['prompt']))]
                original_prompts = [self.remove_template(t) for t in texts]
                new_prompts = self.merge_prompt_response(original_prompts, all_agent_responses_text, template=None, turn_idx=turn_idx)
                new_prompt_messages = [[{"role": "user", "content": p}] for p in new_prompts]


                # 3. 为每个agent更新下一轮的输入，利用每个agent的tokenizer分别处理
                #### tokenizer.apply_chat_template 会自动处理特殊token，以及max length的截断
                #### todo: 每轮都处理了raw prompt ids，所以这里可以使用这个
                for agent_idx, agent in enumerate(self.mac.agents):
                    agent_key = f"agent_{agent_idx}"


                    # 逐条处理每个prompt，避免批处理长度不一致问题
                    processed_input_ids = []
                    processed_attention_masks = []
                    processed_position_ids = []
                    processed_raw_prompt_ids = []  # 存储处理后的raw_prompt_ids

                    if agent.unthinking_mode:  # 专门用于qwen3系列关闭推理
                        new_prompt_strs = [agent.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False) for msg in new_prompt_messages]
                    else:
                        new_prompt_strs = [agent.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in new_prompt_messages]


                    model_inputs = agent.tokenizer(new_prompt_strs,
                                                    return_tensors="pt",
                                                    add_special_tokens=False,
                                                    truncation=True,
                                                    max_length=self.config.data.max_prompt_length,
                                                    padding=True  # 保证batch对齐
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

                    # 处理raw_prompt_ids
                    agent_max_prompt_length = self.config.data.max_prompt_length
                    # 检查batch处理tokenizer是否正确
                    raw_prompt_ids = agent.tokenizer(new_prompt_strs, add_special_tokens=False)
                    processed_raw_prompt_ids = [ids[:agent_max_prompt_length] for ids in raw_prompt_ids['input_ids']]


                    # 将处理后的结果堆叠成批次
                    batch_input_ids = torch.stack(processed_input_ids)
                    batch_attention_masks = torch.stack(processed_attention_masks)
                    batch_position_ids = torch.stack(processed_position_ids)

                    # 更新next_batch
                    new_batch_data = {
                        'input_ids': batch_input_ids,
                        'attention_mask': batch_attention_masks,
                        'position_ids': batch_position_ids,
                    }

                    # new_next_meta_info = torch.sum(new_batch_data['attention_mask'], dim=-1).tolist()

                    
                    # 创建新的DataProto，保持 tools 和 index 不用更新
                    new_next_batch = DataProto.from_dict(
                        tensors=new_batch_data,
                        non_tensors={'raw_prompt_ids': processed_raw_prompt_ids},
                    )
                    # 添加原始dataset相关数据
                    last_turn_non_tensor_batch = copy.deepcopy(agent_batchs[agent_key].non_tensor_batch)
                    new_next_batch.non_tensor_batch.update(last_turn_non_tensor_batch)
                    next_turn_batchs[agent_key] = new_next_batch

                
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

                # 更新gen_batch_all
                gen_batch_all = new_gen_batch_all   


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
                reward_tensor, reward_extra_infos_dict = compute_reward(agent_batch, self.reward_fn_list[agent_id])
                reward_tensor_all[f"agent_{agent_id}"] = reward_tensor
                reward_extra_infos_dict_all[f"agent_{agent_id}"] = reward_extra_infos_dict

            # 记录原始的reward
            agent_reward_metrics = {
                f"rewards_origin/agent_{agent_id}_turn_{turn_idx}_mean": torch.mean(reward_tensor.sum(-1)).detach().item(),
                # f"origin_reward/agent_{agent_id}_turn_{turn_idx}_max": torch.max(reward_tensor.sum(-1)).detach().item(),
                # f"origin_reward/agent_{agent_id}_turn_{turn_idx}_min": torch.min(reward_tensor.sum(-1)).detach().item(),
            }
            metrics.update(agent_reward_metrics)

        # 新版本sum reward，将team reward对齐到自己的response mask最后一个有效位置
        if self.agg_mode == "sum":
            sum_reward = None
            for agent_key, agent_reward_tensor in reward_tensor_all.items():
                if sum_reward is None:
                    sum_reward = agent_reward_tensor.clone()
                else:
                    sum_reward += agent_reward_tensor.clone()
            team_reward_scalar = sum_reward.sum(-1)
            batch_indices = torch.arange(sum_reward.size(0)) 
            # 将总和奖励赋值给每个agent
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

            # 将max模式的团队奖励赋值给每个agent的最后一个response位置
            for agent_key in reward_tensor_all.keys():
                agent_reward_position = agent_batchs[agent_key].batch['response_mask'].sum(-1) - 1
                reward_tensor_all[agent_key][batch_indices, agent_reward_position] = team_reward_scalar.clone()

            team_reward_metrics = {f"total/turn_{turn_idx}_reward_mean": torch.mean(max_reward.sum(-1)).detach().item(),
                                   f"total/turn_{turn_idx}_reward_max": torch.max(max_reward.sum(-1)).detach().item(),
                                   f"total/turn_{turn_idx}_reward_min": torch.min(max_reward.sum(-1)).detach().item(),
                                   }
            metrics.update(team_reward_metrics)

        elif self.agg_mode == "dictator":
            # 只用 agent_0 的奖励作为团队奖励，强模型
            dictator_reward = reward_tensor_all['agent_0'].clone()
            team_reward_scalar = dictator_reward.sum(-1)
            batch_indices = torch.arange(dictator_reward.size(0))

            # 将dictator模式的团队奖励赋值给每个agent的最后一个response位置
            for agent_key in reward_tensor_all.keys():
                agent_reward_position = agent_batchs[agent_key].batch['response_mask'].sum(-1) - 1
                reward_tensor_all[agent_key][batch_indices, agent_reward_position] = team_reward_scalar.clone()

            team_reward_metrics = {f"total/turn_{turn_idx}_reward_mean": torch.mean(team_reward_scalar.sum(-1)).detach().item(),
                                   f"total/turn_{turn_idx}_reward_max": torch.max(team_reward_scalar.sum(-1)).detach().item(),
                                   f"total/turn_{turn_idx}_reward_min": torch.min(team_reward_scalar.sum(-1)).detach().item(),
                                   }
            metrics.update(team_reward_metrics)


        return reward_tensor_all, reward_extra_infos_dict_all, future_reward_all




    """traj-GRPO版本奖励，每一轮都eval回复，dense reward,不做accumulative"""
    def cal_reward_dense(self, multi_turn_batchs, metrics):
        reward_tensor_all_turns = {}
        reward_extra_infos_dict_all_turns = {}
        future_reward_all_turns = {}
        
        # 获取轮次数量
        num_turns = len(multi_turn_batchs)
        # last_turn_idx = f"turn_{num_turns - 1}"

        for turn_idx in range(num_turns):
            turn_key = f"turn_{turn_idx}"
            turn_batchs = multi_turn_batchs[turn_key]
            # 这里只负责计算response的reward，不负责计算adv时候的grpo group uid
            turn_reward_tensor_all, turn_reward_extra_infos_dict_all, turn_future_reward_all = self.cal_reward(turn_batchs, turn_idx, metrics)
        
            # 记录每一轮的奖励
            reward_tensor_all_turns[turn_key] = turn_reward_tensor_all
            reward_extra_infos_dict_all_turns[turn_key] = turn_reward_extra_infos_dict_all
            future_reward_all_turns[turn_key] = turn_future_reward_all
        

        return reward_tensor_all_turns, reward_extra_infos_dict_all_turns, future_reward_all_turns



    """traj-GRPO的稀疏奖励，只计算最后一轮的奖励，其他轮是0"""
    def cal_reward_sparse(self, multi_turn_batchs, metrics):
        """
        计算多轮对话的奖励，只计算最后一轮的奖励
        
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
        
        # 其他轮奖励设为0,在grpo设置下需要根据每一轮的数据量动态调整zeros tensor维度
        for turn_idx in range(num_turns - 1):
            turn_key = f"turn_{turn_idx}"
            zero_rewards = {}
            turn_bs = multi_turn_batchs[turn_key]['agent_0'].batch.batch_size[0]
            turn_len = last_reward_tensor_all['agent_0'].shape[1]
            for agent_key, reward_tensor in last_reward_tensor_all.items():
                zero_rewards[agent_key] = torch.zeros(turn_bs, turn_len, dtype=reward_tensor.dtype, device=reward_tensor.device)
            reward_tensor_all_turns[turn_key] = zero_rewards
            reward_extra_infos_dict_all_turns[turn_key] = last_reward_extra_infos_dict_all  # 空字典
            future_reward_all_turns[turn_key] = last_future_reward_all  # 空字典
        
        
        return reward_tensor_all_turns, reward_extra_infos_dict_all_turns, future_reward_all_turns



    """traj-GRPO的dense奖励，每一轮都用自己eval的奖励，并且做accumulative"""
    def cal_reward_accumulative(self, multi_turn_batchs, metrics):
        """
        计算多轮对话的奖励，只计算最后一轮的奖励
        
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
                # 要考虑不同位置的reward进行对齐
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



    # 暂时不用实现支持llm rollout,后续可以用于支持每轮concate变成summary
    def init_workers(self):
        pass



    # 带 early stop 判断，如果两个agent的回复都正确就停止debate
    def rollout_multi_turn_grpo_earlystop(self, gen_batch_all, multi_turn_batchs, metrics, turn=2, validate=False):
        """
        gen_batch_all: {batch_key: DataProto}, 与single turn一致，只是存储dataloader的prompt
        multi_turn_batchs: {turn_idx: {agent_id: DataProto}}，只有turn_0的数据
        metrics: Dict[str, float]
        turn: int  代表交互几轮
        """
        
        # 跨轮维护终止状态
        terminated_samples = set()

        for turn_idx in range(turn):

            # 取出每一轮的作为agent_batchs，这里需要已经pop掉其他数据了
            agent_batchs = multi_turn_batchs[f"turn_{turn_idx}"]     #[bs]

            # 每一轮的rollout，保证输入的gen_batch_all是当前轮的形式
            gen_batch_outputs_all = {}
            for agent, gen_batch_agent in zip(self.mac.agents, gen_batch_all.values()):    
                gen_batch_output = agent.actor_rollout_wg.generate_sequences(gen_batch_agent)    # 动态的，不一定是 bs
                gen_batch_outputs_all[f"agent_{agent.agent_id}"] = gen_batch_output


            # last_batch_response = self.mac.agents[0].tokenizer.batch_decode(multi_turn_batchs['turn_0']['agent_0'].batch['responses'], skip_special_tokens=True)
            # new_batch_response = self.mac.agents[1].tokenizer.batch_decode(gen_batch_output.batch['responses'], skip_special_tokens=True)
            """完成了每个agent的rollout过程，后续复杂pipeline考虑函数包装，增加轨迹uid以及balance gpu分配"""
            #### todo: 这里的balance也需要考虑根据index对齐，不然两个不同的response会进入不同的pair，目前移除balance功能
            for agent_key, agent_batch  in agent_batchs.items():
                #### todo 应该只有第一轮需要添加uid
                agent_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(agent_batch.batch))], dtype=object)
                # repeat to align with repeated responses in rollout
                if not validate:
                    agent_batch = agent_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)


                agent_batch = agent_batch.union(gen_batch_outputs_all[agent_key])

                agent_batch.batch['response_mask'] = compute_response_mask(agent_batch)

                #### remark: (qd) 因为balance batch会打乱顺序，不同agent得response长度不一致，这里暂时移除balance功能
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


            

            # 处理所有agent的response，concate成下一轮的prompt
            if turn_idx < turn - 1:
                # 1. 收集所有agent的响应文本
                all_agent_responses_text = {}
                for agent_idx, agent in enumerate(self.mac.agents):
                    agent_key = f"agent_{agent_idx}"
                    agent_batch = agent_batchs[agent_key]
                    
                    response_ids = agent_batch.batch['responses']
                    response_mask = agent_batch.batch['response_mask']

                    # 对每个样本的响应进行解码，移除特殊token
                    valid_ids_list = [response_ids[i][response_mask[i].bool()].tolist() for i in range(response_ids.shape[0])]
                    texts = agent.tokenizer.batch_decode(valid_ids_list, skip_special_tokens=True)
                    responses_text = [f"agent {agent_idx} response is: {t}" for t in texts]
                    all_agent_responses_text[agent_key] = responses_text

                
                """用于测试qwen3b的数据泄露问题"""
                # all_agent_responses_text["agent_1"] = all_agent_responses_text["agent_0"].copy()
                
                # 2. 获取下一轮的batchs并更新
                next_turn_batchs = multi_turn_batchs[f"turn_{turn_idx + 1}"]
                

                # 这里 next turn batchs与经过balance和gen union的agent batchs不一致，non tensor的reward model、extra info、index打乱顺序，少了raw prompt ids，多了uid
                # 而且meta info少了sum tokens
                # 3. 为每个agent更新下一轮的输入，利用每个agent的tokenizer分别处理
                #### tokenizer.apply_chat_template 会自动处理特殊token，以及max length的截断
                #### todo: 每轮都处理了raw prompt ids，所以这里可以使用这个


                # 对公共文本进行合并处理
                # 利用上一轮的agent batchs repeated数据，提取对应的原始prompt文本
                texts = [agent_batchs['agent_0'].non_tensor_batch['prompt'][i][0]['content'] for i in range(len(agent_batchs['agent_0'].non_tensor_batch['prompt']))]
                original_prompts = [self.remove_template(t) for t in texts]


                # 用原始问题和上一轮的response构建next turn输入
                # 10.20 debug，尝试给不同model注入不同role，检查其指令遵循和生成
                # template_a0 = "Here are some attempts to solve the problem::\n {responses} \n It is possible that any, all, or none of these solutions are correct or complete. Carefully analyze the provided solutions, using them as starting points—correcting mistakes, filling in gaps, and/or combining useful ideas—to produce a final, comprehensive, and correct solution to the problem. \n. {instruction_following}" 
                # template_a1 = "Given the following problem: {original_prompt}\n and these solution attempts::\n {responses} \n Based on the provided solutions, make an updated and clear solution to the problem. If you believe the attempts are incorrect, feel free to revise it. However, avoid repeating the same answers that have already provided.  {instruction_following}" 

                # # template_a0 = "What's the result of 1+1? {instruction_following}" 
                # # template_a1 = "Who is the president of the United States? {instruction_following}" 

                # new_prompt_a0 = self.merge_prompt_response(original_prompts, all_agent_responses_text, template=None, ego_id=0)
                # new_prompt_a1 = self.merge_prompt_response(original_prompts, all_agent_responses_text, template=None, ego_id=1)

                # new_prompt_messages_a0 = [[{"role": "user", "content": p}] for p in new_prompt_a0]
                # new_prompt_messages_a1 = [[{"role": "user", "content": p}] for p in new_prompt_a1]
                # new_prompt_messages_all = [new_prompt_messages_a0, new_prompt_messages_a1]


                new_prompts = self.merge_prompt_response(original_prompts, all_agent_responses_text, template=None, turn_idx=turn_idx)


                """这里需要根据mask列表，剔除对应的next turn batchs和gen batch all"""
                if turn_idx>0:
                    # 计算当前轮次的奖励，判断终止列表
                    reward_tensor_all_tmp, _, _ = self.cal_reward(agent_batchs, turn_idx, metrics)
                    first_agent_key = list(reward_tensor_all_tmp.keys())[0]
                    current_bs = reward_tensor_all_tmp[first_agent_key].shape[0]
                    # terminate_list = [False] * current_bs

                    # 计算每个样本的总奖励（所有agent的奖励之和）
                    total_rewards = torch.zeros(current_bs)
                    for agent_key, reward_tensor in reward_tensor_all_tmp.items():
                        # 对每个样本的reward求和，得到该样本的总奖励
                        sample_total_rewards = reward_tensor.sum(dim=-1)  # shape: [batch_size]
                        total_rewards += sample_total_rewards

                    terminate_list = (total_rewards == self.mac.num_agents).tolist()

                    # 更新终止样本集合（跨轮维护）
                    for i, terminated in enumerate(terminate_list):
                        if terminated:
                            terminated_samples.add(i)

                    # 显示每轮终止样本的数量
                    print(f"Turn {turn_idx}: Terminated samples count = {len(terminated_samples)}, ratio = {len(terminated_samples)}/{current_bs}")


                # 替换为终止文本
                for i in range(len(new_prompts)):
                    if i in terminated_samples:
                        new_prompts[i] = "The conversation is over, just output <End>"



                new_prompt_messages = [[{"role": "user", "content": p}] for p in new_prompts]


                for agent_idx, agent in enumerate(self.mac.agents):
                    agent_key = f"agent_{agent_idx}"
                    # next_batch = next_turn_batchs[agent_key]   # 检查发现，agent batchs经过balance后的extra info与response一致，但是没有raw prompt ids了
                    
                    # 获取原始prompt文本,换成使用balance以后的agent batch数据，不用raw prompt ids
                    # 这里用的是上一轮计算的agent batchs，可以保持uid和prompts等变量的数量n倍和index位置对应，但是对于2轮以上的prompt处理可能会麻烦,考虑使用non tensor batch的prompt属性 
                    # prompt_ids = agent_batchs[agent_key].batch['prompts']     # [batch, seq] 
                    # texts = agent.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)


                    # 逐条处理每个prompt，避免批处理长度不一致问题
                    processed_input_ids = []
                    processed_attention_masks = []
                    processed_position_ids = []
                    processed_raw_prompt_ids = []  # 存储处理后的raw_prompt_ids

                    """todo 支持per agent的qwen3 thinking模式"""
                    # new_prompt_strs = [agent.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in new_prompt_messages]
                    # if self.use_qwen3_unthinking:
                    #     new_prompt_strs = [agent.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False) for msg in new_prompt_messages]
                    # else:
                    #     new_prompt_strs = [agent.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in new_prompt_messages]

                    """测试不同role prompt"""
                    # new_prompt_messages = new_prompt_messages_all[agent_idx]
                    if agent.unthinking_mode:  # 专门用于qwen3系列关闭推理
                        new_prompt_strs = [agent.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False) for msg in new_prompt_messages]
                    else:
                        new_prompt_strs = [agent.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in new_prompt_messages]


                    model_inputs = agent.tokenizer(new_prompt_strs,
                                                    return_tensors="pt",
                                                    add_special_tokens=False,
                                                    truncation=True,
                                                    max_length=self.config.data.max_prompt_length,
                                                    padding=True  # 保证batch对齐
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

                    # 处理raw_prompt_ids, 暂时采取left truncate
                    agent_max_prompt_length = self.config.data.max_prompt_length
                    raw_prompt_ids = agent.tokenizer(new_prompt_strs, add_special_tokens=False)
                    processed_raw_prompt_ids = [ids[:agent_max_prompt_length] for ids in raw_prompt_ids['input_ids']]


                    # 将处理后的结果堆叠成批次
                    batch_input_ids = torch.stack(processed_input_ids)
                    batch_attention_masks = torch.stack(processed_attention_masks)
                    batch_position_ids = torch.stack(processed_position_ids)

                    # 更新next_batch
                    new_batch_data = {
                        'input_ids': batch_input_ids,
                        'attention_mask': batch_attention_masks,
                        'position_ids': batch_position_ids,
                    }

                    # 创建新的DataProto，保持 tools 和 index 不用更新
                    new_next_batch = DataProto.from_dict(
                        tensors=new_batch_data,
                        non_tensors={'raw_prompt_ids': processed_raw_prompt_ids},
                    )
                    # 添加原始dataset相关数据
                    last_turn_non_tensor_batch = copy.deepcopy(agent_batchs[agent_key].non_tensor_batch)
                    new_next_batch.non_tensor_batch.update(last_turn_non_tensor_batch)

                    next_turn_batchs[agent_key] = new_next_batch


                # active_indices = [i for i, val in enumerate(terminate_list) if not val]

                # # 利用terminal list对next turn batchs做过滤
                # 涉及到batch的动态填充，比较麻烦，先不考虑
                # for agent_key, next_batch in next_turn_batchs.items():
                #     # 过滤tensor数据
                #     new_tensors = {}
                #     for k, v in next_batch.batch.items():
                #         if isinstance(v, torch.Tensor):
                #             new_tensors[k] = v[active_indices]
                #         else:
                #             new_tensors[k] = v
                    
                #     # 过滤non_tensor数据
                #     new_non_tensors = {}
                #     for k, v in next_batch.non_tensor_batch.items():
                #         if isinstance(v, np.ndarray):
                #             # 对于numpy数组，使用索引选择
                #             new_non_tensors[k] = v[active_indices]
                #         elif isinstance(v, list):
                #             # 对于列表，使用列表推导式
                #             new_non_tensors[k] = [v[i] for i in active_indices]
                #         else:
                #             # 对于其他类型，直接复制
                #             new_non_tensors[k] = v

                #     next_batch = DataProto.from_dict(tensors=new_tensors,non_tensors=new_non_tensors,meta_info=next_batch.meta_info)
                #     next_turn_batchs[agent_key] = next_batch
                

                



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

                    # 如果是test过程，那么启用validate，避免grpo多次rollout
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

                # 更新gen_batch_all用于下一个循环的generation
                gen_batch_all = new_gen_batch_all   


        return multi_turn_batchs