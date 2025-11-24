import asyncio
import aiohttp
import pandas as pd
import multiprocessing as mp
import json
import os
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
from verl.utils.reward_score.math import compute_score, last_boxed_only_string, remove_boxed, is_equiv


def retrieve_answer(answer):
    try:
        string_in_last_boxed = last_boxed_only_string(answer)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
    except Exception as e:
        print(e)
    return answer


# 异步调用
async def async_inference(base_url, model, messages):
    headers = {"Authorization": "Bearer sk-xxxxxxxxxxxxxxxx"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
        "stream": False,
        "max_tokens": 8192,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Request failed ({resp.status}): {text}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

# 单次推理
async def rollout_with_debate(row, port_a, port_b, sem, log_dir, row_idx):
    async with sem:
        # 创建日志结构
        debate_log = {
            "question": row.prompt,
            "prompt": row.prompt,
            "ground_truth": row.ground_truth,
            "agent_0_response_turn_0": "",
            "agent_1_response_turn_0": "",
            "agent_0_response_turn_1": "",
            "agent_1_response_turn_1": "",
            "agent_0_response_turn_2": "",
            "agent_1_response_turn_2": "",
            "agent_0_response_turn_3": "",
            "agent_1_response_turn_3": "",
            "agent_0_response_turn_4": "",
            "agent_1_response_turn_4": "",
            "agent_0_response_turn_5": "",
            "agent_1_response_turn_5": "",
            "scores": {
                "initial": {"agent_0": 0, "agent_1": 0},
                "final": {"agent_0": 0, "agent_1": 0}
            }
        }
        
        messages = [{"role": "user", "content": row.prompt}]
        try:
            qwen25_ans, qwen3_ans = await asyncio.gather(
                async_inference(f"http://localhost:{port_a}/v1", "/root/models/qwen2.5-3b-instruct", messages),
                async_inference(f"http://localhost:{port_b}/v1", "/root/models/qwen3-4b-instruct-2507", messages),
            )
        except Exception as e:
            print(f"初始推理失败: {e}")
            qwen25_ans, qwen3_ans = "", ""

        # 记录初始响应 (turn 0)
        debate_log["agent_0_response_turn_0"] = qwen25_ans
        debate_log["agent_1_response_turn_0"] = qwen3_ans

        qwen25_score = compute_score(qwen25_ans, row.ground_truth)
        qwen3_score = compute_score(qwen3_ans, row.ground_truth)

        # 记录初始分数
        debate_log["scores"]["initial"]["agent_0"] = qwen25_score
        debate_log["scores"]["initial"]["agent_1"] = qwen3_score

        ans1 = retrieve_answer(qwen25_ans)
        ans2 = retrieve_answer(qwen3_ans)
        trial_num = 0
        
        while ans1 != ans2 and trial_num < 5:
            trial_num += 1

            # debate
            new_prompt = f"""
            Given the following problem: {row.prompt}\n
            We have two answers:\n
            Answer 1: {qwen25_ans}\n
            Answer 2: {qwen3_ans}\n
            Please carefully review these two answers and recognize which one is right.
            If one or all of them are right, please summarize the reasoning process of right ones and output the right answer within \\boxed{{}}.
            If both of them are wrong, please correct their mistakes and provide a novel and complete solution to the problem, also output the final answer within \\boxed{{}}.
            """
            debate_messages = [{"role": "user", "content": new_prompt}]
            
            try:
                qwen25_ans, qwen3_ans = await asyncio.gather(
                    async_inference(f"http://localhost:{port_a}/v1", "/root/models/qwen2.5-3b-instruct", debate_messages),
                    async_inference(f"http://localhost:{port_b}/v1", "/root/models/qwen3-4b-instruct-2507", debate_messages),
                )
            except Exception as e:
                print(f"辩论轮次{trial_num}失败: {e}")
                qwen25_ans, qwen3_ans = "", ""
            
            # 记录当前轮次的响应
            debate_log[f"agent_0_response_turn_{trial_num}"] = qwen25_ans
            debate_log[f"agent_1_response_turn_{trial_num}"] = qwen3_ans
            
            ans1 = retrieve_answer(qwen25_ans)
            ans2 = retrieve_answer(qwen3_ans)

        # 记录最终分数
        qwen25_score2 = compute_score(qwen25_ans, row.ground_truth)
        qwen3_score2 = compute_score(qwen3_ans, row.ground_truth)

        debate_log["scores"]["final"]["agent_0"] = qwen25_score2
        debate_log["scores"]["final"]["agent_1"] = qwen3_score2

        # 保存日志到文件
        log_file = os.path.join(log_dir, f"problem_{row_idx}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(debate_log, f, ensure_ascii=False, indent=2)

        return qwen25_score, qwen3_score, qwen25_score2, qwen3_score2


# 每个进程的异步执行逻辑
async def async_worker(df_chunk, port_a, port_b, process_id, log_dir):
    sem = asyncio.Semaphore(256)  # 每个进程最多256并发
    tasks = []
    
    for idx, row in enumerate(df_chunk.itertuples()):
        task = rollout_with_debate(row, port_a, port_b, sem, log_dir, row.Index)
        tasks.append(task)
    
    results = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc=f"Proc {process_id}"):
        result = await coro
        results.append(result)

    return results

def process_worker(df_chunk, port_a, port_b, process_id, output_queue, log_dir):
    results = asyncio.run(async_worker(df_chunk, port_a, port_b, process_id, log_dir))
    output_queue.put(results)


def main():
    df = pd.read_csv("/root/code/maverl/rollout/only_one_correct_qwen2.5_qwen3.csv")
    # df = df.head(10)

    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/root/code/maverl/rollout/debate_logs_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"日志将保存到: {log_dir}")

    # 修复端口配置 - 每个进程使用不同的端口对
    port_pairs = [
        (8000, 8001),  # 进程0: Qwen2.5-3B + Qwen3-4B
        (8000, 8002),  # 进程1: Qwen2.5-3B + Qwen3-4B
        (8000, 8003),  # 进程2: Qwen2.5-3B + Qwen3-4B
    ]
    n_processes = len(port_pairs)

    # 按进程数切分数据
    chunks = np.array_split(df, n_processes)
    output_queue = mp.Queue()
    processes = []

    for i, ((port_a, port_b), df_chunk) in enumerate(zip(port_pairs, chunks)):
        p = mp.Process(
            target=process_worker,
            args=(df_chunk, port_a, port_b, i, output_queue, log_dir),
        )
        p.start()
        processes.append(p)

    # 收集结果
    all_results = []
    for _ in processes:
        all_results.extend(output_queue.get())

    for p in processes:
        p.join()

    # 保存汇总结果
    summary = {
        "timestamp": timestamp,
        "total_problems": len(df),
        "results": all_results
    }
    
    summary_file = os.path.join(log_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    qwen25_scores_before_debate = []
    qwen3_scores_before_debate = []
    qwen25_scores_after_debate = []
    qwen3_scores_after_debate = []
    for qwen25_score, qwen3_score, qwen25_score2, qwen3_score2 in all_results:
        qwen25_scores_before_debate.append(qwen25_score)
        qwen3_scores_before_debate.append(qwen3_score)
        qwen25_scores_after_debate.append(qwen25_score2)
        qwen3_scores_after_debate.append(qwen3_score2)
    
    print(f"Before Debate: Qwen2.5 average score={sum(qwen25_scores_before_debate) / len(qwen25_scores_before_debate)}")
    print(f"Before Debate: Qwen3 average score={sum(qwen3_scores_before_debate) / len(qwen3_scores_before_debate)}")
    print(f"After Debate: Qwen2.5 average score={sum(qwen25_scores_after_debate) / len(qwen25_scores_after_debate)}")
    print(f"After Debate: Qwen3 average score={sum(qwen3_scores_after_debate) / len(qwen3_scores_after_debate)}")

    all_right = one_right = none_right = 0
    for s1, s2 in zip(qwen25_scores_before_debate, qwen3_scores_before_debate):
        if s1 > 0.5 and s2 > 0.5:
            all_right += 1
        elif s1 > 0.5 or s2 > 0.5:
            one_right += 1
        else:
            none_right += 1
    print("Before Debate:")
    print(f"qwen2.5和qwen3都答对: {all_right}道题")
    print(f"qwen2.5和qwen3只有一个答对: {one_right}道题")
    print(f"qwen2.5和qwen3都答错: {none_right}道题")

    all_right = one_right = none_right = 0
    for s1, s2 in zip(qwen25_scores_after_debate, qwen3_scores_after_debate):
        if s1 > 0.5 and s2 > 0.5:
            all_right += 1
        elif s1 > 0.5 or s2 > 0.5:
            one_right += 1
        else:
            none_right += 1
    print("After Debate:")
    print(f"qwen2.5和qwen3都答对: {all_right}道题")
    print(f"qwen2.5和qwen3只有一个答对: {one_right}道题")
    print(f"qwen2.5和qwen3都答错: {none_right}道题")

    print(f"✅ 所有日志已保存到: {log_dir}")

if __name__ == "__main__":
    import numpy as np
    main()