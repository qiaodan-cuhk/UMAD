import asyncio
import aiohttp
import pandas as pd
import multiprocessing as mp
from tqdm.asyncio import tqdm_asyncio
# from score import compute_score, retrieve_answer
from verl.utils.reward_score.math import compute_score, last_boxed_only_string, remove_boxed, is_equiv

import json, os
from datetime import datetime


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
async def rollout_with_debate(row, port_a, port_b, sem):
    async with sem:
        messages = [{"role": "user", "content": row.prompt}]
        try:
            qwen25_ans, qwen3_ans = await asyncio.gather(
                async_inference(f"http://localhost:{port_a}/v1", "/root/models/qwen2.5-3b-instruct", messages),
                async_inference(f"http://localhost:{port_b}/v1", "/root/models/qwen3-4b-instruct-2507", messages),
            )
        except Exception:
            qwen25_ans, qwen3_ans = "", ""

        qwen25_score = compute_score(qwen25_ans, row.ground_truth)
        qwen3_score = compute_score(qwen3_ans, row.ground_truth)

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
            except Exception:
                qwen25_ans, qwen3_ans = "", ""
            ans1 = retrieve_answer(qwen25_ans)
            ans2 = retrieve_answer(qwen3_ans)

        qwen25_score2 = compute_score(qwen25_ans, row.ground_truth)
        qwen3_score2 = compute_score(qwen3_ans, row.ground_truth)

        return qwen25_score, qwen3_score, qwen25_score2, qwen3_score2


# 每个进程的异步执行逻辑
async def async_worker(df_chunk, port_a, port_b, process_id):
    sem = asyncio.Semaphore(256)  # 每个进程最多256并发
    tasks = [rollout_with_debate(row, port_a, port_b, sem) for row in df_chunk.itertuples()]
    results = []

    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc=f"Proc {process_id}"):
        result = await coro
        results.append(result)

    return results

def process_worker(df_chunk, port_a, port_b, process_id, output_queue):
    results = asyncio.run(async_worker(df_chunk, port_a, port_b, process_id))
    output_queue.put(results)


def main():
    df = pd.read_csv("/root/code/maverl/rollout/only_one_correct_qwen2.5_qwen3.csv")

    # 多进程端口映射，例如四个GPU对应四个端口
    port_pairs = [
        (8000, 8001),
        (8000, 8002),
        (8000, 8003),
    ]
    n_processes = len(port_pairs)

    # 按进程数切分数据
    chunks = np.array_split(df, n_processes)
    output_queue = mp.Queue()
    processes = []

    for i, ((port_a, port_b), df_chunk) in enumerate(zip(port_pairs, chunks)):
        p = mp.Process(
            target=process_worker,
            args=(df_chunk, port_a, port_b, i, output_queue),
        )
        p.start()
        processes.append(p)

    # 收集结果
    all_results = []
    for _ in processes:
        all_results.extend(output_queue.get())

    for p in processes:
        p.join()

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

    # result_df = pd.DataFrame(all_results, columns=[
    #     "qwen25_score", "qwen3_score", "qwen25_score2", "qwen3_score2"
    # ])
    # result_df.to_csv("results.csv", index=False)
    # print("✅ All done. Results saved to results.csv")

if __name__ == "__main__":
    import numpy as np
    main()

"""
运行前, 先启动vllm服务
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-3B-Instruct \
--max_model_len 32768 \
--host 0.0.0.0 \
--port 8000

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen3-4B-Instruct-2507 \
--max_model_len 81920 \
--host 0.0.0.0 \
--port 8001

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen3-4B-Instruct-2507 \
--max_model_len 81920 \
--host 0.0.0.0 \
--port 8002

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen3-4B-Instruct-2507 \
--max_model_len 81920 \
--host 0.0.0.0 \
--port 8003
"""
