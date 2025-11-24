
from openai import OpenAI
import pandas as pd
from verl.utils.reward_score.math import compute_score
from tqdm import tqdm
from joblib import Parallel, delayed

def qwen25_inference(messages):
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="sk-xxxxxxxxxxxxxxxx"
    )

    response = client.chat.completions.create(
        model="/root/models/qwen2.5-3b-instruct",
        messages=messages,
        temperature=0.6,
        max_tokens=800,
        stream=False,
    )

    return response.choices[0].message.content

def qwen3_inference(messages):
    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="sk-xxxxxxxxxxxxxxxx"
    )

    response = client.chat.completions.create(
        model="/root/models/qwen3-4b-instruct-2507",
        messages=messages,
        temperature=0.6,
        max_tokens=800,
        stream=False,
    )

    return response.choices[0].message.content

df = pd.read_csv("/root/code/maverl/rollout/only_one_correct_qwen2.5_qwen3.csv")

def rollout_wo_debate(row):
    messages = [
        {"role": "user", "content": row.prompt}
    ]
    qwen25_ans = qwen25_inference(messages)
    qwen3_ans = qwen3_inference(messages)
    qwen25_score = compute_score(qwen25_ans, row.ground_truth)
    qwen3_score = compute_score(qwen3_ans, row.ground_truth)
    return qwen25_score, qwen3_score

qwen25_scores = []
qwen3_scores = []
# 使用线程池而不是进程池，并减少并发数
results = Parallel(n_jobs=8, backend='threading')(delayed(rollout_wo_debate)(row) for row in df.itertuples())
for qwen25_score, qwen3_score in results:
    qwen25_scores.append(qwen25_score)
    qwen3_scores.append(qwen3_score)

print(f"Qwen2.5 average score: {sum(qwen25_scores) / len(qwen25_scores)}")
print(f"Qwen3 average score: {sum(qwen3_scores) / len(qwen3_scores)}")

all_right = one_right = none_right = 0
for s1, s2 in zip(qwen25_scores, qwen3_scores):
    if s1 > 0.5 and s2 > 0.5:
        all_right += 1
    elif s1 > 0.5 or s2 > 0.5:
        one_right += 1
    else:
        none_right += 1

print(f"qwen2.5和qwen3都答对: {all_right}道题")
print(f"qwen2.5和qwen3只有一个答对: {one_right}道题")
print(f"qwen2.5和qwen3都答错: {none_right}道题")




