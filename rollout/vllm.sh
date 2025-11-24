#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=$PYTHONPATH
export MP_START_METHOD=spawn
export VLLM_USE_V1=0

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
--model /root/models/qwen2.5-3b-instruct \
--max_model_len 32768 \
--host 0.0.0.0 \
--port 8000
--disable-frontend-multiprocessing &

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
--model /root/models/qwen3-4b-instruct-2507 \
--max_model_len 81920 \
--host 0.0.0.0 \
--port 8001
--disable-frontend-multiprocessing &

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
--model /root/models/qwen3-4b-instruct-2507 \
--max_model_len 81920 \
--host 0.0.0.0 \
--port 8002
--disable-frontend-multiprocessing &

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
--model /root/models/qwen3-4b-instruct-2507 \
--max_model_len 81920 \
--host 0.0.0.0 \
--port 8003
--disable-frontend-multiprocessing &

# 等待所有后台进程
wait

