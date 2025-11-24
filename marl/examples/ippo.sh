#!/bin/bash

# 基于verl的快速指南，创建的测试脚本，用PPO instruct训练Qwen 0.5B

# 设置环境变量
export PYTHONUNBUFFERED=1
export VERL_USE_MODELSCOPE=False  # 如果需要从modelscope下载模型
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 设置路径变量
DATA_DIR="$HOME/data/gsm8k"
# MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME="$HOME/models/qwen2.5-0.5b-instruct"
LOG_FILE="verl_demo.log"

# 设置算法和config name
export ALGORITHM_NAME="ippo"   # ippo, cory, vdn, qmix
export CONFIG_NAME="ippo_trainer"  # ippo/grpo/remax
# 设置是否使用 sum reward，默认不使用
export SUM_REWARD=false  # 若要使用，改为 true
export CORY_FLIP=false # 是否flip agents in cory


# 设置WandB相关变量
WANDB_PROJECT="verl_test"
WANDB_RUN_NAME="GSM8K_qwen0.5b_$ALGORITHM_NAME-a100_sumreward"

# 设置TensorBoard日志目录
export TENSORBOARD_DIR="/mlx_devbox/users/qiaodan.cuhksz/repo/19101/marl_for_llm_finetune/maverl/results/$WANDB_RUN_NAME"


# 运行训练
python3 -m verl.trainer.main_marl \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=64 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=$MODEL_NAME \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.dtype=bfloat16 \
 critic.optim.lr=1e-5 \
 critic.model.path=$MODEL_NAME \
 critic.ppo_micro_batch_size_per_gpu=2 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console','tensorboard'] \
 trainer.project_name=$WANDB_PROJECT \
 trainer.experiment_name=$WANDB_RUN_NAME \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=-1 \
 trainer.test_freq=20 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log

# used for ray debugpy
#  PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo data.train_files=$HOME/data/gsm8k/train.parquet data.val_files=$HOME/data/gsm8k/test.parquet data.train_batch_size=64 data.max_prompt_length=512 data.max_response_length=256 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 critic.optim.lr=1e-5 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct critic.ppo_micro_batch_size_per_gpu=4 algorithm.kl_ctrl.kl_coef=0.001 trainer.logger=['console'] +trainer.val_before_train=False trainer.default_hdfs_dir=null trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=10 trainer.total_epochs=15 2>&1 | tee verl_demo.log