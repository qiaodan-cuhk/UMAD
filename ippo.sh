#!/bin/bash

# 设置环境变量
export PYTHONUNBUFFERED=1
export VERL_USE_MODELSCOPE=False  # 如果需要从modelscope下载模型
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 设置路径变量
DATA_DIR="$HOME/data/math"
# MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_NAME="$HOME/models/qwen2.5-0.5b-instruct"
MODEL_NAME='/root/models/deepseek-coder-1.3b-instruct'
# MODEL_NAME="$HOME/models/qwen2.5-3B-instruct"
LOG_FILE="verl_demo.log"

# 设置算法和config name
export CONFIG_NAME="ippo_trainer"  # ippo/grpo/remax
export ALGORITHM_NAME="ippo"   # ippo

# 设置是否使用 sum reward，默认不使用
SUM_REWARD=true  # 若要使用，改为 true
CORY_FLIP=false # 是否flip agents in cory
USE_KL=false # 是否使用kl penalty in rewards
DATE_TIME=$(date +%m%d-%H%M)


if [ "$SUM_REWARD" = true ]; then
    SUM_REWARD_FLAG="_sum"
fi

if [ "$CORY_FLIP" = true ]; then
    CORY_FLIP_FLAG="_flip"
fi

if [ "$USE_KL" = true ]; then
    USE_KL_FLAG="_usekl"
fi

# 设置WandB相关变量
WANDB_PROJECT="verl_test"
WANDB_RUN_NAME="MATH_ds-1.3b_${ALGORITHM_NAME}${SUM_REWARD_FLAG}${CORY_FLIP_FLAG}${USE_KL_FLAG}_8a100_$DATE_TIME"

echo "###### Running Test with $ALGORITHM_NAME and $CONFIG_NAME ######"

# 设置TensorBoard日志目录
TENSORBOARD_DIR="qmix_results/$WANDB_RUN_NAME"
export TENSORBOARD_DIR=$TENSORBOARD_DIR
mkdir -p "$TENSORBOARD_DIR"
echo "##### TensorBoard logs will be saved to: $TENSORBOARD_DIR #####"

# 运行训练
python3 -m marl.main \
 data.train_files=$HOME/data/math/train.parquet \
 data.val_files=$HOME/data/math/test.parquet \
 data.train_batch_size=32 \
 data.max_prompt_length=2048 \
 data.max_response_length=512 \
 actor_rollout_ref.model.path=$MODEL_NAME \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.dtype=bfloat16 \
 critic.optim.lr=1e-5 \
 critic.model.path=$MODEL_NAME \
 critic.ppo_micro_batch_size_per_gpu=2 \
 algorithm.use_kl_in_reward=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console','tensorboard'] \
 trainer.project_name=$WANDB_PROJECT \
 trainer.experiment_name=$WANDB_RUN_NAME \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=-1 \
 trainer.test_freq=20 \
 trainer.total_epochs=15 \
 +marl.name=$ALGORITHM_NAME \
 +marl.mixer="none" \
 +marl.config_name=$CONFIG_NAME \
 +marl.sum_reward=$SUM_REWARD \
 +marl.cory_flip=$CORY_FLIP \
 +marl.tensorboard_dir=$TENSORBOARD_DIR 2>&1 | tee verl_demo.log
