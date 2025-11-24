#!/bin/bash
# 异构相同tokenizer/processor模型，不同参数，多轮推理
# 这里kl in reward

# 设置环境变量
export PYTHONUNBUFFERED=1
export VERL_USE_MODELSCOPE=False  # 如果需要从modelscope下载模型
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 设置路径变量
# MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_NAME="$HOME/models/qwen2.5-0.5b-instruct"
# MODEL_NAME="$HOME/models/qwen2.5-3B-instruct"
LOG_FILE="verl_demo.log"

export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_MODE=offline



# 设置算法和config name
export CONFIG_NAME="qmix_rollout"  # ippo/grpo/remax
# export CONFIG_NAME="qmix_trainer_3agent"  # 用于多个agent

export ALGORITHM_NAME="qmix"   # ippo, vdn

# 设置是否使用 sum reward，默认不使用
SUM_REWARD=false  # 若要使用，改为 true
CORY_FLIP=false # 是否flip agents in cory
USE_KL=false # 是否使用kl penalty in rewards
DATE_TIME=$(date +%m%d-%H%M)

DATASET="math"  # "math" "gsm8k"
TD_MODE="td0"   # td0, mc


if [ "$SUM_REWARD" = true ]; then
    SUM_REWARD_FLAG="_sum"
fi

if [ "$CORY_FLIP" = true ]; then
    CORY_FLIP_FLAG="_flip"
fi

if [ "$USE_KL" = true ]; then
    USE_KL_FLAG="_usekl"
fi


TURNS=4
ROLLOUT_MODE="earlystop" # earlystop, normal

# 设置WandB相关变量
WANDB_PROJECT="qmix_grpo"
WANDB_RUN_NAME="MATH_qwen2.5-3b_llama-3.2-3b-instruct_turn${TURNS}_tmp0.7_p5k_r2k_binbin-template_earlystop_rollout_$DATE_TIME"

ROLLOUT_DATA_DIR=/root/code/maverl/rollout/logs

echo "###### Running Test with $ALGORITHM_NAME and $CONFIG_NAME ######"

# 设置TensorBoard日志目录
TENSORBOARD_DIR="/root/code/maverl/rollout/results/$WANDB_RUN_NAME"
export TENSORBOARD_DIR=$TENSORBOARD_DIR
mkdir -p "$TENSORBOARD_DIR"
echo "##### TensorBoard logs will be saved to: $TENSORBOARD_DIR #####"


# 运行训练
python3 -m rollout.main_qmix_rollout \
 data.train_files=$HOME/data/$DATASET/train.parquet \
 data.val_files=$HOME/data/$DATASET/test.parquet \
 data.train_batch_size=128 \
 data.max_prompt_length=5120 \
 data.max_response_length=2048 \
 data.filter_overlong_prompts=True \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
 actor_rollout_ref.rollout.n=5 \
 actor_rollout_ref.rollout.dtype=bfloat16 \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 critic.optim.lr=1e-5 \
 critic.ppo_micro_batch_size_per_gpu=10 \
 algorithm.adv_estimator='grpo' \
 algorithm.use_kl_in_reward=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console','tensorboard'] \
 trainer.project_name=$WANDB_PROJECT \
 trainer.experiment_name=$WANDB_RUN_NAME \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=5 \
 trainer.test_freq=10 \
 trainer.total_epochs=10 \
 trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
 trainer.val_generations_to_log_to_wandb=50 \
 marl.name=$ALGORITHM_NAME \
 marl.turns=$TURNS \
 marl.td_mode=$TD_MODE \
 marl.rollout_mode=$ROLLOUT_MODE \
 marl.mixer="none" \
 marl.sum_reward=$SUM_REWARD \
 marl.cory_flip=$CORY_FLIP \
 marl.tensorboard_dir=$TENSORBOARD_DIR 2>&1 | tee verl_demo.log
