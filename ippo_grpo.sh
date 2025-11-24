#!/bin/bash
# 异构相同tokenizer/processor模型，不同参数，多轮推理
# 这里kl in reward

# 设置环境变量
export PYTHONUNBUFFERED=1
export VERL_USE_MODELSCOPE=False  # 如果需要从modelscope下载模型
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LOG_FILE="verl_demo.log"

export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_MODE=offline


# 设置算法和config name
export CONFIG_NAME="ippo_trainer"  # ippo/grpo/remax

export ALGORITHM_NAME="ippo"   # ippo, vdn

# 设置是否使用 sum reward，默认不使用
SUM_REWARD=true  # 若要使用，改为 true
DATE_TIME=$(date +%m%d-%H%M)


DATASET="math"  # "math" "gsm8k"


AGG_MODE="sum"  # sum 代表加和，max代表取最大，dictator代表强模型,ind代表自己用自己的
TEAM_REWARD_TYPE="dense"  # dense/sparse/accumulative 

# 设置WandB相关变量
WANDB_PROJECT="ippo_grpo"
WANDB_RUN_NAME="Confidence-no_intrinsic_reward-no_gain_MATH_qwen2.5-3b_llama3.2-3b_0.1sample_tmp0.7_topp0.95_traj-grpo_IPPO_reward${AGG_MODE}_team-${TEAM_REWARD_TYPE}_${ALGORITHM_NAME}_bs64_p5k_r2k_2turn_$DATE_TIME"
VAL_BEFORE_TRAIN=False




ROLLOUT_DATA_DIR=/root/code/maverl/conf-gain-logs
echo "###### Running Test with $ALGORITHM_NAME and $CONFIG_NAME ######"
# 设置TensorBoard日志目录
TENSORBOARD_DIR="conf-gain-results/$WANDB_RUN_NAME"
export TENSORBOARD_DIR=$TENSORBOARD_DIR
mkdir -p "$TENSORBOARD_DIR"
echo "##### TensorBoard logs will be saved to: $TENSORBOARD_DIR #####"

# 数据集降采样到0.1比例
# rollout.log_prob_micro_batch_size_per_gpu=10 基本上对于5k+2k的长度，gpu接近74G，配合6G其他模型差不多80G

# 运行训练
python3 -m marl.ippo_main \
 data.train_files=$HOME/data/$DATASET/train.parquet \
 data.val_files=$HOME/data/$DATASET/test.parquet \
 data.train_batch_size=64 \
 data.max_prompt_length=5120 \
 data.max_response_length=2048 \
 data.filter_overlong_prompts=True \
 data.sample_ratio=0.1 \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
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
 trainer.val_before_train=$VAL_BEFORE_TRAIN \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=5 \
 trainer.test_freq=5 \
 trainer.total_epochs=10 \
 trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
 trainer.val_generations_to_log_to_wandb=50 \
 marl.name=$ALGORITHM_NAME \
 marl.mixer="none" \
 marl.sum_reward=$SUM_REWARD \
 marl.agg_mode=$AGG_MODE \
 marl.team_reward_type=$TEAM_REWARD_TYPE \
 marl.tensorboard_dir=$TENSORBOARD_DIR 2>&1 | tee verl_demo.log
