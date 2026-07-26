#!/bin/bash
set -euo pipefail
# Heterogeneous two-agent UMAD/IPPO training with shared tokenizer/processor format.
# KL regularization is applied through the reward path.

# Runtime environment
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
# export VERL_USE_MODELSCOPE=False  # Enable if models should be downloaded from ModelScope.
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAVERL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FILE_ROOT="${FILE_ROOT:-$MAVERL_DIR}"
DATA_ROOT="${DATA_ROOT:-$FILE_ROOT/data}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
AGENT0_MODEL_PATH="${AGENT0_MODEL_PATH:-$MODEL_ROOT/Qwen2.5-3B-Instruct}"
AGENT1_MODEL_PATH="${AGENT1_MODEL_PATH:-$MODEL_ROOT/Qwen3-4B-Instruct-2507}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${GPU_PER_NODE_COUNT:-8}}"
NNODES="${NNODES:-${NODE_COUNT:-1}}"

echo FILE_ROOT: $FILE_ROOT
echo DATA_ROOT: $DATA_ROOT
echo AGENT0_MODEL_PATH: $AGENT0_MODEL_PATH
echo AGENT1_MODEL_PATH: $AGENT1_MODEL_PATH
RESULTS_ROOT="${RESULTS_ROOT:-${MAVERL_EXP_ROOT:-${MODEL_OUTPUT_DIR:-$MAVERL_DIR/outputs}}}"
EXP_GROUP="${EXP_GROUP:-umad}"
CKPT_ROOT="${CKPT_ROOT:-$RESULTS_ROOT/$EXP_GROUP/checkpoints}"



# Algorithm and Hydra config
export CONFIG_NAME="ippo_trainer"
export ALGORITHM_NAME="ippo"

SUM_REWARD=true
DATE_TIME=$(date +%m%d-%H%M)


DATASET="${DATASET:-math}"  # training dataset: "math" "gsm8k"
VAL_DATASET="${VAL_DATASET:-math500}"  # validation dataset, defaults to MATH-500 for paper-aligned eval
TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/$DATASET/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_ROOT/$VAL_DATASET/test.parquet}"


AGG_MODE="sum"  # sum, max, dictator, or ind
TEAM_REWARD_TYPE="dense"  # dense/sparse/accumulative 

sample_ratio="${SAMPLE_RATIO:-0.3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PROMPT_LEN="${PROMPT_LEN:-5120}"
RESP_LEN="${RESP_LEN:-2048}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-5}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-10}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-10}"
CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU="${CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU:-10}"
ROLLOUT_N="${ROLLOUT_N:-5}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
SAVE_FREQ="${SAVE_FREQ:-40}"
TEST_FREQ="${TEST_FREQ:-20}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
TRAINING_LOG_NUMS="${TRAINING_LOG_NUMS:-5}"

PROJECT_NAME="uncertainty_mad"
RUN_NAME="${RUN_NAME:-Mean-PLL-Intrin_MATH_qwen2.5-3b_qwen2507_${sample_ratio}sample_val-${VAL_DATASET}_tmp0.7_topp0.95_traj-grpo_IPPO_reward${AGG_MODE}_team-${TEAM_REWARD_TYPE}_${ALGORITHM_NAME}_bs${TRAIN_BATCH_SIZE}_p${PROMPT_LEN}_r${RESP_LEN}_2turn_$DATE_TIME}"
VAL_BEFORE_TRAIN=True

RUN_ROOT="${RUN_ROOT:-$RESULTS_ROOT/$EXP_GROUP/$RUN_NAME}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$RUN_ROOT/rollouts}"
CKPT_DIR="${CKPT_DIR:-$RUN_ROOT/ckpts}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"


echo "###### Running Test with $ALGORITHM_NAME and $CONFIG_NAME ######"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-$RESULTS_ROOT/tb_runs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$TENSORBOARD_ROOT/$RUN_NAME}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train.log}"
# export TENSORBOARD_DIR=$TENSORBOARD_DIR
# mkdir -p "$TENSORBOARD_DIR"
echo "##### Results root: $RESULTS_ROOT #####"
echo "##### Run root: $RUN_ROOT #####"
echo "##### TensorBoard logs will be saved to: $TENSORBOARD_DIR #####"
echo "##### Train file: $TRAIN_FILE #####"
echo "##### Val file: $VAL_FILE #####"
echo "##### CKPT dir: $CKPT_DIR #####"
test -f "$TRAIN_FILE"
test -f "$VAL_FILE"
test -d "$AGENT0_MODEL_PATH"
test -d "$AGENT1_MODEL_PATH"
mkdir -p "$RUN_ROOT" "$CKPT_DIR" "$ROLLOUT_DATA_DIR" "$LOG_DIR" "$TENSORBOARD_DIR"

# With 5k prompt tokens and 2k response tokens, a rollout log-prob micro batch
# size of 10 is close to the memory limit of an 80GB GPU in the paper setup.

# Launch UMAD/IPPO training.
python3 -m marl.ippo_main \
 data.train_files="$TRAIN_FILE" \
 data.val_files="$VAL_FILE" \
 data.train_batch_size=$TRAIN_BATCH_SIZE \
 data.max_prompt_length=$PROMPT_LEN \
 data.max_response_length=$RESP_LEN \
 data.filter_overlong_prompts=True \
 data.sample_ratio=$sample_ratio \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 actor_rollout_ref.rollout.dtype=bfloat16 \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 critic.optim.lr=1e-5 \
 critic.ppo_micro_batch_size_per_gpu=$CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU \
 algorithm.adv_estimator='grpo' \
 algorithm.use_kl_in_reward=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console','tensorboard'] \
 trainer.default_local_dir="${CKPT_DIR:-$CKPT_ROOT/$RUN_NAME}" \
 trainer.project_name=$PROJECT_NAME \
 trainer.experiment_name=$RUN_NAME \
 trainer.val_before_train=$VAL_BEFORE_TRAIN \
 trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
 trainer.nnodes=$NNODES \
 trainer.save_freq=$SAVE_FREQ \
 trainer.test_freq=$TEST_FREQ \
 trainer.total_epochs=$TOTAL_EPOCHS \
 trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
 trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
 marl.name=$ALGORITHM_NAME \
 marl.mixer="none" \
 marl.sum_reward=$SUM_REWARD \
 marl.agg_mode=$AGG_MODE \
 marl.team_reward_type=$TEAM_REWARD_TYPE \
 +marl.training_log_nums=$TRAINING_LOG_NUMS \
 marl.agent_configs.agent_0.model.path="$AGENT0_MODEL_PATH" \
 marl.agent_configs.agent_1.model.path="$AGENT1_MODEL_PATH" \
 marl.tensorboard_dir=$TENSORBOARD_DIR \
 "$@" 2>&1 | tee "$LOG_FILE"
