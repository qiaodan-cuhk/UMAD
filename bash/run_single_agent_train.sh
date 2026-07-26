#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for single-agent GRPO/PPO-style training.
# Usage:
#   bash bash/run_single_agent_train.sh

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAVERL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FILE_ROOT="${FILE_ROOT:-$MAVERL_DIR}"
DATA_ROOT="${DATA_ROOT:-$FILE_ROOT/data}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
MODEL_NAME="${MODEL_NAME:-$MODEL_ROOT/Qwen3-4B-Instruct-2507}"
DATASET="${DATASET:-math}"
VAL_DATASET="${VAL_DATASET:-math500}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/$DATASET/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_ROOT/$VAL_DATASET/test.parquet}"
PROJECT_NAME="${PROJECT_NAME:-maverl_single_agent}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-single_agent_train_$(date +%m%d-%H%M)}"
RESULTS_ROOT="${RESULTS_ROOT:-${MAVERL_EXP_ROOT:-${MODEL_OUTPUT_DIR:-$MAVERL_DIR/outputs}}}"
EXP_GROUP="${EXP_GROUP:-single_agent_math500}"
RUN_ROOT="${RUN_ROOT:-$RESULTS_ROOT/$EXP_GROUP/$EXPERIMENT_NAME}"
CKPT_DIR="${CKPT_DIR:-$RUN_ROOT/ckpts}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-null}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$RUN_ROOT/validation}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-$RESULTS_ROOT/tb_runs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$TENSORBOARD_ROOT/$EXPERIMENT_NAME}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train.log}"
N_GPUS="${N_GPUS:-${N_GPUS_PER_NODE:-${GPU_PER_NODE_COUNT:-8}}}"
NNODES="${NNODES:-${NODE_COUNT:-1}}"
TRAIN_BS="${TRAIN_BS:-1024}"
PROMPT_LEN="${PROMPT_LEN:-2048}"
RESP_LEN="${RESP_LEN:-2048}"
ROLLOUT_N="${ROLLOUT_N:-5}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-20}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-20}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-20}"
CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU="${CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU:-20}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"

echo "FILE_ROOT=${FILE_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "VAL_FILE=${VAL_FILE}"
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR}"

test -f "$TRAIN_FILE"
test -f "$VAL_FILE"
test -d "$MODEL_NAME"
mkdir -p "$RUN_ROOT" "$CKPT_DIR" "$VALIDATION_DATA_DIR" "$LOG_DIR" "$TENSORBOARD_DIR"
if [[ "$ROLLOUT_DATA_DIR" != "null" ]]; then
  mkdir -p "$ROLLOUT_DATA_DIR"
fi

python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.filter_overlong_prompts=True \
  data.train_batch_size="$TRAIN_BS" \
  data.max_prompt_length="$PROMPT_LEN" \
  data.max_response_length="$RESP_LEN" \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL_NAME" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
  actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
  actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P" \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  critic.model.path="$MODEL_NAME" \
  critic.optim.lr=1e-5 \
  critic.ppo_micro_batch_size_per_gpu="$CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU" \
  algorithm.adv_estimator='grpo' \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0 \
  trainer.logger=['console','tensorboard'] \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes="$NNODES" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  +marl.tensorboard_dir="$TENSORBOARD_DIR" \
  "$@" 2>&1 | tee "$LOG_FILE"
