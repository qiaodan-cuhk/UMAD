#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for checkpoint evaluation (validation-only first).
# Usage:
#   CKPT_PATH=/path/to/checkpoint_dir bash bash/run_single_agent_ckpt_eval.sh
#   CKPT_PATH=/path/to/checkpoint_dir bash bash/run_single_agent_ckpt_eval.sh data.val_files=/path/to/eval.parquet

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAVERL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FILE_ROOT="${FILE_ROOT:-$MAVERL_DIR}"
RESULTS_ROOT="${RESULTS_ROOT:-${MAVERL_EXP_ROOT:-$MAVERL_DIR/outputs}}"
CKPT_ROOT="${CKPT_ROOT:-$RESULTS_ROOT/checkpoints}"
DATA_ROOT="${DATA_ROOT:-$FILE_ROOT/data}"
DATASET="${DATASET:-math}"
VAL_DATASET="${VAL_DATASET:-math500}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/$DATASET/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_ROOT/$VAL_DATASET/test.parquet}"
PROJECT_NAME="${PROJECT_NAME:-maverl_single_agent_eval}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-single_agent_ckpt_eval_$(date +%m%d-%H%M)}"
N_GPUS="${N_GPUS:-1}"
NNODES="${NNODES:-1}"
CKPT_PATH="${CKPT_PATH:-}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Error: CKPT_PATH is required."
  echo "Example: CKPT_PATH=/path/to/checkpoint_dir bash bash/run_single_agent_ckpt_eval.sh"
  exit 1
fi

echo "FILE_ROOT=${FILE_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "CKPT_ROOT=${CKPT_ROOT}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "VAL_FILE=${VAL_FILE}"

test -f "$TRAIN_FILE"
test -f "$VAL_FILE"

python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=256 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  critic.ppo_micro_batch_size_per_gpu=16 \
  algorithm.adv_estimator='grpo' \
  trainer.logger=['console'] \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes="$NNODES" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$CKPT_PATH" \
  trainer.val_before_train=True \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  trainer.total_epochs=0 \
  "$@"
