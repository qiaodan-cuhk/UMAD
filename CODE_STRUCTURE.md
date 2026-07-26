# Code Structure

This repository contains the MAVERL/UMAD/IPPO training code plus a frozen
vendored `verl` runtime.

## Main Entry Points

- `bash/ippo_grpo.sh`: two-agent UMAD/IPPO training launcher.
- `marl/ippo_main.py`: Hydra/Ray entry point used by the UMAD launcher.
- `bash/run_single_agent_train.sh`: single-agent GRPO baseline launcher through
  `verl.trainer.main_ppo`.
- `bash/run_single_agent_ckpt_eval.sh`: single-agent checkpoint validation.
- `eval/run_debate_eval.sh`: standalone multi-turn debate evaluation.

## Configuration

- `marl/config/ippo_trainer.yaml` is the primary public config.
- `data.*` controls parquet files, prompt/response lengths, batch sizes, and
  dataset sampling.
- `actor_rollout_ref.*`, `critic.*`, and `algorithm.*` follow the underlying
  `verl` PPO/GRPO configuration style.
- `trainer.*` controls logging, validation cadence, checkpointing, Ray resource
  shape, and rollout dump directories.
- `marl.*` controls multi-agent behavior such as agent count, debate turns,
  reward aggregation, log-probability gain, and influence intrinsic reward.

The public launcher scripts prefer environment variables for paths and common
scale parameters, then pass the resolved values into Hydra overrides.

## Data Flow

1. Data preprocessors create math-style parquet files:
   - `examples/data_preprocess/gsm8k.py`
   - `examples/data_preprocess/process_all_math.py`
2. `marl.utils.marl_dataset` and `marl.utils.marl_utils` build datasets,
   samplers, masks, and reward-related helpers.
3. `marl.ippo_main` builds per-agent tokenizers, processors, Ray workers, and
   reward managers.
4. `marl.ippo_trainer.RayIPPOTrainer` coordinates training, validation,
   checkpointing, metrics, and rollout dumps.
5. `marl.runners.multi_turn_traj_grpo` generates multi-turn debate
   trajectories for the GRPO/IPPO update path.
6. `marl.learners.ippo_learner` computes per-agent policy updates and UMAD
   reward/advantage shaping.

## `marl/` Package

- `controllers/`: multi-agent controller that owns heterogeneous agent
  interaction.
- `learners/`: IPPO learner and policy-gradient update logic.
- `modules/agents/`: PPO-style agent wrapper around the vendored runtime
  workers.
- `modules/critics/`, `modules/mixers/`, `modules/verifiers/`: retained module
  namespaces for critic, mixer, and verifier compatibility.
- `runners/`: single-turn and multi-turn rollout runners.
- `utils/`: dataset handling, buffer utilities, math/code helpers, and
  multi-turn tensor utilities.

## `verl/` Runtime

`verl/` is vendored and frozen for this release. It provides the Ray trainer,
FSDP workers, PPO/GRPO utilities, rollout backends, reward managers, tracking,
and model/runtime helpers used by both UMAD and single-agent launchers.

Project-specific retained behavior is documented in
`VERL_CUSTOM_CHANGES.md`. The broader upstream comparison is documented in
`VERL_DIFF_AUDIT.md`.

## Evaluation

- `eval/debate_eval_main.py` initializes the same config family as training,
  but runs validation-only debate generation.
- `eval/debate_evaluator.py` writes JSONL rollouts, CSV scores, and aggregate
  summaries.
- `eval/README.md` contains focused usage examples for debate evaluation.

## Outputs

The launchers route outputs through these roots:

- `RESULTS_ROOT` or `MAVERL_EXP_ROOT`: base directory for experiment outputs.
- `RUN_ROOT`: per-run output directory.
- `CKPT_DIR`: checkpoints.
- `ROLLOUT_DATA_DIR`: training rollout dumps.
- `VALIDATION_DATA_DIR`: single-agent validation dumps.
- `TENSORBOARD_DIR`: TensorBoard event files.

Defaults are relative to the repository or the selected results root, and can
be overridden by environment variables.

## Open-Source Maintenance Notes

- Keep the public path centered on UMAD/IPPO, single-agent GRPO baseline, math
  preprocessing, and debate evaluation.
- Avoid editing `verl/` during ordinary cleanup; document and test any runtime
  change separately.
- Keep comments in English and close to the code they clarify.
- Avoid hard-coded user-specific paths in configs, scripts, and logging code.
