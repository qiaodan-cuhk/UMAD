# Retained `verl/` Custom Changes

This repository vendors `verl/` as a frozen runtime. The code was developed from
the `verl` 0.3 line, but the current tree also contains broader runtime drift
that does not map cleanly to `v0.3.0.post0` or `v0.3.0.post1`. See
`VERL_DIFF_AUDIT.md` for the baseline comparison.

This note documents the retained project-specific behavior that is expected by
the public UMAD/IPPO code path.

## Multi-Turn Loss Masks

Files:

- `verl/workers/actor/dp_actor.py`
- `verl/trainer/ppo/ray_trainer.py`
- `marl/utils/marl_utils.py`

Functionality:

- Multi-turn debate rollouts can carry a `loss_mask` tensor.
- The PPO actor loss and GRPO advantage path use this mask to select only the
  trainable response tokens.

Why it is retained:

- UMAD debate trajectories include context from multiple agents and turns.
- The loss should not be applied to prompt tokens, peer-agent context, padding,
  or other non-trainable regions.

## Rollout and Validation Dumps

Files:

- `verl/trainer/ppo/ray_trainer.py`
- `marl/ippo_trainer.py`
- `eval/debate_evaluator.py`

Functionality:

- `trainer.rollout_data_dir` controls training-generation dumps.
- `trainer.validation_data_dir` controls validation-generation dumps in the
  single-agent and debate evaluation paths.
- The UMAD trainer writes debate rollouts under
  `<rollout_data_dir>/<experiment_name>/`.

Why it is retained:

- Academic reporting requires inspectable trajectories, per-sample scores, and
  failure cases.
- The public evaluation script exposes these dumps as part of its output
  contract.

Path policy:

- Launchers inject output paths through environment variables and Hydra
  overrides.
- No personal absolute output path is required.

## Boxed Math Reward Routing

Files:

- `verl/utils/reward_score/__init__.py`
- `examples/data_preprocess/process_all_math.py`

Functionality:

- Math-style datasets using boxed final answers are routed to the math reward
  parser.
- `DigitalLearningGmbH/MATH-lighteval` is supported by the public math
  preprocessing path.

Why it is retained:

- The release targets math debate training and evaluation.
- The generated parquet data stores the reward source expected by this routing.

## TensorBoard Directory Injection

Files:

- `verl/utils/tracking.py`
- `bash/ippo_grpo.sh`
- `bash/run_single_agent_train.sh`

Functionality:

- TensorBoard output can be configured by `marl.tensorboard_dir`,
  `trainer.tensorboard_dir`, or the `TENSORBOARD_DIR` environment variable.
- If none is supplied, the runtime falls back to the relative
  `tensorboard_log` directory.

Why it is retained:

- Open-source users should be able to choose the log directory from the launcher
  or environment.
- This avoids hard-coded personal paths in logging code.

## Confidence Gain Path

The previous token-confidence gain path has been removed from this release. The
current UMAD/IPPO recipe uses log-probability gain and influence intrinsic
reward instead. Confidence tensors are therefore not part of the public actor
log-prob API.

## Maintenance Rule

Treat `verl/` as frozen unless a change is required for the retained behavior
above. For runtime cleanup, first identify a closer upstream baseline, then keep
the custom behavior in this document covered by smoke tests.
