# NLL Branch Compatibility Review

Compared branches:

- Baseline runnable branch: `intern_maverl/nll-gain-fix` at `1fcd815`
- Current release branch at review start: `opensource_cleanup` at `34975e1`

## Result

The current release branch keeps the required UMAD/IPPO training path from the
NLL branch. The large deletion set is mostly old experiments, upstream examples,
non-core MARL algorithms, notebooks, docs, and tests.

Runtime/default-path issues found and fixed during the execution review:

- `verl/trainer/main_ppo.py` now restores `include_dashboard=False` in
  `ray.init`, matching the NLL branch and avoiding unnecessary Ray dashboard
  startup in the single-agent PPO/GRPO path.
- `marl/ippo_main.py` now also disables the Ray dashboard and passes
  `ray_init.num_cpus` into `ray.init`, matching the single-agent runtime
  convention.
- `marl/ippo_main.py` now registers reward-model workers with agent-specific
  `MARLRole` keys when `reward_model.enable=True`. The public default keeps
  `reward_model.enable=False`, but the non-default path is now consistent with
  the MARL role conversion logic.
- `marl/modules/agents/ppo_agent.py` now reports adaptive KL horizon errors via
  `algorithm.kl_ctrl.horizon` instead of the nonexistent `critic.kl_ctrl`.

## Required Functionality Retained

### UMAD/IPPO Training

Retained files:

- `bash/ippo_grpo.sh`
- `marl/ippo_main.py`
- `marl/config/ippo_trainer.yaml`
- `marl/ippo_trainer.py`
- `marl/controllers/basic_controller_ht.py`
- `marl/learners/ippo_learner.py`
- `marl/modules/agents/ppo_agent.py`
- `marl/runners/multi_turn_traj_grpo.py`
- `marl/utils/marl_dataset.py`
- `marl/utils/marl_utils.py`

Retained behavior:

- Heterogeneous two-agent setup.
- Per-agent tokenizer/model configuration.
- Trajectory GRPO rollout runner.
- IPPO learner registry path.
- Dense team reward aggregation.
- Log-probability/NLL advantage gain.
- UMAD influence intrinsic reward.
- Multi-turn response/loss mask handling.

### Single-Agent Baseline and Evaluation

Retained files:

- `bash/run_single_agent_train.sh`
- `bash/run_single_agent_ckpt_eval.sh`
- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/ray_trainer.py`
- `eval/run_debate_eval.sh`
- `eval/debate_eval_main.py`
- `eval/debate_evaluator.py`

Retained behavior:

- Single-agent GRPO/PPO baseline launcher.
- Checkpoint validation launcher.
- Standalone multi-turn debate evaluation.
- Validation and rollout JSONL/CSV dumps.

### Data and Reward Path

Retained files:

- `examples/data_preprocess/gsm8k.py`
- `examples/data_preprocess/process_all_math.py`
- `verl/utils/reward_score/__init__.py`
- `verl/utils/reward_score/math.py`

Retained behavior:

- GSM8K parquet preprocessing.
- MATH-500 / AIME / AMC-style math evaluation parquet preprocessing.
- Boxed final-answer reward routing for math sources.

### Vendored Runtime Support

Retained `verl/` functionality needed by the public path:

- Ray/FSDP actor, critic, reference policy, and rollout workers.
- vLLM rollout backend.
- SGLang and Megatron runtime code kept as frozen compatibility surface.
- TensorBoard path injection through `marl.tensorboard_dir`,
  `trainer.tensorboard_dir`, or `TENSORBOARD_DIR`.
- Multi-turn `loss_mask` actor-loss and GRPO advantage support.

## Intentional Removals

The following removed areas are not required by the current public UMAD/IPPO
recipe:

- `marl/learners/magrpo_learner.py`, `marl/magrpo_trainer.py`, and
  `bash/magrpo.sh`.
- QMix/VDN/MAPPO configs, learners, trainers, mixers, and launchers.
- LoRA-specific MARL agent prototypes.
- Confidence-gain training and analysis scripts.
- Old rollout experiments under `rollout/`.
- Ablation notebooks and draft PDFs.
- Upstream `verl` example zoo and test suite outside the frozen runtime.

## Static Checks

Checks performed:

- Diffed file additions, deletions, and modifications against
  `intern_maverl/nll-gain-fix`.
- Verified current public launcher Hydra override paths exist in
  `marl/config/ippo_trainer.yaml`.
- Verified direct config attribute paths used by the public `marl/` and
  `eval/` code are present in `marl/config/ippo_trainer.yaml`, excluding
  expected `.get(...)` calls.
- Ran an AST-based internal import existence check over `marl/`, `eval/`, and
  `verl/`.
- Ran `python -m py_compile` over all current Python files under `marl/`,
  `eval/`, `examples/`, and `verl/`.
- Scanned for stale core references to removed confidence gain APIs and removed
  algorithm entry points.

Results:

- No missing internal import modules were detected.
- All non-`+` Hydra overrides in public launchers map to current YAML keys.
- Python compilation passed. Only historical regex escape warnings in the
  vendored math reward parser were emitted.
- No stale references to `calculate_confidence`, `confidence_from_logits`,
  `use_confidence_gain`, or `use_relative_confidence_gain` remain in the public
  training path.

## Non-Blocking Notes

- The local ignored directory `maverl.egg-info/` may exist after editable
  installs. It is not tracked by git and is covered by `.gitignore`, so it will
  not be pushed.
- `marl/utils/multiturn_utils.py` imports `openai`, but that utility is not
  imported by the public training or evaluation entry points. This was already
  true in the NLL branch and is not a regression from cleanup. If this helper is
  later documented as public API, add `openai` as an optional dependency or move
  the import inside the OpenAI-backed functions.
- Full training smoke tests were not run in this workspace because the local
  environment does not provide the training stack such as `ray`.

## Conclusion

The current `opensource_cleanup` branch is not missing required UMAD/IPPO or
NLL-gain functionality relative to `intern_maverl/nll-gain-fix`. The execution
risks found during review were limited to runtime defaults or non-default
configuration paths and have been fixed.
