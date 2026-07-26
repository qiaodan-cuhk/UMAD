![MAVERL collaborators logo](logo1.png)

# MAVERL

**MAVERL** is a multi-agent reinforcement learning training codebase for
language-model debate and collaborative reasoning. It is built on top of a
vendored `verl` runtime and focuses on the UMAD/IPPO training recipe used in
our math-reasoning experiments.

This project originated from a collaborative effort by researchers and interns
from **The Chinese University of Hong Kong, Shenzhen (CUHKSZ)** and the
**ByteDance ByteBrain** team.

## Key Features

- **Multi-agent language RL:** Co-train multiple LLM agents in language-based
  interaction trajectories instead of treating alignment as a single-agent
  problem.
- **Heterogeneous agents:** Configure each agent with its own base model,
  tokenizer, processor, and rollout seed.
- **UMAD/IPPO training:** Use IPPO with trajectory GRPO rollouts,
  log-probability/NLL advantage gain, and the UMAD influence intrinsic reward.
- **Scalable runtime:** Reuse `verl` Ray, FSDP, and vLLM infrastructure for
  distributed actor updates and high-throughput generation.
- **Inspectable experiments:** Dump training and validation debate trajectories
  as JSONL/CSV files for academic analysis and error inspection.

## Core Idea

MAVERL studies **diversity and emergence in multi-agent verbal reinforcement
learning**:

1. **Diversity:** Co-training LLM agents with different model priors or rollout
   seeds helps reduce single-policy mode collapse.
2. **Emergence:** Multi-turn verbal interaction can encourage debate,
   verification, correction, and collaboration.
3. **Multi-agent language RL:** Single-agent RLHF/RLAIF algorithms such as
   PPO/GRPO can be extended to multi-agent decision-making with agent-level
   credit assignment and team reward shaping.

## Repository Scope

This open-source release is intentionally focused. It keeps the code needed for
the UMAD/IPPO paper path and removes older experimental algorithms, notebooks,
and upstream `verl` example suites from the outer project tree.

Main retained entry points:

- `bash/ippo_grpo.sh`: main two-agent UMAD/IPPO training launcher.
- `bash/run_single_agent_train.sh`: single-agent GRPO baseline launcher.
- `bash/run_single_agent_ckpt_eval.sh`: single-agent checkpoint validation.
- `eval/run_debate_eval.sh`: standalone multi-turn debate evaluation.
- `examples/data_preprocess/gsm8k.py`: GSM8K parquet preprocessing.
- `examples/data_preprocess/process_all_math.py`: math evaluation parquet
  preprocessing.

## Code Structure

```text
maverl/
|-- bash/                         # Public training and evaluation launchers
|-- eval/                         # Standalone debate evaluation
|-- examples/data_preprocess/     # Dataset preprocessing scripts
|-- marl/                         # MAVERL multi-agent training code
|   |-- config/                   # Main UMAD/IPPO Hydra config
|   |-- controllers/              # Heterogeneous multi-agent controller
|   |-- learners/                 # IPPO learner and UMAD reward/advantage logic
|   |-- modules/agents/           # Per-agent PPO wrapper
|   |-- runners/                  # Single-turn and trajectory-GRPO rollout logic
|   `-- utils/                    # Dataset, mask, reward, and role utilities
|-- verl/                         # Frozen vendored runtime
|-- CODE_STRUCTURE.md             # Detailed code map
|-- VERL_CUSTOM_CHANGES.md        # Retained project-specific verl changes
`-- VERL_DIFF_AUDIT.md            # Baseline audit against upstream verl 0.3
```

The primary training flow is:

```text
bash/ippo_grpo.sh
  -> python -m marl.ippo_main
  -> RayIPPOTrainer
  -> BasicMAC_Hetero
  -> Multi_Turn_Runner_Traj_GRPO
  -> RayIPPOLearner
  -> per-agent PPO workers in vendored verl/
```

For a fuller map of the project, see `CODE_STRUCTURE.md`.

## Frozen `verl` Runtime

The `verl/` directory is treated as a frozen vendored runtime for this release.
It contains project-specific low-level changes in Ray, rollout, and FSDP paths.
Do not simplify or refactor it as part of ordinary open-source cleanup.

Important retained runtime changes include:

- multi-turn `loss_mask` support for actor loss and GRPO advantage calculation;
- rollout and validation generation dump hooks;
- boxed-answer math reward routing;
- TensorBoard log directory injection through `marl.tensorboard_dir`,
  `trainer.tensorboard_dir`, or `TENSORBOARD_DIR`.

See `VERL_CUSTOM_CHANGES.md` for the functional summary and
`VERL_DIFF_AUDIT.md` for the upstream comparison.

## Installation

MAVERL is intended for CUDA GPU machines with Ray, PyTorch, and vLLM. The
default public launcher uses vLLM rollouts, so install the vLLM extra before
running training.

The recommended path for a fresh CUDA environment is:

```bash
conda create -n maverl python=3.10 -y
conda activate maverl

bash install_env.sh
```

For a manual installation, install a CUDA-compatible PyTorch build first, then
install this repository and the vLLM/math extras:

```bash
python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -m pip install -r requirements.txt
python -m pip install -e ".[vllm,math]"
```

`install_env.sh` records the environment path used for this release
(PyTorch 2.6.0, Ray 2.45.0, vLLM 0.8.2). If you manage PyTorch/vLLM versions
manually, keep them compatible with your CUDA driver and cluster image.

## Data

Prepare math-style parquet files with the provided preprocessors:

```bash
python examples/data_preprocess/gsm8k.py --local_dir ./data/gsm8k
python examples/data_preprocess/process_all_math.py --local_dir ./data/math500
```

The GSM8K script creates `./data/gsm8k/train.parquet` and
`./data/gsm8k/test.parquet`, which is the easiest public smoke-test dataset.
The combined math evaluation script creates `./data/math500/test.parquet`,
`test_aime.parquet`, `test_general.parquet`, and a manifest.

The paper-aligned default training launcher expects:

- training file: `./data/math/train.parquet`
- validation file: `./data/math500/test.parquet`

Prepare `./data/math/train.parquet` in the same schema if you have the original
training mixture, or override the paths with `DATA_ROOT`, `DATASET`,
`VAL_DATASET`, `TRAIN_FILE`, and `VAL_FILE`.

## Model Preparation

Download model checkpoints to a local directory, for example:

```bash
export MODEL_ROOT=$HOME/models

huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir $MODEL_ROOT/Qwen2.5-3B-Instruct \
  --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir $MODEL_ROOT/Qwen3-4B-Instruct-2507 \
  --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir $MODEL_ROOT/Qwen2.5-0.5B-Instruct \
  --local-dir-use-symlinks False
```

For a faster smoke run, you can point both agents to any smaller local
Hugging Face causal LM that vLLM supports, for example
`Qwen/Qwen2.5-0.5B-Instruct`.

For gated models, authenticate with your Hugging Face account before
downloading.

## Training

Launch the default two-agent UMAD/IPPO training path:

```bash
export MAVERL_EXP_ROOT=/path/to/results
export DATA_ROOT=$PWD/data
export MODEL_ROOT=/path/to/models
export AGENT0_MODEL_PATH=$MODEL_ROOT/Qwen2.5-3B-Instruct
export AGENT1_MODEL_PATH=$MODEL_ROOT/Qwen3-4B-Instruct-2507
export EXP_GROUP=umad

bash bash/ippo_grpo.sh
```

For a smaller first run using only the public GSM8K preprocessor output, point
both train and validation files to GSM8K and reduce the batch sizes:

```bash
export MAVERL_EXP_ROOT=$PWD/outputs
export DATA_ROOT=$PWD/data
export DATASET=gsm8k
export VAL_DATASET=gsm8k
export MODEL_ROOT=$HOME/models
export AGENT0_MODEL_PATH=$MODEL_ROOT/Qwen2.5-0.5B-Instruct
export AGENT1_MODEL_PATH=$MODEL_ROOT/Qwen2.5-0.5B-Instruct
export N_GPUS_PER_NODE=1
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=4
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export PROMPT_LEN=1024
export RESP_LEN=512
export ROLLOUT_N=2
export TOTAL_TRAINING_STEPS=2

bash bash/ippo_grpo.sh
```

The smoke run checks the full data, rollout, reward, advantage, logging, and
checkpoint path. Increase model size, GPU count, sequence length, rollout count,
and batch sizes for paper-scale experiments.

Useful overrides:

- `TRAIN_BATCH_SIZE`
- `PROMPT_LEN`
- `RESP_LEN`
- `PPO_MINI_BATCH_SIZE`
- `PPO_MICRO_BATCH_SIZE_PER_GPU`
- `ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`
- `REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`
- `CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU`
- `ROLLOUT_N`
- `TOTAL_EPOCHS`
- `TOTAL_TRAINING_STEPS`
- `SAMPLE_RATIO`
- `SAVE_FREQ`
- `TEST_FREQ`
- `VAL_DATASET`
- `AGENT0_MODEL_PATH`
- `AGENT1_MODEL_PATH`
- `TENSORBOARD_DIR`

The main UMAD switches live under the `marl` section of
`marl/config/ippo_trainer.yaml`. The release defaults use IPPO with
sum-aggregated dense team reward, log-probability advantage gain, and the UMAD
influence intrinsic reward.

The default UMAD recipe is enabled by the following two switches:

- `marl.use_log_prob_gain=True`: applies the log-probability/NLL advantage gain.
- `marl.use_influence_intrinsic_reward=True`: adds the UMAD influence intrinsic
  reward between agents.

The main strength knobs are:

- `marl.log_prob_ratio_alpha=0.25`: controls the exponential strength of the
  NLL-based advantage gain. Larger values make low-NLL responses receive
  stronger positive reweighting and high-NLL responses receive stronger
  downweighting.
- `marl.influence_intrinsic_reward_strength=0.2`: controls the signed intrinsic
  reward added when one agent improves or hurts the other agents' subsequent
  rewards. Setting it to `0` removes this reward contribution.

## Single-Agent Baseline

Run a single-agent GRPO baseline through the vendored `verl` trainer:

```bash
export DATA_ROOT=$PWD/data
export MODEL_NAME=/path/to/model_or_checkpoint

bash bash/run_single_agent_train.sh
```

Validate a checkpoint:

```bash
export CKPT_PATH=/path/to/checkpoint/global_step_100
bash bash/run_single_agent_ckpt_eval.sh
```

## Debate Evaluation

Run standalone multi-turn debate evaluation:

```bash
export EVAL_FILE=$PWD/data/math500/test.parquet
export AGENT0_MODEL=/path/to/agent0_or_checkpoint
export AGENT1_MODEL=/path/to/agent1_or_checkpoint

bash eval/run_debate_eval.sh
```

Evaluation outputs are written under `${RESULTS_ROOT}/eval/debate_rollouts` by
default and include:

- `debate_rollouts.jsonl`: per-sample prompts, responses, scores, and
  diagnostics;
- `debate_scores.csv`: compact final-turn agent/team scores;
- `summary.json`: aggregate score statistics.

## Outputs and Logging

The launcher scripts route output directories through environment variables:

- `RESULTS_ROOT` or `MAVERL_EXP_ROOT`: base experiment directory;
- `RUN_ROOT`: per-run output directory;
- `CKPT_DIR`: checkpoints;
- `ROLLOUT_DATA_DIR`: training rollout dumps;
- `VALIDATION_DATA_DIR`: validation dumps;
- `TENSORBOARD_DIR`: TensorBoard event files.

No user-specific absolute output path is required.

## Notes for Developers

- Treat `verl/` as frozen unless a runtime change is required and tested.
- Keep public examples centered on UMAD/IPPO, single-agent GRPO, math
  preprocessing, and debate evaluation.
- Keep comments and docs in English.
- Avoid adding hard-coded local paths, private tokens, or machine-specific
  cluster assumptions.

## Citation and Acknowledgement

If you find this project helpful in your research, please cite:

```bibtex
@article{qiao2026epistemic,
  title={Epistemic Gain, Aleatoric Cost: Uncertainty Decomposition in Multi-Agent Debate for Math Reasoning},
  author={Qiao, Dan and Chen, Binbin and Cai, Fengyu and Chen, Jianlong and Li, Wenhao and Jiang, Fuxin and Chen, Zuzhi and Zha, Hongyuan and Zhang, Tieying and Wang, Baoxiang},
  journal={CoRR},
  volume={abs/2603.01221},
  year={2026},
  doi={10.48550/arXiv.2603.01221},
  url={https://arxiv.org/abs/2603.01221},
  eprint={2603.01221},
  archivePrefix={arXiv},
  primaryClass={cs.MA}
}
```

MAVERL is inspired by and builds on ideas and infrastructure from `verl`,
MARFT, REMA, MARTI, PyMARL, PyMARL2, and EPyMARL. We thank the open-source
community and our advisors for their contributions and guidance.
