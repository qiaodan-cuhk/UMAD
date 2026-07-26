# Debate Evaluation

This folder contains a standalone multi-turn debate evaluator for MAVERL.
It reuses the MARL dataset, heterogeneous tokenizers, `BasicMAC_Hetero`, and
`traj_grpo` rollout runner, but initializes only actor/rollout workers and does
not run PPO updates.

Typical usage:

```bash
EVAL_FILE=$PWD/data/math500/test.parquet \
CKPT_PATH=$PWD/outputs/ippo/<run>/ckpts/global_step_100 \
TURNS=5 \
PROMPT_LEN=5120 \
RESP_LEN=2048 \
VAL_N=1 \
bash eval/run_debate_eval.sh
```

Outputs are written under `${RESULTS_ROOT}/eval/debate_rollouts` by default:

- `debate_rollouts.jsonl`: full per-sample multi-turn prompts, responses, scores, and diagnostics.
- `debate_scores.csv`: compact per-sample final-turn agent and team rewards.
- `summary.json`: aggregate mean/std/min/max scores.

Useful overrides for AIME-style evaluation:

```bash
RESP_LEN=8192 PROMPT_LEN=8192 bash eval/run_debate_eval.sh
```

The evaluator records response token counts and whether a response reached
`data.max_response_length`, which helps separate answer-extraction failures from
length truncation.
