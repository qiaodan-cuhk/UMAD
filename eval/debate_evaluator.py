from __future__ import annotations

import csv
import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.reward import compute_reward
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

from marl.modules.agents.ppo_agent import ResourcePoolManager


class DebateEvaluator:
    """Run multi-agent debate rollouts without initializing learner/training state."""

    def __init__(
        self,
        config,
        num_agents: int,
        tokenizer_list,
        role_worker_mapping,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls,
        processor_list=None,
        val_reward_fn_list=None,
        val_dataset=None,
        collate_fn=None,
        device_name: str = "cuda",
    ):
        self.config = config
        self.num_agents = num_agents
        self.tokenizer_list = tokenizer_list
        self.processor_list = processor_list
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.val_reward_fn_list = val_reward_fn_list
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.device_name = device_name
        self.turns = int(config.marl.turns)

        from marl.controllers import REGISTRY as MAC_REGISTRY
        from marl.runners import REGISTRY as RUNNER_REGISTRY

        mac_cls = MAC_REGISTRY[self.config.marl.mac_cls]
        runner_cls = RUNNER_REGISTRY[self.config.marl.runner_cls]

        extra_params = {
            "use_critic": False,
            "use_reference_policy": False,
            "use_rm": False,
            "ref_in_actor": False,
            "kl_ctrl_in_reward": None,
            "hybrid_engine": self.config.actor_rollout_ref.hybrid_engine,
        }
        self.mac = mac_cls(
            config,
            num_agents,
            tokenizer_list,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor_list,
            **extra_params,
        )
        self.runner = runner_cls(
            config,
            self.mac,
            num_agents,
            reward_fn_list=val_reward_fn_list,
            device_name=device_name,
        )
        self._create_dataloader()

    def _create_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("val_dataset is required for debate evaluation")
        batch_size = self.config.data.get("val_batch_size", None)
        if batch_size is None:
            batch_size = min(len(self.val_dataset), self.config.data.train_batch_size)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        for agent in self.mac.agents:
            agent.init_workers()
        self.runner.init_workers()
        self._maybe_load_checkpoint()

    def _maybe_load_checkpoint(self):
        ckpt_path = self.config.trainer.get("resume_from_path", None)
        if not ckpt_path:
            return

        ckpt_path = os.path.abspath(os.path.expanduser(ckpt_path))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"resume_from_path does not exist: {ckpt_path}")

        latest = find_latest_ckpt_path(ckpt_path)
        if latest is not None:
            ckpt_path = latest

        print(f"Loading debate eval checkpoint from {ckpt_path}")
        for agent_id, agent in enumerate(self.mac.agents):
            agent_dir = self._resolve_agent_ckpt_dir(ckpt_path, agent_id)
            agent._load_checkpoint(agent_dir, del_local_after_load=False)

    @staticmethod
    def _resolve_agent_ckpt_dir(ckpt_path: str, agent_id: int) -> str:
        candidates = [
            os.path.join(ckpt_path, f"agent_{agent_id}"),
            os.path.join(ckpt_path, str(agent_id)),
            ckpt_path,
        ]
        for candidate in candidates:
            if os.path.exists(os.path.join(candidate, "actor")):
                return candidate
        raise FileNotFoundError(
            f"Could not find actor checkpoint for agent_{agent_id} under {ckpt_path}"
        )

    def evaluate(self):
        output_dir = self._build_output_dir()
        jsonl_path = os.path.join(output_dir, "debate_rollouts.jsonl")
        csv_path = os.path.join(output_dir, "debate_scores.csv")
        summary_path = os.path.join(output_dir, "summary.json")

        max_batches = OmegaConf.select(self.config, "eval.max_batches")
        max_batches = None if max_batches in (None, "null") else int(max_batches)

        all_rows = []
        aggregate = {f"agent_{i}": [] for i in range(self.num_agents)}
        aggregate["team_reward"] = []
        aggregate_by_subset = {}

        with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
            iterator = tqdm(self.val_dataloader, desc="debate-eval")
            for batch_idx, test_data in enumerate(iterator):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                batch_records = self._run_batch(test_data)
                for record in batch_records:
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    score_row = self._record_to_score_row(record)
                    all_rows.append(score_row)
                    for key in aggregate:
                        aggregate[key].append(float(score_row[key]))
                    subset = score_row.get("subset") or score_row.get("data_source") or "unknown"
                    if subset not in aggregate_by_subset:
                        aggregate_by_subset[subset] = {key: [] for key in aggregate}
                    for key in aggregate:
                        aggregate_by_subset[subset][key].append(float(score_row[key]))

        self._write_csv(csv_path, all_rows)
        summary = self._build_summary(
            aggregate,
            len(all_rows),
            output_dir,
            aggregate_by_subset=aggregate_by_subset,
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"Debate eval rollouts: {jsonl_path}")
        print(f"Debate eval scores: {csv_path}")
        print(f"Debate eval summary: {summary_path}")
        return summary

    def _run_batch(self, test_data) -> list[dict]:
        world_size = self.mac.agents[0].actor_rollout_wg.world_size
        repeat_n = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        multi_turn_batches: Dict[str, Dict[str, DataProto]] = {}
        pad_sizes: Dict[str, int] = {}

        for turn_idx in range(self.turns):
            multi_turn_batches[f"turn_{turn_idx}"] = {}
            for agent_id in range(self.num_agents):
                agent_key = f"agent_{agent_id}"
                batch = DataProto.from_single_dict(deepcopy(test_data[agent_key]))
                if repeat_n > 1:
                    batch = batch.repeat(repeat_times=repeat_n, interleave=True)
                padded, pad_size = pad_dataproto_to_divisor(batch, world_size)
                pad_sizes[agent_key] = pad_size
                multi_turn_batches[f"turn_{turn_idx}"][agent_key] = padded

        gen_batch_all = {}
        for agent_id in range(self.num_agents):
            agent_key = f"agent_{agent_id}"
            batch = multi_turn_batches["turn_0"][agent_key]
            if "multi_modal_inputs" in batch.non_tensor_batch:
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=[
                        "raw_prompt_ids",
                        "multi_modal_data",
                        "multi_modal_inputs",
                    ],
                )
            else:
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer_list[agent_id].eos_token_id,
                "pad_token_id": self.tokenizer_list[agent_id].pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            gen_batch_all[agent_key] = gen_batch

        metrics = {}
        output_padded = self.runner.rollout_traj_grpo(
            gen_batch_all,
            multi_turn_batches,
            metrics,
            turn=self.turns,
            validate=True,
        )

        output = {f"turn_{turn_idx}": {} for turn_idx in range(self.turns)}
        for turn_idx in range(self.turns):
            for agent_id in range(self.num_agents):
                agent_key = f"agent_{agent_id}"
                output[f"turn_{turn_idx}"][agent_key] = unpad_dataproto(
                    output_padded[f"turn_{turn_idx}"][agent_key],
                    pad_size=pad_sizes[agent_key],
                )

        batch_size = len(output["turn_0"]["agent_0"].batch)
        dialogues, rewards = self._get_dialogue_data(output, batch_size)
        records = []
        indices = output["turn_0"]["agent_0"].non_tensor_batch.get("index")
        data_sources = output["turn_0"]["agent_0"].non_tensor_batch.get("data_source")
        ground_truths = output["turn_0"]["agent_0"].non_tensor_batch.get("reward_model")
        extra_infos = output["turn_0"]["agent_0"].non_tensor_batch.get("extra_info")

        for sample_idx in range(batch_size):
            extra_info = self._jsonable_array_value(extra_infos, sample_idx)
            records.append(
                {
                    "sample_index": self._jsonable_array_value(indices, sample_idx),
                    "data_source": self._jsonable_array_value(data_sources, sample_idx),
                    "subset": self._subset_from_extra_info(extra_info),
                    "extra_info": extra_info,
                    "reward_model": self._jsonable_array_value(ground_truths, sample_idx),
                    "turns": self.turns,
                    "val_repeat_n": repeat_n,
                    "scores": rewards[sample_idx],
                    "diagnostics": self._diagnostics_for_sample(output, sample_idx),
                    "dialogue": dialogues[sample_idx],
                }
            )
        return records

    def _get_dialogue_data(self, multi_turn_batches, batch_size: int):
        dialogues = [{} for _ in range(batch_size)]
        rewards = [{} for _ in range(batch_size)]

        for turn_idx in range(self.turns):
            turn_key = f"turn_{turn_idx}"
            turn_batches = multi_turn_batches[turn_key]
            prompts = self.tokenizer_list[0].batch_decode(
                turn_batches["agent_0"].batch["prompts"],
                skip_special_tokens=True,
            )
            responses = {}
            for agent_id in range(self.num_agents):
                agent_key = f"agent_{agent_id}"
                responses[agent_key] = self.tokenizer_list[agent_id].batch_decode(
                    turn_batches[agent_key].batch["responses"],
                    skip_special_tokens=True,
                )

            for sample_idx in range(batch_size):
                turn_dialogue = [{"role": "user", "content": prompts[sample_idx]}]
                for agent_id in range(self.num_agents):
                    agent_key = f"agent_{agent_id}"
                    turn_dialogue.append(
                        {"role": agent_key, "content": responses[agent_key][sample_idx]}
                    )
                dialogues[sample_idx][turn_key] = turn_dialogue

        last_turn = multi_turn_batches[f"turn_{self.turns - 1}"]
        reward_tensors = self._cal_origin_reward(last_turn)
        for sample_idx in range(batch_size):
            team_total = 0.0
            for agent_id in range(self.num_agents):
                agent_key = f"agent_{agent_id}"
                agent_reward = float(reward_tensors[agent_key][sample_idx].sum().item())
                rewards[sample_idx][agent_key] = agent_reward
                team_total += agent_reward
            rewards[sample_idx]["team_reward"] = team_total
        return dialogues, rewards

    def _cal_origin_reward(self, agent_batches):
        reward_tensor_all = {}
        for agent_id, agent in enumerate(self.mac.agents):
            agent_key = f"agent_{agent_id}"
            agent_batch = agent_batches[agent_key]
            if agent.use_rm:
                reward_tensor = agent.rm_wg.compute_rm_score(agent_batch)
                agent_batch = agent_batch.union(reward_tensor)
            reward_tensor, _ = compute_reward(
                agent_batch,
                self.val_reward_fn_list[agent_id],
            )
            reward_tensor_all[agent_key] = reward_tensor
        return reward_tensor_all

    def _diagnostics_for_sample(self, multi_turn_batches, sample_idx: int):
        diagnostics = {}
        max_response_length = int(self.config.data.max_response_length)
        for turn_idx in range(self.turns):
            turn_key = f"turn_{turn_idx}"
            diagnostics[turn_key] = {}
            for agent_id in range(self.num_agents):
                agent_key = f"agent_{agent_id}"
                batch = multi_turn_batches[turn_key][agent_key]
                response_mask = batch.batch["response_mask"][sample_idx].bool()
                token_len = int(response_mask.sum().item())
                response_ids = batch.batch["responses"][sample_idx][response_mask]
                text = self.tokenizer_list[agent_id].decode(
                    response_ids,
                    skip_special_tokens=True,
                )
                diagnostics[turn_key][agent_key] = {
                    "response_tokens": token_len,
                    "hit_max_response_length": token_len >= max_response_length,
                    "has_boxed": "\\boxed" in text,
                    "has_gsm8k_hash": "####" in text,
                }
        return diagnostics

    @staticmethod
    def _subset_from_extra_info(extra_info):
        if isinstance(extra_info, dict):
            return extra_info.get("subset") or extra_info.get("source_dataset")
        return None

    def _build_output_dir(self) -> str:
        base_dir = self.config.trainer.get("validation_data_dir", None)
        if base_dir is None:
            base_dir = self.config.trainer.get("rollout_data_dir", None)
        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), "eval_outputs")

        run_name = self.config.trainer.get("experiment_name", "debate_eval")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(os.path.abspath(os.path.expanduser(base_dir)), f"{run_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def _jsonable_array_value(array_like, index):
        if array_like is None:
            return None
        value = array_like[index]
        return DebateEvaluator._to_jsonable(value)

    @staticmethod
    def _to_jsonable(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return DebateEvaluator._to_jsonable(value.tolist())
        if isinstance(value, dict):
            return {str(k): DebateEvaluator._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [DebateEvaluator._to_jsonable(v) for v in value]
        return value

    @staticmethod
    def _record_to_score_row(record: dict) -> dict:
        row = {
            "sample_index": record["sample_index"],
            "data_source": record["data_source"],
            "subset": record.get("subset"),
            "source_dataset": None,
            "team_reward": record["scores"]["team_reward"],
        }
        extra_info = record.get("extra_info")
        if isinstance(extra_info, dict):
            row["source_dataset"] = extra_info.get("source_dataset")
        for key, value in record["scores"].items():
            if key != "team_reward":
                row[key] = value
        return row

    @staticmethod
    def _write_csv(path: str, rows: list[dict]):
        if not rows:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _build_summary(
        aggregate: dict[str, list[float]],
        count: int,
        output_dir: str,
        aggregate_by_subset: Optional[dict[str, dict[str, list[float]]]] = None,
    ) -> dict:
        summary = {"num_samples": count, "output_dir": output_dir, "scores": {}}
        for key, values in aggregate.items():
            arr = np.array(values, dtype=np.float32)
            summary["scores"][key] = {
                "mean": float(arr.mean()) if len(arr) else None,
                "std": float(arr.std()) if len(arr) else None,
                "min": float(arr.min()) if len(arr) else None,
                "max": float(arr.max()) if len(arr) else None,
            }
        summary["by_subset"] = {}
        for subset, subset_aggregate in (aggregate_by_subset or {}).items():
            summary["by_subset"][subset] = {
                "num_samples": len(next(iter(subset_aggregate.values()), [])),
                "scores": {},
            }
            for key, values in subset_aggregate.items():
                arr = np.array(values, dtype=np.float32)
                summary["by_subset"][subset]["scores"][key] = {
                    "mean": float(arr.mean()) if len(arr) else None,
                    "std": float(arr.std()) if len(arr) else None,
                    "min": float(arr.min()) if len(arr) else None,
                    "max": float(arr.max()) if len(arr) else None,
                }
        return summary
