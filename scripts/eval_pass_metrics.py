import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step, dict_add_prefix

logger = logging.getLogger(__name__)


def _is_correct(sample: Any, reward_value: float) -> bool:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        acc = reward.get("acc")
        if acc is not None:
            return bool(acc)
        score = reward.get("score")
        if score is not None:
            return float(score) == 1.0
    return float(reward_value) == 1.0


def _pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1)))


def _sample_to_record(dataset_name: str, sample: Any, reward_value: float, correct: bool) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "index": getattr(sample, "index", None),
        "group_index": getattr(sample, "group_index", None),
        "reward": getattr(sample, "reward", None),
        "reward_value": reward_value,
        "correct": correct,
        "status": getattr(getattr(sample, "status", None), "value", str(getattr(sample, "status", ""))),
        "label": getattr(sample, "label", None),
        "response": getattr(sample, "response", ""),
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _write_outputs(rollout_id: int, per_dataset_records: dict[str, list[dict[str, Any]]], log_dict: dict[str, Any]) -> None:
    output_dir = os.environ.get("EVAL_OUTPUT_DIR")
    if not output_dir:
        return

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    metrics_path = path / f"eval_metrics_rollout{rollout_id}.json"
    metrics_path.write_text(json.dumps(log_dict, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    for dataset_name, records in per_dataset_records.items():
        samples_path = path / f"eval_samples_{dataset_name}_rollout{rollout_id}.jsonl"
        with samples_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _group_correctness(samples: list[Any], correct: list[bool], group_size: int) -> list[list[bool]]:
    groups_by_id: dict[Any, list[bool]] = {}
    for sample, ok in zip(samples, correct, strict=False):
        group_index = getattr(sample, "group_index", None)
        if group_index is None:
            groups_by_id = {}
            break
        groups_by_id.setdefault(group_index, []).append(ok)

    if groups_by_id:
        return list(groups_by_id.values())

    usable = (len(correct) // group_size) * group_size if group_size > 0 else 0
    return [correct[i : i + group_size] for i in range(0, usable, group_size)]


def log_eval_metrics(rollout_id, args, data, extra_metrics):
    from slime.ray.rollout import compute_metrics_from_samples

    log_dict = dict(extra_metrics or {})
    per_dataset_records: dict[str, list[dict[str, Any]]] = {}

    for dataset_name, info in data.items():
        rewards = [float(x) for x in info["rewards"]]
        samples = info.get("samples") or []
        correct = [_is_correct(sample, reward) for sample, reward in zip(samples, rewards, strict=False)]

        log_dict[f"eval/{dataset_name}"] = sum(rewards) / len(rewards) if rewards else 0.0
        if samples:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{dataset_name}/")

        if "truncated" in info:
            truncated = info["truncated"]
            log_dict[f"eval/{dataset_name}-truncated_ratio"] = sum(truncated) / len(truncated) if truncated else 0.0

        group_size = int(args.n_samples_per_eval_prompt)
        grouped = _group_correctness(samples, correct, group_size)
        counts = [sum(group) for group in grouped]

        accuracy = sum(correct) / len(correct) if correct else 0.0
        pass_at_1 = float(np.mean([_pass_at_k(len(group), c, 1) for group, c in zip(grouped, counts, strict=False)])) if counts else 0.0
        pass_at_5 = (
            float(np.mean([_pass_at_k(len(group), c, min(5, len(group))) for group, c in zip(grouped, counts, strict=False)]))
            if counts
            else 0.0
        )

        log_dict[f"eval/{dataset_name}/accuracy"] = accuracy
        log_dict[f"eval/{dataset_name}/pass@1"] = pass_at_1
        log_dict[f"eval/{dataset_name}/pass@5"] = pass_at_5
        log_dict[f"eval/{dataset_name}/num_prompts"] = len(grouped)
        log_dict[f"eval/{dataset_name}/num_samples"] = len(correct)

        per_dataset_records[dataset_name] = [
            _sample_to_record(dataset_name, sample, reward, ok)
            for sample, reward, ok in zip(samples, rewards, correct, strict=False)
        ]

    logger.info(f"eval {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")
    _write_outputs(rollout_id, per_dataset_records, log_dict)
    return True
