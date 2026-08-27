"""Rollout and eval sample logging utilities.

Usage in training script:

    # Add log directory argument (path should include model name)
    LOG_ARGS=(
       --rollout-log-dir /mnt/data/zdm/output/021_rl/tir/logs/qwen3-4b-instruct-2507-notool
    )

    # In CUSTOM_ARGS
    CUSTOM_ARGS=(
        --custom-rollout-log-function-path slime.utils.rollout_logger.log_rollout_to_file
        --custom-eval-rollout-log-function-path slime.utils.rollout_logger.log_eval_rollout_to_file
    )

Output structure:
    ${rollout_log_dir}/
    ├── rollout_outputs/
    │   ├── rollout_0.jsonl
    │   └── ...
    └── eval_outputs/
        ├── eval_aime_0.jsonl
        └── ...
"""

import json
import os
import statistics
from typing import Any

# Global set to track saved files (avoid duplicates)
_saved_rollout_files: set[str] = set()
_saved_eval_files: set[str] = set()


def _get_output_base_dir(args) -> str:
    """Get output base directory from args.rollout_log_dir."""
    if hasattr(args, "rollout_log_dir") and args.rollout_log_dir:
        return args.rollout_log_dir
    else:
        return './'
    raise ValueError("--rollout-log-dir is required when using rollout_logger")


def _sample_to_dict(sample) -> dict[str, Any]:
    """Convert Sample object to dictionary for JSON serialization."""
    data = {
        "prompt": sample.prompt,
        "response": getattr(sample, "log_response", sample.response),
        "label": sample.label,
        "reward": sample.reward,
        "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
        "response_length": sample.response_length,
    }
    # Add tool_call_count if available
    if hasattr(sample, "tool_call_count"):
        data["tool_call_count"] = sample.tool_call_count
    # Add tool_return_length if available
    if hasattr(sample, "tool_return_length"):
        data["tool_return_length"] = sample.tool_return_length
    if getattr(sample, "metadata", None) and "compaction" in sample.metadata:
        data["compaction"] = sample.metadata["compaction"]
    # Add tokens (token ids)
    # if sample.tokens is not None:
    #     data["tokens"] = sample.tokens
    # # Add loss_mask
    # if sample.loss_mask is not None:
    #     data["loss_mask"] = sample.loss_mask
    # # Add rollout_log_probs
    # if sample.rollout_log_probs is not None:
    #     data["rollout_log_probs"] = sample.rollout_log_probs
    # Add metadata (includes messages, finish_reason, round_number, timestamp, etc.)
    if sample.metadata:
        metadata = dict(sample.metadata)
        # Extract messages from metadata for top-level access
        if "messages" in metadata:
            data["messages"] = metadata.pop("messages")
        # Extract finish_reason from metadata for top-level access
        if "finish_reason" in metadata:
            data["finish_reason"] = metadata.pop("finish_reason")
    return data

# def _sample_to_dict(sample) -> dict[str, Any]:
#     """Convert Sample object to dictionary for JSON serialization."""
#     data = {
#         "prompt": sample.prompt,
#         "response": sample.response,
#         "label": sample.label,
#         "reward": sample.reward,
#         "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
#         "response_length": sample.response_length,
#     }
#     # Add tool_call_count if available
#     if hasattr(sample, "tool_call_count"):
#         data["tool_call_count"] = sample.tool_call_count
#     # Add tool_return_length if available
#     if hasattr(sample, "tool_return_length"):
#         data["tool_return_length"] = sample.tool_return_length
#     return data


def _log_tool_call_stats(samples, rollout_extra_metrics: dict | None) -> None:
    """Log tool call count statistics to console.
    
    Note: Metrics are now logged via compute_metrics_from_samples.
    This function only prints to console for debugging.
    """
    tool_call_counts = [getattr(s, "tool_call_count", 0) for s in samples]
    
    if not tool_call_counts:
        return
    
    # Print to console only
    print(f"[rollout_logger] Tool call stats: "
          f"max={max(tool_call_counts)}, "
          f"min={min(tool_call_counts)}, "
          f"mean={statistics.mean(tool_call_counts):.2f}, "
          f"median={statistics.median(tool_call_counts)}")


def log_rollout_to_file(rollout_id: int, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Save training rollout samples to JSONL file.

    Args:
        rollout_id: Rollout batch ID
        args: Command line arguments
        samples: List of Sample objects
        rollout_extra_metrics: Extra metrics dict (will be modified in-place)
        rollout_time: Rollout time (unused)

    Returns:
        False to continue with default logging
    """
    global _saved_rollout_files

    # Generate output path
    output_dir = os.path.join(_get_output_base_dir(args), "rollout_outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"rollout_{rollout_id}.jsonl")

    # Avoid duplicate writes
    if output_file in _saved_rollout_files:
        # Still log tool call stats even if file already saved
        _log_tool_call_stats(samples, rollout_extra_metrics)
        return False
    _saved_rollout_files.add(output_file)

    # Save each sample
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(_sample_to_dict(sample), ensure_ascii=False) + "\n")

    print(f"[rollout_logger] Saved {len(samples)} samples to {output_file}")
    
    # Log tool call statistics (adds to rollout_extra_metrics)
    _log_tool_call_stats(samples, rollout_extra_metrics)
    
    return False  # Continue with default logging


def log_eval_rollout_to_file(rollout_id: int, args, data: dict, extra_metrics) -> bool:
    """Save eval samples to JSONL file.

    Args:
        rollout_id: Eval batch ID
        args: Command line arguments
        data: Dict like {"aime": {"rewards": [...], "samples": [...]}, ...}
        extra_metrics: Extra metrics dict (unused)

    Returns:
        False to continue with default logging
    """
    global _saved_eval_files

    # Generate output path
    output_dir = os.path.join(_get_output_base_dir(args), "eval_outputs")
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, dataset_data in data.items():
        output_file = os.path.join(output_dir, f"eval_{dataset_name}_{rollout_id}.jsonl")

        # Avoid duplicate writes
        if output_file in _saved_eval_files:
            continue
        _saved_eval_files.add(output_file)

        samples = dataset_data.get("samples", [])
        if not samples:
            continue

        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(_sample_to_dict(sample), ensure_ascii=False) + "\n")

        print(f"[rollout_logger] Saved {len(samples)} samples to {output_file}")

    return False  # Continue with default logging
