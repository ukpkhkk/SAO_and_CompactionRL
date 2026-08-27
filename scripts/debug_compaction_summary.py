#!/usr/bin/env python3
"""Replay the CompactionRL summary turn with training-matched settings.

The script extracts the exact prompt used for the summary turn when the
rollout log contains ``compaction.segments[*].current_prompt``. The default
backend is SGLang because the training rollout path calls the SGLang
``/generate`` endpoint with token IDs and ``return_logprob=True``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_case(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if line:
                return json.loads(line)
        raise
    if isinstance(obj, list):
        if not obj:
            raise ValueError(f"{path} contains an empty list.")
        obj = obj[0]
    if not isinstance(obj, dict):
        raise TypeError(f"{path} must contain a JSON object, JSONL line, or non-empty list.")
    return obj


def find_raw_summary(compaction: dict[str, Any], event_index: int) -> str:
    summary_texts = compaction.get("summary_texts") or []
    if event_index < len(summary_texts):
        return str(summary_texts[event_index])
    events = compaction.get("events") or []
    if event_index < len(events):
        return str(events[event_index].get("summary_text", ""))
    return ""


def extract_summary_prompt(sample: dict[str, Any], event_index: int) -> tuple[str, dict[str, Any]]:
    compaction = sample.get("compaction")
    if not isinstance(compaction, dict):
        raise ValueError("case does not contain a top-level compaction object.")

    events = compaction.get("events") or []
    if event_index >= len(events):
        raise IndexError(f"event_index={event_index} but only {len(events)} compaction events exist.")
    event = events[event_index]
    compact_id = event.get("compact_id")
    raw_summary = find_raw_summary(compaction, event_index)
    sanitized_summary = str(event.get("summary_text", ""))

    for segment in compaction.get("segments") or []:
        if segment.get("compact_id") != compact_id or not segment.get("contains_summary"):
            continue
        current_prompt = segment.get("current_prompt")
        text = segment.get("text", "")
        if not current_prompt:
            continue

        idx = -1
        for needle in (raw_summary, sanitized_summary):
            if needle:
                idx = text.find(needle)
                if idx >= 0:
                    break
        if idx < 0:
            raise ValueError("found summary segment, but could not locate summary_text inside segment text.")

        prompt = str(current_prompt) + text[:idx]
        return prompt, {"compaction": compaction, "event": event, "raw_summary": raw_summary}

    response = sample.get("response", "")
    idx = -1
    for needle in (raw_summary, sanitized_summary):
        if needle:
            idx = str(response).find(needle)
            if idx >= 0:
                break
    if idx < 0:
        raise ValueError("could not reconstruct summary prompt from segments or response.")

    prompt = str(sample.get("prompt", "")) + str(response)[:idx]
    return prompt, {"compaction": compaction, "event": event, "raw_summary": raw_summary, "approximate": True}


def build_training_sampling_params(args: argparse.Namespace, prompt_token_count: int) -> dict[str, Any]:
    """Match the summary-turn sampling logic in generate_with_retool.py."""
    remaining_budget = max(args.max_context_len - prompt_token_count, 1)
    max_new_tokens = min(
        int(args.rollout_max_response_len),
        int(remaining_budget),
        int(args.compaction_summary_max_new_tokens),
    )
    temperature = (
        args.compaction_summary_temperature
        if args.compaction_summary_temperature is not None
        else args.rollout_temperature
    )
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": args.rollout_top_p,
    }


def call_sglang(args: argparse.Namespace, prompt: str) -> tuple[str, dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer_path = args.tokenizer_path or args.model_path
    if not tokenizer_path:
        raise ValueError("--tokenizer-path or --model-path is required for --backend sglang.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    sampling_params = build_training_sampling_params(args, len(input_ids))
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if sampling_params["temperature"] <= 0:
        payload["sampling_params"]["temperature"] = 0

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        args.sglang_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            output = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to call SGLang endpoint {args.sglang_url}: {exc}") from exc

    meta_info = output.get("meta_info", {}) if isinstance(output, dict) else {}
    token_logprobs = meta_info.get("output_token_logprobs") or []
    request_info = {
        "input_token_count": len(input_ids),
        "sampling_params": sampling_params,
        "return_logprob": True,
        "finish_reason": meta_info.get("finish_reason"),
        "output_token_count": len(token_logprobs),
    }
    if isinstance(output, dict) and "text" in output:
        return str(output["text"]), request_info
    if isinstance(output, dict) and "meta_info" in output:
        token_ids = [item[1] for item in token_logprobs]
        return tokenizer.decode(token_ids, skip_special_tokens=False), request_info
    return json.dumps(output, ensure_ascii=False, indent=2), request_info


def call_transformers(args: argparse.Namespace, prompt: str) -> tuple[str, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.model_path:
        raise ValueError("--model-path is required for --backend transformers.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    sampling_params = build_training_sampling_params(args, len(input_ids))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    do_sample = sampling_params["temperature"] > 0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=sampling_params["max_new_tokens"],
            do_sample=do_sample,
            temperature=sampling_params["temperature"] if do_sample else None,
            top_p=sampling_params["top_p"] if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=False), {
        "input_token_count": len(input_ids),
        "sampling_params": sampling_params,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", default="qwen3_4b_vanillappo_checkpoints/case.json")
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--backend", choices=["prompt-only", "sglang", "transformers"], default="sglang")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/mnt/workspace/models/font-info/qwen3-4b-sft"))
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--sglang-url", default=os.environ.get("SGLANG_URL", "http://127.0.0.1:30000/generate"))
    parser.add_argument("--max-context-len", type=int, default=16384)
    parser.add_argument("--rollout-max-response-len", type=int, default=10240)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--compaction-summary-max-new-tokens", type=int, default=2048)
    parser.add_argument("--compaction-summary-temperature", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", default="qwen3_4b_vanillappo_checkpoints/summary_debug.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample = load_case(Path(args.case_json))
    prompt, meta = extract_summary_prompt(sample, args.event_index)

    model_output = ""
    request_info: dict[str, Any] = {}
    if args.backend == "sglang":
        model_output, request_info = call_sglang(args, prompt)
    elif args.backend == "transformers":
        model_output, request_info = call_transformers(args, prompt)

    result = {
        "case_json": args.case_json,
        "event_index": args.event_index,
        "backend": args.backend,
        "training_setting": {
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path or args.model_path,
            "max_context_len": args.max_context_len,
            "rollout_max_response_len": args.rollout_max_response_len,
            "rollout_temperature": args.rollout_temperature,
            "rollout_top_p": args.rollout_top_p,
            "compaction_summary_max_new_tokens": args.compaction_summary_max_new_tokens,
            "compaction_summary_temperature": args.compaction_summary_temperature,
            "sglang_url": args.sglang_url if args.backend == "sglang" else None,
            "return_logprob": args.backend == "sglang",
        },
        "request_info": request_info,
        "compact_prompt": meta["event"].get("compact_prompt", ""),
        "reference_summary_text": meta.get("raw_summary", ""),
        "prompt": prompt,
        "model_output": model_output,
    }
    if meta.get("approximate"):
        result["warning"] = "Prompt was reconstructed from top-level prompt/response and may lack system/tool text."

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved debug payload to {output_path}")
    if model_output:
        print("\n===== MODEL OUTPUT =====")
        print(model_output)
    else:
        print("\nBackend is prompt-only; inspect the saved prompt field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
