# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import re
from typing import Any

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import reward models
try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

# Import tool sandbox functionality
from tool_sandbox import SEMAPHORE, TOOL_CONFIGS, tool_registry

# Jinja2 template for tool-enabled conversations
TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""

ASSISTANT_PREFIX = "<|im_start|>assistant\n"
DEFAULT_COMPACTION_SUMMARY_PROMPT = (
    "The context is too long. I need to compact the previous problem-solving history now.\n"
    "I should output only a concise state summary needed to continue solving the same problem.\n"
    "I should keep the original problem, key facts, equations, tool results, failed attempts, and current plan.\n"
    "I should not try to call tools or reasoning while compacting."
)
DEFAULT_COMPACTION_RESUME_TEMPLATE = (
    "Original problem:\n{original_prompt}\n\n"
    "Compacted history:\n{summary}\n\n"
    "I should continue solve the same problem. Remember each code_interpreter call is stateless."
)


def _load_optional_text(path: str | None, default: str) -> str:
    if not path:
        return default
    with open(path, encoding="utf-8") as f:
        return f.read()


def format_conversation_with_tools(
    prompt: str, tools: list[dict[str, Any]] = None, system_prompt: str = None, messages: list[dict[str, Any]] = None
) -> str:
    """Format conversation using Jinja2 template with tool support"""
    template = Template(TOOL_TEMPLATE)

    # Prepare messages
    messages_to_render = []

    # Always add system message - use provided one or default
    if system_prompt:
        system_content = system_prompt
    else:
        # system_content = (
        #     "You are a helpful assistant that can use Python "
        #     "tools to solve mathematical problems. When you need "
        #     "to perform calculations, use the code_interpreter "
        #     "tool to execute code and get results."
        # )
        system_content = (
            "You are a helpful assistant that can use Python tools to solve mathematical problems. "
            "When you need calculations, use the code_interpreter tool to execute code and get results. "
            "Each code_interpreter call starts a fresh Python process with an empty namespace. "
            "It cannot see imports, variables, or functions from previous calls. "
            "Every code_interpreter call must include all imports, variables, and functions it uses. "
            "When you have finished solving the problem, you MUST provide the final answer "
            "using exactly this format:\n"
            "Answer: \\boxed{answer}\n"
        )
        

    messages_to_render.append({"role": "system", "content": system_content})

    # Add user message if provided
    if prompt:
        messages_to_render.append({"role": "user", "content": prompt})

    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(messages)

    # Render template
    formatted_text = template.render(messages=messages_to_render, tools=tools or [])

    return formatted_text


def _tokenize_text(state: GenerateState, text: str) -> list[int]:
    return state.tokenizer(text, add_special_tokens=False)["input_ids"]


def build_compaction_observation(summary_prompt: str) -> str:
    return f"\n\n{summary_prompt.strip()}\n\n{ASSISTANT_PREFIX}"


def inject_compaction_observation(observation: str, compaction_observation: str) -> str:
    if observation.endswith(ASSISTANT_PREFIX):
        return observation[: -len(ASSISTANT_PREFIX)] + compaction_observation
    return observation + compaction_observation


def sanitize_summary_text(text: str) -> str:
    return text.replace("<|im_end|>", "").strip()


def _clip_prompt_to_context(tokens: list[int], max_context_length: int, reserve: int = 1) -> list[int]:
    max_len = max(max_context_length - reserve, 1)
    if len(tokens) <= max_len:
        return tokens
    return tokens[-max_len:]


def compute_answer_format_reward(response: str) -> float:
    """Reward final answer format without using prompt text."""
    matches = list(re.finditer(r"Answer:\s*\\boxed\{\s*([^}\s]+)", response))
    if not matches:
        return -1.0

    answer_token = matches[-1].group(1).strip()
    if answer_token.lower().startswith("answer"):
        return -1.0
    return 1.0 if re.match(r"[-+]?\d", answer_token) else -1.0


def compute_tool_call_format_reward(response: str) -> float:
    """Reward tool calls that end the assistant turn immediately."""
    def iter_assistant_chunks(text: str):
        assistant_prefix = "<|im_start|>assistant\n"
        observation_markers = ("<interpreter>", "My previous action is invalid.")
        start = 0
        while start < len(text):
            boundary_candidates = [
                pos for marker in observation_markers if (pos := text.find(marker, start)) != -1
            ]
            if not boundary_candidates:
                yield text[start:]
                return

            boundary = min(boundary_candidates)
            yield text[start:boundary]
            next_start = text.find(assistant_prefix, boundary)
            if next_start == -1:
                return
            start = next_start + len(assistant_prefix)

    closing_tag = "</tool_call>"
    saw_tool_call = False
    for chunk in iter_assistant_chunks(response):
        if "<tool_call>" not in chunk:
            continue

        saw_tool_call = True
        matches = list(re.finditer(re.escape(closing_tag), chunk))
        if not matches:
            return -1.0

        for match in matches:
            suffix = chunk[match.end() :]
            if not re.fullmatch(r"\s*(?:<\|im_end\|>\s*)?", suffix):
                return -1.0

    return 1.0 if saw_tool_call else 0.0


def extract_terminal_boxed_answer(response: str) -> str | None:
    """Extract a final bare \\boxed{...} answer from model output."""
    marker = r"\boxed{"
    start = response.rfind(marker)
    if start < 0:
        return None

    content_start = start + len(marker)
    depth = 1
    end = content_start
    while end < len(response):
        if response[end] == "{":
            depth += 1
        elif response[end] == "}":
            depth -= 1
            if depth == 0:
                break
        end += 1

    if depth != 0:
        return None

    suffix = response[end + 1 :]
    terminal_suffix_pattern = r"[\s\.\,\;\:\!\?。．，；：！？]*(?:<\|im_end\|>[\s\.\,\;\:\!\?。．，；：！？]*)?"
    if not re.fullmatch(terminal_suffix_pattern, suffix):
        return None

    content = response[content_start:end].strip()
    if not content or content.lower().startswith("answer"):
        return None
    return content


def extract_terminal_plain_answer(response: str) -> str | None:
    """Extract a final plain `Answer: <number>` answer from model output."""
    matches = list(re.finditer(r"(?i)Answer\s*[:：]\s*([-+]?\d[\d,]*(?:\.\d+)?)", response))
    if not matches:
        return None

    match = matches[-1]
    suffix = response[match.end() :]
    if not re.fullmatch(r"[\s\.\,\;\:\!\?。．，；：！？]*", suffix):
        return None

    return match.group(1).replace(",", "").strip()


def postprocess_predictions(prediction: str):
    """Extract action and content from prediction string"""
    # Check for Answer: \boxed{...} format (only format we need for math_dapo)
    # Use a more robust regex that handles nested braces
    answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        if not content or content.lower().startswith("answer"):
            return None, ""
        return "answer", content

    # Then check for <tool_call> tags (new format from Jinja2 template)
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            import json

            # Clean up the JSON string by removing newlines and extra
            # whitespace
            json_str = tool_call_match.group(1)
            # Replace newlines in string values with \n
            json_str = json_str.replace("\n", "\\n")
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})

            if tool_name == "code_interpreter":
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

    # Then check for <code> tags
    code_pattern = r"<code>(.*?)</code>"
    code_match = re.search(code_pattern, prediction, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        return "code", content

    # Finally check for ```python code blocks (lowest priority)
    python_code_pattern = r"```python\s*(.*?)\s*```"
    python_code_match = re.search(python_code_pattern, prediction, re.DOTALL)
    if python_code_match:
        content = python_code_match.group(1).strip()
        return "code", content

    boxed_answer = extract_terminal_boxed_answer(prediction)
    if boxed_answer is not None:
        return "answer", boxed_answer

    plain_answer = extract_terminal_plain_answer(prediction)
    if plain_answer is not None:
        return "answer", plain_answer

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    # Handle <tool_call> tags (new format from Jinja2 template)
    if "<tool_call>" in resp:
        # Find the last occurrence of <tool_call>...</tool_call>
        tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(tool_call_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle <code> tags
    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"

    # Handle ```python code blocks
    if "```python" in resp:
        # Find the last occurrence of ```python...```
        python_pattern = r"```python\s*.*?```"
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle Answer: \boxed{...} format (only format we need for math_dapo)
    if "Answer:" in resp and "\\boxed{" in resp:
        # Find the last occurrence of Answer: \boxed{...} with nested braces support
        answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(answer_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    return resp


async def execute_predictions(prediction: str) -> str:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        # Content is already the Python code (extracted by
        # postprocess_predictions)
        code = content.strip()
        if code:
            async with SEMAPHORE:
                result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            next_obs = f"\n\n<interpreter>\n{result}\n</interpreter>\n\n<|im_start|>assistant\n"
            done = False
        else:
            next_obs = "\n\n<interpreter>\nError: No Python code found" "\n</interpreter>\n\n<|im_start|>assistant\n"
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        # next_obs = (
        #     "\nMy previous action is invalid. "
        #     "If I want to execute code, I should put the code between "
        #     "<code> and </code>. "
        #     "If I want to give the final answer, I should use the format "
        #     "'Answer: \\boxed{answer}'. Let me try again.\n"
        # )
        next_obs = (
                "My previous action is invalid. "
                "If I want to execute Python code, I should call the "
                "code_interpreter tool using the following format:"
                "<tool_call>"
                "{\"name\": \"code_interpreter\", "
                "\"arguments\": {\"code\": \"your Python code\"}}"
                "</tool_call>"
                "If I want to give the final answer, I should use the format "
                "'Answer: \\boxed{answer}'. Let me try again.\n"
                "<|im_start|>assistant\n"
        )
        done = False

    return next_obs, done


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """Custom generation function supporting tool calls and optional CompactionRL."""
    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."

    # Retried samples (previously aborted / partial) arrive here with stale
    # rollout state from the first attempt. Clear it so this generation starts
    # clean; otherwise downstream logprob slicing sees length mismatches.
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.train_metadata = None
    sample.status = Sample.Status.PENDING

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)
    initial_prompt_tokens = _tokenize_text(state, prompt)
    sample.tokens = list(initial_prompt_tokens)

    enable_compaction = bool(getattr(args, "enable_compaction_rl", False))
    if evaluation and args.eval_max_response_len is not None:
        max_context_length = args.eval_max_response_len
    elif args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu

    compaction_context_length = getattr(args, "compaction_max_context_len", None) or max_context_length
    compaction_context_length = min(int(compaction_context_length), int(max_context_length))
    compaction_trigger_len = int(getattr(args, "compaction_trigger_len", 10240))
    compaction_max_count = int(getattr(args, "compaction_max_count", 3))
    compaction_recent_steps = int(getattr(args, "compaction_recent_steps", 2))
    compaction_summary_max_new_tokens = int(getattr(args, "compaction_summary_max_new_tokens", 1024))

    response = ""
    full_response_token_ids: list[int] = []
    active_context_tokens = list(initial_prompt_tokens)
    loss_masks = sample.loss_mask
    tool_call_count = 0
    last_finish_type: str | None = None

    history_steps: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    summary_texts: list[str] = []
    compaction_events: list[dict[str, Any]] = []
    compact_count = 0
    segment_id = 0
    current_segment: dict[str, Any] | None = None
    pending_compaction_summary = False
    pending_compact_id: int | None = None
    pending_before_context_len: int | None = None
    pending_compact_prompt_tokens = 0
    final_answer_text = ""
    final_segment_id: int | None = None

    if enable_compaction:
        summary_prompt = _load_optional_text(
            getattr(args, "compaction_summary_prompt_path", None),
            DEFAULT_COMPACTION_SUMMARY_PROMPT,
        )
        resume_template = _load_optional_text(
            getattr(args, "compaction_resume_template_path", None),
            DEFAULT_COMPACTION_RESUME_TEMPLATE,
        )
    else:
        summary_prompt = DEFAULT_COMPACTION_SUMMARY_PROMPT
        resume_template = DEFAULT_COMPACTION_RESUME_TEMPLATE

    def start_segment(prompt_tokens: list[int], current_prompt: str) -> None:
        nonlocal current_segment
        if not enable_compaction:
            return
        current_segment = {
            "segment_id": segment_id,
            "type": "execute",
            "prompt_tokens": list(prompt_tokens),
            "current_prompt": current_prompt,
            "response_tokens": [],
            "rollout_log_probs": [],
            "loss_mask": [],
            "trainable_token_count": 0,
            "compact_id": None,
            "contains_compaction_prompt": False,
            "contains_summary": False,
            "is_final_answer_segment": False,
            "text": "",
        }

    def append_to_segment(
        *,
        tokens: list[int],
        rollout_log_probs: list[float],
        loss_mask: list[int],
        text: str,
    ) -> None:
        if not enable_compaction or current_segment is None or not tokens:
            return
        assert len(tokens) == len(loss_mask), f"segment token/loss-mask mismatch: {len(tokens)} vs {len(loss_mask)}"
        assert len(tokens) == len(rollout_log_probs), (
            f"segment token/logprob mismatch: {len(tokens)} vs {len(rollout_log_probs)}"
        )
        current_segment["response_tokens"].extend(tokens)
        current_segment["rollout_log_probs"].extend(rollout_log_probs)
        current_segment["loss_mask"].extend(loss_mask)
        current_segment["trainable_token_count"] += int(sum(loss_mask))
        current_segment["text"] += text

    def finalize_segment(*, is_final_answer_segment: bool = False) -> None:
        nonlocal current_segment, segment_id, final_segment_id
        if not enable_compaction or current_segment is None:
            return
        if not current_segment["response_tokens"]:
            current_segment = None
            return
        current_segment["is_final_answer_segment"] = bool(is_final_answer_segment)
        if is_final_answer_segment:
            final_segment_id = int(current_segment["segment_id"])
        segments.append(current_segment)
        segment_id += 1
        current_segment = None

    def append_model_tokens(
        *,
        tokens: list[int],
        log_probs: list[float],
        text: str,
        meta_info: dict[str, Any],
        contains_summary: bool = False,
    ) -> None:
        nonlocal response, full_response_token_ids, active_context_tokens
        if not tokens:
            return
        response += text
        full_response_token_ids += tokens
        active_context_tokens += tokens
        sample.append_response_tokens(
            args,
            tokens=tokens,
            log_probs=log_probs,
            trainable=True,
            meta_info=meta_info,
        )
        append_to_segment(tokens=tokens, rollout_log_probs=log_probs, loss_mask=[1] * len(tokens), text=text)
        if contains_summary and current_segment is not None:
            current_segment["contains_summary"] = True

    def append_env_tokens(*, text: str, tokens: list[int] | None = None) -> list[int]:
        nonlocal response, full_response_token_ids, active_context_tokens
        tokens = _tokenize_text(state, text) if tokens is None else list(tokens)
        if not tokens:
            return []
        response += text
        full_response_token_ids += tokens
        active_context_tokens += tokens
        sample.append_response_tokens(args, tokens=tokens, trainable=False)
        append_to_segment(tokens=tokens, rollout_log_probs=[0.0] * len(tokens), loss_mask=[0] * len(tokens), text=text)
        return tokens

    def append_env_observation(text: str, *, allow_truncate: bool = True) -> tuple[str, list[int], bool]:
        tokens = _tokenize_text(state, text)
        overflow = len(active_context_tokens) + len(tokens) - max_context_length
        if overflow <= 0:
            append_env_tokens(text=text, tokens=tokens)
            return text, tokens, False
        if not allow_truncate:
            return "", [], True

        keep = max(0, len(tokens) - overflow)
        clipped_tokens = tokens[:keep]
        clipped_text = state.tokenizer.decode(clipped_tokens) if clipped_tokens else ""
        append_env_tokens(text=clipped_text, tokens=clipped_tokens)
        sample.status = Sample.Status.TRUNCATED
        return clipped_text, clipped_tokens, True

    def get_recent_step_tokens(steps: list[dict[str, Any]]) -> list[int]:
        tokens: list[int] = []
        for step in steps:
            tokens.extend(step["action_tokens"])
            tokens.extend(step["observation_tokens"])
        return tokens

    def get_recent_step_text(steps: list[dict[str, Any]]) -> str:
        return "".join(step["action_text"] + step["observation_text"] for step in steps)

    def build_compacted_context(summary_text: str, before_context_len: int) -> tuple[list[int], str, str, dict[str, Any]]:
        sanitized_summary = sanitize_summary_text(summary_text)
        resume_prompt = resume_template.format(original_prompt=sample.prompt, summary=sanitized_summary)
        resume_text = format_conversation_with_tools(prompt=resume_prompt, tools=tool_specs)
        resume_tokens = _tokenize_text(state, resume_text)

        keep_count = max(compaction_recent_steps, 0)
        recent_steps = history_steps[-keep_count:] if keep_count else []
        dropped_recent_steps = 0

        rollout_budget = int(getattr(args, "rollout_max_response_len", 0) or 0)
        reserve_for_generation = min(rollout_budget if rollout_budget > 0 else compaction_context_length // 2, compaction_context_length // 2)
        prompt_budget = max(compaction_context_length - max(reserve_for_generation, 1), 1)

        context = resume_tokens + get_recent_step_tokens(recent_steps)
        current_prompt = resume_text + get_recent_step_text(recent_steps)
        while (len(context) > prompt_budget or len(context) >= before_context_len) and recent_steps:
            recent_steps = recent_steps[1:]
            dropped_recent_steps += 1
            context = resume_tokens + get_recent_step_tokens(recent_steps)
            current_prompt = resume_text + get_recent_step_text(recent_steps)

        forced_clip = False
        if len(context) > prompt_budget:
            context = _clip_prompt_to_context(context, prompt_budget, reserve=0)
            forced_clip = True
        if len(context) >= before_context_len:
            target_len = max(min(before_context_len - 1, compaction_context_length - 1), 1)
            context = _clip_prompt_to_context(context, target_len, reserve=0)
            forced_clip = True
        if forced_clip:
            current_prompt = state.tokenizer.decode(context)

        stats = {
            "after_context_len": len(context),
            "recent_step_count": len(recent_steps),
            "dropped_recent_steps": dropped_recent_steps,
            "forced_clip": forced_clip,
            "failed_to_shrink": len(context) >= before_context_len,
        }
        return context, sanitized_summary, current_prompt, stats

    if enable_compaction:
        start_segment(active_context_tokens, prompt)

    for turn in range(TOOL_CONFIGS["max_turns"]):
        total_length = len(active_context_tokens)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        remaining_budget = max_context_length - total_length
        per_turn_sampling_params = dict(sampling_params)
        per_turn_sampling_params["max_new_tokens"] = min(
            sampling_params.get("max_new_tokens", remaining_budget),
            remaining_budget,
        )
        if pending_compaction_summary:
            per_turn_sampling_params["max_new_tokens"] = min(
                per_turn_sampling_params["max_new_tokens"],
                compaction_summary_max_new_tokens,
            )
            summary_temperature = getattr(args, "compaction_summary_temperature", None)
            if summary_temperature is not None:
                per_turn_sampling_params["temperature"] = summary_temperature

        current_token_ids = list(active_context_tokens)
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": per_turn_sampling_params,
            "return_logprob": True,
        }

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "debug/payload_length": len(current_token_ids),
                        "debug/available_tools": len(tool_specs),
                        "debug/tools_used": response.count("<interpreter>"),
                        "debug/turn": turn,
                    }
                )
        except ImportError:
            pass

        output = await post(url, payload)
        last_finish_type = output["meta_info"]["finish_reason"]["type"]
        if last_finish_type == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        if "output_token_logprobs" not in output["meta_info"]:
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        cur_response = state.tokenizer.decode(cur_response_token_ids)

        append_model_tokens(
            tokens=cur_response_token_ids,
            log_probs=cur_log_probs,
            text=cur_response,
            meta_info=output["meta_info"],
            contains_summary=pending_compaction_summary,
        )

        if pending_compaction_summary:
            summary_text = cur_response
            summary_texts.append(summary_text)
            if last_finish_type in {"stop", "length"}:
                sample.status = Sample.Status.PENDING

            before_context_len = pending_before_context_len or len(current_token_ids)
            compact_id = pending_compact_id if pending_compact_id is not None else compact_count
            new_context, sanitized_summary, new_prompt, compact_stats = build_compacted_context(summary_text, before_context_len)
            if current_segment is not None:
                current_segment["compact_id"] = compact_id
            finalize_segment(is_final_answer_segment=False)

            compaction_events.append(
                {
                    "turn": turn,
                    "compact_id": compact_id,
                    "trigger_turn": turn - 1,
                    "before_context_len": before_context_len,
                    "compact_prompt": summary_prompt,
                    "compact_prompt_tokens": pending_compact_prompt_tokens,
                    "summary_tokens": len(cur_response_token_ids),
                    "summary_text": sanitized_summary,
                    **compact_stats,
                }
            )
            active_context_tokens = new_context
            pending_compaction_summary = False
            pending_compact_id = None
            pending_before_context_len = None
            pending_compact_prompt_tokens = 0

            if last_finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                return sample
            if sample.status == Sample.Status.PENDING:
                start_segment(active_context_tokens, new_prompt)
            else:
                break
            continue

        if last_finish_type == "length":
            break

        action, _ = postprocess_predictions(cur_response)
        if action == "answer":
            final_answer_text = cur_response
            finalize_segment(is_final_answer_segment=True)
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            final_answer_text = cur_response
            finalize_segment(is_final_answer_segment=True)
            break

        if "<interpreter>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = _tokenize_text(state, next_obs)
        normal_context_len_after_obs = len(active_context_tokens) + len(obs_tokens_ids)
        should_compact = (
            enable_compaction
            and compact_count < compaction_max_count
            and turn + 1 < TOOL_CONFIGS["max_turns"]
            and normal_context_len_after_obs >= compaction_trigger_len
        )

        if should_compact:
            compact_obs = build_compaction_observation(summary_prompt)
            compact_obs_tokens = _tokenize_text(state, compact_obs)
            combined_obs = inject_compaction_observation(next_obs, compact_obs)
            combined_obs_tokens = _tokenize_text(state, combined_obs)
            if len(active_context_tokens) + len(combined_obs_tokens) > max_context_length:
                obs_text, actual_obs_tokens, truncated_by_observation = append_env_observation(next_obs, allow_truncate=True)
                if enable_compaction:
                    history_steps.append(
                        {
                            "action_text": cur_response,
                            "action_tokens": list(cur_response_token_ids),
                            "observation_text": obs_text,
                            "observation_tokens": list(actual_obs_tokens),
                        }
                    )
                if truncated_by_observation:
                    sample.status = Sample.Status.TRUNCATED
                break

            append_env_tokens(text=combined_obs, tokens=combined_obs_tokens)
            if current_segment is not None:
                current_segment["contains_compaction_prompt"] = True
            history_steps.append(
                {
                    "action_text": cur_response,
                    "action_tokens": list(cur_response_token_ids),
                    "observation_text": next_obs,
                    "observation_tokens": list(obs_tokens_ids),
                }
            )
            pending_compaction_summary = True
            pending_compact_id = compact_count
            pending_before_context_len = normal_context_len_after_obs
            pending_compact_prompt_tokens = len(compact_obs_tokens)
            compact_count += 1
            continue

        obs_text, actual_obs_tokens, truncated_by_observation = append_env_observation(next_obs, allow_truncate=True)
        if enable_compaction:
            history_steps.append(
                {
                    "action_text": cur_response,
                    "action_tokens": list(cur_response_token_ids),
                    "observation_text": obs_text,
                    "observation_tokens": list(actual_obs_tokens),
                }
            )

        if sample.rollout_log_probs is not None:
            assert len(full_response_token_ids) == len(
                sample.rollout_log_probs
            ), (
                f"Token/logp length mismatch at turn {turn}: "
                f"{len(full_response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"
            )

        if truncated_by_observation:
            sample.status = Sample.Status.TRUNCATED
            break

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    if enable_compaction:
        finalize_segment(is_final_answer_segment=False)

    sample.tokens = list(initial_prompt_tokens) + full_response_token_ids
    sample.response_length = len(full_response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    sample.payload_text = prompt + response
    sample.payload_has_system = "<|im_start|>system" in prompt + response
    sample.payload_has_tools = "# Tools" in prompt + response
    sample.tool_call_count = tool_call_count

    if enable_compaction:
        if compact_count > 0:
            sample.log_response = "".join(
                (s.get("current_prompt", "") if idx > 0 else "") + s["text"]
                for idx, s in enumerate(segments)
            )
        sample.metadata["final_answer_text"] = final_answer_text
        sample.metadata["final_segment_id"] = final_segment_id
        train_segments = [{k: v for k, v in s.items() if k != "current_prompt"} for s in segments]
        compaction_metadata = {
            "enabled": True,
            "trace_id": sample.index,
            "compact_count": compact_count,
            "context_budget": compaction_context_length,
            "trigger_len": compaction_trigger_len,
            "recent_steps": compaction_recent_steps,
            "segments": train_segments,
            "summary_texts": summary_texts,
            "events": compaction_events,
            "final_context_len": len(active_context_tokens),
            "final_answer_text": final_answer_text,
            "final_segment_id": final_segment_id,
        }
        sample.train_metadata = {"compaction": compaction_metadata}
        sample.metadata["compaction"] = {
            "enabled": True,
            "compact_count": compact_count,
            "context_budget": compaction_context_length,
            "trigger_len": compaction_trigger_len,
            "recent_steps": compaction_recent_steps,
            "summary_texts": summary_texts,
            "events": compaction_events,
            "final_context_len": len(active_context_tokens),
            "final_answer_text": final_answer_text,
            "final_segment_id": final_segment_id,
            "segments": [
                {
                    "segment_id": s["segment_id"],
                    "type": s["type"],
                    "response_length": len(s["response_tokens"]),
                    "trainable_token_count": s["trainable_token_count"],
                    "compact_id": s["compact_id"],
                    "contains_compaction_prompt": s.get("contains_compaction_prompt", False),
                    "contains_summary": s.get("contains_summary", False),
                    "is_final_answer_segment": s["is_final_answer_segment"],
                    "current_prompt": s.get("current_prompt", ""),
                    "text": s["text"],
                }
                for s in segments
            ],
        }

    if sample.status is Sample.Status.PENDING:
        match last_finish_type:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case _:
                sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Under CompactionRL, summaries can mention answer-like strings. Score only
    # the assistant turn that actually ended the trajectory.
    if (
        getattr(args, "enable_compaction_rl", False)
        and isinstance(getattr(sample, "metadata", None), dict)
        and "final_answer_text" in sample.metadata
    ):
        solution_str = sample.metadata["final_answer_text"]
    else:
        solution_str = sample.response

    # Get ground truth answer - label is a string, not a dict
    ground_truth = sample.label if sample.label is not None else ""

    # Get tool call count as num_turns
    num_turns = getattr(sample, "tool_call_count", 0)

    plain_answer = extract_terminal_plain_answer(solution_str)
    if plain_answer is not None:
        result = math_dapo_compute_score(f"Answer: {plain_answer}", ground_truth, strict_box_verify=False)
    else:
        # use \\boxed{...} answer
        result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    result["answer_format_reward"] = compute_answer_format_reward(solution_str)
    result["tool_call_format_reward"] = compute_tool_call_format_reward(sample.response)

    return result
