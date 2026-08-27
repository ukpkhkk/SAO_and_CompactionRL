#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/mnt/workspace/wangdy/code/zj-slime-0722}"
MODEL_DIR="${MODEL_DIR:-/mnt/workspace/models/font-info/qwen3-4b-sft}"
CASE_JSON="${CASE_JSON:-${PROJECT_DIR}/qwen3_4b_vanillappo_checkpoints/case.json}"
OUTPUT="${OUTPUT:-${PROJECT_DIR}/qwen3_4b_vanillappo_checkpoints/summary_debug.json}"
DEBUG_SCRIPT="${DEBUG_SCRIPT:-${PROJECT_DIR}/scripts/debug_compaction_summary.py}"

SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_LAUNCH_HOST="${SGLANG_LAUNCH_HOST:-0.0.0.0}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_LOG_FILE="${SGLANG_LOG_FILE:-${PROJECT_DIR}/qwen3_4b_checkpoints/sglang_compaction_summary_${SGLANG_PORT}.log}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-900}"
KEEP_SERVER="${KEEP_SERVER:-0}"
REUSE_EXISTING_SERVER="${REUSE_EXISTING_SERVER:-0}"

# Match scripts/run-qwen3-4B-compactionrl.sh rollout engine settings.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"

# Match the CompactionRL summary-generation settings used by generate_with_retool.py.
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-16384}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-10240}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
COMPACTION_SUMMARY_MAX_NEW_TOKENS="${COMPACTION_SUMMARY_MAX_NEW_TOKENS:-2048}"
COMPACTION_SUMMARY_TEMPERATURE="${COMPACTION_SUMMARY_TEMPERATURE:-}"
EVENT_INDEX="${EVENT_INDEX:-0}"

export PYTHONPATH="/root/Megatron-LM/:${PROJECT_DIR}:${PROJECT_DIR}/examples/retool:${SCRIPT_DIR}:${PYTHONPATH:-}"
export MODEL_PATH="${MODEL_DIR}"
export SGLANG_URL="http://${SGLANG_HOST}:${SGLANG_PORT}/generate"

SGLANG_PID=""

cleanup() {
    if [ "${KEEP_SERVER}" != "1" ] && [ -n "${SGLANG_PID}" ] && kill -0 "${SGLANG_PID}" 2>/dev/null; then
        echo "Stopping SGLang server pid=${SGLANG_PID}"
        kill "${SGLANG_PID}" 2>/dev/null || true
        wait "${SGLANG_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_sglang() {
    local start_ts
    local now_ts
    start_ts="$(date +%s)"
    while true; do
        if curl -sf "http://${SGLANG_HOST}:${SGLANG_PORT}/health_generate" >/dev/null; then
            echo "SGLang is ready at http://${SGLANG_HOST}:${SGLANG_PORT}"
            return 0
        fi

        if [ -n "${SGLANG_PID}" ] && ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
            echo "SGLang server exited before becoming healthy. Last logs:" >&2
            tail -n 120 "${SGLANG_LOG_FILE}" >&2 || true
            exit 1
        fi

        now_ts="$(date +%s)"
        if [ $((now_ts - start_ts)) -ge "${STARTUP_TIMEOUT_SECONDS}" ]; then
            echo "Timed out waiting for SGLang. Last logs:" >&2
            tail -n 120 "${SGLANG_LOG_FILE}" >&2 || true
            exit 1
        fi
        sleep 5
    done
}

if [ ! -f "${DEBUG_SCRIPT}" ]; then
    echo "Cannot find debug script: ${DEBUG_SCRIPT}" >&2
    exit 1
fi

if [ ! -f "${CASE_JSON}" ]; then
    echo "Cannot find case json: ${CASE_JSON}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${SGLANG_LOG_FILE}")"

if curl -sf "http://${SGLANG_HOST}:${SGLANG_PORT}/health_generate" >/dev/null; then
    if [ "${REUSE_EXISTING_SERVER}" = "1" ]; then
        echo "Reusing existing SGLang server at http://${SGLANG_HOST}:${SGLANG_PORT}"
    else
        echo "SGLang already responds at http://${SGLANG_HOST}:${SGLANG_PORT}." >&2
        echo "Set REUSE_EXISTING_SERVER=1 to reuse it, or set SGLANG_PORT to a free port." >&2
        exit 1
    fi
else
    echo "Starting SGLang server..."
    echo "  model: ${MODEL_DIR}"
    echo "  cuda:  ${CUDA_VISIBLE_DEVICES}"
    echo "  tp:    ${TP_SIZE}"
    echo "  log:   ${SGLANG_LOG_FILE}"

    sglang_args=(
        --model-path "${MODEL_DIR}"
        --host "${SGLANG_LAUNCH_HOST}"
        --port "${SGLANG_PORT}"
        --tp "${TP_SIZE}"
        --mem-fraction-static "${MEM_FRACTION_STATIC}"
        --trust-remote-code
        --skip-server-warmup
    )

    if [ -n "${EXTRA_SGLANG_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        extra_sglang_args=(${EXTRA_SGLANG_ARGS})
        sglang_args+=("${extra_sglang_args[@]}")
    fi

    python3 -m sglang.launch_server "${sglang_args[@]}" >"${SGLANG_LOG_FILE}" 2>&1 &
    SGLANG_PID="$!"
    wait_for_sglang
fi

curl -sf "http://${SGLANG_HOST}:${SGLANG_PORT}/get_model_info" || true

debug_args=(
    --case-json "${CASE_JSON}"
    --event-index "${EVENT_INDEX}"
    --backend sglang
    --model-path "${MODEL_DIR}"
    --sglang-url "${SGLANG_URL}"
    --max-context-len "${MAX_CONTEXT_LEN}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE}"
    --rollout-top-p "${ROLLOUT_TOP_P}"
    --compaction-summary-max-new-tokens "${COMPACTION_SUMMARY_MAX_NEW_TOKENS}"
    --output "${OUTPUT}"
)

if [ -n "${COMPACTION_SUMMARY_TEMPERATURE}" ]; then
    debug_args+=(--compaction-summary-temperature "${COMPACTION_SUMMARY_TEMPERATURE}")
fi

if [ -n "${EXTRA_DEBUG_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    extra_debug_args=(${EXTRA_DEBUG_ARGS})
    debug_args+=("${extra_debug_args[@]}")
fi

cd "${PROJECT_DIR}"
python3 "${DEBUG_SCRIPT}" "${debug_args[@]}"

echo "Saved summary debug output to ${OUTPUT}"
