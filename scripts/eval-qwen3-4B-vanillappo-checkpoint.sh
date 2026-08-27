#!/bin/bash

set -ex

export PYTHONUNBUFFERED=1

RESET_RUNTIME="${RESET_RUNTIME:-1}"
if [ "${RESET_RUNTIME}" = "1" ]; then
   pkill -9 sglang || true
   ray stop --force || true
   pkill -9 ray || true
   sleep 3
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/mnt/workspace/wangdy/code/zj-slime-0722}"
DATA_DIR="${DATA_DIR:-/mnt/workspace/public/data/RLdata/tir}"
MODEL_DIR="${MODEL_DIR:-/mnt/workspace/models/font-info/qwen3-4b-sft}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/qwen3_4b_vanillappo_checkpoints}"
EXP_NAME="${EXP_NAME:-qwen3-4b-Retool-colocate-vanilla-ppo-8k-exp}"
export PYTHONPATH="/root/Megatron-LM/:${PROJECT_DIR}:${PROJECT_DIR}/examples/retool:${SCRIPT_DIR}:${PYTHONPATH:-}"

REF_TORCH_DIST_DIR="${REF_TORCH_DIST_DIR:-${MODEL_DIR}_torch_dist}"
ACTOR_LOAD_DIR="${ACTOR_LOAD_DIR:-${CHECKPOINT_DIR:-${LOG_DIR}/checkpoints/${EXP_NAME}}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ACTOR_LOAD_DIR}}"
EVAL_NAME="${EVAL_NAME:-aime2025}"
EVAL_DATA="${EVAL_DATA:-${DATA_DIR}/tir_val_data/aime-2025.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${LOG_DIR}/eval_${EXP_NAME}}"

NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
DP_SIZE=$((NUM_GPUS / TP_SIZE))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * DP_SIZE))}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
N_SAMPLES="${N_SAMPLES:-1}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1}"
EVAL_TOP_P="${EVAL_TOP_P:-1}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-8192}"
CHECK_WEIGHT_UPDATE_EQUAL="${CHECK_WEIGHT_UPDATE_EQUAL:-0}"

if [ "$((NUM_GPUS % TP_SIZE))" -ne 0 ]; then
   echo "NUM_GPUS (${NUM_GPUS}) must be divisible by TP_SIZE (${TP_SIZE})." >&2
   exit 1
fi

if [ "$((NUM_GPUS % ROLLOUT_GPUS_PER_ENGINE))" -ne 0 ]; then
   echo "NUM_GPUS (${NUM_GPUS}) must be divisible by ROLLOUT_GPUS_PER_ENGINE (${ROLLOUT_GPUS_PER_ENGINE})." >&2
   exit 1
fi

if [ ! -f "${ACTOR_LOAD_DIR}/latest_checkpointed_iteration.txt" ]; then
   echo "Cannot find actor Megatron checkpoint tracker: ${ACTOR_LOAD_DIR}/latest_checkpointed_iteration.txt" >&2
   echo "Set ACTOR_LOAD_DIR=/path/to/rl_checkpoint_dir if your checkpoint is elsewhere." >&2
   exit 1
fi

if [ ! -f "${REF_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]; then
   echo "Cannot find ref Megatron checkpoint tracker: ${REF_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" >&2
   echo "Set REF_TORCH_DIST_DIR=/path/to/qwen3-4b-sft_torch_dist if your ref checkpoint is elsewhere." >&2
   exit 1
fi

mkdir -p "${EVAL_OUTPUT_DIR}"
export TENSORBOARD_DIR="${EVAL_OUTPUT_DIR}/tensorboards"

MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-5000000}"
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${REF_TORCH_DIST_DIR}"
   --load "${ACTOR_LOAD_DIR}"
   --no-load-optim
   --no-load-rng
   --no-save-optim
)

DATA_ARGS=(
   --prompt-data "${EVAL_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --reward-key score
   --num-rollout 0
   --rollout-batch-size 1
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
)

EVAL_ARGS=(
   --eval-interval 1
   --eval-prompt-data "${EVAL_NAME}" "${EVAL_DATA}"
   --n-samples-per-eval-prompt "${N_SAMPLES}"
   --eval-temperature "${EVAL_TEMPERATURE}"
   --eval-top-p "${EVAL_TOP_P}"
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
   --skip-eval-before-train
   --custom-eval-rollout-log-function-path eval_pass_metrics.log_eval_metrics
   --save-debug-rollout-data "${EVAL_OUTPUT_DIR}/rollout_data/{rollout_id}.pt"
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --micro-batch-size "${MICRO_BATCH_SIZE}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --use-stateless-adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-decay-iters 1
   --lr-warmup-iters 0
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

EVAL_ONLY_ARGS=(
   --advantage-estimator grpo
   --kl-coef 0.00
   --entropy-coef 0.00
)

WANDB_ARGS=(
   --use-tensorboard
   --tensorboard-dir "${TENSORBOARD_DIR}/${EXP_NAME}/"
   --tb-project-name "${EXP_NAME}"
   --tb-experiment-name "eval-pass"
)

SGLANG_ARGS=(
   --rollout-num-gpus "${NUM_GPUS}"
   --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static 0.7
)

DEBUG_ARGS=()
if [ "${CHECK_WEIGHT_UPDATE_EQUAL}" = "1" ]; then
   DEBUG_ARGS+=(--check-weight-update-equal)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\",
    \"EVAL_OUTPUT_DIR\": \"${EVAL_OUTPUT_DIR}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${PROJECT_DIR}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --megatron-config-path "${PROJECT_DIR}/megatron_config.yaml" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${DATA_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${EVAL_ONLY_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DEBUG_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
