#!/bin/bash

set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="/mnt/workspace/wangdy/code/zj-slime-0722"
DATA_DIR="/mnt/workspace/public/data/RLdata/tir"
MODEL_DIR="/mnt/workspace/models/font-info/qwen3-4b-sft"
LOG_DIR="${PROJECT_DIR}/qwen3_4b_checkpoints"
EXP_NAME=sao-compactionrl-windowed-fifo

export TENSORBOARD_DIR="${LOG_DIR}/tensorboards/${EXP_NAME}/"
CHECKPOINT_ROOT="${LOG_DIR}/checkpoints/${EXP_NAME}"
ACTOR_CKPT_DIR="${CHECKPOINT_ROOT}/actor"
CRITIC_CKPT_DIR="${CHECKPOINT_ROOT}/critic"
ROLE_MEGATRON_CONFIG="${CHECKPOINT_ROOT}/megatron_role_config.yaml"

ACTOR_TRACKER="${ACTOR_CKPT_DIR}/latest_checkpointed_iteration.txt"
CRITIC_TRACKER="${CRITIC_CKPT_DIR}/latest_checkpointed_iteration.txt"
LEGACY_SHARED_TRACKER="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"

if [ -f "${LEGACY_SHARED_TRACKER}" ] && { [ ! -f "${ACTOR_TRACKER}" ] || [ ! -f "${CRITIC_TRACKER}" ]; }; then
   echo "Found legacy shared checkpoint at ${CHECKPOINT_ROOT}." >&2
   echo "This run expects separate actor/critic checkpoints under:" >&2
   echo "  ${ACTOR_CKPT_DIR}" >&2
   echo "  ${CRITIC_CKPT_DIR}" >&2
   exit 1
fi

if [ -f "${ACTOR_TRACKER}" ] && [ ! -f "${CRITIC_TRACKER}" ]; then
   echo "Found actor checkpoint but no critic checkpoint: ${CRITIC_TRACKER}" >&2
   exit 1
fi

if [ ! -f "${ACTOR_TRACKER}" ] && [ -f "${CRITIC_TRACKER}" ]; then
   echo "Found critic checkpoint but no actor checkpoint: ${ACTOR_TRACKER}" >&2
   exit 1
fi

if [ -f "${ACTOR_TRACKER}" ] && [ -f "${CRITIC_TRACKER}" ]; then
   ACTOR_STEP="$(tr -d '[:space:]' < "${ACTOR_TRACKER}")"
   CRITIC_STEP="$(tr -d '[:space:]' < "${CRITIC_TRACKER}")"
   if [ "${ACTOR_STEP}" != "${CRITIC_STEP}" ]; then
      echo "Actor/critic checkpoint steps differ: actor=${ACTOR_STEP}, critic=${CRITIC_STEP}" >&2
      exit 1
   fi
   ACTOR_LOAD_DIR="${ACTOR_CKPT_DIR}"
   CRITIC_LOAD_DIR="${CRITIC_CKPT_DIR}"
else
   ACTOR_LOAD_DIR="${MODEL_DIR}_torch_dist"
   CRITIC_LOAD_DIR="${MODEL_DIR}_torch_dist"
fi

mkdir -p "${CHECKPOINT_ROOT}"
cat > "${ROLE_MEGATRON_CONFIG}" <<EOF
megatron:
  - name: default
    role: actor
    overrides:
      load: ${ACTOR_LOAD_DIR}
      save: ${ACTOR_CKPT_DIR}
  - name: default
    role: critic
    overrides:
      load: ${CRITIC_LOAD_DIR}
      save: ${CRITIC_CKPT_DIR}
EOF

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
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
   --rotary-base 5000000
)

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}
   --ref-load ${MODEL_DIR}_torch_dist
   --load ${ACTOR_CKPT_DIR}
   --save ${ACTOR_CKPT_DIR}
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/tir_train_data/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 500
   --rollout-batch-size 64
   --n-samples-per-prompt 1
   --rollout-max-response-len 10240
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   --eval-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --eval-prompt-data aime ${DATA_DIR}/tir_val_data/aime-2024.jsonl
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 16384
   --eval-top-p 1
   --eval-interval 20
   --skip-eval-before-train
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

PPO_ARGS=(
   --advantage-estimator ppo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type k1
   --entropy-coef 0.00

   --use-dis
   --dis-clip 5.0
   --dis-clip-low 0.3

   --enable-compaction-rl
   --compaction-max-context-len 16384
   --compaction-trigger-len 12000
   --compaction-max-count 3
   --compaction-recent-steps 2
   --compaction-summary-max-new-tokens 2048
   --compaction-segment-reward-mode paper_each_segment
   --compaction-log-samples

   --adaptive-alpha 1.5
   --enable-length-adaptive
   --skip-observation-gae

   --critic-lambd 1.0
   --critic-lr 3e-6
   --critic-lr-warmup-iters 10
   --critic-update-ratio 2
   --num-critic-only-steps 50

   --normalize-advantages
   --calculate-per-token-loss

   --overlong-reward-coef 0.0
   --overlong-reward-threshold-ratio 0.85
   --overlong-reward-max-penalty 1.0
   --answer-format-reward-coef 0.2
   --tool-call-format-reward-coef 0.0

   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-tensorboard
   --tensorboard-dir ${TENSORBOARD_DIR}
   --tb-project-name ${EXP_NAME}
   --tb-experiment-name ${EXP_NAME}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-server-concurrency 32
)

WINDOWED_FIFO_ARGS=(
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --windowed-fifo-max-delay-step 4
   --windowed-fifo-max-prefetch-steps 4
   --update-weights-interval 1
)

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
   --custom-rollout-log-function-path slime.utils.rollout_logger.log_rollout_to_file
   --custom-eval-rollout-log-function-path slime.utils.rollout_logger.log_eval_rollout_to_file
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${PROJECT_DIR}:${PROJECT_DIR}/examples/retool:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${PROJECT_DIR}/train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   --megatron-config-path ${ROLE_MEGATRON_CONFIG} \
   --rollout-log-dir ${LOG_DIR}/${EXP_NAME}/rollout_log \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WINDOWED_FIFO_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
