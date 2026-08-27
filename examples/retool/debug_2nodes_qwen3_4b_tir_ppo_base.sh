#!/bin/bash

# for rerun the task
set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOG_DIR="/mnt/math/zxn/code/slime-20260305/slime/qwen3_4b_ppo_checkpoints"
EXP_NAME=qwen3-4b-Retool-colocate-SAO-8k-exp
export TENSORBOARD_DIR="/mnt/math/zxn/code/slime-20260305/slime/"

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
   --rotary-base 5000000 \
)

CKPT_ARGS=(
   --hf-checkpoint /mnt/data/zdm/models/font-info/qwen3-4b-sft
   --ref-load /mnt/data/zdm/models/font-info/qwen3-4b-sft_torch_dist
   # --load ${LOG_DIR}/checkpoints/${EXP_NAME}
   --save ${LOG_DIR}/checkpoints/${EXP_NAME}
   --save-interval 20
)


ROLLOUT_ARGS=(
   --prompt-data /mnt/data/zdm/data/tir/tir_train_data/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-temperature 1
   # --partial-rollout

   --global-batch-size 16
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime  /mnt/data/zdm/data/tir/tir_val_data/aime-2024.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-top-p 1
   --skip-eval-before-train
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
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
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator ppo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type k1
   --entropy-coef 0.00

   # DIS
   --use-dis
   --dis-clip 3.0
   --dis-clip-low 0.8

   # length-adaptive GAE
   --adaptive-alpha 1.5
   --enable-length-adaptive # for SAO
   --skip-observation-gae

   # critic lr
   --critic-lambd 1.0
   --critic-lr 1e-5
   --critic-lr-warmup-iters 10
   --critic-update-ratio 2 # different model different value
   --num-critic-only-steps 10 # Number of steps to train only the critic at the beginning of training
   # --critic-freeze-params-name-list self_attention

   --normalize-advantages
   # --calculate-per-token-loss

   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-tensorboard
   --tensorboard-dir ${LOG_DIR}/tensorboards/${EXP_NAME}/
   --tb-project-name ${EXP_NAME}
   --tb-experiment-name ${EXP_NAME}
   # --use-wandb
   # --wandb-project slime-dapo
   # --wandb-group qwen3-4B-test-multi-turn
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   # --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /mnt/math/zxn/code/slime-20260305/slime/train.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 8 \
   --megatron-config-path /mnt/math/zxn/code/slime-20260305/slime/megatron_config.yaml \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
