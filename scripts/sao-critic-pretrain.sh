#!/bin/bash

set -ex

export PYTHONBUFFERED=16


#############################################
# environment
#############################################
PROJECT_DIR="/mnt/workspace/wangdy/code/zj-slime-0722"
DATA_DIR="/mnt/workspace/public/data/RLdata"
MODEL_DIR="/mnt/workspace/models/font-info/qwen3-4b-sft"
LOG_DIR="${PROJECT_DIR}/qwen3_4b_checkpoints"
EXP_NAME=sao-classify-pretrained-critic
export TENSORBOARD_DIR="${LOG_DIR}/tensorboards/${EXP_NAME}"
CHECKPOINT_ROOT="${LOG_DIR}/checkpoints/${EXP_NAME}"
ACTOR_CKPT_DIR="${MODEL_DIR}_torch_dist"
CRITIC_CKPT_DIR="${CHECKPOINT_ROOT}/critic"
ROLE_MEGATRON_CONFIG="${CHECKPOINT_ROOT}/megatron_role_config.yaml"

mkdir -p ${CHECKPOINT_ROOT}

#############################################
# Megatron role config
#############################################
cat > ${ROLE_MEGATRON_CONFIG} <<EOF

megatron:

- name: default
  role: actor
  overrides:
    load: ${ACTOR_CKPT_DIR}
- name: default
  role: critic
  overrides:
    load: ${ACTOR_CKPT_DIR}
    save: ${CRITIC_CKPT_DIR}
EOF



#############################################
# model
#############################################

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
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --vocab-size 151936
    --kv-channels 128
    --qk-layernorm
    --rotary-base 5000000
)

#############################################
# checkpoint
#############################################

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}
    --ref-load ${MODEL_DIR}_torch_dist
    # actor 不保存
    --save ${CRITIC_CKPT_DIR}
    --save-interval 50
)

#############################################
# rollout data
#############################################

ROLLOUT_ARGS=(
# deepmath17k
    --prompt-data ${DATA_DIR}/tir/tir_train_data/deepmath-17k-critic-pretrain.jsonl
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --reward-key score
    # 训练280 step
    --num-rollout 280
    --rollout-batch-size 64
    --global-batch-size 64
    --n-samples-per-prompt 1
    --rollout-max-response-len 16384
    --rollout-temperature 1
    --balance-data
)

#############################################
# critic pretraining
#############################################

PPO_ARGS=(
    --advantage-estimator ppo
    # 不训练actor
    --num-critic-only-steps 280
    # critic train twice per rollout
    --critic-update-ratio 1
    # critic objective

    --critic-lambd 1.0
    # skip observation gae, since we use it in rl phase
    --skip-observation-gae

    --critic-lr 5e-6
    # freeze attention
    --critic-freeze-params-name-list self_attention
    --normalize-advantages
    --entropy-coef 0.0
    --kl-loss-coef 0.0
    --eps-clip 0.2

    # classification loss
    --value-loss-type classification
    --value-num-bins 51
    --value-reward-range -1.5 1.5
    --value-target-type hl_gauss
    --hl-gauss-sigma-ratio 0.75
)

#############################################
# optimizer
#############################################

OPTIMIZER_ARGS=(
    --optimizer adam
    # actor lr=0
    --lr 0
    --critic-lr 5e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)
#############################################
# performance
#############################################

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1

    --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

#############################################
# logging
#############################################

WANDB_ARGS=(
    --use-tensorboard
    --tensorboard-dir ${TENSORBOARD_DIR}
    --tb-project-name ${EXP_NAME}
    --tb-experiment-name ${EXP_NAME}
)

#############################################
# rollout engine
#############################################

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
    --sglang-server-concurrency 32
)

#############################################
# async rollout
#############################################

WINDOWED_FIFO_ARGS=(
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
    --windowed-fifo-max-delay-step 4
    --windowed-fifo-max-prefetch-steps 4
    --update-weights-interval 1
)

#############################################
# misc
#############################################
MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

#############################################
# custom reward
#############################################

CUSTOM_ARGS=(
    --custom-generate-function-path generate_with_retool.generate
    --custom-rm-path generate_with_retool.reward_func
    --custom-rollout-log-function-path slime.utils.rollout_logger.log_rollout_to_file
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${PROJECT_DIR}:${PROJECT_DIR}/examples/retool:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit \
    --address="http://127.0.0.1:8265" \
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
    ${PPO_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${WINDOWED_FIFO_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]}