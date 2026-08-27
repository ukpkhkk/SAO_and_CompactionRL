# Agentic RL on Slime

[中文文档](README_zh.md) | English

This repository is a research fork of [slime](https://github.com/THUDM/slime) for long-horizon LLM reinforcement learning. It keeps the Megatron + SGLang training stack from slime and adds the pieces needed for agentic RL experiments:

- Windowed FIFO asynchronous rollout/training.
- Vanilla GRPO baseline.
- PPO/SAO with a critic, GAE, DIS, and optional critic pretraining.
- Regression-as-classification critic loss.
- Retool-style tool-integrated reasoning with Python sandbox rewards.
- Rollout/eval JSONL logging and TensorBoard metrics.

The current scripts are written for Qwen3-4B TIR experiments and are intended to be edited in place for your model/data paths.

## What Is Implemented

| Component | Description |
| --- | --- |
| Async pipeline | `train_async.py` with `slime.rollout.fully_async_rollout.generate_rollout_fully_async` implements Windowed FIFO rollout. |
| Vanilla GRPO | Async GRPO baseline with grouped sampling. |
| SAO | Single-rollout PPO with critic baseline, length-adaptive GAE, skip-observation GAE, and DIS. |
| Critic pretraining | Critic-only training before SAO, using the same rollout/reward path as RL. |
| Classification critic | `--value-loss-type classification` supports two-hot / HL-Gauss targets. |
| Vanilla PPO | Synchronous colocated PPO entry through `train.py`. |
| Retool TIR | Tool-calling rollout with Python interpreter, answer-format reward, and tool-call-format reward. |

## Results

Representative metrics are shown below. For full training curves, use the TensorBoard logs produced by each script.

<p align="center">
  <img src="imgs/metrics.png" alt="training metrics" width="900">
</p>

## Quick Start

### 1. Prepare the Environment

The training stack expects Linux, CUDA GPUs, Megatron-LM, SGLang, Ray, and Python 3.12. The recommended setup is the repository conda build script:

```bash
bash build_conda.sh
source ~/.bashrc
micromamba activate slime
cd /path/to/this/repo
pip install -e . --no-deps
```

If your server cannot access GitHub or Docker Hub, prepare third-party source archives under `third_party/` and use the offline build workflow in `my_build_conda.sh` if present in your checkout.

`build_conda.sh` is primarily a dependency bootstrap script. Before running experiments, make sure the editable `slime` package points to this modified repository, not a fresh upstream clone.

Before launching training, verify the key imports:

```bash
python -c "import ray, sglang, torch, slime; print('env ok')"
```

### 2. Prepare Model Checkpoints

Each experiment script expects both a HuggingFace checkpoint and a Megatron torch distributed checkpoint:

```text
MODEL_DIR=/path/to/qwen3-4b-sft
MODEL_DIR_torch_dist=/path/to/qwen3-4b-sft_torch_dist
```

The HuggingFace directory should contain files such as:

```text
config.json
tokenizer.json / tokenizer.model
tokenizer_config.json
generation_config.json
```

The torch distributed directory should contain Megatron checkpoint metadata, including:

```text
latest_checkpointed_iteration.txt
iter_*/ or release/
```

The provided Qwen3-4B scripts currently use:

```bash
MODEL_DIR="/mnt/workspace/models/font-info/qwen3-4b-sft"
```

Edit this path before running.

### 3. Prepare Data

The Retool/TIR scripts expect JSONL data with at least:

```json
{"prompt": "...", "label": "..."}
```

The default paths in the scripts are:

```bash
DATA_DIR="/mnt/workspace/public/data/RLdata/tir"
TRAIN_DATA="${DATA_DIR}/tir_train_data/dapo-math-17k.jsonl"
EVAL_DATA="${DATA_DIR}/tir_val_data/aime-2024.jsonl"
```

Update `PROJECT_DIR`, `DATA_DIR`, `MODEL_DIR`, and `LOG_DIR` near the top of each script before launching.

### 4. Reproduce Async GRPO

Entry script:

```bash
scripts/run-qwen3-4B-grpo.sh
```

This script runs:

- entry: `train_async.py`
- algorithm: `--advantage-estimator grpo`
- async rollout: Windowed FIFO
- train GPUs: 4
- rollout GPUs: 4
- rollout group: `--rollout-batch-size 8` and `--n-samples-per-prompt 8`
- global batch size: 64

Run:

```bash
ray stop --force || true
bash scripts/run-qwen3-4B-grpo.sh
```

Main outputs:

```text
qwen3_4b_grpo_checkpoints/
  checkpoints/<exp_name>/
  tensorboards/<exp_name>/
  <exp_name>/rollout_log/
```

Useful parameters to edit:

```bash
--num-rollout
--global-batch-size
--rollout-batch-size
--n-samples-per-prompt
--rollout-max-response-len
--sglang-server-concurrency
--windowed-fifo-max-delay-step
--windowed-fifo-max-prefetch-steps
```

### 5. Reproduce Async SAO with Critic Pretraining

SAO is a two-stage workflow:

1. Pretrain the critic.
2. Run PPO/SAO initialized from the pretrained critic.

#### Stage A: critic pretraining

Entry script:

```bash
scripts/sao-critic-pretrain.sh
```

This script runs:

- entry: `train_async.py`
- actor learning rate: 0
- critic-only steps: all rollout steps
- value loss: classification critic with HL-Gauss targets
- train GPUs: 4
- rollout GPUs: 4
- output critic checkpoint:

```text
qwen3_4b_checkpoints/checkpoints/sao-classify-pretrained-critic/critic
```

Run:

```bash
ray stop --force || true
bash scripts/sao-critic-pretrain.sh
```

#### Stage B: SAO training

Entry script:

```bash
scripts/run-qwen3-4B-sao-fifo.sh
```

This script runs:

- entry: `train_async.py`
- algorithm: `--advantage-estimator ppo`
- async rollout: Windowed FIFO
- single rollout per prompt: `--n-samples-per-prompt 1`
- critic baseline: enabled
- DIS: `--use-dis`
- length-adaptive GAE: `--enable-length-adaptive`
- skip observation GAE: `--skip-observation-gae`
- classification critic: `--value-loss-type classification`
- pretrained critic path:

```bash
PRETRAINED_CRITIC_DIR="${LOG_DIR}/checkpoints/sao-classify-pretrained-critic/critic"
```

Run:

```bash
ray stop --force || true
bash scripts/run-qwen3-4B-sao-fifo.sh
```

Main outputs:

```text
qwen3_4b_checkpoints/
  checkpoints/<exp_name>/actor/
  checkpoints/<exp_name>/critic/
  tensorboards/<exp_name>/
  <exp_name>/rollout_log/
```

Resume behavior:

- If both `actor/latest_checkpointed_iteration.txt` and `critic/latest_checkpointed_iteration.txt` exist, the script resumes from them.
- If only one side exists, the script exits to avoid loading mismatched actor/critic state.
- If neither exists, actor starts from `${MODEL_DIR}_torch_dist` and critic starts from `PRETRAINED_CRITIC_DIR`.

Important SAO parameters:

```bash
--use-dis
--dis-clip
--dis-clip-low
--critic-lr
--critic-update-ratio
--critic-freeze-params-name-list
--value-loss-type classification
--value-num-bins
--value-reward-range
--value-target-type hl_gauss
--hl-gauss-sigma-ratio
--update-weights-interval
```

### 6. Reproduce Vanilla PPO

Entry script:

```bash
scripts/run-qwen3-4B-vanillappo.sh
```

This script runs:

- entry: `train.py`
- algorithm: `--advantage-estimator ppo`
- synchronous rollout/train flow
- colocated training and rollout: `--colocate`
- actor GPUs: 8
- rollout batch size: 32
- global batch size: 32

Run:

```bash
ray stop --force || true
bash scripts/run-qwen3-4B-vanillappo.sh
```

Main outputs:

```text
qwen3_4b_vanillappo_checkpoints/
  checkpoints/<exp_name>/
  <exp_name>/tensorboards/
  <exp_name>/rollout_log/
```

Vanilla PPO is the right baseline when you want the synchronous PPO behavior without the Windowed FIFO asynchronous pipeline.

### 7. Monitor Training

Start TensorBoard from the project root:

```bash
tensorboard --logdir qwen3_4b_checkpoints/tensorboards --host 0.0.0.0 --port 6006
```

For vanilla PPO:

```bash
tensorboard --logdir qwen3_4b_vanillappo_checkpoints --host 0.0.0.0 --port 6006
```

For GRPO:

```bash
tensorboard --logdir qwen3_4b_grpo_checkpoints/tensorboards --host 0.0.0.0 --port 6006
```

Useful TensorBoard groups:

| Group | What to check |
| --- | --- |
| `train/*` | PPO/GRPO loss, policy loss, entropy, critic loss, DIS/TIS metrics. |
| `rollout/*` | reward, response length, tool call count, KL, staleness, queue behavior. |
| `eval/*` | eval reward/accuracy and eval rollout statistics. |
| `perf/*` | rollout time, actor train time, logprob time, token throughput. |

Rollout text logs are written as JSONL:

```text
<LOG_DIR>/<EXP_NAME>/rollout_log/rollout_outputs/rollout_<step>.jsonl
<LOG_DIR>/<EXP_NAME>/rollout_log/eval_outputs/eval_<dataset>_<step>.jsonl
```

These files are the fastest way to inspect tool calls, final answer parsing, reward fields, truncation, and compaction metadata.

## Script Reference

| Experiment | Script | Entry | Notes |
| --- | --- | --- | --- |
| Async GRPO | `scripts/run-qwen3-4B-grpo.sh` | `train_async.py` | Windowed FIFO, grouped sampling. |
| Critic pretrain | `scripts/sao-critic-pretrain.sh` | `train_async.py` | Critic-only training, classification value loss. |
| Async SAO | `scripts/run-qwen3-4B-sao-fifo.sh` | `train_async.py` | PPO + critic + DIS + Windowed FIFO. |
| Vanilla PPO | `scripts/run-qwen3-4B-vanillappo.sh` | `train.py` | Synchronous colocated PPO baseline. |
| Checkpoint eval | `scripts/eval-qwen3-4B-vanillappo-checkpoint.sh` | `train.py` | Eval-only checkpoint inference. |
| SFT sanity check | `scripts/debug-infer-qwen3-4B-sft.sh` | `train.py` | Verifies SFT checkpoint generation. |

## Running on DLC or Multi-Node Ray

For managed jobs, use `scripts/ray.sh` as the outer launcher and select the actual training script through `SCRIPTS`:

```bash
export WORK_DIR=/mnt/workspace/wangdy/code/zj-slime-0722
export NNODES=1
export N_GPUS_PER_NODE=8
export SCRIPTS=run-qwen3-4B-sao-fifo.sh
export EXP_NAME=sao-classify-critic-pretrain

bash /mnt/workspace/wangdy/code/zj-slime-0722/scripts/ray.sh
tail -f /dev/null
```

`ray.sh` reads `WORK_DIR` from the environment, starts or joins the Ray cluster, and launches `${SCRIPTS}` from the project scripts directory.

## Notes and Common Pitfalls

- Always edit hard-coded `PROJECT_DIR`, `DATA_DIR`, `MODEL_DIR`, and `LOG_DIR` before running.
- The HuggingFace checkpoint and `${MODEL_DIR}_torch_dist` checkpoint must match.
- Classification critic checkpoints are not compatible with MSE critic heads unless the value head is reinitialized intentionally.
- For SAO resume, actor and critic checkpoint steps must match.
- `--rollout-max-response-len` controls rollout generation length/context budget in the Retool rollout.
- `--max-tokens-per-gpu` controls Megatron dynamic batching capacity per GPU, not the SGLang KV cache size.
- If SGLang reports KV cache pressure, reduce `--sglang-server-concurrency`, reduce max response length, or increase rollout GPU memory fraction.
- If Ray job submission fails because an old cluster is still running, run `ray stop --force` before restarting.

## Acknowledgement

This work builds on the original slime training framework and its Megatron + SGLang integration. The added experiment scripts and training logic are focused on reproducible agentic RL experiments with GRPO, PPO, SAO, critic pretraining, and classification value losses.
