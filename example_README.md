
<div align="center">

---

## What is miles-values?

**miles-values** is a fork of [**miles**](https://github.com/radixark/miles) (itself a fork of [slime](https://github.com/THUDM/slime)) that adds a **value-model / critic-centric RL stack** for LLM and agentic post-training. It is built to make **PPO-with-a-critic practical again** in the era of long-horizon agentic RL, where the group-sampling assumptions behind GRPO start to break down.

It brings together four ideas that reinforce each other:

1. **SAO (Single-rollout Asynchronous Optimization)** — one rollout per prompt + a good critic, instead of a large sampling group per prompt.
2. **Colocate-PPO** — actor and critic time-share the *same* GPUs, so the critic costs **zero extra GPUs**.
3. **Regression-as-Classification value loss** — train the critic as a categorical distribution (two-hot / HL-Gauss) instead of scalar MSE.
4. **Value Pretrain** — warm-start the critic on offline rollouts before RL, so early advantages are not pure noise.

These notes are written up in three Zhihu articles:

| # | Article                                                                              | Topic                                                                    |
| :- | :----------------------------------------------------------------------------------- | :----------------------------------------------------------------------- |
| 1 | [SAO and small-scale experiments](https://zhuanlan.zhihu.com/p/2064452938018322060)   | SAO, Colocate-PPO, regression-as-classification, first math experiments  |
| 2 | [credit-assignment is all you need](https://zhuanlan.zhihu.com/p/2064646116927344943) | Why agentic RL stalls, PivotRL / prefix-replay, credit assignment        |
| 3 | [Notes on value-pretrain](https://zhuanlan.zhihu.com/p/2066503866091410176)           | Value pretraining at scale, HL-Gauss vs MSE, 35A3 + 128k agentic results |

---

## 1. SAO — Single-rollout Asynchronous Optimization

> *Single-rollout Asynchronous Optimization for Agentic Reinforcement Learning* (arXiv:2607.07508, Hou et al., 2026 — used to train GLM-5.2, 750B-A40B).

**The problem with GRPO in async agentic RL.** Mainstream RLVR uses GRPO: sample `n` trajectories per prompt and use the in-group mean as a baseline. In an **asynchronous agentic** setting (long trajectories, multi-turn tool calls, variable environment latency) this becomes awkward:

- trajectories in a group finish at very different times — you **must wait for the slowest** before you can compute the group baseline, so throughput is dragged down by the long tail (or you accept much higher staleness);
- the "all samples in a group share one frozen policy version" assumption is hard to maintain asynchronously, which amplifies off-policy drift;
- group sampling generalizes poorly to *online* tasks where the environment itself changes.

**The SAO answer: single rollout + a good critic.**

- **(a) Single-rollout** — sample just **1** trajectory per prompt. Without a group there is no in-group baseline, so variance degrades to REINFORCE levels — which is exactly why you need…
- **(b) A sufficiently good value model (critic)** as the baseline, with advantages from **GAE**. This puts PPO-with-critic back at the center of LLM RL. SAO adds two agentic-specific GAE tricks:
  - **Length-adaptive GAE** (from VAPO): `λ_policy = 1 − 1/(α·l)`, so the earliest tokens of a short trajectory and a 10k-token trajectory get comparable gradient weight;
  - **Skip-observation GAE**: environment-injected `observation` tokens are not model-generated, so GAE steps *over* the obs span and connects the value of adjacent actions directly, avoiding noise propagation.
- **(c) DIS — dual-sided token-level importance clipping** for async stability: token ratio `r_t = exp(log π_θ − log π_rollout)` is clipped on both sides; tokens outside `[1−ε_l, 1+ε_h]` are zeroed out (no gradient). The rollout policy is used directly as the behavior proxy — no historical checkpoint set to maintain.

**The takeaway:** the SAO recipe boils down to *single rollout + a good critic*. A good critic has two costs — **GPU** (a classic PPO critic wants its own cards) and **quality** (cold start, long trajectories, sparse reward). The next three sections are precisely the solutions to those two costs.

---

## 2. Colocate-PPO: critic with zero extra GPU

This solves the *"the critic wants a whole extra set of GPUs"* cost: **actor and critic time-share the same GPUs**, so the extra GPU footprint is **zero**.

### Time-division multiplexing

```
critic wake_up → critic train → backup → critic sleep
→ actor  wake_up → actor  train → backup → actor  sleep
→ weight update → rollout → …
```

- Uses `torch_memory_saver` (TMS) pause/resume to hand off VRAM: `sleep()` = TMS pause (frees VRAM), `wake_up()` = TMS resume.
- **Weight backup is a correctness requirement.** TMS resume can return *zeroed* VRAM (observed with the LD_PRELOAD hook mode), so the model is backed up at init, before every sleep, and restored after every wake. Skipping this fails **silently**: no crash, but the critic loss sits at ~0.002 (the value head is silently zeroed) and explained variance stays at 0.
- NCCL detail: after a TMS pause the communicator's GPU buffers are invalid, so the NCCL group to the rollout engine is torn down before pause and, on `update_weights`, re-established (wake → reconnect → broadcast → sleep again).

### Value transfer over Ray object store (not NCCL)

After the critic finishes training it `detach().cpu()`s the values and passes them to the actor as a return value through the Ray object store (`submit_train(external_data=refs)`). In sync mode the **Ray dependency chain** enforces ordering: the actor task's args reference the critic's `ObjectRef`, so Ray guarantees the critic fully completes (including its final `sleep`) before the actor starts (before it wakes) — naturally preventing the two models from occupying the GPU at once and OOM-ing.

### Two modes

| Mode                                                       | Layout                                          | Weight update                                 |
| :--------------------------------------------------------- | :---------------------------------------------- | :-------------------------------------------- |
| **sync** (`train.py --colocate --colocate-critic`) | 8 GPUs fully shared (actor + critic + engine)   | IPC / CPU backup (can update while asleep)    |
| **async** (`train_async.py --colocate-critic`)     | actor/critic colocated + dedicated rollout GPUs | NCCL: wake → reconnect → broadcast → sleep |

---

## 3. Regression-as-Classification value loss

This attacks one half of the *"the critic is hard to train well"* cost. Following Farebrother et al., **"Stop Regressing: Training Value Functions via Classification for Scalable Deep RL"** (ICML 2024) — an idea traceable to C51 distributional RL.

**How it works:**

1. place `num_bins` support points over the reward range `[low, high]` (a categorical support);
2. the value head outputs `num_bins` logits instead of 1 scalar; the **scalar prediction = softmax expectation** `V = softmax(logits) · support`;
3. project the scalar target `R` into a **target distribution** and train with cross-entropy:
   - **two-hot**: linear interpolation onto the two adjacent bins (expectation is exactly `R`);
   - **HL-Gauss** (recommended): treat `R` as a sample of `N(R, σ²)`, integrate the Gaussian CDF over each bin, `σ = 0.75 × bin_width`. Compared to two-hot this spreads the supervision over ~6 bins, and the label-smoothing effect noticeably improves robustness and generalization.

**Why it helps** (the *Stop Regressing* result): cross-entropy gradients are naturally bounded (softmax saturation) and robust to noisy targets; classification forces the network to learn the *distributional structure* of the value rather than a single point; and the advantage over MSE *grows with model scale*.

**In this repo:**

```bash
--value-loss-type classification \
--value-num-bins 51 \
--value-reward-range 0.0 1.0 \
--value-target-type hl_gauss \
--hl-gauss-sigma-ratio 0.75
```

- `model_provider.py`: value head output dim = `num_bins` (⚠️ not compatible with an MSE critic checkpoint);
- `logit_processors.py`: `predict_values_from_logits` = softmax expectation (the GAE/advantage side is completely unaware — it still receives a scalar `V`);
- `losses.py`: HL-Gauss / two-hot target distribution + cross-entropy, reporting `value_accuracy / value_entropy / value_confidence`;
- for both MSE and classification, a **loss-mask-aware** `value_explained_variance` is reported (only over `loss_mask=1` tokens, aggregated across microbatch / DP / CP via sufficient statistics) — this is the core signal for whether the critic is actually learning: a healthy run climbs monotonically from deeply negative toward 0 / positive.

The classification value loss is **orthogonal** to SAO (whose paper uses plain MSE) and composes freely with value pretraining, colocate-critic, and `gae_adaptive`.

---

## 4. Value Pretrain

This attacks the other half of the critic-quality cost: a randomly initialized value head cold-starts terribly (early advantages are pure noise). The fix is to **pretrain the critic on offline rollout data before RL** — the larger and more diverse the pretraining corpus, the more robust the initialization (VAPO and SAO both call for *large-scale* value pretraining).

Article 3 works through a paper — *Do You Really Need to Pretrain Q-Functions for Online RL Fine-tuning?* (arXiv:2607.27203) — with two useful conclusions:

1. a **naive** pretrained Q-function can actually *hurt* RL training;
2. pretraining the value model on trajectories **mixed from several different policies** converges faster and to better metrics than a single-policy corpus.

Practical notes from the experiments:

- value pretrain is run as a **separate training job** (not just `--num-critic-only-steps`);
- when the value-pretrain corpus and the RL training set **overlap** (e.g. pretrain the critic on rollouts of the SFT-filtered RL data), explained variance stays at a healthy level and downstream eval improves — an economical setup when large-scale pretraining is out of reach;
- **HL-Gauss > two-sided > MSE** as the value-pretrain objective (EV grows faster and higher);
- to build critic targets offline you can fix a prefix and sample suffixes (a tree-search-like Monte-Carlo estimate), then train the critic on `(prefix, return)` pairs.

### When the critic still isn't good enough → EVPO

When the critic is unreliable, *whom do you trust?* **EVPO — Explained Variance Policy Optimization** (arXiv:2604.19485) uses per-group EV as an online switch, hard-toggling each prompt group between **PPO (critic baseline)** and **GRPO (group-mean baseline)**:

```
ev_B = 1 − Var_B(G_m − V(s_{m,t})) / Var_B(G_m)

ev_B >  threshold  →  b = V(s)          (critic mode / PPO)
ev_B ≤  threshold  →  b = mean_B(G)     (group-mean mode / GRPO)
```

The paper's ablated optimum is **threshold = 0.0** — a critic is used the moment it beats "predict the group mean," which is exactly what EV measures.

---

## Credit assignment for agentic RL (article 2)

Reasoning-RL is forgiving: with a hard enough task, a large enough group, low train/infer mismatch and non-exploding entropy, models tend to climb regardless of credit assignment. **Agentic RL is not** — swapping data, adding KL, adding entropy, reward shaping, or brute-forcing with bigger batches / larger groups tends to move the needle only randomly. Two failure modes dominate:

1. **bad behaviors inside *correct* trajectories** get reinforced and amplified when data quality is low, eventually failing the harder problems;
2. **correct reasoning / tool-call paths inside *wrong* trajectories** get uniformly punished by critic-free GRPO — one signal for everyone trains mediocrity.

Both point at **partial credit assignment**. A cheap, effective baseline is **PivotRL / prefix-replay**:

<div align="center">
<img src="imgs/pivot_rl_diagram.jpg" alt="PivotRL / prefix-replay" width="720">
</div>

1. offline-SFT-filter a batch of data;
2. find a good cut point via LLM-judge / entropy (e.g. first-error-step detection, high-entropy branch points);
3. cut offline, feed the **prefix as a fixed prompt** (prefix-replay, not optimized) and only roll out + optimize the **suffix**, otherwise standard GRPO.

Even this naive version (offline cut, replay prefix, rollout+optimize suffix only) gives **consistent, non-random** gains on several benchmarks, scales with model size and steps, and trains **faster** (shorter suffixes) — a good fit for tight-deadline, low-resource settings. With more compute, tree-rollout + pivot-node selection (estimating q-values) is the stronger option. If resources allow, a **value pretrain** is the more principled long-term fix — and IID value-pretrain / RL data helps even OOD.

---

## Experiments

### Agentic RL: HL-Gauss value loss + SAO (single-rollout PPO)

Eval accuracy over RL steps on three held-out agentic benchmarks (dataset names withheld). Raw (light) vs. EMA-smoothed (bold).

<div align="center">
<img src="imgs/sao_agentic_eval_curves.png" alt="SAO + HL-Gauss agentic eval curves" width="960">
</div>

### Math (single-turn), qwen35-4b-base, deepmath-103k

Config: `bs=16`, `n_rollout=8`, `actor_lr=1e-6`, `critic_lr=5e-6`, `warmup=20`, `critic_update_steps=1`, `max_length=16k`.

> On a 4B base with a small rollout count the critic is intrinsically weak, so a reasonably balanced set of positive/negative samples matters. Larger models tolerate a much smaller rollout count without hurting convergence.

**Exp 1 — MSE (pink) vs HL-Gauss (purple).** HL-Gauss is the better critic loss: MSE's aime24 plateaus after ~200 steps while HL-Gauss is better on average.

<div align="center"><img src="imgs/exp1_mse_vs_hlgauss.jpg" alt="MSE vs HL-Gauss" width="820"></div>

**Exp 2 — HL-Gauss (purple) vs HL-Gauss + value-pretrain (light green).** Value-pretrain (on trajectories rolled before the DAPO-17k run) gives a markedly better EV starting point; the gap narrows as training proceeds, and aime25 ends up better.

<div align="center"><img src="imgs/exp2_hlgauss_vs_pretrain.jpg" alt="HL-Gauss vs value-pretrain" width="820"></div>

**Exp 3 — value-pretrain data IID with RL data matters.** Value model was pretrained on DAPO-17k; comparing RL on deepmath-17k (light green) vs DAPO-17k (grey). When the value-pretrain set and the RL set overlap, EV holds at a high level (~0.2) and both aime24/25 climb better.

<div align="center"><img src="imgs/exp3_iid_value_pretrain.jpg" alt="Value-pretrain IID with RL data" width="820"></div>

**Exp 4 — PPO + HL-Gauss + value-pretrain (grey) vs GRPO (blue).** PPO-with-critic edges out the GRPO baseline and is broadly close — consistent with prior single-turn zero-RL findings.

<div align="center"><img src="imgs/exp4_ppo_vs_grpo.jpg" alt="PPO vs GRPO" width="820"></div>

**PivotRL / prefix-replay results** — consistent (non-random) gains on several benchmarks:

<div align="center"><img src="imgs/pivot_rl_results.jpg" alt="PivotRL results" width="820"></div>

*Math summary:* (1) bigger model + critic-pretrain → can use a smaller rollout count; (2) overlapping value-pretrain / RL data (train the critic on rollouts of your SFT-filtered data) is the economical sweet spot; (3) HL-Gauss > MSE (EV grows faster and higher during value-pretrain).

### Scaling up: qwen35-35a3-base + long-horizon agentic

The paper *Do You Really Need to Pretrain Q-Functions…* motivated the 35A3 study:

<div align="center"><img src="imgs/qfunction_pretrain_paper.jpg" alt="Q-function pretraining paper" width="720"></div>

Config: dapo-17k, qwen35-35a3-base, 16k length; three settings (mse / hl-gauss / hl-gauss-pretrain, the latter pretrained 500 steps on prior 35A3 rollouts).

- **Convergence** — trained long enough, all three reach a similar final metric (aime24 0.91 / aime25 0.85 at 16k rollout length):

  <div align="center"><img src="imgs/vp_35a3_convergence.jpg" alt="35A3 convergence" width="820"></div>
- **Sample efficiency** — `hl-gauss-pretrain > hl-gauss > baseline`, as expected (final metrics are close because the pretrain corpus is still single-policy):

  <div align="center"><img src="imgs/vp_35a3_sample_efficiency.jpg" alt="35A3 sample efficiency" width="820"></div>
- **Explained variance** — hl-gauss-pretrain is better at the start and throughout:

  <div align="center"><img src="imgs/vp_35a3_ev.jpg" alt="35A3 explained variance" width="820"></div>
- **128k search-agentic task** — single-rollout PPO (SAO) with a small value-pretrain (~60–70k samples, warmup 10 steps). Some offline benchmarks gain **8–10 points**; EV starts low because the pretrain corpus distribution differs from the current policy, and single-rollout still benefits from a stronger value model (or EVPO-style critic/group-mean switching):

  <div align="center"><img src="imgs/sao_search_agent_128k.jpg" alt="128k search-agentic SAO" width="820"></div>

*Value-pretrain summary:* (1) it wants **large scale** — in data volume *and* policy diversity; (2) **HL-Gauss** (regression-as-classification) is the better objective; (3) even a naive, small agentic value-pretrain + single-rollout PPO can lift several benchmarks by 8–10 points offline.

---

## Quick Start

miles-values follows the miles / slime workflow. Install as usual:

```bash
pip install -r requirements.txt
pip install -e .
```

### Colocate-PPO with an HL-Gauss classification critic (sync)

```bash
python train.py \
    --advantage-estimator gae_adaptive \
    --colocate --colocate-critic \
    --value-loss-type classification \
    --value-num-bins 51 \
    --value-reward-range 0.0 1.0 \
    --value-target-type hl_gauss \
    --hl-gauss-sigma-ratio 0.75 \
    --critic-lr 5e-6 \
    --model-name qwen3-4b-base \
    --hf-checkpoint /path/to/qwen3-4b-base-hf
```

Add a value-pretrained critic with `--critic-load /path/to/value_pretrain_ckpt`. For the async / SAO layout use `train_async.py --colocate-critic` with dedicated rollout GPUs. See the flags in `miles/utils/arguments.py` (`--value-loss-type`, `--value-num-bins`, `--value-target-type`, `--hl-gauss-sigma-ratio`, `--colocate-critic`, `--num-critic-only-steps`, `--num-critic-epochs`).

---

## References

- **SAO**: *Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning*, arXiv:2607.07508
- **VAPO** (length-adaptive GAE): Yue et al., 2025
- **Stop Regressing**: *Training Value Functions via Classification for Scalable Deep RL*, Farebrother et al., ICML 2024
- **C51**: *A Distributional Perspective on Reinforcement Learning*, Bellemare et al., 2017
- **EVPO**: *Explained Variance Policy Optimization*, arXiv:2604.19485
- **Q-function pretraining**: *Do You Really Need to Pretrain Q-Functions for Online RL Fine-tuning?*, arXiv:2607.27203
- **PivotRL**: *High Accuracy Agentic Post-Training at Low Compute Cost*; *Agent RL via Pivotal-Aware Self-Feedback Retry*; *Multi-Turn On-Policy Distillation with Prefix Replay*; *TreeRL*

### Blog notes (Zhihu)

- [SAO and small-scale experiments](https://zhuanlan.zhihu.com/p/2064452938018322060)
- [credit-assignment is all you need](https://zhuanlan.zhihu.com/p/2064646116927344943)
- [Notes on value-pretrain](https://zhuanlan.zhihu.com/p/2066503866091410176)

---

## Acknowledgements

Built on [**miles**](https://github.com/radixark/miles) and [**slime**](https://github.com/THUDM/slime), with [SGLang](https://github.com/sgl-project/sglang) for rollout and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for training.

<div align="center">
