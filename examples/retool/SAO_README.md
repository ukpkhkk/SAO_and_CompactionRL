# SAO(Single-rollout Asynchronous Optimization)
SAO是一个基于单轨迹异步训练的大模型强化学习框架，专门针对长程Agent任务的稳定高效训练而设计。
## 核心特性
| 特性 | 说明 |
|------|------|
| **单轨迹异步训练** | 每个prompt只生成一条轨迹，完成后立即触发训练更新，避免组采样等待 |
| **长度自适应GAE (VAPO)** | λ = 1 - 1/(α·L)，让长短序列获得均衡的优势估计 |
| **跳过观测GAE** | 优势只在动作token之间传播，避免环境反馈的不确定性干扰 |
| **价值模型快速更新 (K=2)** | 每批次Critic更新2次，降低单轨迹训练的梯度方差 |
| **双端截断重要性采样 (DIS)** | 支持TIS和IcePop两种策略，减少离线策略偏差 |
| **冻结价值模型Attention** | 仅训练MoE投影层，抑制价值估计震荡 |

---
## 核心配置参数

   --advantage-estimator ppo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type k1
   --entropy-coef 0.00

   # DIS for SAO
   --use-dis
   --dis-clip 3.0
   --dis-clip-low 0.8

   # length-adaptive GAE
   --adaptive-alpha 1.5
   --enable-length-adaptive # for SAO
   --skip-observation-gae # for SAO

   # critic lr
   --critic-lambd 1.0
   --critic-lr 1e-5
   --critic-lr-warmup-iters 10
   --critic-update-ratio 2 # number of steps to critic
   --num-critic-only-steps 10 # Number of steps to train only the critic at the beginning of training
   --critic-freeze-params-name-list self_attention # for SAO


## slime-ppo相关说明
