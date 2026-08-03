---
title: RLHF 奖励与 KL 权衡实验
description: 调整 KL 系数，观察策略在奖励、偏离基线与奖励投机之间的选择
---

# RLHF：为什么不能只追求更高奖励

## 先做预测

1. 当 $\beta=0$ 时，策略会不会在意自己偏离参考模型多远？
2. KL 系数不断增大，回答会更大胆还是更接近原模型？
3. 奖励模型给高分，是否足以证明回答真正更好？

<PreferenceRLLab />

这个实验把 PPO/RLHF 中的一条关键张力压成四个候选回答。真实系统是在每个 Token 上学习策略，并需要优势估计、裁剪、奖励归一化和大量在线采样。

继续学习：[第 30 课奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference)和[第 29 课 PPO](/beginner/44-rl-ppo)。
