---
title: 综合练习与参考要点
description: Kimi K3 自学教程附录
---

# 附录 D　综合练习与参考要点

### 练习 1：三种“省资源”不要混

分别说明 KDA、MLA、MoE 对训练 FLOPs、decode cache、权重存储、通信的主要影响。

**参考要点**：KDA 用固定 recurrent state 取代大部分随 `T` 增长的 KV，并改变 attention 计算；MLA 仍是 global softmax，但低秩压缩 KV；MoE 扩总参数、按 token 稀疏激活，主要挑战是 expert weight/communication。三者作用轴不同。

### 练习 2：重复 key

令归一化 key `k` 连续两次出现，value 先为 `v_1` 后为 `v_2`，`α=1, β=1`。若第一次更新后 `S_1^Tk=v_1`，求第二次后 `S_2^Tk`。

**答案**：`S_2=S_1+k(v_2-v_1)^T`，若 `k^Tk=1`，则 `S_2^Tk=v_2`。这体现 delta overwrite。

### 练习 3：MoE 数量

`m=8192` token、`n=256` experts、Top-8，理想每 expert 收多少 token？若某 expert 收 400 个，为什么既可能拖慢系统又影响学习？

**答案**：`q=mk/n=256`。400 是 1.5625× 目标负载，所在 rank 可能成为 straggler；其他专家数据不足、该专家过度共享，也改变 specialization。

### 练习 4：AttnRes 退化

若所有 depth logits 相同，Full AttnRes 输出是什么？与标准 residual sum 有何尺度差异？

**答案**：softmax 均匀平均历史 values，而标准 residual 是求和；二者方向相似但尺度随历史长度不同，后续 norm 会进一步改变效果。

### 练习 5：长上下文真假

设计三个层次测试 1M context：接口可运行、远距检索、跨多处组合推理。

**参考要点**：

1. 1M random/valid tokens 能 prefill/decode 且数值稳定；
2. needle 位于不同深度与干扰分布，测定位取回；
3. 答案必须联合开头/中间/结尾多个事实并执行约束，防局部 shortcut。

### 练习 6：Partial rollout stale policy

一条轨迹前 70% 由 `π_0` 生成，暂停两次后最后 30% 由 `π_2` 生成；训练时已是 `π_3`。列出三种 mismatch。

**参考要点**：trajectory 内行为 policy 不一致；数据相对 current policy stale；environment state 由旧动作塑造，后续 state distribution 也偏离 `π_3`。需要 token-level ratio/regularization、版本记录与稳定 update。

### 练习 7：Agent benchmark 公平性

模型 A 用 20 次工具、模型 B 用 200 次，B 成功率高 5 分。你会如何判断谁更好？

**参考要点**：给出 success–cost Pareto；控制相同 budget 再比；报告 wall time、token/API cost、failure/recovery；检查 harness/tool 差异。没有唯一答案，取决于产品预算与 SLO。

### 练习 8：KDA prefix 命中

MLA prefix 匹配到 token 4096，但最近共同 KDA checkpoint 在 3584。可从哪里恢复？

**答案**：最多从 3584，因为两类状态必须在同一 boundary 一致；3584–4096 的 MLA cache 即使存在也不能单独使用来跳过 KDA recurrence。

### 练习 9：Scaling claim

要把 K3 的 2.5× 分解成 KDA、AttnRes、Stable LatentMoE、数据和 optimizer 各自贡献，需要什么实验？

**参考要点**：多尺度 factorial/逐步 ablation，每个 family 独立调优 LR/batch/shape，固定数据与 compute 口径，多 seed/OOD loss；模块有交互，贡献未必可加，总成本极高。

### 练习 10：职业迁移

从以下选一项写一页设计：

- 用视觉 Agent 自动修 UI；
- 用 verifier RL 优化 CUDA kernel；
- 为 256K 代码 Agent 设计 cache/sandbox；
- 在 3B 模型比较 softmax/KDA hybrid。

必须包含：可验证目标、数据/环境、模型改动、资源预算、baseline、ablation、失败模式、安全边界。

---
