---
title: Scaling：参数、数据与算力怎样配平
description: 从 Transformer 矩阵逐层算出参数量，再理解显存、6ND 与数据参数配比
---

# Scaling：参数、数据与算力怎样配平

## 先做预测

先不要按按钮，写下你的答案：

1. 一个 $d=768$、12 heads 的 Attention，参数量是 $4d^2$，还是 $12\times4d^2$？
2. 把普通 FFN 换成相同 $d_{ff}$ 的 SwiGLU，为什么参数会增加？
3. Query heads 保持 32，把 KV heads 从 32 降到 8，哪两张矩阵会变小？
4. 参数与训练 Token 都翻倍，$6ND$ 近似计算量会变成几倍？

::: info 来源与计算口径
实验跟随 Stanford CS336 [Lecture 2：Resource Accounting](/lectures/?trace=var/traces/lecture_02.json) 的“逐张量记账、12 bytes/参数、$6ND$”方法，并结合 [Lecture 3：Architecture and Hyperparameters](https://stanford-cs336.github.io/spring2026/) 中的 FFN 比例、SwiGLU、head dimension 和 GQA 架构。为让第一遍容易看懂，默认计算现代 bias-free Decoder-only Transformer。
:::

<ScalingLab />

## 建议按这四轮操作

### 第一轮：证明多头没有白送一份参数

点“GPT-2 124M 式”，记下 Attention 参数量。只改变 Query heads，保持 $d$ 不变。每个 head 会变窄，但 Q/K/V/O 的总输出宽度仍是 $d$，所以参数量不应按 head 数倍增。

### 第二轮：找到参数大户

点“LLaMA 7B 式”，观察彩色构成条。多数参数通常在 FFN，而不是 Norm。把 $d_{ff}$ 往左拖，查看总参数如何变化；再切换普通 FFN 与 SwiGLU，解释“2 个矩阵”和“3 个矩阵”的差异。

### 第三轮：看 GQA 到底省了什么

在相同 $V,L,d,d_{ff}$ 下，把 KV heads 从 32 改成 8。Q 投影和 O 投影不变，只有 K、V 投影缩小。GQA 还会减少推理时 KV Cache，但那是运行时状态，不属于模型参数；可在 [KV Cache 实验](/labs/kv-cache)继续验证。

### 第四轮：从“能存下”走到“能训完”

得到总参数后，先读 12 bytes/参数的训练状态小计，再拖训练 token：权重显存不变，但 $6ND$ 与 GPU·天线性增加。最后把利用率从 30% 调到 60%：总 FLOPs 不变，完成同一工作的时间约减半。

## 完成标准

你应该能够不查资料解释：

- 为什么普通 MHA 一层约为 $4d^2$，而不是 $4hd^2$；
- 普通 FFN、SwiGLU 与 GQA 分别改动了哪些矩阵；
- “7B 参数”怎样变成 BF16 权重大小和 Adam 训练状态大小；
- 为什么参数量 $N$ 必须与训练 token $D$、硬件效率一起讨论。

::: warning 这是预算直觉，不是训练报价器
实验使用 $6ND$、A100 峰值算力和简单利用率做粗估，没有加入激活/临时张量、通信、故障恢复、数据管线、序列长度、Attention 二次项与硬件差异。模型预设是“结构近似”，不是对某个正式 checkpoint 配置的逐项复刻。
:::

回到系统教程：[第 14 课 Scaling：参数、数据与算力怎样配平](/beginner/25-data-scaling)。
