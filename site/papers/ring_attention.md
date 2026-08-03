---
title: Ring Attention with Blockwise Transformers for Near-Infinite Context
description: Ring Attention 把序列块分到多张卡，Query 留在本卡，K/V 块沿设备环传递；每到一站就计算一块 attention 并在线合并 softmax。
---

# Ring Attention with Blockwise Transformers for Near-Infinite Context

<div class="paper-lesson-meta"><span>方向选读</span><span>16 页</span><span>arXiv 2310.01889</span></div>

<div class="lesson-lead">Ring Attention 把序列块分到多张卡，Query 留在本卡，K/V 块沿设备环传递；每到一站就计算一块 attention 并在线合并 softmax。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像围桌传阅资料：每个人保留自己的问题，资料包沿圆桌传一圈后，每个人都看过全部资料。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**单卡无法容纳超长序列的输入与 activation；简单切序列又要求每个 Query 看到所有设备上的 K/V。

**它在整条学习链中的位置：**用环形通信把完整 attention 扩到多卡长序列

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：每张卡持有一段 Q/K/V。
2. **再看核心变化**：K/V 块按 ring 传给下一张卡，Q 块保持本地。
3. **最后看输出**：通信下一块的同时计算当前块，利用 blockwise online softmax 精确累计结果。

## 论文拿什么证明

- 论文展示上下文长度可随设备数近线性扩展，并在足够块计算量下隐藏大部分通信。
- 方法保持全 attention 语义，而不是局部窗口近似。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- Ring Attention 是 MLA/full attention 的 context parallel 对照；KDA 只需扫描固定状态片段，通信对象和复杂度不同。

继续补背景：[第 9 章 · Scaling 与长上下文](/guide/ch09) · [第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- 上下文‘近无限’受总显存、网络带宽、训练稳定性和数据限制。
- 通信能否隐藏取决于块计算够不够重以及网络拓扑。

## 原文应该怎么读

**推荐范围：**读 blockwise attention 与 ring 图；重点理解通信怎样和计算重叠

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2310.01889" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2310.01889" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：54,320 字符 · 21 个标题</span></div>

### 原文章节地图

1. Ring Attention with Blockwise Transformers for Near-Infinite Context
2. Hao Liu, Matei Zaharia, Pieter Abbeel UC Berkeley
3. 1 Introduction
4. 2 Large Context Memory Constraint
5. 3 Ring Attention with Blockwise Parallel Transformers
6. 4 Setting
7. 5 Results
8. 5.1 Evaluating Max Context Size
9. 5.2 Evaluating Model Flops Utilization
10. 5.4 Impact on LLM Performance
11. 6 Related Work
12. 7 Conclusion

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Transformers have emerged as the architecture of choice for many state-of-the-art AI models, showcasing exceptional performance across a wide range of AI applications. However, the memory demands imposed by Transformers limit their ability to handle long sequences, thereby posing challenges in utilizing videos, actions, and other long-form sequences and modalities in complex environments. We present a novel approach, Ring Attention with Blockwise Transformers (Ring Attention), which leverages blockwise computation of self-attention and feedforward to distribute long sequences across multiple devices while fully overlapping the communication of key-value blocks with the computation of blockwise attention. Our approach enables training and inference of sequences that are up to device count times longer than those achievable by prior memory-efficient Transformers, without resorting to approximations or incurring additional communication and computation overheads. Extensive experiments on language modeling and reinforcement learning tasks demonstrate the effectiveness of our approach in allowing millions of tokens context size and improving performance. 1.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>用 4 张 GPU、4 个序列块画 4 轮 K/V 环传递，确认每个 Q 最终见过全部 K/V。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
