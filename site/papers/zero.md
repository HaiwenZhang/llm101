---
title: ZeRO： Memory Optimizations Toward Training Trillion Parameter Models
description: ZeRO 发现数据并行每张卡重复保存优化器状态、梯度和参数，于是按 Stage 1/2/3 逐步把三者分片。
---

# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

<div class="paper-lesson-meta"><span>方向选读</span><span>24 页</span><span>arXiv 1910.02054</span></div>

<div class="lesson-lead">ZeRO 发现数据并行每张卡重复保存优化器状态、梯度和参数，于是按 Stage 1/2/3 逐步把三者分片。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像多人合作时不再每个人都背一整套工具，而是把工具、材料和清单分开携带，用时再汇合。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**普通数据并行虽然把 batch 分开，却在每张 GPU 上复制完整模型状态，显存随 GPU 数增加没有得到充分利用。

**它在整条学习链中的位置：**把数据并行的重复显存一层层去掉

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：Stage 1 分片 optimizer states。
2. **再看核心变化**：Stage 2 再分片 gradients。
3. **最后看输出**：Stage 3 连 parameters 也分片，在某层计算前临时 AllGather，用完释放。

## 论文拿什么证明

- 论文给出各阶段的显存与通信分析，并展示把可训练模型规模推向万亿参数的可行性。
- ZeRO-R 进一步讨论 activation、临时 buffer 与内存碎片。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 的显存账同样不止参数；MoE 权重、优化器、activation、视觉分支和百万 token 状态都需要分片或重算。

继续补背景：[第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- Stage 越高，显存越省，但参数临时聚合的通信与实现复杂度更高。
- ZeRO 是状态分片，不直接减少模型本身需要执行的 FLOPs。

## 原文应该怎么读

**推荐范围：**精读三阶段分片表；通信推导第二遍再看

<div class="paper-source-row"><a href="https://arxiv.org/pdf/1910.02054" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/1910.02054" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：70,673 字符 · 49 个标题</span></div>

### 原文章节地图

1. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
2. 1 Extended Introduction
3. 2 Related Work
4. 2.1 Data, Model and Pipeline Parallelism
5. 2.2 Non-parallelism based approach to reduce memory
6. 2.3 Training Optimizers
7. 3 Where Did All the Memory Go?
8. 3.1 Model States: Optimizer States, Gradients and Parameters
9. 3.2 Residual Memory Consumption
10. 4 ZeRO: Insights and Overview
11. 4.1 Insights and Overview: ZeRO-DP ZeRO powered DP is based on three key insights:
12. 4.2 Insights and Overview: ZeRO-R

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Large deep learning models oﬀer signiﬁcant accuracy gains, but training billions to trillions of parameters is challenging. Existing solutions such as data and model parallelisms exhibit fundamental limitations to ﬁt these models into limited device memory, while obtaining computation, communication and development eﬃciency. We develop a novel solution, Zero Redundancy Optimizer (ZeRO), to optimize memory, vastly improving training speed while increasing the model size that can be eﬃciently trained. ZeRO eliminates memory redundancies in data- and model-parallel training while retaining low communication volume and high computational granularity, allowing us to scale the model size proportional to the number of devices with sustained high eﬃciency. Our analysis on memory requirements and communication volume demonstrates: ZeRO has the potential to scale beyond 1 Trillion parameters using today’s hardware. We implement and evaluate ZeRO: it trains large models of over 100B parameter with super-linear speedup on 400 GPUs, achieving throughput of 15 Petaﬂops. This represents an 8x increase in model size and 10x increase in achievable performance over state-of-the-art. In terms of usability, ZeRO can train large models of up to 13B parameters (e.g., larger than Megatron GPT 8.3B and T5 11B) without requiring model parallelism which is harder for scientists to apply. Last but not the least, researchers have used the system breakthroughs of ZeRO to create the world’s largest language model (17B parameters) with record breaking accuracy.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>假设参数、梯度、Adam 状态分别占 2、2、12 GB，计算 8 卡下三个 Stage 的理想每卡占用。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
