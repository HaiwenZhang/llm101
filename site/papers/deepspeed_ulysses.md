---
title: DeepSpeed Ulysses： System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models
description: Ulysses 先按序列把输入分卡，在 attention 前通过 All-to-All 变成按 head 分卡，算完再换回序列切分。
---

# DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models

<div class="paper-lesson-meta"><span>方向选读</span><span>12 页</span><span>arXiv 2309.14509</span></div>

<div class="lesson-lead">Ulysses 先按序列把输入分卡，在 attention 前通过 All-to-All 变成按 head 分卡，算完再换回序列切分。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像先按书的章节分工，讨论人物时临时改成按人物分工，讨论完再换回章节分工。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**超长序列 activation 单卡放不下；传统 sequence parallel 往往需要高通信量或受 attention head 数限制。

**它在整条学习链中的位置：**用 All-to-All 在 head 与 sequence 两种切分间换形

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：非 attention 模块保持 sequence-sharded，每卡只存一段 token。
2. **再看核心变化**：第一次 All-to-All 把序列片段重排成完整序列的部分 heads。
3. **最后看输出**：各卡独立算自己的 heads，第二次 All-to-All 把输出还原为序列切分。

## 论文拿什么证明

- 论文在多 GPU 上展示长序列训练的显存扩展与吞吐，并讨论与 tensor/pipeline/data parallel 的正交组合。
- 通信量与序列长度线性相关，主要依赖高效 All-to-All。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 的百万 token 系统同样需要沿序列切分，但 KDA Context Parallelism 传递的是可组合递推片段，不等同 Ulysses 的 head 重排。

继续补背景：[第 9 章 · Scaling 与长上下文](/guide/ch09) · [第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- 可用并行度受 attention head 数与网络 All-to-All 能力约束。
- Ulysses 解决显存与并行，不改变 attention 的数学计算量。

## 原文应该怎么读

**推荐范围：**精读 sequence parallel group 的两次 All-to-All；再看与 ZeRO/TP 的组合

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2309.14509" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2309.14509" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：34,807 字符 · 24 个标题</span></div>

### 原文章节地图

1. DEEPSPEED ULYSSES: SYSTEM OPTIMIZATIONS FOR ENABLING TRAINING OF EXTREME LONG SEQUENCE TRANSFORMER MODELS
2. Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang
3. 1 Introduction
4. 2 Background and Related Work
5. 2.1 Background
6. 2.1.1 Transformer Architecture
7. 2.1.2 Mode of Parallelism
8. 2.2 Related Work
9. 3 DeepSpeed-Ulysses Core Design
10. 3.1 System Design
11. 3.2 Communication Analysis
12. 3.3 Memory Efficiency

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Computation in a typical Transformer-based large language model (LLM) can be characterized by batch size, hidden dimension, number of layers, and sequence length. Until now, system works for accelerating LLM training have focused on the first three dimensions: data parallelism for batch size, tensor parallelism for hidden size and pipeline parallelism for model depth or layers. These widely studied forms of parallelism are not targeted or optimized for long sequence Transformer models. Given practical application needs for long sequence LLM, renewed attentions are being drawn to sequence parallelism. However, existing works in sequence parallelism are constrained by memorycommunication inefficiency, limiting their scalability to long sequence large models. In this work, we introduce DeepSpeed-Ulysses, a novel, portable and effective methodology for enabling highly efficient and scalable LLM training with extremely long sequence length. DeepSpeed-Ulysses at its core partitions input data along the sequence dimension and employs an efficient all-to-all collective communication for attention computation. Theoretical communication analysis shows that whereas other methods incur communication overhead as sequence length increases, DeepSpeed-Ulysses maintains constant communication volume when sequence length and compute devices are increased proportionally. Furthermore, experimental evaluations show that DeepSpeed-Ulysses trains 2.5x faster with 4x longer sequence length than the existing method SOTA baseline.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>画出 2 卡、4 token、2 head 的 sequence-shard → head-shard → sequence-shard 张量布局。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
