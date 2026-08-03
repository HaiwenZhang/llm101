---
title: Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM
description: Megatron-LM 把单层里的大矩阵沿 GPU 切开，通过少量 AllReduce 协同计算，使单卡放不下的 Transformer 能在多卡上训练。
---

# Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

<div class="paper-lesson-meta"><span>方向选读</span><span>13 页</span><span>arXiv 2104.04473</span></div>

<div class="lesson-lead">Megatron-LM 把单层里的大矩阵沿 GPU 切开，通过少量 AllReduce 协同计算，使单卡放不下的 Transformer 能在多卡上训练。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像多人搬一张巨桌：关键不是人多，而是从合适接缝切开，减少来回交接。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**模型权重、激活和矩阵计算超过单卡容量与算力，必须切分；切错位置会产生大量通信和同步等待。

**它在整条学习链中的位置：**大模型并行训练的基础坐标系

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：MLP 第一层按列切、第二层按行切，中间激活可局部计算，block 只需少数 AllReduce。
2. **再看核心变化**：Attention 的 heads 天然可分到不同 GPU，再在输出投影处合并。
3. **最后看输出**：Tensor parallel 可与 data parallel、pipeline parallel 组成多维并行。

## 论文拿什么证明

- 论文在大规模 GPU 集群上训练数十亿参数语言模型，并报告扩展效率与下游结果。
- 最持久的贡献是列并行/行并行的切分模式，成为后续大模型框架基础。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 系统章节里的 TP、PP、DP、EP、CP 都建立在这套多维并行语言上。

继续补背景：[第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- 更多 GPU 不会线性变快；通信、pipeline bubble 和小矩阵效率会限制 scaling。
- Tensor parallel 切参数，data parallel 复制参数，二者不要混。

## 原文应该怎么读

**推荐范围：**先看 tensor parallel 图与通信位置；再看 data/pipeline 组合

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2104.04473" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2104.04473" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：76,159 字符 · 31 个标题</span></div>

### 原文章节地图

1. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM
2. 1 INTRODUCTION
3. 2 MODES OF PARALLELISM
4. 2.1 Data Parallelism
5. 2.2 Pipeline Model Parallelism
6. Figure 2: Combination of tensor and pipeline model parallelism (MP) used in this work for transformer-based models.
7. 2.3 Tensor Model Parallelism
8. 3 PERFORMANCE ANALYSIS OF PARALLELIZATION CONFIGURATIONS
9. 3.3 Data and Model Parallelism
10. 3.4 Microbatch Size
11. 3.5 Activation Recomputation
12. 4 IMPLEMENTATION

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Large language models have led to state-of-the-art accuracies across several tasks. However, training these models efficiently is challenging because: a) GPU memory capacity is limited, making it impossible to fit large models on even a multi-GPU server, and b) the number of compute operations required can result in unrealistically long training times. Consequently, new methods of model parallelism such as tensor and pipeline parallelism have been proposed. Unfortunately, naive usage of these methods leads to scaling issues at thousands of GPUs. In this paper, we show how tensor, pipeline, and data parallelism can be composed to scale to thousands of GPUs. We propose a novel interleaved pipelining schedule that can improve throughput by 10+% with memory footprint comparable to existing approaches. Our approach allows us to perform training iterations on a model with 1 trillion parameters at 502 petaFLOP/s on 3072 GPUs (per-GPU throughput of 52% of theoretical peak).

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>画一个两层 MLP 在 2 张 GPU 上的列切—激活—行切过程，标出唯一需要合并的位置。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
