---
title: Muon is Scalable for LLM Training
description: Muon 把二维权重的梯度更新做近似正交化，让不同方向更均衡；这篇报告用权重衰减与更新尺度校准把它扩展到大模型。
---

# Muon is Scalable for LLM Training

<div class="paper-lesson-meta"><span>方向选读</span><span>19 页</span><span>arXiv 2502.16982</span></div>

<div class="lesson-lead">Muon 把二维权重的梯度更新做近似正交化，让不同方向更均衡；这篇报告用权重衰减与更新尺度校准把它扩展到大模型。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像不只逐个拧音量旋钮，而是校正整张调音台的方向，让不同通道的更新更均衡。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**Muon 在小模型上有效，但直接搬到大模型会遇到更新尺度随矩阵形状变化、正则化缺失与分布式成本问题。

**它在整条学习链中的位置：**理解 Kimi 系列为什么改用矩阵优化器

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：对矩阵梯度做 Newton–Schulz 迭代，得到近似正交方向，而不是逐元素归一化。
2. **再看核心变化**：根据参数矩阵形状调整 update RMS，使不同宽度和深度的层得到可比较的相对更新。
3. **最后看输出**：加入 weight decay，并把 orthogonalization 组织成通信与显存友好的分布式实现。

## 论文拿什么证明

- 报告的 compute-optimal scaling 实验给出相对 AdamW 约 2× 的计算效率主张。
- Moonlight MoE 以 5.7T token 训练，作为优化器可扩展到真实大模型的系统验证。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K2 使用 Muon，K3 进一步采用 Per-Head Muon；它属于优化器维度，不能与 KDA、MoE 等结构收益混为一谈。

继续补背景：[第 9 章 · Scaling 与长上下文](/guide/ch09) · [第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- 约 2× 来自特定 scaling fit，不等于训练墙钟时间固定减半。
- Muon 主要用于二维矩阵参数，bias、norm 等参数通常仍由 AdamW 类方法处理。

## 原文应该怎么读

**推荐范围：**先读摘要、§2 的 scale 修正和 scaling-law 对比；分布式实现按需读

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2502.16982" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2502.16982" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：57,329 字符 · 30 个标题</span></div>

### 原文章节地图

1. MUON IS SCALABLE FOR LLM TRAINING
2. 1 Introduction
3. 2 Methods
4. 3 Experiments
5. A Update RMS Proof of Lemma 1
6. B AdamW Baseline Scaling Law
7. E Comparison with More Expensive Models
8. F Singular Value Distributions of Weight Matrices

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Recently, the Muon optimizer (K. Jordan et al. 2024) based on matrix orthogonalization has demonstrated strong results in training small-scale language models, but the scalability to larger models has not been proven. We identify two crucial techniques for scaling up Muon: (1) adding weight decay and (2) carefully adjusting the per-parameter update scale. These techniques allow Muon to work out-ofthe-box on large-scale training without the need of hyper-parameter tuning. Scaling law experiments indicate that Muon achieves ∼ 2× computational efficiency compared to AdamW with compute optimal training. Based on these improvements, we introduce Moonlight, a 3B/16B-parameter Mixture-of-Expert (MoE) model trained with 5.7T tokens using Muon. Our model improves the current Pareto frontier, achieving better performance with much fewer training FLOPs compared to prior models. We open-source our distributed Muon implementation that is memory optimal and communication efficient. We also release the pretrained, instruction-tuned, and intermediate checkpoints to support future research. r 3.0

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>比较逐元素 AdamW 与整张矩阵正交化的观察单位，并解释为什么矩阵长宽改变时要校准更新尺度。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
