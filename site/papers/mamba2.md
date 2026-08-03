---
title: Transformers are SSMs： Generalized Models and Efficient Algorithms Through Structured State Space Duality
description: Mamba-2 通过 Structured State Space Duality 说明一类状态空间模型既能写成逐 token 递推，也能写成受结构约束的 attention 矩阵。
---

# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

<div class="paper-lesson-meta"><span>方向选读</span><span>52 页</span><span>arXiv 2405.21060</span></div>

<div class="lesson-lead">Mamba-2 通过 Structured State Space Duality 说明一类状态空间模型既能写成逐 token 递推，也能写成受结构约束的 attention 矩阵。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像同一段音乐既能逐音符播放，也能分小节并行排练；两种视角描述同一结构。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**RNN/SSM 推理时递推高效，Transformer 训练时矩阵并行高效；如果二者完全分开理解，就很难同时得到两种优势。

**它在整条学习链中的位置：**把 attention 与状态空间模型放进同一张地图

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：递推视角维护固定状态，适合一步步 decode；对偶视角把一段序列展开成结构化矩阵，适合训练时分块并行。
2. **再看核心变化**：SSD 限制状态转移的结构，使序列变换能被高效分解，而不需要物化完整二次 attention 矩阵。
3. **最后看输出**：Mamba-2 block 还重新安排投影与卷积，让计算更接近硬件友好的大矩阵乘法。

## 论文拿什么证明

- 论文在多个模型规模与语言建模基准上比较 Mamba-2、Mamba 与 Transformer，并报告更好的训练吞吐和有竞争力的质量。
- 系统收益依赖专门的 chunkwise/SSD kernel，而不只是一条递推公式。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- Gated DeltaNet 与 KDA 都受益于这种“递推形式负责解释，分块形式负责训练”的思想。

继续补背景：[第 5 章 · KDA](/guide/ch05)

## 不要从论文中过度推出什么

- “Transformers are SSMs”指特定结构下的数学对偶，不是所有 Transformer 与所有 SSM 完全相同。
- 线性时间并不保证任何序列长度、batch 和硬件上都更快。

## 原文应该怎么读

**推荐范围：**先读 §1–3 的 SSD 直觉和算法框图；矩阵分块证明放到第二遍

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2405.21060" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2405.21060" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：181,144 字符 · 73 个标题</span></div>

### 原文章节地图

1. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
2. 1 Introduction
3. 2 Background and Overview
4. 3 State Space Models are Structured Matrices
5. 4 Structured Masked Attention: Generalizing Linear Attention with Structured Matrices
6. 5 State Space Duality
7. Outputs 𝑌 States 𝐻
8. Inputs 𝑋
9. 7 The Mamba-2 Architecture
10. 8 Systems Optimization for SSMs
11. 10 Related Work and Discussion
12. 11 Conclusion

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

While Transformers have been the main architecture behind deep learning’s success in language modeling, state-space models (SSMs) such as Mamba have recently been shown to match or outperform Transformers at small to medium scale. We show that these families of models are actually quite closely related, and develop a rich framework of theoretical connections between SSMs and variants of attention, connected through various decompositions of a well-studied class of structured semiseparable matrices. Our state space duality (SSD) framework allows us to design a new architecture (Mamba-2) whose core layer is an a refinement of Mamba’s selective SSM that is 2-8× faster, while continuing to be competitive with Transformers on language modeling.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>分别画出同一模型的 recurrent mode 和 chunkwise training mode，标出跨 token 保留的状态与块内临时矩阵。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
