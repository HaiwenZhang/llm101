---
title: Kimi Linear： An Expressive, Efficient Attention Architecture
description: Kimi Linear 将 Gated DeltaNet 的单一时间尺度扩展为逐通道时间尺度，并用 3:1 的 KDA–MLA 混合承认了递归记忆与显式全局检索各自不可替代。
---

# Kimi Linear: An Expressive, Efficient Attention Architecture

<div class="paper-lesson-meta"><span>核心精读</span><span>28 页</span><span>arXiv 2510.26692</span></div>

<div class="lesson-lead">Kimi Linear 将 Gated DeltaNet 的单一时间尺度扩展为逐通道时间尺度，并用 3:1 的 KDA–MLA 混合承认了递归记忆与显式全局检索各自不可替代。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像“随身笔记 + 定期回图书馆”：大多数时候读固定大小的笔记，隔几层再做一次全库检索。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**纯 full attention 的 prefill 和 KV cache 随上下文增长，纯 recurrent/linear attention 又容易在精确回忆和复杂全局匹配上吃亏。

**它在整条学习链中的位置：**KDA 与 3:1 混合注意力的直接前传

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：KDA 使用逐通道门 $Diag(α_t)$：$S_t=(I-β_tk_tk_t^T)Diag(α_t)S_{t-1}+β_tk_tv_t^T$；每个 value channel 可选择不同遗忘速度。
2. **再看核心变化**：作者把 KDA 归入可高效 chunk 化的 DPLR 递推族，并给出硬件友好的 UT 算法。
3. **最后看输出**：架构每 3 层 KDA 插入 1 层 MLA；NoPE MLA 负责无位置偏置的全局内容检索，顺序线索主要由 KDA 递归提供。
4. **系统如何执行**：Kimi Linear 使用低秩 output gate；K3 把它改成 full-rank gate，并在更深网络上加入 AttnRes。

## 论文拿什么证明

- 论文在 matched 1.4T-token 预训练中比较 MLA、GDN-H 与 Kimi Linear，并报告后者在多数预训练、SFT 和长上下文指标上更优。
- 3:1 在质量与速度间优于论文测试的 1:1、7:1 和纯 attention 配方；论文还报告最多减少 75% KV cache、1M 上下文 decode 最多 6×，以及约 1.16× compute efficiency。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 基本继承 3:1 骨架，但把它扩展到 93 层级别、加入末端全局 MLA、Block AttnRes、Stable LatentMoE 和更完整的 KDA 系统实现。

继续补背景：[第 3 章 · MLA](/guide/ch03) · [第 5 章 · KDA](/guide/ch05) · [第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- 3:1 是该模型族和训练配方下的经验最优点，不是所有任务的理论最优比例。
- 1M 下 6× decode 与 75% cache 是配置相关的峰值；短上下文、不同 batch 或 kernel 下收益会显著变化。

## 原文应该怎么读

**推荐范围：**精读 PDF p.4–7 的 KDA/架构；读 p.8–16 的消融、长上下文与速度

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2510.26692" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2510.26692" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：94,528 字符 · 42 个标题</span></div>

### 原文章节地图

1. KIMI LINEAR: AN EXPRESSIVE, EFFICIENT ATTENTION ARCHITECTURE
2. 1 Introduction
3. 2 Preliminary
4. 3 Kimi Delta Attention: Improving Delta Rule with Fine-grained Gating
5. 4 The Kimi Linear Model Architecture
6. 5 Experiments
7. 1.16×
8. 6 Discussions
9. Conclusion
10. A Contributions
11. B Derivations for Chunkwise Parallelism of KDA We first recall the recurrent form of KDA:
12. C Pseudo Code for chunkwise KDA

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

We introduce Kimi Linear, a hybrid linear attention architecture that, for the first time, outperforms full attention under fair comparisons across various scenarios—including short-context, long-context, and reinforcement learning (RL) scaling regimes. At its core lies Kimi Delta Attention (KDA), an expressive linear attention module that extends Gated DeltaNet [111] with a finer-grained gating mechanism, enabling more effective use of limited finite-state RNN memory. Our bespoke chunkwise algorithm achieves high hardware efficiency through a specialized variant of the Diagonal-Plus-LowRank (DPLR) transition matrices, which substantially reduces computation compared to the general DPLR formulation while remaining more consistent with the classical delta rule. We pretrain a Kimi Linear model with 3B activated parameters and 48B total parameters, based on a layerwise hybrid of KDA and Multi-Head Latent Attention (MLA). Our experiments show that with an identical training recipe, Kimi Linear outperforms full MLA with a sizeable margin across all evaluated tasks, while reducing KV cache usage by up to 75% and achieving up to 6× decoding throughput for a 1M context. These results demonstrate that Kimi Linear can be a drop-in replacement for full attention architectures with superior performance and efficiency, including tasks with longer input and output lengths. To support further research, we open-source the KDA kernel and vLLM implementations 1, and release the pre-trained and instruction-tuned model checkpoints. 2 90 60 Kimi Linear 84.3 MLA 81.3 GDN-H 80.5 Performance Kimi Linear 51.0 GDN-H 47.9 MLA 47.2 RULER (128k) MMLU-Pro (4k) 45 50 1× 2× 3× 4× Decoding Acceleration (a)

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>分别给出一个 KDA 擅长、MLA 擅长的检索例子，并解释为什么 7:1 可能更快却验证损失更差。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
