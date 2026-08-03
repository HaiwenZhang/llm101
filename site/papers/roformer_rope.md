---
title: RoFormer： Enhanced Transformer with Rotary Position Embedding
description: RoPE 不把位置向量直接加到 token 上，而是按位置旋转 Query 与 Key，使它们的点积天然依赖相对距离。
---

# RoFormer: Enhanced Transformer with Rotary Position Embedding

<div class="paper-lesson-meta"><span>方向选读</span><span>14 页</span><span>arXiv 2104.09864</span></div>

<div class="lesson-lead">RoPE 不把位置向量直接加到 token 上，而是按位置旋转 Query 与 Key，使它们的点积天然依赖相对距离。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像给每个位置的箭头转不同角度，两支箭头的夹角自然记录它们相隔多远。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**Attention 本身只看内容，不知道 token 顺序；绝对位置相加能告诉位置，却不自然地把相对距离放进 Query–Key 匹配。

**它在整条学习链中的位置：**理解现代位置编码最重要的基础论文

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：在二维平面中，位置 m 对向量旋转角度 $mθ$。
2. **再看核心变化**：位置 m 的 query 与位置 n 的 key 做点积时，两个旋转合成为角度差 $(m-n)θ$。
3. **最后看输出**：高维向量被分成二维小组，每组使用不同频率，因此同时表达近距离与远距离模式。

## 论文拿什么证明

- 论文在多类 NLP 任务上比较位置编码，并展示长度外推与相对位置性质。
- 更深远的影响是后续多数开源 LLM 采用 RoPE，并围绕其频率做长上下文扩展。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- 理解 RoPE 后才能明白 MLA 为什么需要解耦位置通道，也能理解 K3 的 NoPE 路线究竟省去了什么约束。

继续补背景：[第 3 章 · MLA](/guide/ch03) · [第 9 章 · Scaling 与长上下文](/guide/ch09)

## 不要从论文中过度推出什么

- RoPE 不是无限长度记忆；训练范围外的角度与频率分布仍会失配。
- 旋转发生在 Q/K 上，不是旋转 Value。

## 原文应该怎么读

**推荐范围：**精读 §3.2 的二维旋转推导；高维推广只需理解分组旋转

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2104.09864" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2104.09864" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：45,767 字符 · 32 个标题</span></div>

### 原文章节地图

1. ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING
2. 1 Introduction
3. 2 Background and Related Work
4. 2.1 Preliminary
5. 3 Proposed approach
6. 3.2.2 General form
7. 3.3 Properties of RoPE
8. 3.4.2 Computational efficient realization of rotary matrix multiplication
9. 3.4.3 Long-term decay of RoPE
10. 4 Experiments and Evaluation
11. 4.2.1 Experimental Settings
12. 4.2.2 Implementation details

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: https://huggingface.co/docs/transformers/model_doc/roformer. Keywords Pre-trained Language Models · Position Information Encoding · Pre-training · Natural Language Processing.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>在纸上旋转两个二维箭头，说明为什么共同平移位置不会改变它们的相对角度。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
