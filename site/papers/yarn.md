---
title: YaRN： Efficient Context Window Extension of Large Language Models
description: YaRN 不把所有 RoPE 频率一刀切地缩放，而是区分需要插值、需要保持和需要平滑过渡的频段，并调整 attention 温度。
---

# YaRN: Efficient Context Window Extension of Large Language Models

<div class="paper-lesson-meta"><span>方向选读</span><span>20 页</span><span>arXiv 2309.00071</span></div>

<div class="lesson-lead">YaRN 不把所有 RoPE 频率一刀切地缩放，而是区分需要插值、需要保持和需要平滑过渡的频段，并调整 attention 温度。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像拉长一把多刻度尺：粗刻度可以压缩，细刻度要尽量保留，否则近距离分辨率会丢失。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**直接把位置除以扩展倍数会破坏高频局部关系；完全不缩放又让低频维度在超长位置进入训练外区域。

**它在整条学习链中的位置：**把已有 RoPE 模型扩展到更长上下文

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：NTK-by-parts 按频率处理：高频维度更偏向保留，低频维度更偏向插值，中间平滑过渡。
2. **再看核心变化**：动态 scaling 允许模型在不同推理长度下调整尺度。
3. **最后看输出**：temperature scaling 补偿扩窗后 attention 分布熵的变化。

## 论文拿什么证明

- 论文在 LLaMA 系列上用较少长文本微调扩展到更长窗口，并比较位置插值与 NTK 类方法。
- 结果显示扩窗不仅看 perplexity，还要看长距离检索和下游任务。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- DeepSeek-V3 使用 YaRN 激活 128K 长上下文；K3 选择 KDA + NoPE MLA，代表另一条避免 RoPE 外推压力的路线。

继续补背景：[第 3 章 · MLA](/guide/ch03) · [第 9 章 · Scaling 与长上下文](/guide/ch09)

## 不要从论文中过度推出什么

- 能运行到目标长度不等于能有效利用每个远距离信息。
- 扩窗倍数、训练数据和基座模型共同决定效果，参数不能机械照搬。

## 原文应该怎么读

**推荐范围：**读方法总图与三种 scaling 组件；实验重点看训练长度和目标长度的对应

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2309.00071" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2309.00071" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：54,315 字符 · 8 个标题</span></div>

### 原文章节地图

1. YARN: EFFICIENT CONTEXT WINDOW EXTENSION OF LARGE LANGUAGE MODELS
2. Bowen Peng†1 Jeffrey Quesnelle†1 Honglu Fan23 Enrico Shippole
3. Wq,Wk : R|D| → R|D|.
4. fq = fW
5. , fk = fW
6. fW′ (xm,m,θ) = fW (xm,g(m),h(θ)), (7)
7. |D|−2. (22)

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models. However, these models fail to generalize past the sequence length they were trained on. We present YaRN (Yet another RoPE extensioN method), a compute-efficient method to extend the context window of such models, requiring 10x less tokens and 2.5x less training steps than previous methods. Using YaRN, we show that LLaMA models can effectively utilize and extrapolate to context lengths much longer than their original pre-training would allow, while also surpassing previous the state-of-the-art at context window extension. In addition, we demonstrate that YaRN exhibits the capability to extrapolate beyond the limited context of a fine-tuning dataset. Code is available at https://github.com/jquesnelle/yarn. 1 INTRODUCTION Transformer-based Large Language Models(Vaswani et al., 2017) (LLMs) have become the nearubiquitous choice for many natural language processing (NLP) tasks where long-range abilities such as in-context learning (ICL) has been crucial. In performing the NLP tasks, the maximal length of the sequences (the context window) determined by its training processes has been one of the major limits of a pretrained LLM. Being able to dynamically extend the context window via a small amount of fine-tuning (or without fine-tuning) has become more and more desirable. To this end, the position encodings of transformers are the center of the discussions. The original Transformer architecture used an absolute sinusoidal position encoding, which was later improved to a learnable absolute position encoding (Gehring et al., 2017). Since then, relative positional encoding schemes (Shaw et al., 2018) have further increased the performance of Trans

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>解释为什么“所有频率都除以 8”会伤害局部位置分辨率，并给出高低频分开处理的直觉。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
