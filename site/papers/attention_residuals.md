---
title: Attention Residuals
description: AttnRes 把残差流从固定系数求和改成对历史层表示的内容相关检索，让深层模块不必从越来越稀释的单一残差流中恢复早期特征。
---

# Attention Residuals

<div class="paper-lesson-meta"><span>核心精读</span><span>21 页</span><span>arXiv 2603.15031</span></div>

<div class="lesson-lead">AttnRes 把残差流从固定系数求和改成对历史层表示的内容相关检索，让深层模块不必从越来越稀释的单一残差流中恢复早期特征。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像写论文时不只看上一版草稿，而是根据当前问题，从所有历史版本里挑最有用的一版。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**PreNorm Transformer 的残差流不断以权重 1 累加，深度增长时幅值扩大，任一早期层对当前层的相对贡献被稀释，而且当前 token 无法按内容选择需要哪一层。

**它在整条学习链中的位置：**把 token attention 的思想旋转到网络深度轴

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：Full AttnRes 用每层学习到的 pseudo-query 对所有前序层输出做 softmax attention；历史输出同时充当 key/value，key 先 RMSNorm。
2. **再看核心变化**：Full 版本沿深度的计算为二次量级，并要求保留全部层表示；Block AttnRes 先在 block 内普通累加，只在 block 边界保留代表，成本由层数 L 降为 block 数 N。
3. **最后看输出**：pseudo-query 零初始化使初始权重近似均匀，避免训练初期某个深度源被随机放大。
4. **系统如何执行**：在 pipeline parallel 和 activation recomputation 中，真正昂贵的不只 FLOPs，还有跨 stage 传输和保存的历史 activation，因此 block 化是系统设计而非纯近似。

## 论文拿什么证明

- 论文的 scaling fit 显示 Block AttnRes 在最大实验点达到约等价于 baseline 1.25× compute 的 loss；Full 版本略强但系统成本更高。
- 48B total/3B activated、1.4T token 实验中，Block AttnRes 对多个下游指标有增益；权重可视化显示相邻层通常最重要，但 embedding 与非局部层仍会被选择。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 用 Block AttnRes 支撑更深的混合注意力骨架：KDA/MLA 解决 token 轴的信息流，AttnRes 解决 layer 轴的信息流，两者正交。

继续补背景：[第 6 章 · AttnRes](/guide/ch06) · [第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- 约 1.25× 是 loss scaling 拟合出的 compute-equivalent，不等于 wall-clock 训练直接快 25%。
- 注意力权重可帮助诊断信息路径，但不能自动当成因果解释。

## 原文应该怎么读

**推荐范围：**精读 PDF p.2–6 的动机与方法；读 p.8–12 的 scaling/下游/消融

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2603.15031" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2603.15031" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：74,767 字符 · 28 个标题</span></div>

### 原文章节地图

1. ATTENTION RESIDUALS
2. Kimi Team
3. 1 Introduction
4. Contributions
5. 2 Motivation
6. 3 Attention Residuals: A Unified View of Time and Depth
7. 3.1 Full Attention Residuals
8. 3.2 Block Attention Residuals
9. 4 Infrastructure Design
10. 4.2 Inference
11. 5 Experiments
12. 5.1 Scaling Laws

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Residual connections [12] with PreNorm [60] are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit weights. This uniform aggregation causes uncontrolled hidden-state growth with depth, progressively diluting each layer’s contribution [27]. We propose Attention Residuals (AttnRes), which replaces this fixed accumulation with softmax attention over preceding layer outputs, allowing each layer to selectively aggregate earlier representations with learned, inputdependent weights. To address the memory and communication overhead of attending over all preceding layer outputs for large-scale model training, we introduce Block AttnRes, which partitions layers into blocks and attends over block-level representations, reducing the memory footprint while preserving most of the gains of full AttnRes. Combined with cache-based pipeline communication and a two-phase computation strategy, Block AttnRes becomes a practical drop-in replacement for standard residual connections with minimal overhead. Scaling law experiments confirm that the improvement is consistent across model sizes, and ablations validate the benefit of content-dependent depth-wise selection. We further integrate AttnRes into the Kimi Linear architecture [69] (48B total / 3B activated parameters) and pre-train on 1.4T tokens, where AttnRes mitigates PreNorm dilution, yielding more uniform output magnitudes and gradient distribution across depth, and improves downstream performance across all evaluated tasks. Output

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>把标准 residual、Full AttnRes、Block AttnRes 画成三张计算图，并分别标出需要跨 pipeline stage 保存/传输的张量。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
