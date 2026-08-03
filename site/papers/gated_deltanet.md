---
title: Gated Delta Networks： Improving Mamba2 with Delta Rule
description: DeltaNet 把线性 attention 的累加状态解释成快速权重，并在写入新键值关联前先擦除旧关联；Gated DeltaNet 再用衰减门管理整张记忆的寿命。
---

# Gated Delta Networks: Improving Mamba2 with Delta Rule

<div class="paper-lesson-meta"><span>核心精读</span><span>22 页</span><span>arXiv 2412.06464</span></div>

<div class="lesson-lead">DeltaNet 把线性 attention 的累加状态解释成快速权重，并在写入新键值关联前先擦除旧关联；Gated DeltaNet 再用衰减门管理整张记忆的寿命。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像可擦写白板：写新答案前先看同一位置的旧答案，只擦掉差值，而不是在上面无限叠字。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**普通线性 attention 只会把 outer product 不断累加，键冲突时无法有选择地改写旧值；单纯指数衰减又会丢失需要长期保留的信息。

**它在整条学习链中的位置：**从线性 attention 到可控关联记忆

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：按论文的矩阵方向，delta 更新为 $S_t=S_{t-1}(I-β_tk_tk_t^T)+β_tv_tk_t^T$：先从状态中减去当前 key 已绑定的预测，再写入目标 value。
2. **再看核心变化**：它等价于对快速权重损失 $0.5 ||S k_t-v_t||^2$ 做一步 SGD，β 是样本相关学习率。
3. **最后看输出**：Gated DeltaNet 增加标量 α：$S_t=α_tS_{t-1}(I-β_tk_tk_t^T)+β_tv_tk_t^T$；α 管全局遗忘，delta 项管定向改写。
4. **系统如何执行**：论文用 WY/UT 形式构造 chunkwise 并行算法，让递归模型能在训练时利用矩阵乘法吞吐。

## 论文拿什么证明

- 作者在约 1.3B、100B token 的受控比较中报告 Gated DeltaNet 优于所比较的 recurrent baselines；与 sliding-window attention 混合后进一步改善。
- Single Needle 分析展示互补性：只有 decay 容易遗忘，只有 delta 容易发生容量碰撞，二者结合兼顾过滤与关联记忆。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- KDA 把标量 α 升级为逐通道对角门，使不同 value channel 拥有不同时间尺度；K3 再把 KDA 与周期性全局 MLA 组合。

继续补背景：[第 5 章 · KDA](/guide/ch05)

## 不要从论文中过度推出什么

- 状态矩阵的左右乘方向在不同论文中会因 row/column 约定而转置，比较时应看语义，不要把转置误当成算法差异。
- 固定大小状态不可能无损保存无限历史；门控改善的是资源分配，不是消除记忆容量上限。

## 原文应该怎么读

**推荐范围：**精读 PDF p.3–7 的 delta rule/Gated DeltaNet；扫读 p.10–13 的结果与 p.22 消融

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2412.06464" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2412.06464" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：87,559 字符 · 0 个标题</span></div>

### 原文章节地图

这份解析文本没有稳定提取出章节标题；请按 PDF 页码阅读。

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Linear Transformers have gained attention as efﬁcient alternatives to standard Transformers, but their performance in retrieval and long-context tasks has been limited. To address these limitations, recent work has explored two distinct mechanisms: gating for adaptive memory control and the delta update rule for precise memory modiﬁcations. We observe that these mechanisms are complementary—gating enables rapid memory erasure while the delta rule facilitates targeted updates. Building on this insight, we introduce the gated delta rule and develop a parallel training algorithm optimized for modern hardware. Our proposed architecture, Gated DeltaNet, consistently surpasses existing models like Mamba2 and DeltaNet across multiple benchmarks, including language modeling, commonsense reasoning, in-context retrieval, length extrapolation, and long-context understanding. We further enhance performance by developing hybrid architectures that combine Gated DeltaNet layers with sliding window attention or Mamba2 layers, achieving both improved training efﬁciency and superior task performance. Code: https://github.com/NVlabs/GatedDeltaNet 1 INTRODUCTION The Transformer architecture has signiﬁcantly advanced the capabilities of Large Language Models (LLMs), showcasing exceptional performance across a wide range of tasks due to its effective attention mechanism. This mechanism excels in precise sequence modeling and leverages the parallel processing capabilities of modern GPUs during training. However, the self-attention component scales quadratically with sequence length, leading to substantial computational demands that pose challenges for both training and inference. To mitigate these issues, researchers have explored alternatives such as linear Transformers (Katharopoulos et al.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>把 delta 更新写成“预测—误差—写回”三步，并说明当两个 key 高度相似时它如何替换旧关联。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
