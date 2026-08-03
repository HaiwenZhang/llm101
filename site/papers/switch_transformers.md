---
title: Switch Transformers： Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
description: Switch Transformer 把每个 token 只路由到一个 FFN 专家，用极简 Top-1 稀疏计算把参数容量扩到万亿级。
---

# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

<div class="paper-lesson-meta"><span>方向选读</span><span>40 页</span><span>arXiv 2101.03961</span></div>

<div class="lesson-lead">Switch Transformer 把每个 token 只路由到一个 FFN 专家，用极简 Top-1 稀疏计算把参数容量扩到万亿级。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像分诊台每次只把病人送去一个科室，医院很大，但一次就诊不需要全院出动。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**Dense 模型增加参数时，每个 token 的计算也同步增加；传统 Top-2 MoE 又带来更复杂的路由、通信和稳定性问题。

**它在整条学习链中的位置：**最简单的 Top-1 MoE 起点

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：Router 为 token 计算专家概率，只选最高的一个专家。
2. **再看核心变化**：capacity factor 为每个专家预留有限 token 槽；超出容量的 token 会绕过或被丢弃专家计算。
3. **最后看输出**：负载均衡辅助损失同时约束路由概率与实际 token 分配，避免少数专家过热。

## 论文拿什么证明

- 论文展示了稀疏模型在固定计算预算下的预训练与下游迁移收益，并训练了万亿参数量级模型。
- 作者系统分析了低精度训练不稳定、初始化、expert dropout 和 capacity factor。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 的 896 选 16 比 Top-1 复杂得多，但 router、容量、负载与通信这四个基本问题从 Switch 开始就已经出现。

继续补背景：[第 4 章 · MoE](/guide/ch04)

## 不要从论文中过度推出什么

- 总参数巨大不等于每 token 做了同等规模计算。
- Top-1 简单但组合能力有限，后续细粒度 Top-k MoE 选择了不同权衡。

## 原文应该怎么读

**推荐范围：**精读 §2 的 Switch layer 与负载损失；扫读稳定性和系统章节

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2101.03961" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2101.03961" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：102,278 字符 · 72 个标题</span></div>

### 原文章节地图

1. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Eﬃcient Sparsity
2. William Fedus∗
3. Barret Zoph∗
4. Noam Shazeer
5. Contents
6. 1. Introduction
7. 2. Switch Transformer
8. 2.1 Simplifying Sparse Routing
9. 2.2 Eﬃcient Sparse Routing
10. 2.3 Putting It All Together: The Switch Transformer
11. 2.4 Improved Training and Fine-Tuning Techniques
12. 3. Scaling Properties

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

In deep learning, models typically reuse the same parameters for all inputs. Mixture of Experts (MoE) models defy this and instead select diﬀerent parameters for each incoming example. The result is a sparsely-activated model—with an outrageous number of parameters—but a constant computational cost. However, despite several notable successes of MoE, widespread adoption has been hindered by complexity, communication costs, and training instability. We address these with the introduction of the Switch Transformer. We simplify the MoE routing algorithm and design intuitive improved models with reduced communication and computational costs. Our proposed training techniques mitigate the instabilities, and we show large sparse models may be trained, for the ﬁrst time, with lower precision (bﬂoat16) formats. We design models based oﬀ T5-Base and T5-Large (Raﬀel et al., 2019) to obtain up to 7x increases in pre-training speed with the same computational resources. These improvements extend into multilingual settings where we measure gains over the mT5-Base version across all 101 languages. Finally, we advance the current scale of language models by pre-training up to trillion parameter models on the “Colossal Clean Crawled Corpus”, and achieve a 4x speedup over the T5-XXL model.12 Keywords: mixture-of-experts, natural language processing, sparsity, large-scale machine learning, distributed computing ∗. Equal contribution. - 1. JAX code for Switch Transformer and all model checkpoints are available at https://github.com/ google-research/t5x - 2. Tensorﬂow code for Switch Transformer is available at https://github.com/tensorflow/mesh/blob/ master/mesh_tensorflow/transformer/moe.py ©2022 William Fedus, Barret Zoph and Noam Shazeer. License: CC-BY 4.0, see https://creativecommons.o

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>用 8 个 token、4 个专家画一次 Top-1 分配；让 5 个 token 都选同一专家，解释 capacity overflow。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
