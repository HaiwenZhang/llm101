---
title: Kimi K3： Open Frontier Intelligence
description: K3 不是靠单点技巧扩到 2.8T，而是同时重构序列、深度、宽度三条信息流，并为 1M 上下文的训练、RL 与服务做系统共设计。
---

# Kimi K3: Open Frontier Intelligence

<div class="paper-lesson-meta"><span>主线论文</span><span>47 页</span><span>arXiv 2607.24653</span></div>

<div class="lesson-lead">K3 不是靠单点技巧扩到 2.8T，而是同时重构序列、深度、宽度三条信息流，并为 1M 上下文的训练、RL 与服务做系统共设计。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像设计一座超大城市：道路、楼层、专业部门、学校和物流系统必须一起规划，单独拓宽一条路救不了整座城。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**把更强的预训练底座、长上下文推理、原生视觉和长时程 Agent 放进同一个可训练、可服务的 3T 级稀疏模型中。

**它在整条学习链中的位置：**总纲：先建立问题地图，最后再回来逐节核对

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：序列维：每个 block 采用 3 个 KDA 加 1 个 Gated MLA，末层再放全局 MLA；KDA 提供线性递归记忆，MLA 周期性恢复显式全局交互。
2. **再看核心变化**：深度维：Block AttnRes 让模块从 embedding 和历史 block 表示中内容相关地取信息，而不是把所有历史层固定权重相加。
3. **最后看输出**：宽度维：Stable LatentMoE 在低维 latent 中运行大量 routed experts，配合归一化、SiTU-GLU 与 Quantile Balancing，在每 token 激活 896 个 routed experts 中的 16 个。
4. **系统如何执行**：能力维：原生视觉预训练、多领域多 effort RL、multi-teacher on-policy distillation 与部署感知后训练共同形成最终策略。
5. **为什么有效**：系统维：KDA kernel/context parallel/prefix cache、MoonEP、外置 KV cache、可暂停 rollout 与可恢复 microVM 都是 1M Agent 训练成立的必要条件。

## 论文拿什么证明

- 报告给出的主规格是 2.8T 总参数、104B 激活参数、1M context，并声称相对 K2 的总体 scaling efficiency 约提升 2.5×。
- 体系同时报告公开 benchmark、内部 benchmark、成本效率和多个系统 case study；它更像完整工程系统报告，而不是只隔离一个变量的算法论文。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- 这是其余 12 篇论文的汇合点：MLA、delta rule/KDA、AttnRes、DeepSeekMoE/LatentMoE、scaling law、Muon、reasoning RL、视觉 RL 与 Agent Swarm 都在这里被重新组合。

继续补背景：[第 0 章 · K3 全景](/guide/ch00) · [第 5 章 · KDA](/guide/ch05) · [第 6 章 · AttnRes](/guide/ch06) · [第 7 章 · Stable LatentMoE](/guide/ch07) · [第 8 章 · 原生视觉](/guide/ch08) · [第 9 章 · Scaling 与长上下文](/guide/ch09) · [第 10 章 · SFT 与 RL](/guide/ch10) · [第 11 章 · 蒸馏与部署](/guide/ch11) · [第 12 章 · Agent](/guide/ch12) · [第 13 章 · 训练与服务系统](/guide/ch13) · [第 14 章 · 论文评测](/guide/ch14) · [第 15 章 · 三遍阅读法](/guide/ch15)

## 不要从论文中过度推出什么

- 约 2.5× 是整套架构、数据和训练配方的联合收益，不能归因于某一个模块。
- 不同模型使用的 agent harness、reasoning effort、工具与成本口径并不总是相同；横向表格适合判断数量级，不适合当严格同条件科学实验。
- 报告披露了大量系统设计，但完整数据配比、全部训练超参和失败实验仍不足以独立复现。

## 原文应该怎么读

**推荐范围：**第一遍读 PDF p.1–12；学完前置论文后读 p.17–29；系统部分按需读 p.21–24

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2607.24653" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2607.24653" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：189,123 字符 · 91 个标题</span></div>

### 原文章节地图

1. KIMI K3: OPEN FRONTIER INTELLIGENCE
2. Kimi Team
3. 1 Introduction
4. 2 Model Architecture
5. 2.1 Hybrid Attention
6. 2.1.1 Kimi Delta Attention
7. 2.1.2 Gated MLA
8. 2.2 Attention Residuals
9. 2.3 Stable LatentMoE
10. 2.3.1 Normalized LatentMoE
11. 2.3.2 Sigmoid Tanh Unit GLU
12. Wgx

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

We introduce Kimi K3, a 2.8T parameter Mixture-of-Experts model with 104 billion activated parameters, native vision capabilities, and a 1-million-token context window. Kimi K3 is built on Kimi Delta Attention [63] and Attention Residuals [57], which improve information flow across sequence length and model depth. Together with Stable LatentMoE, which effectively activates 16 of 896 routed experts per token, and refined training and data recipes, these advances yield an approximately 2.5× improvement in overall scaling efficiency over Kimi K2 [58]. Post-training highlights reinforcement learning across general, agentic, and coding domains and multiple reasoningeffort levels, enabling compositional generalization and robust long-horizon execution. At 2.8T scale, Kimi K3 is supported by infrastructure advances in multiple areas: algorithm–system co-design for KDA, perfectly balanced expert-parallel training with efficient memory management, million-token agentic RL with persistent rollout and sandbox states, and deployment innovations. Extensive evaluations show that Kimi K3 achieves frontier-level performance across long-horizon coding, agentic, knowledge, reasoning, and vision tasks. While its overall performance still trails the most powerful proprietary models, namely Claude Fable 5 and GPT-5.6 Sol, Kimi K3 consistently outperforms other open and proprietary models evaluated in our suite. We release the full Kimi K3 model weights to facilitate future research and accelerate the broader deployment and adoption of frontier intelligence.1 Figure 1: Kimi K3 main results. 1https://huggingface.co/moonshotai/Kimi-K3

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>闭卷画出 K3 的五层因果链：架构 → 优化稳定性 → 并行系统 → 后训练 → 在线服务，并为每条箭头写出一个失败模式。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
