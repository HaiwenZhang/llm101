---
title: Kimi K2.5： Visual Agentic Intelligence
description: K2.5 的两条主线是让视觉与文本从预训练起共同优化，以及让一个可训练 orchestrator 学会何时拆任务、如何并行调用冻结 subagents。
---

# Kimi K2.5: Visual Agentic Intelligence

<div class="paper-lesson-meta"><span>核心精读</span><span>30 页</span><span>arXiv 2602.02276</span></div>

<div class="lesson-lead">K2.5 的两条主线是让视觉与文本从预训练起共同优化，以及让一个可训练 orchestrator 学会何时拆任务、如何并行调用冻结 subagents。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像让助手同时拥有眼睛、思考预算和多个协作者，并在真实工具环境中练习。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**晚期把视觉接到文本模型容易造成模态冲突；顺序 Agent 的延迟又随分支数线性增长，难以完成宽搜索和多专业协作。

**它在整条学习链中的位置：**原生多模态与并行 Agent 的直接前传

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：在固定 vision/text token 预算下，较早、较低比例地混入视觉优于后期高比例注入；MoonViT-3D 共享图像/视频参数，并以时间 patch 聚合获得约 4× 视频压缩。
2. **再看核心变化**：zero-vision SFT 只用高质量文本工具轨迹激活视觉工具能力，避免低质人工视觉轨迹限制泛化；随后用视觉 outcome RL 修复忽略图像等问题。
3. **最后看输出**：Agent Swarm 的 PARL 只更新 orchestrator，subagents 冻结且其轨迹视为环境观察，从而简化 credit assignment 和稳定性。
4. **系统如何执行**：parallel-instantiation 与 subtask-finish 辅助奖励先防 serial collapse、再防无意义 spawn，训练后期退火到零；critical steps 用最长并行分支而不是总工作量计延迟。
5. **为什么有效**：RL 的 token-level log-ratio masking、Toggle 长度预算和统一异步环境分别处理 train–inference mismatch、token efficiency 与大规模 rollout。

## 论文拿什么证明

- early-fusion 消融在固定 token 预算下同时改善视觉与文本侧指标；joint visual RL 还带来部分文本 benchmark 的正迁移。
- 宽搜索场景中，论文报告 Agent Swarm 相对单 Agent 最多 4.5× 延迟降低，同时 item-F1 从 72.8% 提升到 79.0%。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 继承原生视觉、统一 agentic RL 与并行协调能力；差异是 K3 从头训练 MoonViT-V2，并把长时程任务、上下文和持久环境进一步扩展。

继续补背景：[第 8 章 · 原生视觉](/guide/ch08) · [第 10 章 · SFT 与 RL](/guide/ch10) · [第 12 章 · Agent](/guide/ch12) · [第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- 并行只对可分解任务有利；强依赖链、共享可变状态和高汇总成本会让更多 subagents 反而更慢。
- zero-vision SFT 的成功依赖已经完成的联合多模态预训练，不能理解为视觉模型普遍不需要视觉监督。

## 原文应该怎么读

**推荐范围：**精读 PDF p.1–9；读 p.23–24 的统一 agentic RL 环境

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2602.02276" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2602.02276" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：116,165 字符 · 105 个标题</span></div>

### 原文章节地图

1. KIMI K2.5: VISUAL AGENTIC INTELLIGENCE
2. 1 Introduction
3. 2 Joint Optimization of Text and Vision
4. 3 Agent Swarm
5. ∑
6. 4 Method Overview
7. ∑
8. ∑
9. 5 Evaluations
10. 6 Conclusions
11. A Contributors
12. B Pre-training

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

We introduce Kimi K2.5, an open-source multimodal agentic model designed to advance general agentic intelligence. K2.5 emphasizes the joint optimization of text and vision so that two modalities enhance each other. This includes a series of techniques such as joint text-vision pre-training, zero-vision SFT, and joint text-vision reinforcement learning. Building on this multimodal foundation, K2.5 introduces Agent Swarm, a self-directed parallel agent orchestration framework that dynamically decomposes complex tasks into heterogeneous sub-problems and executes them concurrently. Extensive evaluations show that Kimi K2.5 achieves state-of-the-art results across various domains including coding, vision, reasoning, and agentic tasks. Agent Swarm also reduces latency by up to 4.5× over single-agent baselines. We release the post-trained Kimi K2.5 model checkpoint1 to facilitate future research and real-world applications of agentic intelligence. Figure 1: Kimi K2.5 main results.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>给一个适合 swarm 和一个不适合 swarm 的 CV/软件任务，画出依赖 DAG，并用 critical path 而非 subagent 数判断加速上限。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
