---
title: Kimi K2： Open Agentic Intelligence
description: K2 把 1T 级超稀疏 MoE、MuonClip、数据重述和大规模 agentic 后训练放进同一条流水线，是理解 K3 所谓“相对 K2 提升”的基准。
---

# Kimi K2: Open Agentic Intelligence

<div class="paper-lesson-meta"><span>核心精读</span><span>32 页</span><span>arXiv 2507.20534</span></div>

<div class="lesson-lead">K2 把 1T 级超稀疏 MoE、MuonClip、数据重述和大规模 agentic 后训练放进同一条流水线，是理解 K3 所谓“相对 K2 提升”的基准。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像从“会回答”升级到“会办事”：底座、训练数据、工具环境和执行反馈一起决定能力。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**高质量人类 token 逐渐稀缺，同时工具使用、长期规划和错误恢复在自然语料中又很少，预训练 token utility 与后训练交互数据都必须扩展。

**它在整条学习链中的位置：**K3 的直接基线：稀疏预训练、Muon 与 Agent RL

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：MuonClip 用 Muon 提高 token efficiency，再用 per-head QK-Clip 在参数更新后缩放 query/key 权重，抑制 attention logits 爆炸。
2. **再看核心变化**：知识与数学重述在保持事实/推理内容的同时生成风格、视角和表达多样性，试图比简单多 epoch 重复提供更有效的新信号。
3. **最后看输出**：架构为 1.04T total/32.6B activated、384 routed experts 选 8、1 shared expert；sparsity scaling law 在固定激活计算下选择 sparsity 48。
4. **系统如何执行**：后训练用工具规格、模拟器、任务和 judge 合成 agent trajectories；RL 同时覆盖可验证 reward 与 self-critique rubric reward。
5. **为什么有效**：co-located RL、checkpoint engine、并发环境和 partial rollout 让长时程交互不被最慢轨迹拖死。

## 论文拿什么证明

- K2 在 15.5T token 训练中报告无 loss spike；受控重述实验在 SimpleQA 上优于直接重复，sparsity 实验显示固定 active experts 时增加总专家持续降低验证 loss。
- 论文报告 agent、代码和推理基准增益，同时明确列出过长输出、模糊工具定义和一键软件项目成功率等限制。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 延续 Muon、超稀疏 MoE、agentic 数据与 partial rollout，但用 KDA/AttnRes/Stable LatentMoE 重做底座，并把 agent context 从 128K 量级推到 1M。

继续补背景：[第 4 章 · MoE](/guide/ch04) · [第 9 章 · Scaling 与长上下文](/guide/ch09) · [第 10 章 · SFT 与 RL](/guide/ch10) · [第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- “15.5T 无 loss spike”证明该配方在这次 run 上稳定，不证明 QK-Clip 是唯一原因。
- 重述增加 token utility 的同时会带入生成模型偏差；论文也承认事实一致性、幻觉和跨域扩展仍是问题。

## 原文应该怎么读

**推荐范围：**精读 PDF p.2–9 的预训练；读 p.9–15 的 agentic SFT/RL/系统

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2507.20534" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2507.20534" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：117,208 字符 · 41 个标题</span></div>

### 原文章节地图

1. KIMI K2: OPEN AGENTIC INTELLIGENCE
2. 1 Introduction
3. 2 Pre-training
4. 2.1 MuonClip: Stable Training with Weight Clipping
5. QhKh⊤ Vh.
6. 2.2 Pre-training Data: Improving Token Utility with Rephrasing
7. 2.3 Model Architecture
8. DeepSeek-V3 Kimi K2 ∆
9. 2.4.3 Activation Reduction
10. 2.5 Training recipe
11. 3.1 Supervised Fine-Tuning
12. 3.1.1 Large-Scale Agentic Data Synthesis for Tool Use Learning

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

We introduce Kimi K2, a Mixture-of-Experts (MoE) large language model with 32 billion activated parameters and 1 trillion total parameters. We propose the MuonClip optimizer, which improves upon Muon with a novel QK-clip technique to address training instability while enjoying the advanced token efficiency of Muon. Based on MuonClip, K2 was pre-trained on 15.5 trillion tokens with zero loss spike. During post-training, K2 undergoes a multi-stage post-training process, highlighted by a large-scale agentic data synthesis pipeline and a joint reinforcement learning (RL) stage, where the model improves its capabilities through interactions with real and synthetic environments. Kimi K2 achieves state-of-the-art performance among open-source non-thinking models, with strengths in agentic capabilities. Notably, K2 obtains 66.1 on Tau2-Bench, 76.5 on ACEBench (En), 65.8 on SWE-Bench Verified, and 47.3 on SWE-Bench Multilingual — surpassing most open and closed-sourced baselines in non-thinking settings. It also exhibits strong capabilities in coding, mathematics, and reasoning tasks, with a score of 53.7 on LiveCodeBench v6, 49.5 on AIME 2025, 75.1 on GPQA-Diamond, and 27.1 on OJBench, all without extended thinking. These results position Kimi K2 as one of the most capable open-source large language models to date, particularly in software engineering and agentic tasks. We release our base and post-trained model checkpoints1 to facilitate future research and applications of agentic intelligence. SWE-bench Veriﬁed ###### SWE-bench Multilingual LiveCodeBench v6 OJBench 80 80 80 80 72.5 65.8 60 60 60 60 53.7 54.6 51.0 47.3 47.4 44.7 46.9 44.7 40 40 40 40 38.8 37.0 34.4 31.5 20.9 27.1 24.0 25.8 19.5 19.6 19.5 11.3 20 20 20 20 0 0 0 0 Kimi-K2-InstructDeepSeek-V3-0324Qwen3-235B-A22BO

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>用一页表格区分 K2 的三种 scaling：总专家数、有效训练 token、agent rollout 计算；说明它们分别受什么资源限制。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
