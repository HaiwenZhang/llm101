---
title: DeepSeek-R1： Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
description: R1-Zero 证明强 base model 可仅靠结果可验证的 RL 涌现反思与自检，但最终可用的 R1 仍需要 cold start、SFT、第二阶段 RL 与偏好/安全信号。
---

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

<div class="paper-lesson-meta"><span>核心精读</span><span>86 页</span><span>arXiv 2501.12948</span></div>

<div class="lesson-lead">R1-Zero 证明强 base model 可仅靠结果可验证的 RL 涌现反思与自检，但最终可用的 R1 仍需要 cold start、SFT、第二阶段 RL 与偏好/安全信号。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像只告诉学生最终答案是否正确，让他通过大量尝试逐渐长出自己的解题策略。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**人工长 CoT 昂贵且会限制探索，process reward 又难标注且容易被 hack；问题是能否只验证最终答案而让模型自己发现推理过程。

**它在整条学习链中的位置：**可验证 outcome RL、纯 RL 边界与多阶段对齐

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：GRPO 对每题采样一组输出，用组内 reward 的均值和标准差得到 advantage，不训练单独的 critic/value network；PPO-style ratio clipping 与 KL 控制策略漂移。
2. **再看核心变化**：R1-Zero 跳过 SFT，reward 主要是数学/代码/逻辑的 accuracy 加格式约束，因而自然增长响应长度并出现验证、反思和策略切换。
3. **最后看输出**：最终 R1 加入数千条 cold-start 数据、reasoning RL、rejection-sampling SFT、通用非推理数据和第二阶段混合 RL，以修复可读性、语言混杂与通用能力。
4. **系统如何执行**：蒸馏把大模型产生的推理轨迹传给较小模型；论文发现直接蒸馏常比在小 base 上从零做同规模 RL 更有效。

## 论文拿什么证明

- R1-Zero 的 AIME 2024 pass@1 在训练中由 15.6% 增至 77.9%，平均生成长度同时上升；最终多阶段 R1 在 reasoning 与通用偏好指标间取得更均衡结果。
- 讨论部分明确报告小 base 模型的纯 RL 尝试没有得到有意义提升，并说明 PRM 与 MCTS 在其大规模训练中的成本、value 学习和 reward hacking 问题。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 延续“强底座 + 可验证环境 + outcome RL”的路线，但从数学/代码扩到搜索、专业工作、视觉工具、软件工程和多 effort，并用多教师 on-policy distillation 合并策略。

继续补背景：[第 10 章 · SFT 与 RL](/guide/ch10) · [第 11 章 · 蒸馏与部署](/guide/ch11)

## 不要从论文中过度推出什么

- R1 不是“纯 RL 模型”：纯 RL 指 R1-Zero，正式 R1 是多阶段系统。
- 组内相对 advantage 在一组全对或全错时信号很弱；数据难度、verifier 可靠性与 rollout 多样性比算法名称更关键。
- 过程看起来像人类反思不等于内部机制与人类认知相同，也不保证中间文字忠实反映计算。

## 原文应该怎么读

**推荐范围：**精读 PDF p.1–10；读 p.63–64 的关键发现与失败尝试

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2501.12948" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2501.12948" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：230,683 字符 · 103 个标题</span></div>

### 原文章节地图

1. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
2. 1. Introduction
3. 2. DeepSeek-R1-Zero
4. 3. DeepSeek-R1
5. 4. Experiment
6. 5. Ethics and Safety Statement
7. 6. Conclusion, Limitation, and Future Work
8. 7. Author List
9. A. Background
10. B. Training Details
11. Question {question}
12. Thought process {thoughtprocess}

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

General reasoning represents a long-standing and formidable challenge in artificial intelligence. Recent breakthroughs, exemplified by large language models (LLMs) (Brown et al., 2020; OpenAI, 2023) and chain-of-thought prompting (Wei et al., 2022b), have achieved considerable success on foundational reasoning tasks. However, this success is heavily contingent upon extensive human-annotated demonstrations, and models’ capabilities are still insufficient for more complex problems. Here we show that the reasoning abilities of LLMs can be incentivized through pure reinforcement learning (RL), obviating the need for human-labeled reasoning trajectories. The proposed RL framework facilitates the emergent development of advanced reasoning patterns, such as self-reflection, verification, and dynamic strategy adaptation. Consequently, the trained model achieves superior performance on verifiable tasks such as mathematics, coding competitions, and STEM fields, surpassing its counterparts trained via conventional supervised learning on human demonstrations. Moreover, the emergent reasoning patterns exhibited by these large-scale models can be systematically harnessed to guide and enhance the reasoning capabilities of smaller models.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>设计一个 CV 可验证 RL 任务：定义 prompt、环境、outcome verifier、group sampling、可被 hack 的漏洞及防护；再判断是否需要 SFT cold start。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
