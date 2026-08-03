---
title: ReAct： Synergizing Reasoning and Acting in Language Models
description: ReAct 让语言模型交替产生推理文字与外部行动，再把环境观察放回上下文，从而一边计划、一边查证和修正。
---

# ReAct: Synergizing Reasoning and Acting in Language Models

<div class="paper-lesson-meta"><span>方向选读</span><span>33 页</span><span>arXiv 2210.03629</span></div>

<div class="lesson-lead">ReAct 让语言模型交替产生推理文字与外部行动，再把环境观察放回上下文，从而一边计划、一边查证和修正。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像边查资料边写报告：想一步、做一步、看结果，再决定下一步。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**只做 chain-of-thought 容易凭内部记忆编造事实；只做行动又缺少显式规划与错误分析。

**它在整条学习链中的位置：**Agent 的 reason–act–observe 最小原型

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：Thought 总结当前目标和下一步计划。
2. **再看核心变化**：Action 调用搜索、百科或环境动作。
3. **最后看输出**：Observation 返回真实世界信息；下一轮 Thought 基于新证据继续，直到给出答案或完成任务。

## 论文拿什么证明

- 论文在 HotpotQA/FEVER 等知识任务和 ALFWorld/WebShop 等交互任务上比较纯推理、纯行动与 ReAct。
- 轨迹示例显示外部观察既能补知识，也能让模型识别先前计划失败。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 的 Agent 已扩展到持久环境、长轨迹、并行工具与 verifier，但最小闭环仍是 ReAct 的 reason–act–observe。

继续补背景：[第 12 章 · Agent](/guide/ch12)

## 不要从论文中过度推出什么

- 显式 Thought 不是可靠的因果解释，也可能冗长或错误。
- 工具、prompt、停止条件和环境实现都属于 harness，会显著影响结果。

## 原文应该怎么读

**推荐范围：**精读方法示例轨迹；分别看知识任务和交互环境两类实验

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2210.03629" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2210.03629" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：103,062 字符 · 6 个标题</span></div>

### 原文章节地图

1. REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS
2. C.2 FEVER
3. Action: click B078GWRC1J Observation:

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

While large language models (LLMs) have demonstrated impressive performance across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-speciﬁc actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with and gather additional information from external sources such as knowledge bases or environments. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines in addition to improved human interpretability and trustworthiness. Concretely, on question answering (HotpotQA) and fact veriﬁcation (Fever), ReAct overcomes prevalent issues of hallucination and error propagation in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generating human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. Furthermore, on two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples. 1 INTRODUCTION A unique feature of human intelligence is the ability to seamlessly combine task-oriented actions with verbal reasoning (or inner speech, Alderson-Day & Fernyhough, 2015), which has been theorized to play an important role in human cogniti

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>为“查明一家公司某年的营收”写三轮 Thought–Action–Observation，并标出哪一步来自外部证据。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
