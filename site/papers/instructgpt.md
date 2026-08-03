---
title: Training Language Models to Follow Instructions with Human Feedback
description: InstructGPT 用示范数据做 SFT、用人类排序训练奖励模型，再用 PPO 让策略获得更高奖励，从而把续写模型变成更愿意遵循用户意图的助手。
---

# Training Language Models to Follow Instructions with Human Feedback

<div class="paper-lesson-meta"><span>方向选读</span><span>68 页</span><span>arXiv 2203.02155</span></div>

<div class="lesson-lead">InstructGPT 用示范数据做 SFT、用人类排序训练奖励模型，再用 PPO 让策略获得更高奖励，从而把续写模型变成更愿意遵循用户意图的助手。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像先看老师示范，再学会给多个答案排序，最后根据评分规则反复练习。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**预训练只要求预测互联网文本的下一个 token，不保证回答用户、承认不知道、避免有害输出或遵循格式。

**它在整条学习链中的位置：**现代指令对齐三阶段范式的经典起点

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：标注者先写理想回答，模型用监督学习获得可用起点。
2. **再看核心变化**：对同一 prompt 采样多个回答，由标注者排序；奖励模型学习这种偏好。
3. **最后看输出**：PPO 优化策略得分，同时用 KL 惩罚限制它不要偏离 SFT 模型太远。

## 论文拿什么证明

- 1.3B InstructGPT 在人类偏好比较中可胜过 175B GPT-3，说明行为对齐不只由参数规模决定。
- 论文同时测量真实性、有害性与公开 NLP 指标，并披露对齐税与标注者分歧。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 的 SFT、生成式奖励、分领域 RL 更复杂，但“先给行为起点，再用偏好/结果优化”的骨架来自这条路线。

继续补背景：[第 10 章 · SFT 与 RL](/guide/ch10) · [第 11 章 · 蒸馏与部署](/guide/ch11)

## 不要从论文中过度推出什么

- 奖励模型近似的是一组标注者在特定指南下的偏好，不是普遍人类价值。
- RLHF 可以改善可用性，也可能放大奖励漏洞和过度迎合。

## 原文应该怎么读

**推荐范围：**精读 Fig.2、§3 方法与标注流程；结果重点看人类偏好而非单一 benchmark

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2203.02155" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2203.02155" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：189,378 字符 · 79 个标题</span></div>

### 原文章节地图

1. Training language models to follow instructions with human feedback
2. F1 score

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Making language models bigger does not inherently make them better at following a user’s intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by ﬁne-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to ﬁne-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further ﬁne-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that ﬁne-tuning with human feedback is a promising direction for aligning language models with human intent.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>画出 SFT 数据、比较数据、RL rollout 三种数据的来源，说明为什么不能互换。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
