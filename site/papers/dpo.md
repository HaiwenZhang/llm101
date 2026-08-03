---
title: Direct Preference Optimization： Your Language Model is Secretly a Reward Model
description: DPO 把奖励最大化问题改写成一个二分类损失：提高偏好回答相对参考模型的概率，同时降低被拒回答。
---

# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

<div class="paper-lesson-meta"><span>方向选读</span><span>27 页</span><span>arXiv 2305.18290</span></div>

<div class="lesson-lead">DPO 把奖励最大化问题改写成一个二分类损失：提高偏好回答相对参考模型的概率，同时降低被拒回答。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像直接从“A 比 B 好”的成对选择学习，不再先训练一个裁判、再让选手和裁判在线对练。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**传统 RLHF 需要单独训练奖励模型，再运行容易不稳定且昂贵的在线 RL。

**它在整条学习链中的位置：**不用在线 RL 也能学习成对偏好

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：KL 约束最优策略与隐式奖励之间存在闭式关系。
2. **再看核心变化**：把未知奖励消去后，只需比较 chosen 与 rejected 在策略和 reference 下的 log-ratio。
3. **最后看输出**：训练形式类似 logistic regression，但仍保留 reference model 作为行为锚点。

## 论文拿什么证明

- 论文在情感控制、摘要与对话偏好任务上比较 PPO 类 RLHF 和监督偏好方法，展示简单稳定的竞争力。
- DPO 的工程优势是只需离线偏好对，不需要训练期间采样环境轨迹。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- DPO 适合静态回答偏好；K3 的长程 Agent 结果依赖环境状态与工具行为，因此仍需要 on-policy/环境内 RL 与验证器。

继续补背景：[第 10 章 · SFT 与 RL](/guide/ch10) · [第 11 章 · 蒸馏与部署](/guide/ch11)

## 不要从论文中过度推出什么

- DPO 不是“没有奖励模型”，而是奖励被隐式参数化在策略与参考模型的比值中。
- 离线数据覆盖不到的状态仍会产生分布外问题。

## 原文应该怎么读

**推荐范围：**先读 §3 的核心推导直觉，再看与 PPO 的实验对比；第一次不必推完拉格朗日

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2305.18290" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2305.18290" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：94,437 字符 · 38 个标题</span></div>

### 原文章节地图

1. Direct Preference Optimization: Your Language Model is Secretly a Reward Model
2. 1 Introduction
3. 2 Related Work
4. 3 Preliminaries
5. 4 Direct Preference Optimization
6. 5 Theoretical Analysis of DPO
7. 5.1 Your Language Model Is Secretly a Reward Model
8. 5.2 Instability of Actor-Critic Algorithms
9. 6 Experiments
10. 6.1 How well can DPO optimize the RLHF objective?
11. 6.2 Can DPO scale to real preference datasets?
12. 6.3 Generalization to a new input distribution Win rate vs. ground truth

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>用一对 chosen/rejected 回答解释 DPO 同时做的两次相对比较：回答之间，以及当前策略相对参考策略。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
