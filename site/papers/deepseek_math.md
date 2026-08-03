---
title: DeepSeekMath： Pushing the Limits of Mathematical Reasoning in Open Language Models
description: DeepSeekMath 先构建高质量数学语料，再从代码模型继续预训练，最后用 GRPO 以组内相对奖励强化可验证数学答案。
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

<div class="paper-lesson-meta"><span>方向选读</span><span>30 页</span><span>arXiv 2402.03300</span></div>

<div class="lesson-lead">DeepSeekMath 先构建高质量数学语料，再从代码模型继续预训练，最后用 GRPO 以组内相对奖励强化可验证数学答案。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像数学集训：先补专业教材，再做大量有标准答案的练习，并按同题同组相对评分。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**数学推理同时受限于专业语料质量、形式化结构能力与稀疏的最终答案奖励。

**它在整条学习链中的位置：**数学语料、代码训练与 GRPO 的组合样板

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：从 Common Crawl 迭代检索并分类数学网页，形成大规模数学预训练语料。
2. **再看核心变化**：从代码模型初始化，利用代码与数学共享的结构化推理模式。
3. **最后看输出**：GRPO 对同一问题采样一组回答，用组内奖励标准化估计优势，省去与策略同规模的 critic。

## 论文拿什么证明

- 论文发布 7B 系列模型并在 MATH 等基准上报告 base、instruction 与 RL 阶段的逐步收益。
- 消融说明数据、代码初始化与 RL 各有贡献，不能把最终分数全部归因给 GRPO。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- DeepSeek-R1 与后续 reasoning RL 继承 GRPO/可验证奖励路线；K3 则把可验证结果扩展到搜索、代码和环境任务。

继续补背景：[第 10 章 · SFT 与 RL](/guide/ch10) · [第 11 章 · 蒸馏与部署](/guide/ch11)

## 不要从论文中过度推出什么

- 最终答案可验证不代表推理过程真实或唯一。
- 组内相对优势依赖采样多样性；回答全对或全错时信号会变弱。

## 原文应该怎么读

**推荐范围：**读数据构建、§4 GRPO 与主要消融；数学 benchmark 表只抓训练阶段差异

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2402.03300" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2402.03300" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：82,896 字符 · 43 个标题</span></div>

### 原文章节地图

1. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
2. 1. Introduction
3. 2. Math Pre-Training
4. 3. Supervised Fine-Tuning
5. 4. Reinforcement Learning
6. 6. Conclusion, Limitation, and Future Work
7. A. Appendix

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pretraining DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO. Figure 1 | Top1 accuracy of open-source models on the competition-level MATH benchmark (Hendrycks et al., 2021) without the use of external toolkits and voting techniques. ∗ Core contributors. † Work done during internship at DeepSeek-AI.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>用四个候选答案的奖励手算一次组内均值和相对优势，说明为什么 GRPO 不需要单独 critic。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
