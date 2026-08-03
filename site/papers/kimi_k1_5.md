---
title: Kimi k1.5： Scaling Reinforcement Learning with LLMs
description: K1.5 把更长 context 看成 RL 搜索预算：模型在单条自回归轨迹中学习规划、回退与修正，再用 partial rollout 复用未完成轨迹以控制采样成本。
---

# Kimi k1.5: Scaling Reinforcement Learning with LLMs

<div class="paper-lesson-meta"><span>核心精读</span><span>25 页</span><span>arXiv 2501.12599</span></div>

<div class="lesson-lead">K1.5 把更长 context 看成 RL 搜索预算：模型在单条自回归轨迹中学习规划、回退与修正，再用 partial rollout 复用未完成轨迹以控制采样成本。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像训练马拉松选手：不能只看最后是否到终点，还要处理超长轨迹、等待和中途续跑。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**显式 MCTS、value model 和 process reward 难以在开放 token 空间稳定扩展，而长 CoT rollout 又昂贵、长尾严重并容易无效变长。

**它在整条学习链中的位置：**Kimi 系列长 CoT RL 与 partial rollout 的源头

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：先用小而高质量的 long-CoT SFT 冷启动规划、反思和探索模式，再在可验证问题上做 outcome RL。
2. **再看核心变化**：策略优化是带相对熵正则的 online mirror descent 变体；每题采样一组响应，用组内平均 reward 作 baseline，并惩罚相对旧策略的 log-ratio。
3. **最后看输出**：length penalty 只在组内正确答案间鼓励更短轨迹，避免把错误但短的答案奖励成捷径。
4. **系统如何执行**：partial rollout 暂停未完成长轨迹并在下一轮续跑，复用前缀而不是从头重新生成；hybrid deployment 在同一批设备切换 rollout 与训练。
5. **为什么有效**：long2short 通过长度约束、长 CoT 激活/数据与模型合并，把长推理能力压缩到短输出。

## 论文拿什么证明

- 论文把 RL context 扩到 128K，并报告随 context 增长继续改善的趋势；文本、代码和视觉 reasoning 结果共同支持跨模态 outcome RL。
- 消融强调 prompt 集合的覆盖、难度和可验证性，以及 sampling 与 length control 对训练效率的重要性。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K2 继承其 policy objective 与 partial rollout，K3 再加入外置 KV cache、persistent sandbox 和 1M context，把“保存模型前缀”升级为“保存模型与环境联合状态”。

继续补背景：[第 10 章 · SFT 与 RL](/guide/ch10) · [第 12 章 · Agent](/guide/ch12) · [第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- 长 context 是可用搜索预算，不保证模型有效使用预算；若 reward 或数据不对，额外 token 会变成重复和 overthinking。
- 论文的强结果来自较强 base model、精心筛选的可验证任务和大规模系统，不能由小模型上的简单 GRPO 实验直接外推。

## 原文应该怎么读

**推荐范围：**精读 PDF p.2–8；读 p.11–14 的主结果/长上下文，系统细节按需

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2501.12599" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2501.12599" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：101,337 字符 · 65 个标题</span></div>

### 原文章节地图

1. Kimi Team
2. Figure 1: Kimi k1.5 long-CoT results.
3. 2.1 RL Prompt Set Curation
4. 2.2 Long-CoT Supervised Fine-Tuning
5. 2.3.2 Policy Optimization
6. 2.3.3 Length Penalty
7. 2.3.4 Sampling Strategies
8. 2.3.5 More Details on Training Recipe
9. 2.4 Long2short: Context Compression for Short-CoT Models
10. 2.5 Other Training Details
11. 2.6 RL Infrastructure
12. 2.6.1 Large Scale Reinforcement Learning Training System for LLM

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive results. In light of this, we report on the training practice of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques, multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy optimization methods are key ingredients of our approach, which establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models. Notably, our system achieves state-of-the-art reasoning performance across multiple benchmarks and modalities—e.g., 77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista—matching OpenAI’s o1. Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results—e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench—outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin (up to +550%). Kimi k1.5 long-CoT OpenAI o1 OpenAI o1-mini QVQ-72B-Preview QwQ-32B Preview Math 77.5 74.4 63.6 50 AIME 2024 (Pass@1) 96.2 94.8 90 90.6 MATH 500 (EM) Code 94 94 88 62 Codeforces (Percentile) 67.2 62.5 53.1 40.6 LiveCodeBench v5 24.12-25.2 (Pass@1) Vision 74.9 71 71.4 MathVista (Pass@1) 77.3 70 70.3 MMMU (Pass@1)

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>比较重新 rollout 128K 轨迹与保存 partial rollout 的成本；列出恢复时除 token 前缀外还必须保存的随机性、环境与版本状态。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
