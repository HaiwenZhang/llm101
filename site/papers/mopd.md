---
title: MOPD： Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training
description: MOPD 让学生自己生成轨迹，再由对应领域教师逐 token 打分，把数学、指令和软件工程等专长密集地蒸馏进一个模型。
---

# MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training

<div class="paper-lesson-meta"><span>方向选读</span><span>15 页</span><span>arXiv 2606.30406</span></div>

<div class="lesson-lead">MOPD 让学生自己生成轨迹，再由对应领域教师逐 token 打分，把数学、指令和软件工程等专长密集地蒸馏进一个模型。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像让学生自己答题，再把当前每一步分别拿给数学、写作、代码老师现场批改。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**混合 RL 会让领域信号相互干扰，串行 RL 会遗忘，直接参数合并不稳定，拿教师旧轨迹做 SFT 又有 exposure bias。

**它在整条学习链中的位置：**把多个领域 RL 教师合并回一个学生

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：所有领域教师从同一 SFT 起点独立做专门 RL，可以并行开发。
2. **再看核心变化**：蒸馏阶段由 student rollout，保持训练状态与部署状态一致。
3. **最后看输出**：根据 prompt 路由到领域教师；教师在学生已经走到的 token 状态上提供密集 log-prob 信号。

## 论文拿什么证明

- 在 Qwen3-30B-A3B 的数学、指令遵循和软件工程三域实验中，归一化分数 0.937，高于最强比较基线 0.882。
- 论文还报告同源教师、top-k 蒸馏和多轮迭代的重要性，并在 MiMo-V2-Flash 中部署。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 报告使用九个教师做 Multi-Teacher On-Policy Distillation；MOPD 提供理解这种能力整合范式的直接方法背景。

继续补背景：[第 10 章 · SFT 与 RL](/guide/ch10) · [第 11 章 · 蒸馏与部署](/guide/ch11)

## 不要从论文中过度推出什么

- 学生 on-policy 不等于教师 on-policy；教师被冻结，只负责评估学生访问的状态。
- 不同起点的教师可能给出不兼容的概率几何，论文强调 same-origin 的稳定性。

## 原文应该怎么读

**推荐范围：**精读 Fig.1、§3 pipeline 与 top-k distillation；再看三领域整合实验

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2606.30406" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2606.30406" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：43,643 字符 · 19 个标题</span></div>

### 原文章节地图

1. MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training
2. 1 Introduction
3. 2 Related Work
4. 3 Method
5. 4 Experiments
6. 5 Discussion
7. 6 Conclusion
8. OpenAI. 2024. Introducing SWE-bench Verified. https://openai.com/index/introduci ng-swe-bench-verified/.

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Modern large language models (LLMs) rely on reinforcement learning during post-training to push specific capabilities, yet integrating multiple capabilities into one model remains hard. Existing methods, such as Off-Policy Finetune and Mix-RL, are either inefficient or lose performance. In this work, we propose Multi-teacher On-Policy Distillation (MOPD), a post-training paradigm for combining the capabilities of multiple domain RL teachers: we first run per-domain specialised RL to obtain a set of domain teachers, then distill these teachers into the student on its own rollouts. This eliminates exposure bias and provides a dense optimization signal. On Qwen3-30BA3B, MOPD outperforms Mix-RL, Cascade RL, Off-Policy Finetune, and Param-Merge baselines, inheriting nearly all of each teacher’s capability. MOPD also enables parallel, independent development of domain teachers, removing the cross-domain coupling typical of multi-domain post-training. MOPD has been deployed in the post-training of MiMo-V2-Flash, an industrial-scale frontier model, demonstrating its practical value for capability integration in frontier-scale LLMs.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>比较“教师先生成、学生模仿”和“学生先生成、教师逐 token 评分”的状态分布差异。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
