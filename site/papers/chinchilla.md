---
title: Training Compute-Optimal Large Language Models
description: 在固定训练 FLOPs 下，模型并非越大越好；Chinchilla 的核心经验是参数量 N 与训练 token 数 D 应随计算预算近似等比例增长。
---

# Training Compute-Optimal Large Language Models

<div class="paper-lesson-meta"><span>核心精读</span><span>36 页</span><span>arXiv 2203.15556</span></div>

<div class="lesson-lead">在固定训练 FLOPs 下，模型并非越大越好；Chinchilla 的核心经验是参数量 N 与训练 token 数 D 应随计算预算近似等比例增长。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像用固定预算备考：不能把钱全花在买更厚的书，也要留下足够时间真正做题。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**早期 scaling 实践常固定约 300B token 只扩大模型，导致大模型没有被充分训练，也让推理成本不必要地升高。

**它在整条学习链中的位置：**训练计算预算如何分给参数和数据

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：论文用三种方法估计 compute-optimal frontier：训练曲线包络、IsoFLOP 谷底和参数化损失 $L(N,D)=E+A/N^α+B/D^β$。
2. **再看核心变化**：三种估计都得到近似 $N_{opt} ∝ C^{0.5}$、$D_{opt} ∝ C^{0.5}$；训练计算粗略为参数量与 token 数的乘积。
3. **最后看输出**：关键是优化约束：这是固定 pre-training compute 下的最终 loss 最优，不等于固定延迟、固定显存、固定数据质量或固定项目总成本下的最优。

## 论文拿什么证明

- 作者训练 400 多个 70M–16B 模型，覆盖约 5B–500B token，并用 70B/1.4T 的 Chinchilla 在与 280B Gopher 相近训练计算下验证预测。
- Chinchilla 不仅下游表现更强，较小参数量也降低微调与推理成本，说明训练最优与部署经济性可以同时受益。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K2/K3 的 scaling law 已把 MoE sparsity、架构效率和 token utility 纳入设计，不能机械套用 dense Chinchilla 比例；但“先做小模型族拟合，再决定大 run”仍是核心方法论。

继续补背景：[第 9 章 · Scaling 与长上下文](/guide/ch09)

## 不要从论文中过度推出什么

- 常见的“每参数约 20 token”只是该论文拟合区域和假设下的便捷近似，不是自然常数。
- 重复数据、合成数据、数据质量、MoE 激活参数与推理次数都会改变真正的项目最优点。

## 原文应该怎么读

**推荐范围：**精读 PDF p.1–8；方法细节按需读附录

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2203.15556" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2203.15556" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：101,621 字符 · 58 个标题</span></div>

### 原文章节地图

1. arXiv:2203.15556v1 [cs.CL] 29 Mar 2022
2. Training Compute-Optimal Large Language Models
3. 1. Introduction
4. 2. Related Work
5. 3. Estimating the optimal parameter/training tokens allocation
6. 4. Chinchilla
7. 5. Discussion & Conclusion
8. 6. Acknowledgements
9. A. Training dataset
10. B. Optimal cosine cycle length
11. C. Consistency of scaling results across datasets
12. D. Details on the scaling analyses



## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>分别写出训练计算最优和生命周期成本最优的目标函数；解释为什么面向高 QPS 服务时可能选择比 Chinchilla frontier 更小、更充分训练的模型。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
