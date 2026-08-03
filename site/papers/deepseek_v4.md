---
title: DeepSeek-V4： Towards Highly Efficient Million-Token Context Intelligence
description: DeepSeek-V4 用 CSA + HCA 混合压缩长上下文 attention，以 mHC 改造残差连接，并配合 Muon 和完整后训练把窗口扩到一百万 token。
---

# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

<div class="paper-lesson-meta"><span>方向选读</span><span>58 页</span><span>arXiv 2606.19348</span></div>

<div class="lesson-lead">DeepSeek-V4 用 CSA + HCA 混合压缩长上下文 attention，以 mHC 改造残差连接，并配合 Muon 和完整后训练把窗口扩到一百万 token。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像同时使用快速索引和高度压缩档案处理百万页资料，并重新设计楼层间的信息通道。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**百万 token 下标准 attention 的单 token FLOPs 与 KV Cache 过高，网络加深时信息流与训练稳定性也成为瓶颈。

**它在整条学习链中的位置：**另一条百万 token 混合注意力与深度连接路线

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：Compressed Sparse Attention 让 token 只访问经过选择或压缩的上下文。
2. **再看核心变化**：Heavily Compressed Attention 用更强的缓存压缩保留全局通路，两类层混合弥补单一路线的损失。
3. **最后看输出**：Manifold-Constrained Hyper-Connections 让多条残差流可学习混合，同时用约束控制稳定性；优化器采用 Muon。

## 论文拿什么证明

- 报告包含 1.6T/49B activated 的 Pro 与 284B/13B activated 的 Flash，二者在超过 32T token 上预训练并支持 1M context。
- 在 1M setting 下，Pro 报告为 DeepSeek-V3.2 的 27% 单 token 推理 FLOPs 与 10% KV Cache。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- V4 与 K3 都用混合注意力、改造深度连接、Muon 与多教师后训练，但序列机制分别是 CSA/HCA 与 KDA/MLA，适合做同问题不同答案的对照。

继续补背景：[第 3 章 · MLA](/guide/ch03) · [第 6 章 · AttnRes](/guide/ch06) · [第 9 章 · Scaling 与长上下文](/guide/ch09) · [第 11 章 · 蒸馏与部署](/guide/ch11) · [第 13 章 · 训练与服务系统](/guide/ch13) · [第 14 章 · 论文评测](/guide/ch14)

## 不要从论文中过度推出什么

- 27%/10% 的基线是特定 V3.2 配置与 1M 场景，不能直接推广到所有服务请求。
- 百万窗口支持仍需区分可输入、可检索和可完成复杂推理三个层次。

## 原文应该怎么读

**推荐范围：**第一遍读摘要、架构总图与效率表；第二遍拆 CSA/HCA、mHC 和长上下文训练

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2606.19348" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2606.19348" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：177,078 字符 · 106 个标题</span></div>

### 原文章节地图

1. DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
2. Contents
3. 1. Introduction
4. 2. Architecture
5. 3. General Infrastructures
6. 4. Pre-Training
7. 5. Post-Training
8. Tools You have access to a set of tools to help answer the user’s question. You can
9. 6. Conclusion, Limitations, and Future Directions
10. A. Author List and Acknowledgment
11. B. Evaluation Details

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

We present a preview version of DeepSeek-V4 series, including two strong Mixture-ofExperts (MoE) language models — DeepSeek-V4-Pro with 1.6T parameters (49B activated) and DeepSeek-V4-Flash with 284B parameters (13B activated) — both supporting a context length of one million tokens. DeepSeek-V4 series incorporate several key upgrades in architecture and optimization: (1) a hybrid attention architecture that combines Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) to improve long-context efficiency; (2) ManifoldConstrained Hyper-Connections (mHC) that enhance conventional residual connections; (3) and the Muon optimizer for faster convergence and greater training stability. We pre-train both models on more than 32T diverse and high-quality tokens, followed by a comprehensive post-training pipeline that unlocks and further enhances their capabilities. DeepSeek-V4-ProMax, the maximum reasoning effort mode of DeepSeek-V4-Pro, redefines the state-of-the-art for open models, outperforming its predecessors in core tasks. Meanwhile, DeepSeek-V4 series are highly efficient in long-context scenarios. In the one-million-token context setting, DeepSeekV4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2. This enables us to routinely support one-million-token contexts, thereby making long-horizon tasks and further test-time scaling more feasible. The model checkpoints are available at https://huggingface.co/collections/deepseek-ai/deepseek-v4. DeepSeek-V4-Pro-Max Claude-Opus-4.6-Max GPT-5.4-xHigh Gemini-3.1-Pro-High

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>做一张 K3 与 DeepSeek-V4 对照表，分别写出序列、深度、宽度、优化器和后训练方案。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
