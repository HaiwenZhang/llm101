---
title: DeepSeekMoE： Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
description: DeepSeekMoE 用更小但更多的 routed experts 提高组合精度，再让 always-on shared experts 吸收公共知识，从而减少 routed experts 的知识混杂与重复。
---

# DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

<div class="paper-lesson-meta"><span>核心精读</span><span>33 页</span><span>arXiv 2401.06066</span></div>

<div class="lesson-lead">DeepSeekMoE 用更小但更多的 routed experts 提高组合精度，再让 always-on shared experts 吸收公共知识，从而减少 routed experts 的知识混杂与重复。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像综合医院把大科室拆成更细专科，同时保留一个所有病人都先去的全科门诊。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**传统少量大专家容易同时学习多类无关知识，不同专家又会重复存储公共模式，导致稀疏参数没有真正转化为专门化容量。

**它在整条学习链中的位置：**理解细粒度专家与共享专家为何成为主流

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：在总专家参数和激活计算近似不变时，把 N 个专家各切成 m 份，并把 top-K 改为 top-mK；粒度更细后，一个 token 能组合多个小技能。
2. **再看核心变化**：从 routed 集合中隔离 Ks 个 shared experts，使它们对每个 token 总是激活；公共知识进入 shared experts，routed experts 更专注差异化模式。
3. **最后看输出**：组合空间会快速扩大：论文示例中 16 选 2 只有 120 种组合，而切细后 64 选 8 超过 44 亿种。
4. **系统如何执行**：负载均衡仍是必要约束，否则路由坍缩会让部分专家学不到数据，并造成跨设备 straggler。

## 论文拿什么证明

- 2B 受控实验与消融分别验证细粒度切分和 shared expert；16B/2.8B activated 模型用约 40% 计算达到论文所比较 7B dense 模型的相近表现。
- 论文还给出 145B 初步扩展结果，但措辞和实验完整度弱于 2B/16B 部分，应降低证据权重。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K2、DeepSeek-V3 与 K3 都沿用 fine-grained routed + shared expert 范式；K3 进一步把 routed experts 放进 latent 维度并把专家数扩到 896。

继续补背景：[第 4 章 · MoE](/guide/ch04) · [第 7 章 · Stable LatentMoE](/guide/ch07)

## 不要从论文中过度推出什么

- 组合数是可用函数族的上界直觉，不代表训练后每种组合都被充分使用或语义独立。
- 总参数、激活参数、FLOPs 和显存带宽是四个不同量；MoE 降低每 token 计算，并不会自动降低存储和通信。

## 原文应该怎么读

**推荐范围：**精读 PDF p.4–7；读 p.9–16 的实验、消融与扩展

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2401.06066" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2401.06066" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：91,661 字符 · 53 个标题</span></div>

### 原文章节地图

1. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
2. 1. Introduction
3. 2. Preliminaries: Mixture-of-Experts for Transformers
4. 5. Scaling up to DeepSeekMoE 16B
5. 6. Alignment for DeepSeekMoE 16B
6. 7. DeepSeekMoE 145B Ongoing
7. 8. Related Work
8. 9. Conclusion
9. Appendices
10. C. Training Benchmark Curves of DeepSeekMoE 16B

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

In the era of large language models, Mixture-of-Experts (MoE) is a promising architecture for managing computational costs when scaling up model parameters. However, conventional MoE architectures like GShard, which activate the top-𝐾 out of 𝑁 experts, face challenges in ensuring expert specialization, i.e. each expert acquires non-overlapping and focused knowledge. In response, we propose the DeepSeekMoE architecture towards ultimate expert specialization. It involves two principal strategies: (1) finely segmenting the experts into 𝑚𝑁 ones and activating 𝑚𝐾 from them, allowing for a more flexible combination of activated experts; (2) isolating 𝐾𝑠 experts as shared ones, aiming at capturing common knowledge and mitigating redundancy in routed experts. Starting from a modest scale with 2B parameters, we demonstrate that DeepSeekMoE 2B achieves comparable performance with GShard 2.9B, which has 1.5× expert parameters and computation. In addition, DeepSeekMoE 2B nearly approaches the performance of its dense counterpart with the same number of total parameters, which set the upper bound of MoE models. Subsequently, we scale up DeepSeekMoE to 16B parameters and show that it achieves comparable performance with LLaMA2 7B, with only about 40% of computations. Further, our preliminary efforts to scale up DeepSeekMoE to 145B parameters consistently validate its substantial advantages over the GShard architecture, and show its performance comparable with DeepSeek 67B, using only 28.5% (maybe even 18.2%) of computations.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>在固定激活 FLOPs 下比较“8 个大专家 top-1”和“64 个小专家 top-8”，列出质量、通信、权重读取和负载均衡的变化。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
