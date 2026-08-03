---
title: DeepSeek-V3 Technical Report
description: DeepSeek-V3 的价值不只在 671B 模型，而在于展示 MLA、细粒度 MoE、无辅助损失均衡、MTP、FP8 和通信重叠如何共同把稀疏模型变成可训练系统。
---

# DeepSeek-V3 Technical Report

<div class="paper-lesson-meta"><span>核心精读</span><span>53 页</span><span>arXiv 2412.19437</span></div>

<div class="lesson-lead">DeepSeek-V3 的价值不只在 671B 模型，而在于展示 MLA、细粒度 MoE、无辅助损失均衡、MTP、FP8 和通信重叠如何共同把稀疏模型变成可训练系统。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像高铁系统：车体、轨道、信号、调度与供电必须共同设计，单看某个零件解释不了整体速度。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**大规模 MoE 同时遭遇路由均衡与模型质量冲突、跨节点 all-to-all、低精度数值误差、pipeline bubble 和 activation memory。

**它在整条学习链中的位置：**算法—框架—硬件共设计的现代 MoE 样板

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：auxiliary-loss-free balancing 给每个 expert 一个只影响 top-K 选择、不影响 gating weight 的 bias；过载就下调、欠载就上调，从优化目标中移除主要均衡压力。
2. **再看核心变化**：MTP 除 next token 外再预测后续 token，作为训练时的稠密辅助目标，也可为 speculative decoding 提供候选。
3. **最后看输出**：DualPipe 双向调度并重排 attention、dispatch、MLP、combine，使 forward/backward 计算与 all-to-all/PP 通信重叠。
4. **系统如何执行**：细粒度 FP8 训练、block-wise scaling、重计算与专用通信 kernel 共同降低显存和训练时间；任何一项单独看都解释不了整体成本。

## 论文拿什么证明

- 模型为 671B total/37B activated，在 14.8T token 上预训练；报告总正式训练成本 2.788M H800 GPU-hours，其中不含前期研究和消融。
- 61 层、256 routed experts 中激活 8 个，外加 1 个 shared expert；长上下文通过 4K→32K→128K 两阶段 YaRN 激活。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K2 从 V3 骨架继续扩大 sparsity，K3 又把 MLA 主干换成 KDA-dominant hybrid，并用 Quantile Balancing/MoonEP 处理更极端的路由与执行均衡。
- 读 V3 能理解 K3 的系统主张：模型架构选择必须和网络拓扑、通信 kernel、精度格式及调度一起评估。

继续补背景：[第 4 章 · MoE](/guide/ch04) · [第 7 章 · Stable LatentMoE](/guide/ch07) · [第 9 章 · Scaling 与长上下文](/guide/ch09) · [第 11 章 · 蒸馏与部署](/guide/ch11) · [第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- “auxiliary-loss-free”并非完全没有任何均衡项：论文仍保留很小的 sequence-wise 辅助损失以防单序列极端失衡。
- 公开训练美元数依赖假定的 H800 时租，也排除了研究试错；不能直接等同项目总成本。

## 原文应该怎么读

**推荐范围：**精读 PDF p.6–14；读 p.21–24 的数据与长上下文；后训练按需读 p.28–31

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2412.19437" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2412.19437" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：153,671 字符 · 80 个标题</span></div>

### 原文章节地图

1. DeepSeek-V3 Technical Report
2. Contents
3. 1. Introduction
4. 2. Architecture
5. 3. Infrastructures
6. 4. Pre-Training
7. 5. Post-Training
8. 6. Conclusion, Limitations, and Future Directions
9. A. Contributions and Acknowledgments
10. B. Ablation Studies for Low-Precision Training
11. C. Expert Specialization Patterns of the 16B Aux-Loss-Based and Aux-LossFree Models

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks. The model checkpoints are available at https://github.com/deepseek-ai/DeepSeek-V3. DeepSeek-V3 DeepSeek-V2.5

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>解释 routing bias 为什么能改变 expert 负载而不直接扭曲被选 expert 的输出权重，并说明它与普通 auxiliary loss 的梯度路径差异。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
