---
title: FlashAttention-2： Faster Attention with Better Parallelism and Work Partitioning
description: FlashAttention-2 不近似 attention，而是分块搬入片上 SRAM、在线维护 softmax 统计并重算中间量，从而避免反复读写巨大的分数矩阵。
---

# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

<div class="paper-lesson-meta"><span>方向选读</span><span>14 页</span><span>arXiv 2307.08691</span></div>

<div class="lesson-lead">FlashAttention-2 不近似 attention，而是分块搬入片上 SRAM、在线维护 softmax 统计并重算中间量，从而避免反复读写巨大的分数矩阵。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像在小厨房分批取食材并立刻烹饪，避免每一步都把整桌半成品搬回远处仓库。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**标准实现把 $QK^T$ 和 softmax 矩阵写到 HBM；长序列下，昂贵的数据搬运与二次中间内存压过了有效计算。

**它在整条学习链中的位置：**理解 attention 为什么常常卡在搬数据而不是算乘法

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：Tiling：只把 Q/K/V 小块搬到高速 SRAM。
2. **再看核心变化**：Online softmax：保存行最大值与归一化和，使分块结果能精确合并。
3. **最后看输出**：FA2 减少非矩阵 FLOPs，并沿序列增加 thread-block 并行，重新分配 warp 工作以减少共享内存通信。

<figure class="paper-figure"><img src="/paper-figures/flash-attention-2/tiling.png" alt="FlashAttention 分块与在线 softmax 原论文示意图"><figcaption>原论文图：分块计算 attention，并通过重缩放得到精确结果。第一次看只追踪 Q、K、V 块的移动方向。</figcaption></figure>

## 论文拿什么证明

- 论文报告相对 FlashAttention 约 2× 加速，A100 上达到理论峰值的 50–73%。
- 端到端 GPT 训练达到每张 A100 最高约 225 TFLOPs/s，且结果与标准 attention 数学等价。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 的 MLA/KDA kernel 同样遵循 IO-aware 原则；算法复杂度、片上分块与工作划分必须一起设计。

继续补背景：[第 2 章 · KV Cache](/guide/ch02) · [第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- FlashAttention 降低的是中间内存 IO，不把标准 attention 的理论计算从 $O(T^2)$ 改成线性。
- 峰值比例依赖 GPU、head dimension、causal mask 和序列形状。

## 原文应该怎么读

**推荐范围：**先读 §2.2–2.3 与 Fig.1；再读 §3 三项 work partition 改进

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2307.08691" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2307.08691" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：41,299 字符 · 26 个标题</span></div>

### 原文章节地图

1. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
2. 1 Introduction
3. 2 Background
4. 2.1 Hardware characteristics
5. 2.2 Standard Attention Implementation
6. 2.3 FlashAttention
7. 3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning
8. 3.1 Algorithm
9. 3.2 Parallelism
10. 3.3 Work Partitioning Between Warps
11. 4 Empirical Validation
12. 4.1 Benchmarking Attention

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Scaling Transformers to longer sequence lengths has been a major problem in the last several years, promising to improve performance in language modeling and high-resolution image understanding, as well as to unlock new applications in code, audio, and video generation. The attention layer is the main bottleneck in scaling to longer sequences, as its runtime and memory increase quadratically in the sequence length. FlashAttention [5] exploits the asymmetric GPU memory hierarchy to bring significant memory saving (linear instead of quadratic) and runtime speedup (2-4× compared to optimized baselines), with no approximation. However, FlashAttention is still not nearly as fast as optimized matrix-multiply (GEMM) operations, reaching only 25-40% of the theoretical maximum FLOPs/s. We observe that the inefficiency is due to suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes. We propose FlashAttention-2, with better work partitioning to address these issues. In particular, we (1) tweak the algorithm to reduce the number of non-matmul FLOPs (2) parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy, and (3) within each thread block, distribute the work between warps to reduce communication through shared memory. These yield around 2× speedup compared to FlashAttention, reaching 50-73% of the theoretical maximum FLOPs/s on A100 and getting close to the efficiency of GEMM operations. We empirically validate that when used end-to-end to train GPT-style models, FlashAttention-2 reaches training speed of up to 225 TFLOPs/s per A100 GPU (72% model FLOPs utilization).1

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>画出标准 attention 三次 HBM 往返，再画分块版本；说明为什么重算有时反而更快。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
