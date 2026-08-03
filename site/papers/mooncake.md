---
title: Mooncake： A KVCache-centric Disaggregated Architecture for LLM Serving
description: Mooncake 将 prefill 与 decode GPU 池分离，把 CPU/DRAM/SSD 组织成分布式 KV Cache 池，并由缓存感知调度器在吞吐与延迟 SLO 之间取舍。
---

# Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

<div class="paper-lesson-meta"><span>方向选读</span><span>23 页</span><span>arXiv 2407.00079</span></div>

<div class="lesson-lead">Mooncake 将 prefill 与 decode GPU 池分离，把 CPU/DRAM/SSD 组织成分布式 KV Cache 池，并由缓存感知调度器在吞吐与延迟 SLO 之间取舍。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像把备菜厨房与出餐窗口分开，再建设共享冷库，由总调度根据订单和缓存位置派工。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**Prefill 偏计算、decode 偏带宽，两者混在同一 GPU 互相干扰；长 prompt 的 KV 又昂贵，命中、迁移和过载准入决定整体成本。

**它在整条学习链中的位置：**把 KV Cache 当作服务系统的核心资源

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：Prefill pool 负责建立 KV，decode pool 负责逐 token 生成。
2. **再看核心变化**：KV 可经 RDMA 在 GPU 与分布式 CPU/DRAM/SSD 池之间传输和复用。
3. **最后看输出**：Conductor 同时做 cache-aware prefill、KV 负载均衡与 decode 调度，并在预测无法满足 SLO 时提前拒绝。

## 论文拿什么证明

- 模拟场景中吞吐最高提升 525%；真实 Kimi 负载中架构可多处理约 75% 请求。
- 论文特别报告过载场景，而不是假设所有请求都必须接收。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 面对 2K 到 1M token 请求跨度时使用类似的 cache affinity、资源预算准入和混合缓存思路。

继续补背景：[第 2 章 · KV Cache](/guide/ch02) · [第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- 跨池传 KV 不是免费：RDMA 带宽、序列长度与重算成本决定是否值得。
- 提前拒绝提高有效吞吐，却改变了服务可用性与公平性目标。

## 原文应该怎么读

**推荐范围：**读架构 Fig.1、prefill/decode 分离、KV 池与 Conductor 调度；重点看真实流量实验

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2407.00079" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2407.00079" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：86,676 字符 · 40 个标题</span></div>

### 原文章节地图

1. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
2. 1 Introduction
3. 1.1 Motivation of Developing Mooncacke
4. 1.2 Design and Results of Mooncacke
5. 2 Preliminary and Problem Definition
6. 3 Overview of Mooncake’s Disaggregated Architecture
7. 4 Sampled Real-world Request Trace
8. 4.1 Data Details
9. 4.2 Statistical Features
10. 5 Implementation of the Prefill Pool
11. 5.1 Multi-node Prefill
12. 5.2 Layer-wise Prefill

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI. It features a KVCache-centric disaggregated architecture that separates the prefill and decoding clusters. It also leverages the underutilized CPU, DRAM, and SSD resources of the GPU cluster to implement a disaggregated cache of KVCache. The core of Mooncake is its KVCache-centric scheduler, which balances maximizing overall effective throughput while meeting latencyrelated Service Level Objectives (SLOs). Unlike traditional studies that assume all requests will be processed, Mooncake faces challenges due to highly overloaded scenarios. To mitigate these, we developed a prediction-based early rejection policy. Experiments show that Mooncake excels in long-context scenarios. Compared to the baseline method, Mooncake can achieve up to a 525% increase in throughput in certain simulated scenarios while adhering to SLOs. Under real workloads, Mooncake’s innovative architecture enables Kimi to handle 75% more requests.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>为一个 100K prompt + 200 token 输出请求，分别列出 prefill 节点和 decode 节点最重的资源，并决定 KV 应传输还是重算。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
