---
title: Efficient Memory Management for Large Language Model Serving with PagedAttention
description: PagedAttention 把连续逻辑 KV Cache 切成固定大小块，通过 block table 映射到不连续显存，像虚拟内存一样按需分配和共享。
---

# Efficient Memory Management for Large Language Model Serving with PagedAttention

<div class="paper-lesson-meta"><span>方向选读</span><span>16 页</span><span>arXiv 2309.06180</span></div>

<div class="lesson-lead">PagedAttention 把连续逻辑 KV Cache 切成固定大小块，通过 block table 映射到不连续显存，像虚拟内存一样按需分配和共享。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像操作系统的虚拟内存：读者看到连续书页，仓库里却可以把每页放在任何空闲格子。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**每个请求输出长度未知；提前预留最大连续空间会产生内部碎片，不同长度请求又造成外部碎片，限制可并发 batch。

**它在整条学习链中的位置：**把操作系统分页思想搬到 KV Cache

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：逻辑 token block 与物理显存 block 解耦，只在新 token 到来时分配。
2. **再看核心变化**：Attention kernel 通过 block table 找到各历史 K/V，不要求物理连续。
3. **最后看输出**：多个采样分支可共享前缀 block，写入分叉时用 copy-on-write。

## 论文拿什么证明

- 论文报告 vLLM 相对 FasterTransformer/Orca 等系统可实现约 2–4× 吞吐提升，收益在长短请求混合和多采样时更明显。
- 核心收益来自接近零的 KV 内存浪费与更大的连续 batch。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 的混合 KDA–MLA prefix cache 仍需要分页、细粒度哈希和状态检查点；PagedAttention 提供服务内存管理基础。

继续补背景：[第 2 章 · KV Cache](/guide/ch02) · [第 13 章 · 训练与服务系统](/guide/ch13)

## 不要从论文中过度推出什么

- 分页解决显存利用率，不减少每个有效 token 的 KV 信息量。
- block 太小增加表项与 kernel 开销，太大又增加尾部浪费。

## 原文应该怎么读

**推荐范围：**精读 Fig.1–4 的碎片问题、block table 和 copy-on-write；调度实验第二遍看

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2309.06180" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2309.06180" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：85,904 字符 · 37 个标题</span></div>

### 原文章节地图

1. arXiv:2309.06180v1 [cs.LG] 12 Sep 2023
2. 1 Introduction
3. 2 Background
4. 2.1 Transformer-Based Large Language Models
5. 2.2 LLM Service & Autoregressive Generation
6. 2.3 Batching Techniques for LLMs
7. 3 Memory Challenges in LLM Serving
8. 3.1 Memory Management in Existing Systems
9. 4 Method
10. 4.1 PagedAttention
11. 4.2 KV Cache Manager
12. 4.5 Scheduling and Preemption

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems. On top of it, we build vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage. Our evaluations show that vLLM improves the throughput of popular LLMs by 2-4× with the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca. The improvement is more pronounced with longer sequences, larger models, and more complex decoding algorithms. vLLM’s source code is publicly available at https://github.com/vllm-project/vllm.

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>给三个最大长度未知的请求画连续预留与 4-token 分页布局，数出两种方案浪费的槽位。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
