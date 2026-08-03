---
title: LLM Systems 逐讲资料
description: Large Language Model Systems 的逐讲 Slides、讲义与论文阅读官方索引
---

# LLM Systems 逐讲资料

> **课程**：Large Language Model Systems · Spring 2025  
> **学校**：Carnegie Mellon University  
> **官方主页**：[https://llmsystem.github.io/llmsystem2025spring/docs/Syllabus/](https://llmsystem.github.io/llmsystem2025spring/docs/Syllabus/)  
> **抓取与校验日期**：2026-08-03

::: tip 这是来源与深挖页，不是主学习顺序
本页共索引 **62 份官方 Slides / PDF**，合计 **2,413 页 / 153.7 MB**。PDF 统一链接到课程官网、论文官网或 arXiv。
:::

初学请先沿[大模型系统课](/beginner/)学习；需要核对某一知识点来自哪些课程时，使用[名校课程知识覆盖表](/curriculum/sources)。

从 GPU 与分布式训练出发，系统学习并行、量化、MoE、推理优化、PagedAttention 与在线服务。
课程表共整理 **26 份讲义条目**与 **43 项论文 / 延伸阅读**；其中 **36 份阅读有公开 PDF**，另有 **7 项**只有网页、博客、视频或受限入口，因此保留官方在线链接。

## 建议怎么学

1. 先读每一讲的“本讲抓什么”，确认自己要回答的问题。
2. 第一遍快速翻 Slides，只看标题、图和结论；不在公式处停太久。
3. 第二遍结合 Notes 或 Recitation，把不懂的概念补齐。
4. 最后从论文阅读里挑 1–2 篇精读，并回到本站对应的零基础章节复习。

[查看课程资料机器可读清单（JSON）](/course-materials/manifest.json)

---

## L01 · 大语言模型导论

**日期**：2025-01-13  
**英文主题**：Introduction to LLM

**本讲抓什么**：建立课程总地图，认识模型、数据、训练、推理和系统工程之间的依赖关系。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-01-intro-cd7350c64ac8b720a51aa45a52e9fa50.pdf) · 49 页 · 2.17 MB

---

## L02 · GPU 编程基础一

**日期**：2025-01-13  
**英文主题**：GPU Programming Basics 1

**本讲抓什么**：从线程、Block、内存层级和 Kernel 入手建立 GPU 并行计算直觉，为后续算子优化打基础。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-02-gpu-programming-b6078a9a5fc33d8ae03dff5b3e5fd8bb.pdf) · 33 页 · 1.05 MB

### 论文与延伸阅读

- **论文阅读 · 仅在线** · [Programming Massively Parallel Processors, 4th Ed](https://learning.oreilly.com/library/view/programming-massively-parallel/9780323984638/?sso_link=yes&sso_link_from=cmu-edu) — 课程页没有公开直链 PDF

---

## L03 · GPU 编程基础二

**日期**：2025-01-22  
**英文主题**：GPU Programming Basics 2

**本讲抓什么**：从线程、Block、内存层级和 Kernel 入手建立 GPU 并行计算直觉，为后续算子优化打基础。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-03-gpu-programming2-4075ed5f62b3601db6bbe1991e5980c0.pdf) · 26 页 · 0.44 MB

### 论文与延伸阅读

- **论文阅读 · 仅在线** · [Programming Massively Parallel Processors, 4th Ed](https://learning.oreilly.com/library/view/programming-massively-parallel/9780323984638/?sso_link=yes&sso_link_from=cmu-edu) — 课程页没有公开直链 PDF

---

## L04 · 学习算法与自动微分

**日期**：2025-01-27  
**英文主题**：Learning algorithm and Auto Differentiation

**本讲抓什么**：用计算图理解前向与反向模式自动微分，以及深度学习框架如何生成和执行梯度计算。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-04-autodiff-2ff210695355c7f9089b6209c515731c.pdf) · 39 页 · 1.07 MB

### 论文与延伸阅读

- **论文阅读** · [Auto Diff survey（官方 PDF）](https://arxiv.org/pdf/1502.05767.pdf) · 43 页 · 0.56 MB · [官方来源页](https://arxiv.org/abs/1502.05767)
- **论文阅读** · [Differentiable Programming（官方 PDF）](https://arxiv.org/pdf/2403.14606.pdf) · 485 页 · 8.28 MB · [官方来源页](https://arxiv.org/abs/2403.14606)

---

## L05 · 深度学习框架设计

**日期**：2025-01-27  
**英文主题**：Deep Learning Frameworks Design

**本讲抓什么**：从张量、算子、自动微分、执行图与设备调度理解深度学习框架的核心设计。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-05-dl-framework-3786a6ca3a677b14b94300fa6734893b.pdf) · 36 页 · 1.09 MB

### 论文与延伸阅读

- **论文阅读** · [Tensorflow（官方 PDF）](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf) · 21 页 · 2.14 MB

---

## L06 · Transformer 系统基础

**日期**：2025-02-03  
**英文主题**：Transformer

**本讲抓什么**：围绕本讲主题建立概念、方法、系统实现与评价标准之间的联系，并结合课件和论文理解关键设计取舍。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-06-transformer-8cbfe810b0027cd5aed9f0c649499352.pdf) · 26 页 · 0.84 MB

### 论文与延伸阅读

- **论文阅读** · [Attention is all you need（官方 PDF）](https://arxiv.org/pdf/1706.03762.pdf) · 15 页 · 2.11 MB · [官方来源页](https://arxiv.org/abs/1706.03762)

---

## L07 · 预训练大语言模型

**日期**：2025-02-03  
**英文主题**：Pre-trained LLMs

**本讲抓什么**：围绕本讲主题建立概念、方法、系统实现与评价标准之间的联系，并结合课件和论文理解关键设计取舍。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-07-llms-2054572db24531f7bea2feb74baaf987.pdf) · 22 页 · 1.14 MB

### 论文与延伸阅读

- **论文阅读** · [LLaMA（官方 PDF）](https://arxiv.org/pdf/2302.13971.pdf) · 27 页 · 0.69 MB · [官方来源页](https://arxiv.org/abs/2302.13971)
- **论文阅读** · [GPT3（官方 PDF）](https://arxiv.org/pdf/2005.14165.pdf) · 75 页 · 6.45 MB · [官方来源页](https://arxiv.org/abs/2005.14165)
- **论文阅读 · 仅在线** · [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — 课程页没有公开直链 PDF

---

## L08 · Tokenization

**日期**：2025-02-10  
**英文主题**：Tokenization

**本讲抓什么**：理解 BPE、SentencePiece 与词表训练，观察 Token 粒度如何影响成本、多语言公平和模型输入长度。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-08-tokenization-46bef31fb38ec1f6a0ab0c593c5959b3.pdf) · 45 页 · 1.77 MB

### 论文与延伸阅读

- **论文阅读** · [BPE（官方 PDF）](https://aclanthology.org/P16-1162.pdf) · 11 页 · 0.32 MB · [官方来源页](https://aclanthology.org/P16-1162/)
- **论文阅读** · [Sentence-Piece（官方 PDF）](https://aclanthology.org/D18-2012.pdf) · 6 页 · 0.11 MB · [官方来源页](https://aclanthology.org/D18-2012/)
- **论文阅读** · [VOLT（官方 PDF）](https://aclanthology.org/2021.acl-long.571.pdf) · 13 页 · 0.64 MB · [官方来源页](https://aclanthology.org/2021.acl-long.571/)

---

## L09 · 大模型解码

**日期**：2025-02-10  
**英文主题**：LLM Decoding

**本讲抓什么**：比较贪心、Beam Search、采样和推测解码，理解质量、多样性、延迟与吞吐之间的权衡。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-09-decoding-7735f8be9186c8840ed83128173a0c8f.pdf) · 21 页 · 0.44 MB

---

## L10 · GPU 加速

**日期**：2025-02-17  
**英文主题**：GPU Acceleration

**本讲抓什么**：围绕访存、并行度、融合 Kernel 和硬件利用率分析算子为什么慢，以及如何定位优化空间。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-10-gpu-acceleration-e14f3a7de2dae98b3cb01ce8034e8e7e.pdf) · 35 页 · 0.96 MB

### 论文与延伸阅读

- **论文阅读 · 仅在线** · [Programming Massively Parallel Processors, 4th Ed](https://learning.oreilly.com/library/view/programming-massively-parallel/9780323984638/?sso_link=yes&sso_link_from=cmu-edu) — 课程页没有公开直链 PDF

---

## L11 · GPU 上的 Transformer 加速一

**日期**：2025-02-17  
**英文主题**：Accelerating Transformer on GPU Part 1

**本讲抓什么**：把 Transformer 拆成矩阵乘、归一化、Attention 与通信步骤，理解 Kernel 融合和端到端加速。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-11-transformer-acc-118c60f3304ec2de30bd6a80cb670613.pdf) · 67 页 · 9.28 MB

### 论文与延伸阅读

- **论文阅读** · [LightSeq（官方 PDF）](https://arxiv.org/pdf/2010.13887.pdf) · 8 页 · 0.47 MB · [官方来源页](https://arxiv.org/abs/2010.13887)

---

## L12 · GPU 上的 Transformer 加速二

**日期**：2025-02-24  
**英文主题**：Accelerating Transformer on GPU Part 2

**本讲抓什么**：把 Transformer 拆成矩阵乘、归一化、Attention 与通信步骤，理解 Kernel 融合和端到端加速。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-11-transformer-acc-118c60f3304ec2de30bd6a80cb670613.pdf) · 67 页 · 9.28 MB

### 论文与延伸阅读

- **论文阅读** · [LightSeq2（官方 PDF）](https://arxiv.org/pdf/2110.05722.pdf) · 13 页 · 1.51 MB · [官方来源页](https://arxiv.org/abs/2110.05722)

---

## L13 · 分布式模型训练一

**日期**：2025-02-24  
**英文主题**：Distributed Model Training

**本讲抓什么**：理解参数、梯度、优化器状态和激活如何在设备间切分，以及同步通信为何决定训练效率。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-13-distributed-training-4012db2fbea5c325dea36cc9f6ccbae5.pdf) · 50 页 · 1.16 MB

---

## L14 · 分布式模型训练二

**日期**：2025-03-10  
**英文主题**：Distributed Model Training II

**本讲抓什么**：理解参数、梯度、优化器状态和激活如何在设备间切分，以及同步通信为何决定训练效率。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-14-ddp-4c41acd0996bb77e65703f85b1340b3f.pdf) · 25 页 · 0.79 MB

### 论文与延伸阅读

- **论文阅读** · [DDP（官方 PDF）](https://www.vldb.org/pvldb/vol13/p3005-li.pdf) · 14 页 · 1.16 MB

---

## L15 · 分布式模型训练三

**日期**：2025-03-10  
**英文主题**：Distributed Model Training III

**本讲抓什么**：理解参数、梯度、优化器状态和激活如何在设备间切分，以及同步通信为何决定训练效率。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-15-model-parallel-1278ecd34702c1538bf26894762ec90f.pdf) · 35 页 · 2.04 MB

### 论文与延伸阅读

- **论文阅读** · [GPipe（官方 PDF）](https://arxiv.org/pdf/1811.06965.pdf) · 11 页 · 0.51 MB · [官方来源页](https://arxiv.org/abs/1811.06965)
- **论文阅读** · [Megatron-LM（官方 PDF）](https://arxiv.org/pdf/2104.04473.pdf) · 13 页 · 0.99 MB · [官方来源页](https://arxiv.org/abs/2104.04473)

---

## L16 · 模型量化一

**日期**：2025-03-17  
**英文主题**：Model Quantization

**本讲抓什么**：理解权重与激活从高精度映射到低比特的过程，比较误差、显存、速度和硬件支持之间的取舍。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-16-quantization-80e192f2e967b00c68b29faa9d9e71de.pdf) · 26 页 · 0.85 MB

---

## L17 · 模型量化二

**日期**：2025-03-17  
**英文主题**：Model Quantization II

**本讲抓什么**：理解权重与激活从高精度映射到低比特的过程，比较误差、显存、速度和硬件支持之间的取舍。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-17-quantization2-bec91c67e6870c9c398fcc4a22f0b446.pdf) · 36 页 · 1.39 MB

### 论文与延伸阅读

- **论文阅读** · [GPTQ（官方 PDF）](https://arxiv.org/pdf/2210.17323.pdf) · 16 页 · 0.5 MB · [官方来源页](https://arxiv.org/abs/2210.17323)

---

## L18 · 大模型高效微调

**日期**：2025-03-24  
**英文主题**：Efficient fine-tuning for Large Models

**本讲抓什么**：比较全参数微调、LoRA、量化微调和指令适配，判断不同数据与硬件预算下该更新哪些参数。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-18-peft-1555d9a7770e87fb10e2e95bf46ef12d.pdf) · 45 页 · 1.97 MB

### 论文与延伸阅读

- **论文阅读** · [CIAT（官方 PDF）](https://arxiv.org/pdf/2104.08154.pdf) · 12 页 · 0.44 MB · [官方来源页](https://arxiv.org/abs/2104.08154)
- **论文阅读** · [LORA（官方 PDF）](https://arxiv.org/pdf/2106.09685.pdf) · 26 页 · 1.53 MB · [官方来源页](https://arxiv.org/abs/2106.09685)
- **论文阅读** · [QLoRA（官方 PDF）](https://arxiv.org/pdf/2305.14314.pdf) · 26 页 · 1.02 MB · [官方来源页](https://arxiv.org/abs/2305.14314)

---

## L19 · 大模型与 MoE

**日期**：2025-03-24  
**英文主题**：Large models with Mixture-of-Expert

**本讲抓什么**：学习路由器如何只激活少量专家，理解稀疏计算、负载均衡、通信与专家容量问题。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-19-MoE-1fd3b9f72ba69c8d3d71e01186674c5e.pdf) · 38 页 · 1.4 MB

### 论文与延伸阅读

- **论文阅读** · [GShard（官方 PDF）](https://arxiv.org/pdf/2006.16668.pdf) · 35 页 · 1.67 MB · [官方来源页](https://openreview.net/forum?id=qrwe7XHTmYb)；OpenReview 下载端点拒绝自动访问；改用作者公开的 arXiv 版本。
- **论文阅读** · [Switch Transformer（官方 PDF）](https://arxiv.org/pdf/2101.03961.pdf) · 40 页 · 1.24 MB · [官方来源页](https://arxiv.org/abs/2101.03961)
- **论文阅读** · [DeepSpeed-MOE（官方 PDF）](https://arxiv.org/pdf/2201.05596.pdf) · 31 页 · 1.29 MB
- **论文阅读** · [Deepseek-MoE（官方 PDF）](https://arxiv.org/pdf/2401.06066.pdf) · 33 页 · 0.7 MB · [官方来源页](https://arxiv.org/abs/2401.06066)

---

## L20 · 面向现代硬件优化 Attention

**日期**：2025-03-31  
**英文主题**：Optimizing Attention for Modern Hardware (Tri Dao)

**本讲抓什么**：围绕本讲主题建立概念、方法、系统实现与评价标准之间的联系，并结合课件和论文理解关键设计取舍。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-20-FlashAttention_tridao-cac5b634b4ad77cb027451422b07ae75.pdf) · 56 页 · 3.25 MB

### 论文与延伸阅读

- **论文阅读** · [FlashAttention（官方 PDF）](https://arxiv.org/pdf/2205.14135.pdf) · 34 页 · 2.51 MB

---

## L21 · 通信高效的分布式训练

**日期**：2025-03-31  
**英文主题**：Communication Efficient Distributed Training

**本讲抓什么**：理解参数、梯度、优化器状态和激活如何在设备间切分，以及同步通信为何决定训练效率。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-21-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf) · 76 页 · 1.63 MB

### 论文与延伸阅读

- **论文阅读** · [ZeRO (DeepSpeed)（官方 PDF）](https://arxiv.org/pdf/1910.02054.pdf) · 24 页 · 0.74 MB

---

## L22 · PagedAttention 与大模型服务

**日期**：2025-04-07  
**英文主题**：LLM Serving with PageAttention (Woosuk Kwon)

**本讲抓什么**：从请求调度、连续批处理、KV Cache、并行和可观测性建立大模型在线服务的系统视角。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-22-vLLM_woosuk_kwon-1f34697dbb1a1fb5b798daf6eff14b67.pdf) · 89 页 · 3.6 MB

### 论文与延伸阅读

- **论文阅读** · [vLLM（官方 PDF）](https://arxiv.org/pdf/2309.06180.pdf) · 16 页 · 1.39 MB · [官方来源页](https://arxiv.org/abs/2309.06180)

---

## L23 · 更高效的 KV Cache 服务

**日期**：2025-04-07  
**英文主题**：Better KV Cache for LLM Serving (Yuhan Liu)

**本讲抓什么**：研究 KV Cache 的传输、复用、压缩与混合策略，降低长上下文服务的首 Token 延迟和显存占用。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-23-LMCache_yuhan_liu-168b4d638987bf0e6408d553486059b1.pdf) · 59 页 · 2.73 MB

### 论文与延伸阅读

- **论文阅读** · [CacheGen（官方 PDF）](https://arxiv.org/pdf/2310.07240.pdf) · 19 页 · 1.58 MB · [官方来源页](https://arxiv.org/abs/2310.07240)
- **论文阅读** · [CacheBlend（官方 PDF）](https://arxiv.org/pdf/2405.16444.pdf) · 16 页 · 1.5 MB · [官方来源页](https://arxiv.org/abs/2405.16444)

---

## L24 · DistServe：Prefill/Decode 分离

**日期**：2025-04-14  
**英文主题**：DistServe: Disaggregated Prefill-Decoding (Hao Zhang)

**本讲抓什么**：比较贪心、Beam Search、采样和推测解码，理解质量、多样性、延迟与吞吐之间的权衡。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-24-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf) · 68 页 · 1.96 MB

### 论文与延伸阅读

- **论文阅读** · [DistServe（官方 PDF）](https://arxiv.org/pdf/2401.09670.pdf) · 18 页 · 0.62 MB · [官方来源页](https://arxiv.org/abs/2401.09670)

---

## L25 · SGLang 大模型服务

**日期**：2025-04-14  
**英文主题**：LLM serving with SGL (Ying Sheng)

**本讲抓什么**：从请求调度、连续批处理、KV Cache、并行和可观测性建立大模型在线服务的系统视角。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-25-sglang-72edc5043338f59db34d47e5b96ac870.pdf) · 31 页 · 1.41 MB

### 论文与延伸阅读

- **论文阅读** · [SGLang（官方 PDF）](https://arxiv.org/pdf/2312.07104.pdf) · 20 页 · 1.32 MB · [官方来源页](https://arxiv.org/abs/2312.07104)

---

## L26 · 高效大模型强化学习系统

**日期**：2025-04-21  
**英文主题**：Efficient Reinforcement Learning System for LLMs

**本讲抓什么**：理解状态、动作、奖励、策略梯度与 PPO，并连接到偏好对齐、推理训练和 Agent 行为优化。

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

### 论文与延伸阅读

- **论文阅读** · [ReaLHF（官方 PDF）](https://arxiv.org/pdf/2406.14088.pdf) · 20 页 · 1.08 MB · [官方来源页](https://arxiv.org/abs/2406.14088)

---

## L27 · 应用栈与模型服务

**日期**：2025-04-28  
**英文主题**：App Stack and Model Serving

**本讲抓什么**：从请求调度、连续批处理、KV Cache、并行和可观测性建立大模型在线服务的系统视角。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-15-serving-c4a70ab21cde01fb60068a256c6e163a.pdf) · 49 页 · 4.06 MB

### 论文与延伸阅读

- **论文阅读 · 仅在线** · [Triton](https://developer.nvidia.com/triton-inference-server) — 课程页没有公开直链 PDF
- **论文阅读 · 仅在线** · [LightLLM](https://github.com/ModelTC/lightllm/blob/main/docs/LightLLM.md) — 课程页没有公开直链 PDF

---

## L28 · GPU 即时编译

**日期**：2025-04-28  
**英文主题**：GPU just-in-time compilation

**本讲抓什么**：围绕本讲主题建立概念、方法、系统实现与评价标准之间的联系，并结合课件和论文理解关键设计取舍。

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

### 论文与延伸阅读

- **论文阅读** · [JAX（官方 PDF）](https://mlsys.org/Conferences/doc/2018/146.pdf) · 3 页 · 0.6 MB

---

## L29 · 推测解码

**日期**：2025-04-28  
**英文主题**：Speculative Decoding

**本讲抓什么**：比较贪心、Beam Search、采样和推测解码，理解质量、多样性、延迟与吞吐之间的权衡。

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

### 论文与延伸阅读

- **论文阅读** · [Speculative Decoding（官方 PDF）](https://arxiv.org/pdf/2211.17192.pdf) · 13 页 · 0.55 MB · [官方来源页](https://arxiv.org/abs/2211.17192)

---

## L30 · 检索增强语言模型

**日期**：2025-04-28  
**英文主题**：Retrieval-augmented Language Models

**本讲抓什么**：从文档切分、索引、向量召回、重排到答案生成，建立 RAG 与知识检索系统的完整数据流。

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

### 论文与延伸阅读

- **论文阅读** · [RAG（官方 PDF）](https://arxiv.org/pdf/2005.11401.pdf) · 19 页 · 0.84 MB · [官方来源页](https://arxiv.org/abs/2005.11401)

---

## L31 · Embedding 近邻向量检索

**日期**：2025-04-28  
**英文主题**：Nearest Vector Search for Embeddings

**本讲抓什么**：理解文本向量的训练目标、相似度与索引方式，并连接搜索、聚类、推荐与 RAG。

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

### 论文与延伸阅读

- **论文阅读** · [HNSW（官方 PDF）](https://arxiv.org/pdf/1603.09320.pdf) · 13 页 · 2.51 MB · [官方来源页](https://arxiv.org/abs/1603.09320)

---

## L32 · 多模态大语言模型

**日期**：2025-04-28  
**英文主题**：Multimodal LLMs

**本讲抓什么**：理解文本、图像等模态如何被编码、对齐与统一生成，以及视觉 Token 和跨模态训练的关键设计。

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

### 论文与延伸阅读

- **论文阅读** · [Flamingo（官方 PDF）](https://arxiv.org/pdf/2204.14198.pdf) · 54 页 · 30.12 MB · [官方来源页](https://arxiv.org/abs/2204.14198)

---

## L33 · DeepSeek V3 与 R1

**日期**：2025-04-28  
**英文主题**：Deepseek V3 and R1

**本讲抓什么**：围绕本讲主题建立概念、方法、系统实现与评价标准之间的联系，并结合课件和论文理解关键设计取舍。

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

---

## L34 · Attention Sink 与流式语言模型

**日期**：2025-04-28  
**英文主题**：Efficient Streaming Language Models with Attention Sinks

**本讲抓什么**：围绕本讲主题建立概念、方法、系统实现与评价标准之间的联系，并结合课件和论文理解关键设计取舍。

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

### 论文与延伸阅读

- **论文阅读** · [Attention Sink（官方 PDF）](https://arxiv.org/pdf/2309.17453.pdf) · 21 页 · 16.17 MB · [官方来源页](https://arxiv.org/abs/2309.17453)

---

## L35 · 高级大模型服务

**日期**：2025-04-28  
**英文主题**：Advanced Large Model Serving

**本讲抓什么**：从请求调度、连续批处理、KV Cache、并行和可观测性建立大模型在线服务的系统视角。

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

### 论文与延伸阅读

- **论文阅读 · 仅在线** · [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) — 课程页没有公开直链 PDF

---

## 版权与更新说明

本站不随 GitHub Pages 重新分发课程 PDF；条目优先链接课程官网、出版方或 arXiv。版权归原作者、课程团队及出版方所有；引用时请使用 PDF 内的正式作者与出版信息。机器清单保留官方 URL、页数、大小与 SHA-256，方便后续复核。
