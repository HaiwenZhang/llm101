---
title: CMU Advanced NLP 逐讲资料
description: Advanced Natural Language Processing 的逐讲 Slides、讲义与论文阅读官方索引
---

# CMU Advanced NLP 逐讲资料

> **课程**：CMU Advanced NLP · Spring 2026  
> **学校**：Carnegie Mellon University  
> **官方主页**：[https://cmu-l3.github.io/anlp-spring2026/](https://cmu-l3.github.io/anlp-spring2026/)  
> **抓取与校验日期**：2026-08-03

::: tip 这是来源与深挖页，不是主学习顺序
本页共索引 **119 份官方 Slides / PDF**，合计 **4,793 页 / 702.8 MB**。PDF 统一链接到课程官网、论文官网或 arXiv。
:::

初学请先沿[大模型系统课](/beginner/)学习；需要核对某一知识点来自哪些课程时，使用[名校课程知识覆盖表](/curriculum/sources)。

研究导向的高级 NLP 课程，按讲次同时整理 Slides、核心论文、补充阅读与公开代码。
课程表共整理 **23 份讲义条目**与 **106 项论文 / 延伸阅读**；其中 **96 份阅读有公开 PDF**，另有 **10 项**只有网页、博客、视频或受限入口，因此保留官方在线链接。

## 建议怎么学

1. 先读每一讲的“本讲抓什么”，确认自己要回答的问题。
2. 第一遍快速翻 Slides，只看标题、图和结论；不在公式处停太久。
3. 第二遍结合 Notes 或 Recitation，把不懂的概念补齐。
4. 最后从论文阅读里挑 1–2 篇精读，并回到本站对应的零基础章节复习。

[查看课程资料机器可读清单（JSON）](/course-materials/manifest.json)

---

## L01 · 导论与高级 NLP 基础

**日期**：2026-01-13  
**英文主题**：Introduction & Fundamentals

**本讲抓什么**：建立课程总地图，认识模型、数据、训练、推理和系统工程之间的依赖关系。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-01-intro.pdf) · 62 页 · 3.24 MB

### 论文与延伸阅读

- **论文阅读** · [Natural Language Understanding with Distributed Representation (Ch. 1)（官方 PDF）](https://arxiv.org/pdf/1511.07916.pdf) · 124 页 · 5.25 MB · [官方来源页](https://arxiv.org/abs/1511.07916)
- **论文阅读** · [Machine Learning: a Lecture Note (Ch. 1)（官方 PDF）](https://arxiv.org/pdf/2505.03861.pdf) · 111 页 · 0.92 MB · [官方来源页](https://arxiv.org/abs/2505.03861)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/01_intro)

---

## L02 · 学习表示：文字如何进入模型

**日期**：2026-01-15  
**英文主题**：Fundamentals: Learned Representations

**本讲抓什么**：理解离散符号如何变成连续向量，以及表示学习如何把相似性、类别与上下文信息编码进模型可计算的空间。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-02-representations.pdf) · 68 页 · 4.11 MB

### 论文与延伸阅读

- **论文阅读** · [Natural Language Understanding with Distributed Representation (Ch. 2, Ch. 3)（官方 PDF）](https://arxiv.org/pdf/1511.07916.pdf) · 124 页 · 5.25 MB · [官方来源页](https://arxiv.org/abs/1511.07916)
- **论文阅读 · 仅在线** · [(Video) Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — 课程页没有公开直链 PDF

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/02_wordrep_classification)

---

## L03 · 自回归语言建模

**日期**：2026-01-20  
**英文主题**：Fundamentals: Autoregressive Language Modeling

**本讲抓什么**：从概率链式法则出发理解下一个 Token 预测，连接训练目标、似然、困惑度与逐步生成。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-03-lm.pdf) · 60 页 · 6.55 MB

### 论文与延伸阅读

- **论文阅读** · [Natural Language Understanding with Distributed Representation (Ch. 5 up to 5.4.2)（官方 PDF）](https://arxiv.org/pdf/1511.07916.pdf) · 124 页 · 5.25 MB · [官方来源页](https://arxiv.org/abs/1511.07916)
- **论文阅读** · [A Neural Probabilistic Language Model（官方 PDF）](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) · 19 页 · 0.13 MB
- **论文阅读** · [Understanding the difficulty of training deep feedforward neural networks（官方 PDF）](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) · 8 页 · 1.57 MB

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/03_lm_fundamentals)

---

## L04 · 架构一：循环神经网络

**日期**：2026-01-22  
**英文主题**：Architectures I: Recurrent Neural Networks

**本讲抓什么**：学习 RNN 如何沿时间保存状态，以及门控结构、梯度传播和序列建模的主要限制。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-04-rnns.pdf) · 52 页 · 3.38 MB

### 论文与延伸阅读

- **论文阅读** · [Natural Language Understanding with Distributed Representation (Ch. 4, Ch. 5.5-5.6, Ch. 6)（官方 PDF）](https://arxiv.org/pdf/1511.07916.pdf) · 124 页 · 5.25 MB · [官方来源页](https://arxiv.org/abs/1511.07916)
- **论文阅读** · [Recurrent neural network based language model（官方 PDF）](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.pdf) · 4 页 · 0.21 MB
- **论文阅读** · [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation（官方 PDF）](https://arxiv.org/pdf/1406.1078.pdf) · 15 页 · 1.09 MB · [官方来源页](https://arxiv.org/pdf/1406.1078)
- **论文阅读 · 仅在线** · [Why LSTMs Stop Your Gradients From Vanishing: A View from the Backwards Pass](https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html) — 课程页没有公开直链 PDF
- **论文阅读** · [Neural Machine Translation by Jointly Learning to Align and Translate（官方 PDF）](https://arxiv.org/pdf/1409.0473.pdf) · 15 页 · 0.42 MB · [官方来源页](https://arxiv.org/abs/1409.0473)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/04_recurrent)

---

## L05 · 架构二：Attention 与 Transformer

**日期**：2026-01-27  
**英文主题**：Architectures II: Attention and Transformers

**本讲抓什么**：拆解 Q/K/V、自注意力、多头机制、位置表示与残差归一化，并理解 Transformer 的并行计算路径。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-05-attention-transformers.pdf) · 68 页 · 9.78 MB

### 论文与延伸阅读

- **论文阅读** · [Attention Is All You Need（官方 PDF）](https://arxiv.org/pdf/1706.03762.pdf) · 15 页 · 2.11 MB · [官方来源页](https://arxiv.org/pdf/1706.03762)
- **论文阅读 · 仅在线** · [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — 课程页没有公开直链 PDF
- **论文阅读** · [Root Mean Square Layer Normalization（官方 PDF）](https://arxiv.org/pdf/1910.07467.pdf) · 14 页 · 0.53 MB · [官方来源页](https://arxiv.org/pdf/1910.07467)
- **论文阅读** · [On Layer Normalization in the Transformer Architecture（官方 PDF）](https://arxiv.org/pdf/2002.04745.pdf) · 17 页 · 0.74 MB · [官方来源页](https://arxiv.org/pdf/2002.04745)
- **论文阅读** · [RoFormer: Enhanced Transformer with Rotary Position Embedding（官方 PDF）](https://arxiv.org/pdf/2104.09864.pdf) · 14 页 · 0.57 MB · [官方来源页](https://arxiv.org/abs/2104.09864)
- **论文阅读** · [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints（官方 PDF）](https://arxiv.org/pdf/2305.13245.pdf) · 7 页 · 0.26 MB · [官方来源页](https://arxiv.org/pdf/2305.13245)
- **论文阅读 · 仅在线** · [(Helpful Blog Post): Why Are Sines and Cosines Used For Positional Encoding?](https://mfaizan.github.io/2023/04/02/sines.html) — 课程页没有公开直链 PDF

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/05_transformers)

---

## L06 · 学习一：预训练

**日期**：2026-01-29  
**英文主题**：Learning I: Pretraining

**本讲抓什么**：把预训练放进数据、目标函数、模型规模与计算预算的共同框架中，理解基础模型能力从何而来。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-06-pretraining.pdf) · 57 页 · 10.29 MB

### 论文与延伸阅读

- **论文阅读** · [Language Models are Unsupervised Multitask Learners（官方 PDF）](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · 24 页 · 0.56 MB
- **论文阅读** · [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale（官方 PDF）](https://arxiv.org/pdf/2406.17557.pdf) · 38 页 · 2.99 MB · [官方来源页](https://arxiv.org/abs/2406.17557)
- **论文阅读** · [OLMo 3（官方 PDF）](https://arxiv.org/pdf/2512.13961.pdf) · 118 页 · 6.5 MB · [官方来源页](https://arxiv.org/abs/2512.13961)
- **论文阅读** · [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（官方 PDF）](https://arxiv.org/pdf/1810.04805.pdf) · 16 页 · 0.74 MB · [官方来源页](https://arxiv.org/abs/1810.04805)
- **论文阅读** · [LLaMA: Open and Efficient Foundation Language Models（官方 PDF）](https://arxiv.org/pdf/2302.13971.pdf) · 27 页 · 0.69 MB · [官方来源页](https://arxiv.org/abs/2302.13971)
- **论文阅读** · [OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text（官方 PDF）](https://arxiv.org/pdf/2310.06786.pdf) · 20 页 · 0.71 MB · [官方来源页](https://arxiv.org/abs/2310.06786)
- **论文阅读** · [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research（官方 PDF）](https://arxiv.org/pdf/2402.00159.pdf) · 64 页 · 8.89 MB · [官方来源页](https://arxiv.org/abs/2402.00159)
- **论文阅读** · [Scaling Laws for Neural Language Models（官方 PDF）](https://arxiv.org/pdf/2001.08361.pdf) · 30 页 · 2.38 MB · [官方来源页](https://arxiv.org/abs/2001.08361)
- **论文阅读** · [Training Compute-Optimal Large Language Models（官方 PDF）](https://arxiv.org/pdf/2203.15556.pdf) · 36 页 · 5.73 MB · [官方来源页](https://arxiv.org/abs/2203.15556)
- **论文阅读** · [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism（官方 PDF）](https://arxiv.org/pdf/2401.02954.pdf) · 48 页 · 7.13 MB · [官方来源页](https://arxiv.org/abs/2401.02954)
- **论文阅读** · [Language Modeling Is Compression（官方 PDF）](https://arxiv.org/pdf/2309.10668.pdf) · 17 页 · 2.16 MB · [官方来源页](https://arxiv.org/abs/2309.10668)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/06_pretraining)

---

## L07 · Scaling Laws 与上下文学习

**日期**：2026-02-03  
**英文主题**：Scaling Laws and In-Context Learning

**本讲抓什么**：用经验幂律连接模型大小、数据量、计算量与损失，并理解上下文学习如何在不更新参数时适配任务。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-07-pretraining2-prompting.pdf) · 51 页 · 8.87 MB

### 论文与延伸阅读

- **论文阅读** · [Language Models are Few-Shot Learners（官方 PDF）](https://arxiv.org/pdf/2005.14165.pdf) · 75 页 · 6.45 MB · [官方来源页](https://arxiv.org/abs/2005.14165)
- **论文阅读** · [Deep Learning Scaling is Predictable, Empirically（官方 PDF）](https://arxiv.org/pdf/1712.00409.pdf) · 19 页 · 0.58 MB · [官方来源页](https://arxiv.org/abs/1712.00409)
- **论文阅读** · [Scaling Laws for Neural Language Models（官方 PDF）](https://arxiv.org/pdf/2001.08361.pdf) · 30 页 · 2.38 MB · [官方来源页](https://arxiv.org/abs/2001.08361)
- **论文阅读** · [Training Compute-Optimal Large Language Models（官方 PDF）](https://arxiv.org/pdf/2203.15556.pdf) · 36 页 · 5.73 MB · [官方来源页](https://arxiv.org/abs/2203.15556)
- **论文阅读** · [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism（官方 PDF）](https://arxiv.org/pdf/2401.02954.pdf) · 48 页 · 7.13 MB · [官方来源页](https://arxiv.org/abs/2401.02954)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/07_icl_prompting)

---

## L08 · 微调与知识蒸馏

**日期**：2026-02-05  
**英文主题**：Learning III: Fine-tuning and Distillation

**本讲抓什么**：比较全参数微调、LoRA、量化微调和指令适配，判断不同数据与硬件预算下该更新哪些参数。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-08-finetuning.pdf) · 40 页 · 5.75 MB

### 论文与延伸阅读

- **论文阅读** · [LoRA: Low-Rank Adaptation of Large Language Models（官方 PDF）](https://arxiv.org/pdf/2106.09685.pdf) · 26 页 · 1.53 MB · [官方来源页](https://arxiv.org/abs/2106.09685)
- **论文阅读** · [Sequence-Level Knowledge Distillation（官方 PDF）](https://arxiv.org/pdf/1606.07947.pdf) · 11 页 · 0.53 MB · [官方来源页](https://arxiv.org/abs/1606.07947)
- **论文阅读** · [Universal Language Model Fine-tuning for Text Classification（官方 PDF）](https://arxiv.org/pdf/1801.06146.pdf) · 12 页 · 0.93 MB · [官方来源页](https://arxiv.org/pdf/1801.06146)
- **论文阅读** · [Cross-Task Generalization via Natural Language Crowdsourcing Instructions（官方 PDF）](https://arxiv.org/pdf/2104.08773.pdf) · 18 页 · 1.28 MB · [官方来源页](https://arxiv.org/abs/2104.08773)
- **论文阅读** · [Finetuned Language Models Are Zero-Shot Learners（官方 PDF）](https://arxiv.org/pdf/2109.01652.pdf) · 46 页 · 1.55 MB · [官方来源页](https://arxiv.org/abs/2109.01652)
- **论文阅读** · [Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks（官方 PDF）](https://arxiv.org/pdf/2204.07705.pdf) · 25 页 · 1.66 MB · [官方来源页](https://arxiv.org/abs/2204.07705)
- **论文阅读** · [Self-Instruct: Aligning Language Models with Self-Generated Instructions（官方 PDF）](https://arxiv.org/pdf/2212.10560.pdf) · 23 页 · 4.13 MB · [官方来源页](https://arxiv.org/abs/2212.10560)
- **论文阅读** · [Orca: Progressive Learning from Complex Explanation Traces of GPT-4（官方 PDF）](https://arxiv.org/pdf/2306.02707.pdf) · 51 页 · 1.39 MB · [官方来源页](https://arxiv.org/abs/2306.02707)
- **论文阅读 · 仅在线** · [Symbolic Knowledge Distillation: from General Language Models to Commonsense Models](https://drive.google.com/file/d/1xMohjQcTmQuUd_OiZ3hB1r47WB1WM3Am/view) — 课程页没有公开直链 PDF
- **论文阅读** · [QLoRA: Efficient Finetuning of Quantized LLMs（官方 PDF）](https://arxiv.org/pdf/2305.14314.pdf) · 26 页 · 1.02 MB · [官方来源页](https://arxiv.org/abs/2305.14314)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/08_finetuning)

---

## L09 · 推理：解码算法

**日期**：2026-02-10  
**英文主题**：Inference II: Decoding Algorithms

**本讲抓什么**：比较贪心、Beam Search、采样和推测解码，理解质量、多样性、延迟与吞吐之间的权衡。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-09-decoding.pdf) · 62 页 · 5.66 MB

### 论文与延伸阅读

- **论文阅读** · [From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models (Sections 1-3)（官方 PDF）](https://arxiv.org/pdf/2406.16838.pdf) · 53 页 · 1.25 MB · [官方来源页](https://arxiv.org/abs/2406.16838)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/09_decoding)

---

## L10 · 建模一：检索与 RAG

**日期**：2026-02-12  
**英文主题**：Modeling I: Retrieval and RAG

**本讲抓什么**：从文档切分、索引、向量召回、重排到答案生成，建立 RAG 与知识检索系统的完整数据流。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-10-rag.pdf) · 189 页 · 14.8 MB

### 论文与延伸阅读

- **论文阅读** · [Retrieval-based Language Models and Applications（官方 PDF）](https://aclanthology.org/2023.acl-tutorials.6.pdf) · 6 页 · 0.12 MB · [官方来源页](https://aclanthology.org/2023.acl-tutorials.6/)
- **论文阅读** · [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection（官方 PDF）](https://arxiv.org/pdf/2310.11511.pdf) · 30 页 · 1.34 MB · [官方来源页](https://arxiv.org/abs/2310.11511)
- **论文阅读** · [Task-aware Retrieval with Instructions（官方 PDF）](https://arxiv.org/pdf/2211.09260.pdf) · 25 页 · 1.22 MB · [官方来源页](https://arxiv.org/abs/2211.09260)
- **论文阅读** · [When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories（官方 PDF）](https://arxiv.org/pdf/2212.10511.pdf) · 19 页 · 1.07 MB · [官方来源页](https://arxiv.org/abs/2212.10511)
- **论文阅读** · [Reliable, Adaptable, and Attributable Language Models with Retrieval（官方 PDF）](https://arxiv.org/pdf/2403.03187.pdf) · 21 页 · 2.87 MB · [官方来源页](https://arxiv.org/abs/2403.03187)
- **论文阅读** · [Scaling Retrieval-Based Language Models with a Trillion-Token Datastore（官方 PDF）](https://arxiv.org/pdf/2407.12854.pdf) · 32 页 · 11.58 MB · [官方来源页](https://arxiv.org/abs/2407.12854)
- **论文阅读** · [OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented LMs（官方 PDF）](https://arxiv.org/pdf/2411.14199.pdf) · 53 页 · 3.71 MB · [官方来源页](https://arxiv.org/abs/2411.14199)

---

## L11 · 建模二：多模态基础

**日期**：2026-02-17  
**英文主题**：Modeling II: Multimodal I

**本讲抓什么**：理解文本、图像等模态如何被编码、对齐与统一生成，以及视觉 Token 和跨模态训练的关键设计。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-11-multimodal.pdf) · 47 页 · 38.06 MB

### 论文与延伸阅读

- **论文阅读** · [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（官方 PDF）](https://arxiv.org/pdf/2010.11929.pdf) · 22 页 · 3.57 MB · [官方来源页](https://arxiv.org/abs/2010.11929)
- **论文阅读** · [Learning Transferable Visual Models From Natural Language Supervision（官方 PDF）](https://arxiv.org/pdf/2103.00020.pdf) · 48 页 · 6.5 MB · [官方来源页](https://arxiv.org/abs/2103.00020)
- **论文阅读** · [Visual Instruction Tuning（官方 PDF）](https://arxiv.org/pdf/2304.08485.pdf) · 25 页 · 5.63 MB · [官方来源页](https://arxiv.org/abs/2304.08485)
- **论文阅读** · [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models（官方 PDF）](https://arxiv.org/pdf/2409.17146.pdf) · 30 页 · 6.27 MB · [官方来源页](https://arxiv.org/abs/2409.17146)
- **论文阅读** · [PaliGemma: A versatile 3B VLM for transfer（官方 PDF）](https://arxiv.org/pdf/2407.07726.pdf) · 59 页 · 3.29 MB · [官方来源页](https://arxiv.org/abs/2407.07726)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/11_multimodal_i)

---

## L12 · 建模三：多模态生成

**日期**：2026-02-19  
**英文主题**：Modeling III: Multimodal II

**本讲抓什么**：理解文本、图像等模态如何被编码、对齐与统一生成，以及视觉 Token 和跨模态训练的关键设计。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-12-multimodal-ii.pdf) · 45 页 · 43.97 MB

### 论文与延伸阅读

- **论文阅读** · [Neural Discrete Representation Learning（官方 PDF）](https://arxiv.org/pdf/1711.00937.pdf) · 11 页 · 3.03 MB · [官方来源页](https://arxiv.org/abs/1711.00937)
- **论文阅读** · [Taming Transformers for High-Resolution Image Synthesis（官方 PDF）](https://arxiv.org/pdf/2012.09841.pdf) · 52 页 · 48.17 MB · [官方来源页](https://arxiv.org/abs/2012.09841)
- **论文阅读** · [Chameleon: Mixed-Modal Early-Fusion Foundation Models（官方 PDF）](https://arxiv.org/pdf/2405.09818.pdf) · 27 页 · 24.08 MB · [官方来源页](https://arxiv.org/abs/2405.09818)
- **论文阅读 · 仅在线** · [(Blog Post) Image GPT](https://openai.com/index/image-gpt/) — 课程页没有公开直链 PDF
- **论文阅读** · [Generative Pretraining from Pixels（官方 PDF）](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf) · 12 页 · 1.21 MB
- **论文阅读** · [Pixel Recurrent Neural Networks（官方 PDF）](https://arxiv.org/pdf/1601.06759.pdf) · 11 页 · 2.88 MB · [官方来源页](https://arxiv.org/abs/1601.06759)
- **论文阅读** · [Zero-Shot Text-to-Image Generation（官方 PDF）](https://arxiv.org/pdf/2102.12092.pdf) · 20 页 · 9.66 MB · [官方来源页](https://arxiv.org/abs/2102.12092)
- **论文阅读** · [Auto-Encoding Variational Bayes（官方 PDF）](https://arxiv.org/pdf/1312.6114.pdf) · 14 页 · 3.74 MB · [官方来源页](https://arxiv.org/abs/1312.6114)
- **论文阅读** · [Rethinking Generative Image Pretraining: How Far Are We from Scaling Up Next-Pixel Prediction?（官方 PDF）](https://arxiv.org/pdf/2511.08704.pdf) · 21 页 · 9.82 MB · [官方来源页](https://arxiv.org/abs/2511.08704)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/12_multimodal_ii)

---

## L13 · 评测技术

**日期**：2026-02-24  
**英文主题**：Evaluation Techniques

**本讲抓什么**：从数据集、指标、人工评价到 LLM-as-a-Judge，识别数据污染、偏差、方差和评测失真的风险。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-13-evaluation.pdf) · 33 页 · 5.76 MB

### 论文与延伸阅读

- **论文阅读** · [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations（官方 PDF）](https://arxiv.org/pdf/2411.00640.pdf) · 14 页 · 0.32 MB · [官方来源页](https://arxiv.org/abs/2411.00640)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/13_evaluation)

---

## L14 · 研究技能与实验设计

**日期**：2026-02-26  
**英文主题**：Research Skills and Experimental Design

**本讲抓什么**：学习怎样把研究问题变成可复现的实验：设置基线、控制变量、选择指标、报告不确定性，并避免从偶然结果得出过强结论。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-14-experimentation.pdf) · 60 页 · 3.4 MB

### 论文与延伸阅读

- **论文阅读** · [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations（官方 PDF）](https://arxiv.org/pdf/2411.00640.pdf) · 14 页 · 0.32 MB · [官方来源页](https://arxiv.org/abs/2411.00640)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/14_experimental)

---

## L15 · 建模四：扩散模型与 Flow

**日期**：2026-03-10  
**英文主题**：Modeling IV: Diffusion and Flows

**本讲抓什么**：从逐步加噪与去噪建立扩散模型直觉，并比较 Flow Matching 等连续生成方法。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-15-multimodal-iii.pdf) · 43 页 · 29.41 MB

### 论文与延伸阅读

- **论文阅读** · [Denoising Diffusion Probabilistic Models（官方 PDF）](https://arxiv.org/pdf/2006.11239.pdf) · 25 页 · 9.79 MB · [官方来源页](https://arxiv.org/abs/2006.11239)
- **论文阅读** · [Flow Matching for Generative Modeling（官方 PDF）](https://arxiv.org/pdf/2210.02747.pdf) · 28 页 · 23.98 MB · [官方来源页](https://arxiv.org/abs/2210.02747)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/15_diffusion)

---

## L16 · 强化学习一：基础

**日期**：2026-03-12  
**英文主题**：Reinforcement Learning I: Fundamentals

**本讲抓什么**：理解状态、动作、奖励、策略梯度与 PPO，并连接到偏好对齐、推理训练和 Agent 行为优化。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-16-rl.pdf) · 50 页 · 26.06 MB

### 论文与延伸阅读

- **论文阅读 · 仅在线** · [Deep Reinforcement Learning: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/) — 课程页没有公开直链 PDF
- **论文阅读 · 仅在线** · [Spinning Up in Deep RL (Part 1, Part 3, Vanilla PG, PPO)](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 课程页没有公开直链 PDF
- **论文阅读** · [Proximal Policy Optimization Algorithms（官方 PDF）](https://arxiv.org/pdf/1707.06347.pdf) · 12 页 · 2.79 MB · [官方来源页](https://arxiv.org/abs/1707.06347)
- **论文阅读** · [High-Dimensional Continuous Control Using Generalized Advantage Estimation（官方 PDF）](https://arxiv.org/pdf/1506.02438.pdf) · 14 页 · 1.71 MB · [官方来源页](https://arxiv.org/abs/1506.02438)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/16_rl)

---

## L17 · 强化学习二：大模型应用

**日期**：2026-03-17  
**英文主题**：Reinforcement Learning II: Applications

**本讲抓什么**：理解状态、动作、奖励、策略梯度与 PPO，并连接到偏好对齐、推理训练和 Agent 行为优化。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-17-rl-llms.pdf) · 67 页 · 6.72 MB

### 论文与延伸阅读

- **论文阅读** · [Training language models to follow instructions with human feedback（官方 PDF）](https://arxiv.org/pdf/2203.02155.pdf) · 68 页 · 1.71 MB · [官方来源页](https://arxiv.org/abs/2203.02155)
- **论文阅读** · [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（官方 PDF）](https://arxiv.org/pdf/2501.12948.pdf) · 86 页 · 4.8 MB · [官方来源页](https://arxiv.org/abs/2501.12948)
- **论文阅读** · [AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback（官方 PDF）](https://arxiv.org/pdf/2305.14387.pdf) · 31 页 · 1.32 MB · [官方来源页](https://arxiv.org/abs/2305.14387)
- **论文阅读** · [Deep reinforcement learning from human preferences（官方 PDF）](https://arxiv.org/pdf/1706.03741.pdf) · 17 页 · 3.07 MB · [官方来源页](https://arxiv.org/abs/1706.03741)
- **论文阅读** · [Fine-Tuning Language Models from Human Preferences（官方 PDF）](https://arxiv.org/pdf/1909.08593.pdf) · 26 页 · 0.92 MB · [官方来源页](https://arxiv.org/abs/1909.08593)
- **论文阅读** · [Direct Preference Optimization: Your Language Model is Secretly a Reward Model（官方 PDF）](https://arxiv.org/pdf/2305.18290.pdf) · 27 页 · 1.24 MB · [官方来源页](https://arxiv.org/abs/2305.18290)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/17_rl_llm)

---

## L18 · 基于语言模型的 Agent

**日期**：2026-03-19  
**英文主题**：Language Model-Based Agents

**本讲抓什么**：把模型放进规划、行动、工具调用、观察与修正循环，分析记忆、环境接口和长程任务可靠性。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-18-zora.pdf) · 56 页 · 10.57 MB

### 论文与延伸阅读

- **论文阅读** · [World of Bits: An Open-Domain Platform for Web-Based Agents（官方 PDF）](https://proceedings.mlr.press/v70/shi17a/shi17a.pdf) · 10 页 · 3.23 MB
- **论文阅读** · [WebGPT: Browser-assisted question-answering with human feedback（官方 PDF）](https://arxiv.org/pdf/2112.09332.pdf) · 32 页 · 1.42 MB · [官方来源页](https://arxiv.org/pdf/2112.09332)
- **论文阅读** · [WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents（官方 PDF）](https://arxiv.org/pdf/2207.01206.pdf) · 28 页 · 8.47 MB · [官方来源页](https://arxiv.org/pdf/2207.01206)
- **论文阅读** · [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering（官方 PDF）](https://arxiv.org/pdf/2405.15793.pdf) · 118 页 · 4.73 MB · [官方来源页](https://arxiv.org/abs/2405.15793)
- **论文阅读** · [VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks（官方 PDF）](https://arxiv.org/pdf/2401.13649.pdf) · 25 页 · 4.35 MB · [官方来源页](https://arxiv.org/abs/2401.13649)
- **论文阅读** · [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments（官方 PDF）](https://arxiv.org/pdf/2404.07972.pdf) · 51 页 · 35.16 MB · [官方来源页](https://arxiv.org/abs/2404.07972)
- **论文阅读** · [Programming with Pixels: Computer-Use Meets Software Engineering（官方 PDF）](https://arxiv.org/pdf/2502.18525.pdf) · 26 页 · 5.26 MB · [官方来源页](https://arxiv.org/abs/2502.18525)

---

## L19 · 量化

**日期**：2026-03-26  
**英文主题**：Quantization

**本讲抓什么**：理解权重与激活从高精度映射到低比特的过程，比较误差、显存、速度和硬件支持之间的取舍。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-19-quantization.pdf) · 43 页 · 5.78 MB

### 论文与延伸阅读

- **论文阅读** · [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale（官方 PDF）](https://arxiv.org/pdf/2208.07339.pdf) · 20 页 · 0.69 MB · [官方来源页](https://arxiv.org/abs/2208.07339)
- **论文阅读** · [QLoRA: Efficient Finetuning of Quantized LLMs（官方 PDF）](https://arxiv.org/pdf/2305.14314.pdf) · 26 页 · 1.02 MB · [官方来源页](https://arxiv.org/abs/2305.14314)
- **论文阅读** · [8-bit Optimizers via Block-wise Quantization（官方 PDF）](https://arxiv.org/pdf/2110.02861.pdf) · 20 页 · 0.96 MB · [官方来源页](https://arxiv.org/abs/2110.02861)
- **论文阅读** · [The case for 4-bit precision: k-bit Inference Scaling Laws（官方 PDF）](https://arxiv.org/pdf/2212.09720.pdf) · 24 页 · 0.86 MB · [官方来源页](https://arxiv.org/abs/2212.09720)

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/19_quantization)

---

## L20 · 并行与分布式训练

**日期**：2026-03-31  
**英文主题**：Parallelism and Distributed Training

**本讲抓什么**：拆解数据并行、张量并行、流水线并行与通信开销，学习怎样把大模型训练分布到多张 GPU。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-20-scaling-parallelism.pdf) · 60 页 · 9.38 MB

### 论文与延伸阅读

- **论文阅读 · 仅在线** · [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — 课程页没有公开直链 PDF

### 代码与其他资源

- [Code](https://github.com/cmu-l3/anlp-spring2026-code/tree/main/20_parallelism)

---

## L21 · 混合专家模型 MoE

**日期**：2026-04-02  
**英文主题**：Mixture of Experts

**本讲抓什么**：学习路由器如何只激活少量专家，理解稀疏计算、负载均衡、通信与专家容量问题。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-21-moe.pdf) · 55 页 · 10.96 MB

### 论文与延伸阅读

- **论文阅读** · [A Review of Sparse Expert Models in Deep Learning（官方 PDF）](https://arxiv.org/pdf/2209.01667.pdf) · 23 页 · 3.74 MB · [官方来源页](https://arxiv.org/abs/2209.01667)
- **论文阅读** · [OLMoE: Open Mixture-of-Experts Language Models（官方 PDF）](https://arxiv.org/pdf/2409.02060.pdf) · 63 页 · 6.2 MB · [官方来源页](https://arxiv.org/abs/2409.02060)
- **论文阅读** · [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models（官方 PDF）](https://arxiv.org/pdf/2401.06066.pdf) · 33 页 · 0.7 MB · [官方来源页](https://arxiv.org/abs/2401.06066)

---

## L22 · 扩展序列长度

**日期**：2026-04-07  
**英文主题**：Scaling Sequence Length

**本讲抓什么**：分析长上下文的显存与计算瓶颈，并比较 FlashAttention、分块、环形注意力和状态空间模型。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-22-longcontext.pdf) · 54 页 · 13.88 MB

### 论文与延伸阅读

- **论文阅读** · [Self-attention Does Not Need O(n2) Memory（官方 PDF）](https://arxiv.org/pdf/2112.05682.pdf) · 8 页 · 0.15 MB · [官方来源页](https://arxiv.org/abs/2112.05682)
- **论文阅读** · [Mamba: Linear-Time Sequence Modeling with Selective State Spaces（官方 PDF）](https://arxiv.org/pdf/2312.00752.pdf) · 36 页 · 1.11 MB · [官方来源页](https://arxiv.org/abs/2312.00752)
- **论文阅读** · [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（官方 PDF）](https://arxiv.org/pdf/2205.14135.pdf) · 34 页 · 2.51 MB · [官方来源页](https://arxiv.org/abs/2205.14135)
- **论文阅读** · [Ring Attention with Blockwise Transformers for Near-Infinite Context（官方 PDF）](https://arxiv.org/pdf/2310.01889.pdf) · 16 页 · 1.68 MB · [官方来源页](https://arxiv.org/abs/2310.01889)
- **论文阅读** · [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling（官方 PDF）](https://arxiv.org/pdf/2603.05451) · 20 页 · 7.28 MB · [官方来源页](https://arxiv.org/abs/2603.05451)

---

## L23 · 测试时扩展

**日期**：2026-04-14  
**英文主题**：Test-Time Scaling

**本讲抓什么**：研究怎样在推理阶段投入更多搜索、采样、验证和反思计算，以换取更高答案质量。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-23-inference-scaling.pdf) · 74 页 · 20.84 MB

### 论文与延伸阅读

- **论文阅读** · [From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models (Sections 4-7)（官方 PDF）](https://arxiv.org/pdf/2406.16838.pdf) · 53 页 · 1.25 MB · [官方来源页](https://arxiv.org/abs/2406.16838)
- **论文阅读 · 仅在线** · [NeurIPS 2024 LLM Inference Tutorial (Reading List)](https://cmu-l3.github.io/neurips2024-inference-tutorial/) — 课程页没有公开直链 PDF
- **论文阅读** · [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（官方 PDF）](https://arxiv.org/pdf/2501.12948.pdf) · 86 页 · 4.8 MB · [官方来源页](https://arxiv.org/abs/2501.12948)
- **论文阅读** · [s1: Simple test-time scaling（官方 PDF）](https://arxiv.org/pdf/2501.19393.pdf) · 46 页 · 1.41 MB · [官方来源页](https://arxiv.org/abs/2501.19393)
- **论文阅读** · [L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning（官方 PDF）](https://arxiv.org/pdf/2503.04697.pdf) · 27 页 · 1.27 MB · [官方来源页](https://arxiv.org/abs/2503.04697)

---

## 版权与更新说明

本站不随 GitHub Pages 重新分发课程 PDF；条目优先链接课程官网、出版方或 arXiv。版权归原作者、课程团队及出版方所有；引用时请使用 PDF 内的正式作者与出版信息。机器清单保留官方 URL、页数、大小与 SHA-256，方便后续复核。
