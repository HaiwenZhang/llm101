---
title: 台湾大学 ADL 逐讲资料
description: Applied Deep Learning 2025 的逐讲 Slides、讲义与论文阅读官方索引
---

# 台湾大学 ADL 逐讲资料

> **课程**：NTU Applied Deep Learning · Fall 2025  
> **学校**：National Taiwan University  
> **官方主页**：[https://www.csie.ntu.edu.tw/~miulab/f114-adl/](https://www.csie.ntu.edu.tw/~miulab/f114-adl/)  
> **抓取与校验日期**：2026-08-03

::: tip 这是来源与深挖页，不是主学习顺序
本页共索引 **24 份官方 Slides / PDF**，合计 **1,021 页 / 67.4 MB**。PDF 统一链接到课程官网、论文官网或 arXiv。
:::

初学请先沿[大模型系统课](/beginner/)学习；需要核对某一知识点来自哪些课程时，使用[名校课程知识覆盖表](/curriculum/sources)。

台大页面里的主讲 Slides 与 Recitation Slides 均按上课日期归档。课程主页上 5 个已失效的 Recitation / 旧版补充链接，改用台大同一课程往年官方存档，并在具体条目旁注明来源年份。

## 建议怎么学

1. 先读每一讲的“本讲抓什么”，确认自己要回答的问题。
2. 第一遍快速翻 Slides，只看标题、图和结论；不在公式处停太久。
3. 第二遍结合 Notes 或 Recitation，把不懂的概念补齐。
4. 最后从论文阅读里挑 1–2 篇精读，并回到本站对应的零基础章节复习。

[查看课程资料机器可读清单（JSON）](/course-materials/manifest.json)

---

## P00 · 自学先修：从机器学习走到反向传播

**日期**：自学 / 先修  
**英文主题**：Prerequisites

**本讲抓什么**：用三份课件补齐课程入口：先认识深度学习与任务类型，再学习神经网络的层、激活函数与损失，最后用链式法则理解反向传播。

### Slides 与讲义

- **Slides** · [Introduction（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250901_Introduction.pdf) · 46 页 · 3.05 MB
- **Slides** · [Neural Network Basics（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250901_NNBasics.pdf) · 93 页 · 2.71 MB
- **Slides** · [Backpropagation（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250901_Backprop.pdf) · 33 页 · 1.35 MB

---

## W01 · 第 1 周：课程说明与序列建模

**日期**：2025-09-01  
**英文主题**：Course Logistics + Sequence Modeling

**本讲抓什么**：先了解课程任务与学习方式，再从序列输入输出形式进入 RNN、LSTM 等模型，理解文本为何不能简单当作互不相关的独立词。

### Slides 与讲义

- **Slides** · [Course Logistics（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250901_Course.pdf) · 18 页 · 1.17 MB
- **Slides** · [Sequence Modeling（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250901_SeqModel.pdf) · 66 页 · 2.45 MB

---

## W02 · 第 2 周：Attention 到 BERT

**日期**：2025-09-08  
**英文主题**：Attention, Transformer, Tokenization, BERT

**本讲抓什么**：一周串起现代 NLP 的主干：注意力负责选择信息，Transformer 负责堆叠表示，Tokenizer 负责把文字变成 Token，BERT 用遮盖预训练学习双向上下文。

### Slides 与讲义

- **Slides** · [Attention Mechanism（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250908_Attention.pdf) · 28 页 · 1.92 MB
- **Slides** · [Transformer（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250908_Transformer.pdf) · 58 页 · 2.37 MB
- **Slides** · [Tokenization（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250908_Tokenization.pdf) · 22 页 · 0.49 MB
- **Slides** · [BERT（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250908_BERT.pdf) · 22 页 · 2.1 MB
- **Recitation Slides** · [NLP Lifecycle（Recitation）（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/w2-ProjLife.pdf) · 46 页 · 1.18 MB；2025 课程页原链接失效；改用台大同课程 2024 官方存档。

---

## SUP · 弹性补充：词向量与 BERT 变体

**日期**：弹性补充  
**英文主题**：Word Embeddings + BERT Variants

**本讲抓什么**：回补 Word2Vec、GloVe 与词向量评价，再比较 XLNet、RoBERTa、mBERT 等预训练设计，适合在 BERT 主课之后查漏补缺。

### Slides 与讲义

- **Slides** · [Word Embeddings（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f111-adl/doc/220929_WordEmbeddings.pdf) · 48 页 · 1.87 MB；课程页链接到 2022 版补充课件。
- **Slides** · [BERT Variants（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/240918_BERTVariants.pdf) · 32 页 · 2.38 MB；2025 课程页原链接失效；改用台大同课程 2024 官方存档。

---

## W03 · 第 3 周：预训练与 Prompt Learning

**日期**：2025-09-15  
**英文主题**：Pretraining & Prompt Learning

**本讲抓什么**：理解预训练—下游适配的基本范式，以及离散 Prompt、连续 Prompt 和上下文学习如何改变模型接收任务的方式。

### Slides 与讲义

- **Slides** · [Pretraining & Prompt Learning（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250915_Pretraining.pdf) · 67 页 · 5.15 MB
- **Recitation Slides** · [Underlying Logics of Projects（Recitation）（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f112-adl/doc/w3-UnderlyLogic.pdf) · 43 页 · 2.7 MB；2025 课程页原链接失效；改用台大同课程 2023 官方存档。

---

## W04 · 第 4 周：后训练与大模型适配

**日期**：2025-09-22  
**英文主题**：Post-Training + LLM Adaptation

**本讲抓什么**：从 SFT、偏好学习等后训练方法走向参数高效微调；结合 LoRA 实作理解“冻结大模型，只训练少量低秩参数”的工程价值。

### Slides 与讲义

- **Slides** · [Post-Training（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250922_PostTraining.pdf) · 55 页 · 4.02 MB
- **Slides** · [LLM Adaptation（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250922_Adaptation.pdf) · 14 页 · 0.94 MB
- **Recitation Slides** · [LLM LoRA Training（Recitation）（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/w5-LoRA.pdf) · 30 页 · 7.04 MB；2025 课程页原链接失效；改用台大同课程 2024 官方存档。

---

## W05 · 第 5 周：RAG 与 MoE

**日期**：2025-10-13  
**英文主题**：Retrieval-Augmented Generation

**本讲抓什么**：主讲课覆盖文档切分、向量检索、重排、上下文组织与生成；Recitation 补充大模型架构和 MoE 路由，建立系统级视角。

### Slides 与讲义

- **Slides** · [Retrieval-Augmented Generation（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/251013_RAG.pdf) · 60 页 · 3.3 MB
- **Recitation Slides** · [LLM Basics & MoE（Recitation）（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/w4-LLMBasicsMOE.pdf) · 33 页 · 5.48 MB；2025 课程页原链接失效；改用台大同课程 2024 官方存档。

---

## W06 · 第 6 周：生成解码、推理与评价

**日期**：2025-10-27  
**英文主题**：NLG Decoding + Evaluation

**本讲抓什么**：比较贪心、Beam Search、Top-k、Top-p 等生成策略，再学习自动指标、人工评价与 LLM 评价；Recitation 把推理效率与评测流程连接起来。

### Slides 与讲义

- **Slides** · [NLG Decoding（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/251027_NLG.pdf) · 41 页 · 1.35 MB
- **Slides** · [NLG Evaluation（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/251027_NLGEval.pdf) · 22 页 · 1.77 MB
- **Recitation Slides** · [LLM Inference & Evaluation（Recitation）（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/w6-LLMInferenceEval.pdf) · 37 页 · 5.02 MB；2025 课程页原链接失效；改用台大同课程 2024 官方存档。

---

## W07 · 第 7 周：预训练模型的问题与发展

**日期**：2025-11-03  
**英文主题**：Issues and Development in Pre-Trained Models

**本讲抓什么**：从预训练模型的偏见、遗忘、数据与知识局限出发，讨论模型为什么会失败，以及研究路线如何针对这些问题演进。

### Slides 与讲义

- **Slides** · [Issues and Development in Pre-Trained Models（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/251103_Issues.pdf) · 42 页 · 2.56 MB

---

## W08 · 第 8 周：语言 Agent

**日期**：2025-11-10  
**英文主题**：Language Agents

**本讲抓什么**：把语言模型放进“规划—行动—观察—修正”的循环，学习工具调用、记忆、环境反馈与多步任务分解，并关注 Agent 的评价和可靠性。

### Slides 与讲义

- **Slides** · [Language Agents（官方 PDF）](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/251110_LangAgent.pdf) · 65 页 · 5.07 MB

---

## W09 · 第 9 周：知识与多模态

**日期**：2025-11-17  
**英文主题**：Knowledge, Multimodality

**本讲抓什么**：官网课程表列出本讲主题，但当前没有可直接下载的 Slides。

::: warning 截至本次抓取，官网未提供 Slides。
:::

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

---

## W10 · 第 10 周：个性化

**日期**：2025-11-24  
**英文主题**：Personalization

**本讲抓什么**：官网课程表列出个性化主题，但当前没有可直接下载的 Slides。

::: warning 截至本次抓取，官网未提供 Slides。
:::

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

---

## W11 · 第 11 周：推理

**日期**：2025-12-01  
**英文主题**：Reasoning

**本讲抓什么**：官网课程表列出推理主题，但当前没有可直接下载的 Slides。

::: warning 截至本次抓取，官网未提供 Slides。
:::

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

---

## 版权与更新说明

本站不随 GitHub Pages 重新分发课程 PDF；条目优先链接课程官网、出版方或 arXiv。版权归原作者、课程团队及出版方所有；引用时请使用 PDF 内的正式作者与出版信息。机器清单保留官方 URL、页数、大小与 SHA-256，方便后续复核。
