---
title: CMU LLM Applications 逐讲资料
description: Large Language Model Applications 的逐讲 Slides、讲义与论文阅读官方索引
---

# CMU LLM Applications 逐讲资料

> **课程**：CMU Large Language Model Applications · Spring 2026  
> **学校**：Carnegie Mellon University  
> **官方主页**：[https://cmu-llms.org/schedule/](https://cmu-llms.org/schedule/)  
> **抓取与校验日期**：2026-08-03

::: tip 这是来源与深挖页，不是主学习顺序
本页共索引 **28 份官方 Slides / PDF**，合计 **1,678 页 / 290.0 MB**。PDF 统一链接到课程官网、论文官网或 arXiv。
:::

初学请先沿[大模型系统课](/beginner/)学习；需要核对某一知识点来自哪些课程时，使用[名校课程知识覆盖表](/curriculum/sources)。

围绕检索、Agent、教育、医疗、法律、代码与产品等应用主题组织，强调从模型能力到真实场景。
课程表共整理 **25 份讲义条目**与 **5 项论文 / 延伸阅读**；其中 **3 份阅读有公开 PDF**，另有 **2 项**只有网页、博客、视频或受限入口，因此保留官方在线链接。

## 建议怎么学

1. 先读每一讲的“本讲抓什么”，确认自己要回答的问题。
2. 第一遍快速翻 Slides，只看标题、图和结论；不在公式处停太久。
3. 第二遍结合 Notes 或 Recitation，把不懂的概念补齐。
4. 最后从论文阅读里挑 1–2 篇精读，并回到本站对应的零基础章节复习。

[查看课程资料机器可读清单（JSON）](/course-materials/manifest.json)

---

## L01 · 大语言模型的起源

**日期**：2026-01-13  
**英文主题**：Origins of LLMs

**本讲抓什么**：从统计语言模型、词向量和神经语言模型回顾大模型的技术脉络，建立后续应用课程的历史坐标。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-01-13-llm-history.pdf) · 52 页 · 16.52 MB

### 论文与延伸阅读

- **论文阅读** · [Bengio et al. 2003. “A Neural Probabilistic Language Model”（官方 PDF）](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) · 19 页 · 0.13 MB
- **论文阅读** · [Jurafsky and Martin textbook chapter on noisy channel models（官方 PDF）](https://web.stanford.edu/~jurafsky/slp3/D.pdf) · 14 页 · 0.22 MB

---

## L02 · 自然语言理解与生成

**日期**：2026-01-15  
**英文主题**：Natural language understanding vs generation

**本讲抓什么**：比较理解型任务与生成型任务的目标、训练信号和评测方法，辨清同一模型在两类任务中的角色。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-01-15-nlu-nlg.pdf) · 51 页 · 4.9 MB

---

## L03 · Prompting 的科学方法

**日期**：2026-01-20  
**英文主题**：The science of prompting

**本讲抓什么**：把 Prompt 视作实验变量，学习模板、示例、顺序和措辞如何影响模型，并用评测而不是感觉选择方案。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-01-20-prompting.pdf) · 73 页 · 16.99 MB

### 论文与延伸阅读

- **论文阅读** · [Quantifying Language Models’ Sensitivity to Spurious Features in Prompt Design（官方 PDF）](https://arxiv.org/pdf/2310.11324.pdf) · 29 页 · 2.55 MB · [官方来源页](https://arxiv.org/abs/2310.11324)
- **论文阅读 · 仅在线** · [Prompt Engineering Guide](https://www.promptingguide.ai/) — 课程页没有公开直链 PDF

---

## L04 · 何时微调，以及怎样高效微调

**日期**：2026-01-22  
**英文主题**：Deciding when to finetune and finetuning efficiently

**本讲抓什么**：围绕本讲主题建立概念、方法、系统实现与评价标准之间的联系，并结合课件和论文理解关键设计取舍。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-01-22-finetuning.pdf) · 83 页 · 17.26 MB

### 论文与延伸阅读

- **论文阅读 · 仅在线** · [To fine-tune or not to fine-tune](https://ai.meta.com/blog/when-to-fine-tune-llms-vs-other-techniques/) — 课程页没有公开直链 PDF

---

## L05 · 表示学习与 Embedding

**日期**：2026-01-27  
**英文主题**：Learning representations and embeddings

**本讲抓什么**：理解文本向量的训练目标、相似度与索引方式，并连接搜索、聚类、推荐与 RAG。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-01-27-embeddings.pdf) · 53 页 · 13.09 MB

---

## L06 · 检索一：知识的存储与召回

**日期**：2026-01-29  
**英文主题**：Retrieval 1: Storing and retrieving knowledge

**本讲抓什么**：从文档切分、索引、向量召回、重排到答案生成，建立 RAG 与知识检索系统的完整数据流。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-01-29-retrieval.pdf) · 52 页 · 4.63 MB

---

## L07 · 检索二：RAG 与 Deep Research

**日期**：2026-02-03  
**英文主题**：Retrieval 2: Retrieval augmented generation, deep research

**本讲抓什么**：从文档切分、索引、向量召回、重排到答案生成，建立 RAG 与知识检索系统的完整数据流。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-02-03-retrieval+rag.pdf) · 53 页 · 9.22 MB

---

## L08 · 检索三：进阶 RAG 与 Deep Research

**日期**：2026-02-05  
**英文主题**：Retrieval 3: Retrieval augmented generation (2) and deep research

**本讲抓什么**：从文档切分、索引、向量召回、重排到答案生成，建立 RAG 与知识检索系统的完整数据流。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-02-05-rag+deepresearch.pdf) · 56 页 · 4.3 MB

---

## L09 · Deep Research 系统

**日期**：2026-02-10  
**英文主题**：Deep research

**本讲抓什么**：把多轮检索、查询改写、证据整合、来源核对和长答案生成组织成可追踪的研究工作流。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-02-10-deep-research.pdf) · 74 页 · 9.69 MB

---

## L10 · 任务型对话

**日期**：2026-02-12  
**英文主题**：Task-Oriented Dialogue

**本讲抓什么**：学习意图、槽位、状态跟踪、策略与自然语言生成如何协作完成有明确目标的多轮对话。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-02-12-task.pdf) · 52 页 · 12.76 MB

---

## L11 · 工具使用、闲聊、角色与陪伴

**日期**：2026-02-17  
**英文主题**：Tool-use, chitchat, personas, and companionship

**本讲抓什么**：比较工具调用、闲聊、Persona 与陪伴型助手的目标和边界，关注一致性、安全与长期交互。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-02-17-chatbots.pdf) · 62 页 · 10.11 MB

---

## L12 · 写作、创意助手与 AI 创作

**日期**：2026-02-19  
**英文主题**：Writing and ideation assistants and AI creative writing

**本讲抓什么**：分析大模型怎样支持构思、改写与创作，同时处理原创性、作者控制和评价标准。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-02-19-creativity-v2.pdf) · 46 页 · 10.3 MB

---

## L13 · 用大模型做评测：合成数据、模拟与 AI Judge

**日期**：2026-02-24  
**英文主题**：LLMs for evaluation: Synthetic data generation, simulation, automatic evaluation, AI-as-judge

**本讲抓什么**：从数据集、指标、人工评价到 LLM-as-a-Judge，识别数据污染、偏差、方差和评测失真的风险。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-02-24-evaluation.pdf) · 49 页 · 14.03 MB

---

## L14 · 多 Agent 系统

**日期**：2026-02-26  
**英文主题**：Multi-agent systems

**本讲抓什么**：把模型放进规划、行动、工具调用、观察与修正循环，分析记忆、环境接口和长程任务可靠性。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-02-26-multi-agent-systems.pdf) · 62 页 · 17.71 MB

---

## L15 · 大模型应用造成的风险

**日期**：2026-03-10  
**英文主题**：Harms caused by LLM applications

**本讲抓什么**：系统识别偏见、隐私、误导、依赖与社会影响，并把风险评估加入产品设计和部署流程。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-03-10-risks.pdf) · 57 页 · 20.22 MB

---

## L16 · 攻击大模型及其应用

**日期**：2026-03-12  
**英文主题**：Attacking LLMs and LLM applications

**本讲抓什么**：了解 Prompt Injection、越狱、数据与工具攻击的威胁模型，并学习分层防护和红队评测。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-03-12-misuse.pdf) · 54 页 · 20.87 MB

---

## L17 · 代码助手

**日期**：2026-03-17  
**英文主题**：Code-writing assistants (guest lecture from Zora Wang)

**本讲抓什么**：分析大模型怎样支持构思、改写与创作，同时处理原创性、作者控制和评价标准。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-03-17-coding-agents.pdf) · 56 页 · 7.37 MB

---

## L18 · 图像生成与视觉对话

**日期**：2026-03-19  
**英文主题**：[tentative] Image generation and conversing about images

**本讲抓什么**：学习图像 Token、离散表征与生成模型如何让语言模型处理和生成视觉内容。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-03-19-vision.pdf) · 56 页 · 33.54 MB

---

## L19 · 非英语语言与非美国文化中的大模型

**日期**：2026-03-24  
**英文主题**：LLMs for Non-English Languages and non-American Cultures (guest lecture from Shaily Bhatt)

**本讲抓什么**：观察 Tokenizer、数据覆盖和文化假设如何造成语言差异，并讨论本地化评测与文化适配。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-03-24-non-english.pdf) · 218 页 · 5.66 MB

---

## L20 · 世界模型

**日期**：2026-03-26  
**英文主题**：World models (guest lecture from Mingkai Deng)

**本讲抓什么**：理解模型怎样学习环境状态与动态，并用预测、规划和模拟支持智能体决策。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-03-26-world-models.pdf) · 67 页 · 5.13 MB

---

## L21 · 大模型与生物学理解

**日期**：2026-03-31  
**英文主题**：LLMs for biological understanding (guest lecture by Prof Lei Li)

**本讲抓什么**：探索序列与结构模型如何用于蛋白质、分子和生物知识任务，并辨析语言类比的适用边界。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-03-31-drug-design.pdf) · 70 页 · 4.94 MB

---

## L22 · 音乐生成

**日期**：2026-04-02  
**英文主题**：Music generation (guest lecture by Prof Chris Donahue)

**本讲抓什么**：把音乐表示为可建模的序列或连续信号，理解结构、控制条件、评价与版权问题。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-04-02-music.pdf) · 76 页 · 7.71 MB

---

## L23 · 大模型与数字

**日期**：2026-04-07  
**英文主题**：Numbers

**本讲抓什么**：分析语言模型在算术、数值表示和数量推理上的能力与缺陷，以及工具增强方法。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-03-7-detection-and-numbers.pdf) · 58 页 · 8.66 MB

---

## L24 · 机器人与具身智能

**日期**：2026-04-14  
**英文主题**：Robots and embodied AI (guest lecture from Leena Mathur)

**本讲抓什么**：把语言模型与视觉、动作和物理反馈结合，理解具身智能中的感知、规划、控制与安全。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-04-14-robotics.pdf) · 50 页 · 4.22 MB

---

## L25 · 大模型应用部署

**日期**：2026-04-16  
**英文主题**：Deployment

**本讲抓什么**：围绕本讲主题建立概念、方法、系统实现与评价标准之间的联系，并结合课件和论文理解关键设计取舍。

### Slides 与讲义

- **Slides** · [Slides（官方 PDF）](https://storage.googleapis.com/cmu-llms/2026/2026-03-16-deployment.pdf) · 36 页 · 7.26 MB

---

## L26 · 课程项目展示

**日期**：2026-04-21  
**英文主题**：Project presentations

**本讲抓什么**：围绕本讲主题建立概念、方法、系统实现与评价标准之间的联系，并结合课件和论文理解关键设计取舍。

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

---

## L27 · 课程项目展示

**日期**：2026-04-23  
**英文主题**：Project presentations

**本讲抓什么**：围绕本讲主题建立概念、方法、系统实现与评价标准之间的联系，并结合课件和论文理解关键设计取舍。

### Slides 与讲义

- 官网课程表目前没有公开 Slides / Notes 文件。

---

## 版权与更新说明

本站不随 GitHub Pages 重新分发课程 PDF；条目优先链接课程官网、出版方或 arXiv。版权归原作者、课程团队及出版方所有；引用时请使用 PDF 内的正式作者与出版信息。机器清单保留官方 URL、页数、大小与 SHA-256，方便后续复核。
