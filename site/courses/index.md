---
title: 名校课程原始资料与溯源
description: Stanford、CMU、Berkeley 与台湾大学七门课程的官方 PDF、可执行讲义、出处和逐讲索引
---

# 名校课程原始资料与溯源

::: warning 这里不是主教程入口
这些文件用于核对原始讲义、论文和课程出处。请先学习[大模型系统课](/beginner/)；想了解资料怎样被拆解并合入主线，请看[逐讲知识覆盖表](/curriculum/sources)。
:::

资料按“**课程 → 讲次 → Slides / Notes / 论文阅读**”索引。当前共收录 **383 份资料**（包括官方 PDF 与本站可执行 Slides），其中 PDF 合计 **14,508 页、约 1721.4 MB**。

## 七门来源课程

### [Stanford CS224N · Winter 2026](/courses/cs224n-2026)

系统的深度学习 NLP 主线：从词向量、RNN、Transformer，一直走到后训练、RAG、Agent、评测与推理。 本页含 **90 份官方 PDF / 3,045 页**；论文阅读中有 **63 份公开 PDF**、**17 项仅在线链接。

适合作为整个学习路线的主干课程。

### [NTU Applied Deep Learning · Fall 2025](/courses/ntu-adl-2025)

课件短、图多，并配有 Recitation；覆盖 Transformer、Prompt、LoRA、RAG、MoE、生成评价与 Agent。 本页含 **24 份官方 PDF / 1,021 页**；论文阅读中有 **0 份公开 PDF**、**0 项仅在线链接。

适合快速建立概念结构，并把知识接到项目实践。

### [CMU Advanced NLP · Spring 2026](/courses/cmu-anlp-2026)

研究导向的高级 NLP 课程，按讲次同时整理 Slides、核心论文、补充阅读与公开代码。 本页含 **119 份官方 PDF / 4,793 页**；论文阅读中有 **96 份公开 PDF**、**10 项仅在线链接。

适合在掌握 Transformer 基础后，继续深入架构、训练、推理、评测和前沿研究。

### [Large Language Model Systems · Spring 2025](/courses/llm-systems-2025)

从 GPU 与分布式训练出发，系统学习并行、量化、MoE、推理优化、PagedAttention 与在线服务。 本页含 **62 份官方 PDF / 2,413 页**；论文阅读中有 **36 份公开 PDF**、**7 项仅在线链接。

适合补齐“模型为什么能高效训练和部署”的工程视角。

### [CMU Large Language Model Applications · Spring 2026](/courses/cmu-llm-applications-2026)

围绕检索、Agent、教育、医疗、法律、代码与产品等应用主题组织，强调从模型能力到真实场景。 本页含 **28 份官方 PDF / 1,678 页**；论文阅读中有 **3 份公开 PDF**、**2 项仅在线链接。

适合建立应用地图，按自己的方向选择专题学习。

### [UC Berkeley CS 185/285: Deep Reinforcement Learning (Spring 2026)](/courses/berkeley-deeprl-2026)

从模仿学习、MDP、策略梯度和 Actor-Critic，一直讲到 LLM RL、模型式 RL、离线 RL、探索与开放问题。 本页含 **43 份官方 PDF / 1,062 页**；论文阅读中有 **0 份公开 PDF**、**0 项仅在线链接。

适合补齐大语言模型强化学习真正需要的算法基础，并把 PPO、验证式奖励和 Agent 训练放回完整 RL 框架。

### [Stanford CS336: Language Modeling from Scratch · Spring 2026](/courses/cs336-2026)

从零实现语言模型的工程主线：Tokenizer、Transformer、PyTorch、GPU、并行、Scaling、推理、数据与后训练。 本页含 **9 份本站可执行 Slides**与 **8 份 PDF / 496 页**。

适合把公式变成可运行实现，并养成计算、显存和通信三本账的习惯。

## 推荐组合方法

1. 先按本站[零基础系统课](/beginner/)建立中文直觉。
2. 同主题看台大 ADL Slides，快速确认概念结构。
3. 再沿 CS224N 主线学习，用 CMU Advanced NLP 把研究脉络补深。
4. 想做训练与部署时转到 LLM Systems；想做产品与场景时转到 LLM Applications。
5. 每一讲先读 Slides，再挑 1–2 篇论文精读；不必从第一天就把全部论文读完。

::: info PDF 与部署说明
课程 PDF 只保存在开发者本地的 `resources/` 资料库中，并由 Git 忽略。公开网站直接链接课程官网、出版方或 arXiv，仓库只保存教程、图片、交互实验和资料清单。
:::

[打开完整 JSON 清单](/course-materials/manifest.json)
