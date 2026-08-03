---
title: 大模型系统课 · 完整课程主线
description: 一条覆盖表示、架构、训练、后训练、强化学习、推理系统、应用、评测与安全的系统教程
---

# 大模型系统课：完整课程主线

<div class="lesson-lead">这是一条为自学重新组织的连续主线：从文字怎样变成数字开始，系统学完模型原理、训练、后训练、高效推理、RAG、Agent、多模态、评测与安全，最后用 Kimi K3 检查自己能否把一台真实前沿大模型从头到尾讲清楚。</div>

<figure class="teaching-figure"><img src="/illustrations/beginner-learning-journey.webp" alt="从基础概念到训练、工具实践与 Kimi K3 案例的学习路线"><figcaption>学习顺序由先修关系决定，不照搬任何一门大学课的周次；相同知识合并，系统与应用内容补进同一条路线。</figcaption></figure>

::: tip 名校课程已经融入，不需要另学七遍
Stanford CS336、CS224N、台湾大学 ADL、CMU Advanced NLP、LLM Systems、CMU LLM Applications 与 Berkeley Deep RL 已经映射进主教程。CS336 负责“从零实现与资源核算”，其余课程分别补足 NLP、系统、应用和强化学习。可在[知识覆盖表](/curriculum/sources)逐讲核对。
:::

## CS336 怎样补上“会讲但不会做”的缺口

[Stanford CS336: Language Modeling from Scratch](/courses/cs336-2026)不只是另一套架构 Slides。它要求学习者实现 Tokenizer、Transformer 和训练循环，再一路计算 GPU、并行、Scaling、推理、数据与后训练成本。本教程因此把 CS336 用在三处：

1. 概念章节加入可运行的 PyTorch 最小实现；
2. 公式旁明确张量形状、FLOPs、显存和通信代价；
3. 把课程中的可执行 Slides 本地保存在[互动讲义入口](/lectures/?trace=var/traces/lecture_01.json)，PDF 讲次则保存在逐讲资料页。

## 三门 NLP 主干课怎样被合并

这次不再满足于“讲次已映射”。三门核心课的 **232 份官方 PDF**承担不同角色，并在每个教程页提供可直接打开的对应原稿：

| 来源课程 | 本站主要吸收什么 | 怎样改写给初学者 |
|---|---|---|
| Stanford CS224N 2026 | 词向量、反向传播、RNN、Transformer、预/后训练、Reasoning、多语言、评测与社会风险 | 保留概念演化、反例和 benchmark 思维，把数学补成可手算步骤 |
| NTU Applied Deep Learning 2025 | Tokenization、Word Embeddings、BERT、Prompt、LoRA、RAG、解码、评测与 Agent | Follow Slides 的教学顺序，增加中文解释、工程成本和练习答案 |
| CMU Advanced NLP 2026 | 数据与 Scaling、蒸馏、多模态、扩散、RL、Agent、量化、分布式、长上下文与测试时计算 | 把前沿专题接到已学先修，明确训练数据流、系统代价和验证方法 |

每一课按“共同基础 → 课程互补 → 现实系统 → K3 对照”融合，不把三门课依次粘贴，也不要求读者重复学三遍。

## 这套教程最后要让你会什么

你应该能把一个 LLM 系统画成完整数据流，并回答每个模块的四个问题：

1. 它在解决什么失败？
2. 输入、状态和输出怎样变化？
3. 它用什么计算、显存、通信或数据代价换来收益？
4. 用什么实验才能证明，而不是只听方法名字？

## 八个阶段，而不是一堆孤立名词

### 阶段一：先建立计算直觉

| 课号 | 课程 | 你会回答 |
|---:|---|---|
| 00 | [00 · 模型、参数与训练](/beginner/00-model) | 模型和程序有什么不同？参数怎样学出来？ |
| 01 | [01 · 文字到 Token](/beginner/01-token) | 一句话怎样变成 ID 和向量？ |
| 02 | [02 · 向量与 Word Embeddings](/beginner/02-vector) | 从 one-hot、共现矩阵走到 Word2Vec、GloVe 与上下文化表示。 |
| 03 | [03 · 多语言与 Token 公平性](/beginner/50-multilingual) | 共享词表、跨语言迁移和语言成本为什么会不公平？ |
| 04 | [04 · 损失、梯度与训练](/beginner/03-training) | 一次训练 step 怎样让错误变少？ |
| 05 | [05 · 语言模型演化](/beginner/10-language-models) | n-gram、RNN、Transformer 为什么依次出现？ |

### 阶段二：把主流模型架构拆开

| 课号 | 课程 | 核心对象 |
|---:|---|---|
| 06 | [06 · Attention](/beginner/04-attention) | Q、K、V 与内容检索 |
| 07 | [07 · 完整 Transformer](/beginner/05-transformer) | 多头、位置、残差、Norm、FFN |
| 08–10 | [08 · BERT](/beginner/14-bert) · [09 · T5/BART](/beginner/15-encoder-decoder) · [10 · GPT/LLaMA/SSM](/beginner/16-decoder-ssm) | Encoder-only、Encoder–Decoder、Decoder-only 与非 Transformer |
| 11 | [11 · 架构全景](/beginner/13-architectures) | 怎样按可见性、状态和计算方式比较架构 |
| 12 | [12 · 生成与 KV Cache](/beginner/06-generation) | 训练并行、生成串行、Prefill 与 Decode |
| 13 | [13 · MoE](/beginner/07-moe) | 总参数与激活参数为什么可以分开 |

### 阶段三：模型怎样大规模预训练

| 课程 | 补齐的名校系统内容 |
|---|---|
| [14 · Scaling：参数、数据与算力怎样配平](/beginner/25-data-scaling) | 参数量手算、数据清洗与配比、训练 FLOPs、计算最优和上下文课程 |
| [15 · 自动微分、优化器、框架与 GPU](/beginner/26-training-engineering) | 计算图、激活、混合精度、kernel 与 profiling |
| [16 · 分布式训练与通信](/beginner/27-distributed-training) | DP、TP、PP、ZeRO、序列并行与专家并行 |

### 阶段四：让底座学会按人类目标工作

[17 · 后训练总览](/beginner/08-post-training)先画生命周期，再按顺序学习：

- [18 · Prompt 与上下文学习](/beginner/17-prompting)、[19 · Prompt 进阶](/beginner/18-prompt-advanced)
- [20 · PEFT](/beginner/19-peft)、[21 · LoRA](/beginner/20-lora)、[22 · 模型编辑](/beginner/21-model-editing)
- [23 · SFT、RLHF、DPO 与推理强化学习](/beginner/28-alignment-rl)
- [24 · 推理、验证器与测试时计算](/beginner/49-reasoning-test-time)
- 强化学习专题：[25 · LLM 的 RL 定义](/beginner/40-rl-language-model) → [26 · MDP 与价值](/beginner/41-rl-mdp-value) → [27 · 策略梯度](/beginner/42-rl-policy-gradient) → [28 · Actor-Critic](/beginner/43-rl-actor-critic) → [29 · PPO](/beginner/44-rl-ppo)
- LLM RL 专题：[30 · RLHF 与 DPO](/beginner/45-rlhf-preference) → [31 · GRPO 与可验证奖励](/beginner/46-verifiable-rewards) → [32 · Agent RL](/beginner/47-rl-agent) → [33 · RL 系统与评测](/beginner/48-rl-systems)
- [34 · 知识蒸馏与多教师学习](/beginner/29-distillation)

### 阶段五：把模型变成高效在线服务

| 课程 | 主要问题 |
|---|---|
| [35 · 解码与采样](/beginner/11-decoding) | Greedy、Beam、温度、Top-k、Top-p 怎样改变输出？ |
| [36 · 量化](/beginner/30-quantization) | FP8、INT8、INT4 用什么误差换显存和吞吐？ |
| [37 · FlashAttention 与长上下文](/beginner/31-efficient-attention) | IO 优化、稀疏、线性 Attention、SSM 与 RoPE 扩展有什么区别？ |
| [38 · vLLM、PagedAttention 与在线服务](/beginner/32-serving-systems) | Scheduler、Block Pool、KV 分页、ModelRunner 与推测解码怎样协作？ |

### 阶段六：知识、Agent、多模态与真实应用

先完成 RAG 三课：[39 · 架构](/beginner/22-rag) → [40 · 检索](/beginner/23-rag-retrieval) → [41 · 生成与上线](/beginner/24-rag-generation)，再进入：

- [42 · Agent、工具调用与 Deep Research](/beginner/33-agents)
- [43 · 多模态、扩散生成、世界模型与具身智能](/beginner/34-multimodal)
- [44 · 扩散模型、Guidance 与 Flow Matching](/beginner/51-diffusion-flow)
- [45 · 对话、写作、代码、科学与个性化应用](/beginner/35-applications)

### 阶段七：证明它有效，并让它安全落地

| 课程 | 学完后的检查能力 |
|---|---|
| [46 · 大模型评测基础](/beginner/12-evaluation) | 分清 PPL、词面、语义与任务指标 |
| [47 · 基准、LLM Judge 与实验设计](/beginner/36-evaluation-research) | 识别污染、Judge 偏差、弱基线与无效小提升 |
| [48 · 模型可解释性](/beginner/52-interpretability) | 区分探针、归因与因果机制证据 |
| [49 · 安全与攻击防护](/beginner/37-safety) | 建威胁模型，防注入、越权、泄漏与供应链风险 |
| [50 · 部署、监控与成本](/beginner/38-deployment) | 设计路由、可观测性、版本、回滚和数据闭环 |
| [51 · 可信研究方法](/beginner/39-research-method) | 设计可证伪问题、消融、复现和失败分析 |

### 阶段八：用 Kimi K3 做毕业案例

先学[52 · Kimi K3 全景拼装](/beginner/09-k3-map)，再进入[17 章 K3 案例课](/guide/ch00)，最后完成[第 53 课：K3 完整毕业项目](/beginner/53-k3-capstone)。这时 KDA、MLA、AttnRes、LatentMoE、Muon、多教师蒸馏、原生视觉、百万上下文和在线服务不再是突然出现的缩写，而是前七阶段知识在真实模型上的组合。

## 每一课的学习闭环

1. 看图建立直觉；
2. 只追踪输入、状态、输出；
3. 用小数字手算一次；
4. 再读最小公式；
5. 计算资源账本；
6. 回到名校讲义和论文核对；
7. 闭卷说出边界与验证方法。

::: warning 不建议按 PDF 顺序学习
大学课件默认不同先修，论文又默认读者已经知道背景。原始资料被保留用于溯源和深挖；初学者应沿本页顺序学习，不需要从 373 份课程资料中自己找路线。
:::

<ProgressTracker />

[从第 00 课开始](/beginner/00-model) · [查看七门课程如何逐讲合入](/curriculum/sources) · [直接看 Kimi K3 案例入口](/beginner/09-k3-map)
