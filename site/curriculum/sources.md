---
title: 名校课程知识覆盖表
description: 七门名校课程与可执行讲义映射到大模型系统课的完整来源表
---

# 名校课程怎样进入主教程

<div class="lesson-lead">这里不是另一套需要从头学的课程。七门课的每个讲次已经按知识点映射到主教程：相同概念合并、先修关系重排、系统课与应用课互相补足，最后再进入 Kimi K3 案例。</div>

::: tip OpenDataLoader PDF 解析结果
已结构化解析名校课程 **373 份 PDF / 14,488 页**，失败 **0** 份；另有《大模型基础》**290 页**和 **33 篇核心论文**的 OpenDataLoader 语料。结果保留 Markdown、JSON、分页标记和源文件映射。
:::

## 使用方法

1. 初学者只沿左侧‘大模型系统课’顺序学习；
2. 每个主教程章节先讲中文直觉、最小公式、系统代价和自测；
3. 需要深挖时，从下表回到对应名校讲次与论文；
4. 原始 PDF 是证据与延伸阅读，不再承担主教学结构。

## 三层来源怎样分工

| 来源层 | 在主教程中的作用 | 入口 |
|---|---|---|
| 《大模型基础 完整版》 | 提供语言模型、架构、Prompt、PEFT、模型编辑与 RAG 的系统基础 | [进入语言模型主线](/beginner/10-language-models) |
| 七门名校课程 | 补齐从零实现、前沿算法、强化学习、训练系统、推理优化、应用、安全与实验方法 | 本页逐讲覆盖表 |
| 33 篇核心论文 | 核对机制、实验与 Kimi K3 技术演化证据 | [论文学习库](/papers/) |
| Kimi K3 报告 | 作为毕业案例，把前三层知识装进同一台真实模型 | [K3 案例课](/guide/ch00) |

## Stanford CS224N · Winter 2026

官方主页：[https://web.stanford.edu/class/cs224n/](https://web.stanford.edu/class/cs224n/) · [逐讲原始资料](/courses/cs224n-2026)

| 讲次 | 原主题 | 已合入主教程 | 解析量 |
|---|---|---|---:|
| L01 | NLP 的历史与课程地图<br><small>History of NLP</small> | [05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) | 2 份 / 78 页 |
| L02 | 词向量：让词变成可计算的坐标<br><small>Word Vectors</small> | [02 · 向量、表示与 Embedding](/beginner/02-vector) | 11 份 / 171 页 |
| S01 | Python 复习课<br><small>Python Review Session</small> | [04 · 损失、梯度与训练](/beginner/03-training)<br>[15 · 训练工程与 GPU](/beginner/26-training-engineering) | 1 份 / 66 页 |
| L03 | 神经网络与反向传播<br><small>Backpropagation and Neural Network Basics</small> | [04 · 损失、梯度与训练](/beginner/03-training)<br>[15 · 训练工程与 GPU](/beginner/26-training-engineering) | 8 份 / 184 页 |
| L04 | 语言模型与循环神经网络<br><small>Language Models and RNNs</small> | [05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) | 4 份 / 100 页 |
| L05 | Transformer<br><small>Transformers</small> | [06 · Attention 原理](/beginner/04-attention)<br>[07 · Transformer 架构](/beginner/05-transformer) | 7 份 / 159 页 |
| L06 | 研究项目设计与实践建议<br><small>Final Projects: Custom and Default; Practical Tips</small> | [51 · 研究方法](/beginner/39-research-method) | 1 份 / 53 页 |
| L07 | 预训练：规模、系统与数据<br><small>Pretraining (Scaling, Systems, Data)</small> | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling)<br>[08 · BERT / Encoder-only](/beginner/14-bert) | 5 份 / 199 页 |
| L08 | 后训练：SFT、RLHF 与 DPO<br><small>Post-training (RLHF, SFT, DPO)</small> | [30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[33 · LLM RL 系统、评测与安全](/beginner/48-rl-systems)<br>[23 · 后训练与强化学习](/beginner/28-alignment-rl) | 5 份 / 199 页 |
| L09 | 高效适配：Prompt 与 PEFT<br><small>Efficient Adaptation (Prompting + PEFT)</small> | [20 · PEFT](/beginner/19-peft)<br>[21 · LoRA](/beginner/20-lora)<br>[18 · Prompt 与上下文学习](/beginner/17-prompting)<br>[19 · Prompt 进阶](/beginner/18-prompt-advanced) | 6 份 / 266 页 |
| L10 | Agent、工具调用与 RAG<br><small>Agents, Tool Use, and RAG</small> | [39 · RAG 架构](/beginner/22-rag)<br>[40 · 检索与向量索引](/beginner/23-rag-retrieval)<br>[41 · RAG 生成与实践](/beginner/24-rag-generation)<br>[42 · Agent 与 Deep Research](/beginner/33-agents) | 4 份 / 141 页 |
| S02 | Hugging Face Transformers 实作课<br><small>Hugging Face Transformers Tutorial Session</small> | [07 · Transformer 架构](/beginner/05-transformer)<br>[20 · PEFT](/beginner/19-peft) | 1 份 / 20 页 |
| L11 | 基准测试与评测<br><small>Benchmarking and Evaluation</small> | [46 · 评测基础](/beginner/12-evaluation)<br>[47 · 高级评测与实验设计](/beginner/36-evaluation-research) | 3 份 / 266 页 |
| L12 | 推理（一）：让模型展开思考<br><small>Reasoning 1</small> | [24 · 推理、验证器与测试时计算](/beginner/49-reasoning-test-time)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards) | 5 份 / 237 页 |
| L13 | 推理（二）：验证与测试时计算<br><small>Reasoning 2</small> | [24 · 推理、验证器与测试时计算](/beginner/49-reasoning-test-time)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[12 · 生成、Prefill 与 KV Cache](/beginner/06-generation)<br>[38 · 大模型在线服务](/beginner/32-serving-systems) | 5 份 / 152 页 |
| L14 | 分词与多语言<br><small>Tokenization and Multilinguality</small> | [01 · Token 与分词](/beginner/01-token)<br>[03 · 多语言建模与 Token 公平性](/beginner/50-multilingual)<br>[45 · 大模型应用](/beginner/35-applications) | 5 份 / 155 页 |
| L15 | 可解释性（官网链接待核对）<br><small>Interpretability</small> | [48 · 模型可解释性](/beginner/52-interpretability) | 5 份 / 202 页 |
| L16 | NLP 的社会影响与风险<br><small>Social and Broader Impacts of NLP (Risks)</small> | [49 · 安全、社会风险与攻击防护](/beginner/37-safety) | 1 份 / 49 页 |
| L17 | 多模态<br><small>Multimodality</small> | [43 · 多模态与具身智能](/beginner/34-multimodal) | 10 份 / 258 页 |
| L18 | Tinker 与 LoRA 实践<br><small>Tinker and LoRA Without Regret</small> | [20 · PEFT](/beginner/19-peft)<br>[21 · LoRA](/beginner/20-lora) | 0 份 / 0 页 |
| L19 | 2026 年 NLP 开放问题<br><small>Open Questions in NLP 2026</small> | [49 · 安全、社会风险与攻击防护](/beginner/37-safety) | 1 份 / 90 页 |

## NTU Applied Deep Learning · Fall 2025

官方主页：[https://www.csie.ntu.edu.tw/~miulab/f114-adl/](https://www.csie.ntu.edu.tw/~miulab/f114-adl/) · [逐讲原始资料](/courses/ntu-adl-2025)

| 讲次 | 原主题 | 已合入主教程 | 解析量 |
|---|---|---|---:|
| P00 | 自学先修：从机器学习走到反向传播<br><small>Prerequisites</small> | [04 · 损失、梯度与训练](/beginner/03-training)<br>[15 · 训练工程与 GPU](/beginner/26-training-engineering) | 3 份 / 172 页 |
| W01 | 第 1 周：课程说明与序列建模<br><small>Course Logistics + Sequence Modeling</small> | [05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm)<br>[00 · 模型、参数与训练](/beginner/00-model) | 2 份 / 84 页 |
| W02 | 第 2 周：Attention 到 BERT<br><small>Attention, Transformer, Tokenization, BERT</small> | [01 · Token 与分词](/beginner/01-token)<br>[06 · Attention 原理](/beginner/04-attention)<br>[07 · Transformer 架构](/beginner/05-transformer)<br>[08 · BERT / Encoder-only](/beginner/14-bert) | 5 份 / 176 页 |
| SUP | 弹性补充：词向量与 BERT 变体<br><small>Word Embeddings + BERT Variants</small> | [02 · 向量、表示与 Embedding](/beginner/02-vector)<br>[08 · BERT / Encoder-only](/beginner/14-bert) | 2 份 / 80 页 |
| W03 | 第 3 周：预训练与 Prompt Learning<br><small>Pretraining & Prompt Learning</small> | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling)<br>[18 · Prompt 与上下文学习](/beginner/17-prompting)<br>[19 · Prompt 进阶](/beginner/18-prompt-advanced) | 2 份 / 110 页 |
| W04 | 第 4 周：后训练与大模型适配<br><small>Post-Training + LLM Adaptation</small> | [23 · 后训练与强化学习](/beginner/28-alignment-rl)<br>[30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[33 · LLM RL 系统、评测与安全](/beginner/48-rl-systems) | 3 份 / 99 页 |
| W05 | 第 5 周：RAG 与 MoE<br><small>Retrieval-Augmented Generation</small> | [13 · MoE](/beginner/07-moe)<br>[39 · RAG 架构](/beginner/22-rag)<br>[40 · 检索与向量索引](/beginner/23-rag-retrieval)<br>[41 · RAG 生成与实践](/beginner/24-rag-generation) | 2 份 / 93 页 |
| W06 | 第 6 周：生成解码、推理与评价<br><small>NLG Decoding + Evaluation</small> | [35 · 解码与采样](/beginner/11-decoding)<br>[24 · 推理、验证器与测试时计算](/beginner/49-reasoning-test-time)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[46 · 评测基础](/beginner/12-evaluation) | 3 份 / 100 页 |
| W07 | 第 7 周：预训练模型的问题与发展<br><small>Issues and Development in Pre-Trained Models</small> | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling)<br>[11 · 架构全景](/beginner/13-architectures)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) | 1 份 / 42 页 |
| W08 | 第 8 周：语言 Agent<br><small>Language Agents</small> | [42 · Agent 与 Deep Research](/beginner/33-agents) | 1 份 / 65 页 |
| W09 | 第 9 周：知识与多模态<br><small>Knowledge, Multimodality</small> | [39 · RAG 架构](/beginner/22-rag)<br>[40 · 检索与向量索引](/beginner/23-rag-retrieval)<br>[41 · RAG 生成与实践](/beginner/24-rag-generation)<br>[43 · 多模态与具身智能](/beginner/34-multimodal) | 0 份 / 0 页 |
| W10 | 第 10 周：个性化<br><small>Personalization</small> | [45 · 大模型应用](/beginner/35-applications) | 0 份 / 0 页 |
| W11 | 第 11 周：推理<br><small>Reasoning</small> | [24 · 推理、验证器与测试时计算](/beginner/49-reasoning-test-time)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards) | 0 份 / 0 页 |

## CMU Advanced NLP · Spring 2026

官方主页：[https://cmu-l3.github.io/anlp-spring2026/](https://cmu-l3.github.io/anlp-spring2026/) · [逐讲原始资料](/courses/cmu-anlp-2026)

| 讲次 | 原主题 | 已合入主教程 | 解析量 |
|---|---|---|---:|
| L01 | 导论与高级 NLP 基础<br><small>Introduction & Fundamentals</small> | [00 · 模型、参数与训练](/beginner/00-model)<br>[05 · 语言模型演化](/beginner/10-language-models) | 3 份 / 297 页 |
| L02 | 学习表示：文字如何进入模型<br><small>Fundamentals: Learned Representations</small> | [02 · 向量、表示与 Embedding](/beginner/02-vector)<br>[00 · 模型、参数与训练](/beginner/00-model)<br>[05 · 语言模型演化](/beginner/10-language-models) | 2 份 / 192 页 |
| L03 | 自回归语言建模<br><small>Fundamentals: Autoregressive Language Modeling</small> | [05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm)<br>[00 · 模型、参数与训练](/beginner/00-model) | 4 份 / 211 页 |
| L04 | 架构一：循环神经网络<br><small>Architectures I: Recurrent Neural Networks</small> | [05 · 语言模型演化](/beginner/10-language-models)<br>[09 · Encoder–Decoder](/beginner/15-encoder-decoder)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm)<br>[11 · 架构全景](/beginner/13-architectures) | 5 份 / 210 页 |
| L05 | 架构二：Attention 与 Transformer<br><small>Architectures II: Attention and Transformers</small> | [06 · Attention 原理](/beginner/04-attention)<br>[07 · Transformer 架构](/beginner/05-transformer)<br>[11 · 架构全景](/beginner/13-architectures)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) | 6 份 / 135 页 |
| L06 | 学习一：预训练<br><small>Learning I: Pretraining</small> | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling)<br>[08 · BERT / Encoder-only](/beginner/14-bert) | 12 份 / 495 页 |
| L07 | Scaling Laws 与上下文学习<br><small>Scaling Laws and In-Context Learning</small> | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling)<br>[18 · Prompt 与上下文学习](/beginner/17-prompting)<br>[19 · Prompt 进阶](/beginner/18-prompt-advanced) | 6 份 / 259 页 |
| L08 | 微调与知识蒸馏<br><small>Learning III: Fine-tuning and Distillation</small> | [34 · 知识蒸馏](/beginner/29-distillation)<br>[20 · PEFT](/beginner/19-peft)<br>[21 · LoRA](/beginner/20-lora)<br>[39 · RAG 架构](/beginner/22-rag) | 10 份 / 278 页 |
| L09 | 推理：解码算法<br><small>Inference II: Decoding Algorithms</small> | [35 · 解码与采样](/beginner/11-decoding)<br>[12 · 生成、Prefill 与 KV Cache](/beginner/06-generation)<br>[24 · 推理、验证器与测试时计算](/beginner/49-reasoning-test-time)<br>[38 · 大模型在线服务](/beginner/32-serving-systems) | 2 份 / 115 页 |
| L10 | 建模一：检索与 RAG<br><small>Modeling I: Retrieval and RAG</small> | [39 · RAG 架构](/beginner/22-rag)<br>[40 · 检索与向量索引](/beginner/23-rag-retrieval)<br>[41 · RAG 生成与实践](/beginner/24-rag-generation)<br>[22 · 模型编辑](/beginner/21-model-editing) | 8 份 / 375 页 |
| L11 | 建模二：多模态基础<br><small>Modeling II: Multimodal I</small> | [43 · 多模态与具身智能](/beginner/34-multimodal) | 6 份 / 231 页 |
| L12 | 建模三：多模态生成<br><small>Modeling III: Multimodal II</small> | [43 · 多模态与具身智能](/beginner/34-multimodal) | 9 份 / 213 页 |
| L13 | 评测技术<br><small>Evaluation Techniques</small> | [46 · 评测基础](/beginner/12-evaluation)<br>[47 · 高级评测与实验设计](/beginner/36-evaluation-research) | 2 份 / 47 页 |
| L14 | 研究技能与实验设计<br><small>Research Skills and Experimental Design</small> | [51 · 研究方法](/beginner/39-research-method) | 2 份 / 74 页 |
| L15 | 建模四：扩散模型与 Flow<br><small>Modeling IV: Diffusion and Flows</small> | [44 · 扩散模型与 Flow Matching](/beginner/51-diffusion-flow)<br>[43 · 多模态与具身智能](/beginner/34-multimodal) | 3 份 / 96 页 |
| L16 | 强化学习一：基础<br><small>Reinforcement Learning I: Fundamentals</small> | [26 · MDP、回报与价值函数](/beginner/41-rl-mdp-value)<br>[27 · 策略梯度与 REINFORCE](/beginner/42-rl-policy-gradient)<br>[28 · Actor-Critic 与 GAE](/beginner/43-rl-actor-critic)<br>[29 · 重要性采样、TRPO 与 PPO](/beginner/44-rl-ppo) | 3 份 / 76 页 |
| L17 | 强化学习二：大模型应用<br><small>Reinforcement Learning II: Applications</small> | [23 · 后训练与强化学习](/beginner/28-alignment-rl)<br>[30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[33 · LLM RL 系统、评测与安全](/beginner/48-rl-systems) | 7 份 / 322 页 |
| L18 | 基于语言模型的 Agent<br><small>Language Model-Based Agents</small> | [42 · Agent 与 Deep Research](/beginner/33-agents)<br>[32 · 离线 RL、探索与 Agent](/beginner/47-rl-agent)<br>[05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) | 8 份 / 346 页 |
| L19 | 量化<br><small>Quantization</small> | [36 · 量化](/beginner/30-quantization) | 5 份 / 133 页 |
| L20 | 并行与分布式训练<br><small>Parallelism and Distributed Training</small> | [16 · 分布式训练](/beginner/27-distributed-training) | 1 份 / 60 页 |
| L21 | 混合专家模型 MoE<br><small>Mixture of Experts</small> | [13 · MoE](/beginner/07-moe) | 4 份 / 174 页 |
| L22 | 扩展序列长度<br><small>Scaling Sequence Length</small> | [37 · 高效 Attention 与长上下文](/beginner/31-efficient-attention)<br>[14 · 数据与 Scaling Laws](/beginner/25-data-scaling) | 5 份 / 148 页 |
| L23 | 测试时扩展<br><small>Test-Time Scaling</small> | [24 · 推理、验证器与测试时计算](/beginner/49-reasoning-test-time)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[35 · 解码与采样](/beginner/11-decoding)<br>[12 · 生成、Prefill 与 KV Cache](/beginner/06-generation) | 5 份 / 286 页 |

## Large Language Model Systems · Spring 2025

官方主页：[https://llmsystem.github.io/llmsystem2025spring/docs/Syllabus/](https://llmsystem.github.io/llmsystem2025spring/docs/Syllabus/) · [逐讲原始资料](/courses/llm-systems-2025)

| 讲次 | 原主题 | 已合入主教程 | 解析量 |
|---|---|---|---:|
| L01 | 大语言模型导论<br><small>Introduction to LLM</small> | [05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm)<br>[00 · 模型、参数与训练](/beginner/00-model) | 1 份 / 49 页 |
| L02 | GPU 编程基础一<br><small>GPU Programming Basics 1</small> | [15 · 训练工程与 GPU](/beginner/26-training-engineering) | 1 份 / 33 页 |
| L03 | GPU 编程基础二<br><small>GPU Programming Basics 2</small> | [15 · 训练工程与 GPU](/beginner/26-training-engineering) | 1 份 / 26 页 |
| L04 | 学习算法与自动微分<br><small>Learning algorithm and Auto Differentiation</small> | [15 · 训练工程与 GPU](/beginner/26-training-engineering) | 3 份 / 567 页 |
| L05 | 深度学习框架设计<br><small>Deep Learning Frameworks Design</small> | [15 · 训练工程与 GPU](/beginner/26-training-engineering) | 2 份 / 57 页 |
| L06 | Transformer 系统基础<br><small>Transformer</small> | [06 · Attention 原理](/beginner/04-attention)<br>[07 · Transformer 架构](/beginner/05-transformer) | 2 份 / 41 页 |
| L07 | 预训练大语言模型<br><small>Pre-trained LLMs</small> | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling)<br>[05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) | 3 份 / 124 页 |
| L08 | Tokenization<br><small>Tokenization</small> | [01 · Token 与分词](/beginner/01-token) | 4 份 / 75 页 |
| L09 | 大模型解码<br><small>LLM Decoding</small> | [35 · 解码与采样](/beginner/11-decoding) | 1 份 / 21 页 |
| L10 | GPU 加速<br><small>GPU Acceleration</small> | [15 · 训练工程与 GPU](/beginner/26-training-engineering) | 1 份 / 35 页 |
| L11 | GPU 上的 Transformer 加速一<br><small>Accelerating Transformer on GPU Part 1</small> | [15 · 训练工程与 GPU](/beginner/26-training-engineering)<br>[37 · 高效 Attention 与长上下文](/beginner/31-efficient-attention)<br>[06 · Attention 原理](/beginner/04-attention)<br>[07 · Transformer 架构](/beginner/05-transformer) | 2 份 / 75 页 |
| L12 | GPU 上的 Transformer 加速二<br><small>Accelerating Transformer on GPU Part 2</small> | [15 · 训练工程与 GPU](/beginner/26-training-engineering)<br>[37 · 高效 Attention 与长上下文](/beginner/31-efficient-attention)<br>[06 · Attention 原理](/beginner/04-attention)<br>[07 · Transformer 架构](/beginner/05-transformer) | 2 份 / 80 页 |
| L13 | 分布式模型训练一<br><small>Distributed Model Training</small> | [16 · 分布式训练](/beginner/27-distributed-training) | 1 份 / 50 页 |
| L14 | 分布式模型训练二<br><small>Distributed Model Training II</small> | [16 · 分布式训练](/beginner/27-distributed-training) | 2 份 / 39 页 |
| L15 | 分布式模型训练三<br><small>Distributed Model Training III</small> | [16 · 分布式训练](/beginner/27-distributed-training) | 3 份 / 59 页 |
| L16 | 模型量化一<br><small>Model Quantization</small> | [36 · 量化](/beginner/30-quantization) | 1 份 / 26 页 |
| L17 | 模型量化二<br><small>Model Quantization II</small> | [36 · 量化](/beginner/30-quantization) | 2 份 / 52 页 |
| L18 | 大模型高效微调<br><small>Efficient fine-tuning for Large Models</small> | [20 · PEFT](/beginner/19-peft)<br>[21 · LoRA](/beginner/20-lora) | 4 份 / 109 页 |
| L19 | 大模型与 MoE<br><small>Large models with Mixture-of-Expert</small> | [13 · MoE](/beginner/07-moe) | 5 份 / 177 页 |
| L20 | 面向现代硬件优化 Attention<br><small>Optimizing Attention for Modern Hardware (Tri Dao)</small> | [37 · 高效 Attention 与长上下文](/beginner/31-efficient-attention) | 2 份 / 90 页 |
| L21 | 通信高效的分布式训练<br><small>Communication Efficient Distributed Training</small> | [16 · 分布式训练](/beginner/27-distributed-training) | 2 份 / 100 页 |
| L22 | PagedAttention 与大模型服务<br><small>LLM Serving with PageAttention (Woosuk Kwon)</small> | [12 · 生成、Prefill 与 KV Cache](/beginner/06-generation)<br>[38 · 大模型在线服务](/beginner/32-serving-systems) | 2 份 / 105 页 |
| L23 | 更高效的 KV Cache 服务<br><small>Better KV Cache for LLM Serving (Yuhan Liu)</small> | [12 · 生成、Prefill 与 KV Cache](/beginner/06-generation)<br>[38 · 大模型在线服务](/beginner/32-serving-systems) | 3 份 / 94 页 |
| L24 | DistServe：Prefill/Decode 分离<br><small>DistServe: Disaggregated Prefill-Decoding (Hao Zhang)</small> | [12 · 生成、Prefill 与 KV Cache](/beginner/06-generation)<br>[38 · 大模型在线服务](/beginner/32-serving-systems) | 2 份 / 86 页 |
| L25 | SGLang 大模型服务<br><small>LLM serving with SGL (Ying Sheng)</small> | [12 · 生成、Prefill 与 KV Cache](/beginner/06-generation)<br>[38 · 大模型在线服务](/beginner/32-serving-systems) | 2 份 / 51 页 |
| L26 | 高效大模型强化学习系统<br><small>Efficient Reinforcement Learning System for LLMs</small> | [30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[33 · LLM RL 系统、评测与安全](/beginner/48-rl-systems)<br>[23 · 后训练与强化学习](/beginner/28-alignment-rl) | 1 份 / 20 页 |
| L27 | 应用栈与模型服务<br><small>App Stack and Model Serving</small> | [12 · 生成、Prefill 与 KV Cache](/beginner/06-generation)<br>[38 · 大模型在线服务](/beginner/32-serving-systems)<br>[45 · 大模型应用](/beginner/35-applications)<br>[50 · 部署、监控与成本](/beginner/38-deployment) | 1 份 / 49 页 |
| L28 | GPU 即时编译<br><small>GPU just-in-time compilation</small> | [15 · 训练工程与 GPU](/beginner/26-training-engineering) | 1 份 / 3 页 |
| L29 | 推测解码<br><small>Speculative Decoding</small> | [38 · 大模型在线服务](/beginner/32-serving-systems)<br>[37 · 高效 Attention 与长上下文](/beginner/31-efficient-attention)<br>[35 · 解码与采样](/beginner/11-decoding) | 1 份 / 13 页 |
| L30 | 检索增强语言模型<br><small>Retrieval-augmented Language Models</small> | [39 · RAG 架构](/beginner/22-rag)<br>[40 · 检索与向量索引](/beginner/23-rag-retrieval)<br>[41 · RAG 生成与实践](/beginner/24-rag-generation)<br>[05 · 语言模型演化](/beginner/10-language-models) | 1 份 / 19 页 |
| L31 | Embedding 近邻向量检索<br><small>Nearest Vector Search for Embeddings</small> | [02 · 向量、表示与 Embedding](/beginner/02-vector)<br>[39 · RAG 架构](/beginner/22-rag)<br>[40 · 检索与向量索引](/beginner/23-rag-retrieval)<br>[41 · RAG 生成与实践](/beginner/24-rag-generation) | 1 份 / 13 页 |
| L32 | 多模态大语言模型<br><small>Multimodal LLMs</small> | [43 · 多模态与具身智能](/beginner/34-multimodal)<br>[05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) | 1 份 / 54 页 |
| L33 | DeepSeek V3 与 R1<br><small>Deepseek V3 and R1</small> | [11 · 架构全景](/beginner/13-architectures)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) | 0 份 / 0 页 |
| L34 | Attention Sink 与流式语言模型<br><small>Efficient Streaming Language Models with Attention Sinks</small> | [38 · 大模型在线服务](/beginner/32-serving-systems)<br>[37 · 高效 Attention 与长上下文](/beginner/31-efficient-attention)<br>[05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) | 1 份 / 21 页 |
| L35 | 高级大模型服务<br><small>Advanced Large Model Serving</small> | [12 · 生成、Prefill 与 KV Cache](/beginner/06-generation)<br>[38 · 大模型在线服务](/beginner/32-serving-systems) | 0 份 / 0 页 |

## CMU Large Language Model Applications · Spring 2026

官方主页：[https://cmu-llms.org/schedule/](https://cmu-llms.org/schedule/) · [逐讲原始资料](/courses/cmu-llm-applications-2026)

| 讲次 | 原主题 | 已合入主教程 | 解析量 |
|---|---|---|---:|
| L01 | 大语言模型的起源<br><small>Origins of LLMs</small> | [05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) | 3 份 / 85 页 |
| L02 | 自然语言理解与生成<br><small>Natural language understanding vs generation</small> | [05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) | 1 份 / 51 页 |
| L03 | Prompting 的科学方法<br><small>The science of prompting</small> | [18 · Prompt 与上下文学习](/beginner/17-prompting)<br>[19 · Prompt 进阶](/beginner/18-prompt-advanced) | 2 份 / 102 页 |
| L04 | 何时微调，以及怎样高效微调<br><small>Deciding when to finetune and finetuning efficiently</small> | [20 · PEFT](/beginner/19-peft)<br>[21 · LoRA](/beginner/20-lora) | 1 份 / 83 页 |
| L05 | 表示学习与 Embedding<br><small>Learning representations and embeddings</small> | [02 · 向量、表示与 Embedding](/beginner/02-vector) | 1 份 / 53 页 |
| L06 | 检索一：知识的存储与召回<br><small>Retrieval 1: Storing and retrieving knowledge</small> | [39 · RAG 架构](/beginner/22-rag)<br>[40 · 检索与向量索引](/beginner/23-rag-retrieval)<br>[41 · RAG 生成与实践](/beginner/24-rag-generation) | 1 份 / 52 页 |
| L07 | 检索二：RAG 与 Deep Research<br><small>Retrieval 2: Retrieval augmented generation, deep research</small> | [39 · RAG 架构](/beginner/22-rag)<br>[40 · 检索与向量索引](/beginner/23-rag-retrieval)<br>[41 · RAG 生成与实践](/beginner/24-rag-generation) | 1 份 / 53 页 |
| L08 | 检索三：进阶 RAG 与 Deep Research<br><small>Retrieval 3: Retrieval augmented generation (2) and deep research</small> | [39 · RAG 架构](/beginner/22-rag)<br>[40 · 检索与向量索引](/beginner/23-rag-retrieval)<br>[41 · RAG 生成与实践](/beginner/24-rag-generation) | 1 份 / 56 页 |
| L09 | Deep Research 系统<br><small>Deep research</small> | [39 · RAG 架构](/beginner/22-rag)<br>[40 · 检索与向量索引](/beginner/23-rag-retrieval)<br>[41 · RAG 生成与实践](/beginner/24-rag-generation) | 1 份 / 74 页 |
| L10 | 任务型对话<br><small>Task-Oriented Dialogue</small> | [45 · 大模型应用](/beginner/35-applications) | 1 份 / 52 页 |
| L11 | 工具使用、闲聊、角色与陪伴<br><small>Tool-use, chitchat, personas, and companionship</small> | [42 · Agent 与 Deep Research](/beginner/33-agents)<br>[45 · 大模型应用](/beginner/35-applications) | 1 份 / 62 页 |
| L12 | 写作、创意助手与 AI 创作<br><small>Writing and ideation assistants and AI creative writing</small> | [45 · 大模型应用](/beginner/35-applications) | 1 份 / 46 页 |
| L13 | 用大模型做评测：合成数据、模拟与 AI Judge<br><small>LLMs for evaluation: Synthetic data generation, simulation, automatic evaluation, AI-as-judge</small> | [46 · 评测基础](/beginner/12-evaluation)<br>[47 · 高级评测与实验设计](/beginner/36-evaluation-research) | 1 份 / 49 页 |
| L14 | 多 Agent 系统<br><small>Multi-agent systems</small> | [42 · Agent 与 Deep Research](/beginner/33-agents) | 1 份 / 62 页 |
| L15 | 大模型应用造成的风险<br><small>Harms caused by LLM applications</small> | [45 · 大模型应用](/beginner/35-applications)<br>[49 · 安全、社会风险与攻击防护](/beginner/37-safety) | 1 份 / 57 页 |
| L16 | 攻击大模型及其应用<br><small>Attacking LLMs and LLM applications</small> | [45 · 大模型应用](/beginner/35-applications)<br>[49 · 安全、社会风险与攻击防护](/beginner/37-safety) | 1 份 / 54 页 |
| L17 | 代码助手<br><small>Code-writing assistants (guest lecture from Zora Wang)</small> | [45 · 大模型应用](/beginner/35-applications) | 1 份 / 56 页 |
| L18 | 图像生成与视觉对话<br><small>[tentative] Image generation and conversing about images</small> | [43 · 多模态与具身智能](/beginner/34-multimodal) | 1 份 / 56 页 |
| L19 | 非英语语言与非美国文化中的大模型<br><small>LLMs for Non-English Languages and non-American Cultures (guest lecture from Shaily Bhatt)</small> | [45 · 大模型应用](/beginner/35-applications) | 1 份 / 218 页 |
| L20 | 世界模型<br><small>World models (guest lecture from Mingkai Deng)</small> | [43 · 多模态与具身智能](/beginner/34-multimodal) | 1 份 / 67 页 |
| L21 | 大模型与生物学理解<br><small>LLMs for biological understanding (guest lecture by Prof Lei Li)</small> | [43 · 多模态与具身智能](/beginner/34-multimodal) | 1 份 / 70 页 |
| L22 | 音乐生成<br><small>Music generation (guest lecture by Prof Chris Donahue)</small> | [43 · 多模态与具身智能](/beginner/34-multimodal) | 1 份 / 76 页 |
| L23 | 大模型与数字<br><small>Numbers</small> | [45 · 大模型应用](/beginner/35-applications) | 1 份 / 58 页 |
| L24 | 机器人与具身智能<br><small>Robots and embodied AI (guest lecture from Leena Mathur)</small> | [43 · 多模态与具身智能](/beginner/34-multimodal) | 1 份 / 50 页 |
| L25 | 大模型应用部署<br><small>Deployment</small> | [45 · 大模型应用](/beginner/35-applications)<br>[50 · 部署、监控与成本](/beginner/38-deployment) | 1 份 / 36 页 |
| L26 | 课程项目展示<br><small>Project presentations</small> | [51 · 研究方法](/beginner/39-research-method) | 0 份 / 0 页 |
| L27 | 课程项目展示<br><small>Project presentations</small> | [51 · 研究方法](/beginner/39-research-method) | 0 份 / 0 页 |

## UC Berkeley CS 185/285: Deep Reinforcement Learning (Spring 2026)

官方主页：[https://rail.eecs.berkeley.edu/deeprlcourse/](https://rail.eecs.berkeley.edu/deeprlcourse/) · [逐讲原始资料](/courses/berkeley-deeprl-2026)

| 讲次 | 原主题 | 已合入主教程 | 解析量 |
|---|---|---|---:|
| L01 | 导论：强化学习解决什么问题<br><small>Introduction</small> | [25 · 把语言模型写成强化学习问题](/beginner/40-rl-language-model) | 1 份 / 38 页 |
| L02 | 行为克隆（一）：从专家示范学习<br><small>Behavioral Cloning</small> | [25 · 把语言模型写成强化学习问题](/beginner/40-rl-language-model)<br>[32 · 离线 RL、探索与 Agent](/beginner/47-rl-agent) | 3 份 / 105 页 |
| L03 | 行为克隆（二）：数据聚合与模仿学习<br><small>Behavioral Cloning Part 2</small> | [25 · 把语言模型写成强化学习问题](/beginner/40-rl-language-model)<br>[32 · 离线 RL、探索与 Agent](/beginner/47-rl-agent) | 1 份 / 32 页 |
| L04 | 强化学习基础：MDP、回报与价值<br><small>RL Basics</small> | [25 · 把语言模型写成强化学习问题](/beginner/40-rl-language-model)<br>[26 · MDP、回报与价值函数](/beginner/41-rl-mdp-value) | 3 份 / 50 页 |
| L05 | 策略梯度：直接提高好动作的概率<br><small>Policy Gradients</small> | [27 · 策略梯度与 REINFORCE](/beginner/42-rl-policy-gradient) | 3 份 / 41 页 |
| L06 | Actor-Critic：用价值网络指导策略<br><small>Actor Critic</small> | [28 · Actor-Critic 与 GAE](/beginner/43-rl-actor-critic) | 1 份 / 33 页 |
| L07 | 价值型强化学习<br><small>Value-Based RL</small> | [26 · MDP、回报与价值函数](/beginner/41-rl-mdp-value) | 3 份 / 34 页 |
| L08 | Q-learning 实践<br><small>Q-learning in Practice</small> | [26 · MDP、回报与价值函数](/beginner/41-rl-mdp-value) | 1 份 / 36 页 |
| L09 | 高级策略梯度（一）：重要性采样<br><small>Advanced Policy Gradients Part 1</small> | [29 · 重要性采样、TRPO 与 PPO](/beginner/44-rl-ppo) | 2 份 / 30 页 |
| L10 | 高级策略梯度（二）：约束更新幅度<br><small>Advanced Policy Gradients Part 2</small> | [29 · 重要性采样、TRPO 与 PPO](/beginner/44-rl-ppo) | 1 份 / 33 页 |
| L11 | 变分推断基础<br><small>Variational Inference</small> | [26 · MDP、回报与价值函数](/beginner/41-rl-mdp-value)<br>[30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference) | 2 份 / 26 页 |
| L12 | 强化学习中的变分推断<br><small>VI in RL</small> | [26 · MDP、回报与价值函数](/beginner/41-rl-mdp-value)<br>[30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference) | 1 份 / 37 页 |
| L13 | 控制即推断<br><small>Control as Inference</small> | [26 · MDP、回报与价值函数](/beginner/41-rl-mdp-value)<br>[30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference) | 1 份 / 34 页 |
| L14 | 大语言模型强化学习<br><small>LLM RL</small> | [30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[33 · LLM RL 系统、评测与安全](/beginner/48-rl-systems) | 4 份 / 95 页 |
| L15 | 模型式强化学习（一）<br><small>Model-Based RL Part 1</small> | [32 · 离线 RL、探索与 Agent](/beginner/47-rl-agent) | 2 份 / 26 页 |
| L16 | 模型式强化学习（二）<br><small>Model-Based RL Part 2</small> | [32 · 离线 RL、探索与 Agent](/beginner/47-rl-agent) | 1 份 / 38 页 |
| L17 | 离线强化学习（一）<br><small>Offline RL Part 1</small> | [32 · 离线 RL、探索与 Agent](/beginner/47-rl-agent) | 4 份 / 62 页 |
| L18 | 离线强化学习（二）<br><small>Offline RL Part 2</small> | [32 · 离线 RL、探索与 Agent](/beginner/47-rl-agent) | 1 份 / 35 页 |
| L19 | 探索：怎样发现更好的行为<br><small>Exploration</small> | [32 · 离线 RL、探索与 Agent](/beginner/47-rl-agent) | 1 份 / 23 页 |
| L20 | 强化学习理论<br><small>RL Theory</small> | [26 · MDP、回报与价值函数](/beginner/41-rl-mdp-value)<br>[33 · LLM RL 系统、评测与安全](/beginner/48-rl-systems) | 1 份 / 28 页 |
| L21 | 期中复习（一）<br><small>Midterm Review Part 1</small> | [26 · MDP、回报与价值函数](/beginner/41-rl-mdp-value)<br>[27 · 策略梯度与 REINFORCE](/beginner/42-rl-policy-gradient)<br>[28 · Actor-Critic 与 GAE](/beginner/43-rl-actor-critic)<br>[29 · 重要性采样、TRPO 与 PPO](/beginner/44-rl-ppo) | 1 份 / 34 页 |
| L22 | 期中复习（二）<br><small>Midterm Review Part 2</small> | [26 · MDP、回报与价值函数](/beginner/41-rl-mdp-value)<br>[27 · 策略梯度与 REINFORCE](/beginner/42-rl-policy-gradient)<br>[28 · Actor-Critic 与 GAE](/beginner/43-rl-actor-critic)<br>[29 · 重要性采样、TRPO 与 PPO](/beginner/44-rl-ppo) | 1 份 / 73 页 |
| L23 | 高级探索<br><small>Advanced Exploration</small> | [32 · 离线 RL、探索与 Agent](/beginner/47-rl-agent) | 1 份 / 18 页 |
| L24 | 多任务强化学习<br><small>Multi-task RL</small> | [32 · 离线 RL、探索与 Agent](/beginner/47-rl-agent) | 1 份 / 53 页 |
| L25 | 挑战与开放问题<br><small>Challenges and Open Problems</small> | [33 · LLM RL 系统、评测与安全](/beginner/48-rl-systems)<br>[49 · 安全、社会风险与攻击防护](/beginner/37-safety) | 2 份 / 48 页 |

## Stanford CS336: Language Modeling from Scratch · Spring 2026

官方主页：[https://stanford-cs336.github.io/spring2026/](https://stanford-cs336.github.io/spring2026/) · [逐讲原始资料](/courses/cs336-2026)

| 讲次 | 原主题 | 已合入主教程 | 解析量 |
|---|---|---|---:|
| L01 | 课程总览与从零实现 Tokenizer<br><small>Course Introduction and Tokenization</small> | [01 · Token 与分词](/beginner/01-token)<br>[00 · 模型、参数与训练](/beginner/00-model)<br>[05 · 语言模型演化](/beginner/10-language-models) | 0 份 / 0 页 |
| L02 | 浮点数、张量形状、计算量与 PyTorch<br><small>Resource Accounting and PyTorch</small> | [00 · 模型、参数与训练](/beginner/00-model)<br>[04 · 损失、梯度与训练](/beginner/03-training)<br>[15 · 训练工程与 GPU](/beginner/26-training-engineering) | 0 份 / 0 页 |
| L03 | 语言模型架构与超参数<br><small>LM Architecture and Hyperparameters</small> | [07 · Transformer 架构](/beginner/05-transformer)<br>[11 · 架构全景](/beginner/13-architectures) | 1 份 / 67 页 |
| L04 | Attention 替代路线与混合专家<br><small>Attention Alternatives and Mixtures of Experts</small> | [06 · Attention 原理](/beginner/04-attention)<br>[13 · MoE](/beginner/07-moe)<br>[37 · 高效 Attention 与长上下文](/beginner/31-efficient-attention) | 1 份 / 60 页 |
| L05 | GPU、内存层次与矩阵乘法<br><small>GPUs</small> | [15 · 训练工程与 GPU](/beginner/26-training-engineering) | 1 份 / 55 页 |
| L06 | Kernel、编译与 GPU 性能<br><small>Kernels and Compilation</small> | [15 · 训练工程与 GPU](/beginner/26-training-engineering) | 0 份 / 0 页 |
| L07 | 分布式通信与并行训练<br><small>Distributed Communication and Parallelism</small> | [16 · 分布式训练](/beginner/27-distributed-training) | 0 份 / 0 页 |
| L08 | 大模型并行训练基础<br><small>Parallelism Basics</small> | [16 · 分布式训练](/beginner/27-distributed-training) | 1 份 / 73 页 |
| L09 | Scaling Laws 基础<br><small>Scaling Laws - Basics</small> | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling) | 1 份 / 57 页 |
| L10 | 大模型推理工作负载与优化<br><small>Inference</small> | [12 · 生成、Prefill 与 KV Cache](/beginner/06-generation)<br>[35 · 解码与采样](/beginner/11-decoding)<br>[38 · 大模型在线服务](/beginner/32-serving-systems) | 0 份 / 0 页 |
| L11 | Scaling 实践、参数化与案例<br><small>Scaling - Case Study and Details</small> | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling) | 1 份 / 58 页 |
| L12 | 从困惑度到 Agent 的评测<br><small>Evaluation</small> | [46 · 评测基础](/beginner/12-evaluation)<br>[47 · 高级评测与实验设计](/beginner/36-evaluation-research)<br>[42 · Agent 与 Deep Research](/beginner/33-agents) | 0 份 / 0 页 |
| L13 | 预训练数据来源、许可与筛选<br><small>Data I: Sources and Curation</small> | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling) | 0 份 / 0 页 |
| L14 | 数据处理、去重与混合<br><small>Data II: Processing and Deduplication</small> | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling) | 0 份 / 0 页 |
| L15 | 中训练、指令微调与偏好对齐<br><small>After Pretraining</small> | [23 · 后训练与强化学习](/beginner/28-alignment-rl)<br>[20 · PEFT](/beginner/19-peft)<br>[30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference) | 1 份 / 65 页 |
| L16 | 可验证奖励强化学习<br><small>RL from Verifiable Rewards</small> | [31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference)<br>[33 · LLM RL 系统、评测与安全](/beginner/48-rl-systems) | 1 份 / 61 页 |
| L17 | 从视觉编码器到多模态语言模型<br><small>Multimodal Models</small> | [43 · 多模态与具身智能](/beginner/34-multimodal) | 0 份 / 0 页 |

## 可复核文件

- 课程 PDF 结构化索引：`output/course-corpus/opendataloader/index.json`
- 逐讲到主教程映射：`output/course-corpus/curriculum-coverage.json`
- 原始课程清单：`output/course-materials/combined-manifest.json`

映射表用于检查知识覆盖，不表示每个来源的所有结论都被无条件接受。主教程会区分基础共识、课程讲解、论文证据与针对 Kimi K3 的案例推论。
