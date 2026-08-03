---
title: CS224N 与 CMU ANLP 逐页深度审计
description: 区分课程路由、逐页解析与教程正文深度的可复核报告
---

# CS224N 与 CMU ANLP：逐页深度审计

<div class="lesson-lead">“某讲映射到某章”只说明应该去哪里，不等于 Slides 的观点、例子和推导已经写进正文。本页把逐页来源证据与教程深度拆开统计，用来指导后续逐章扩写。</div>

::: info 审计规模
两门课共有 **50 份 Slides / 讲义，共 2,723 页**；OpenDataLoader 已留下 **2,723 条逐页记录**，页码不一致 **0**。另有 **158 份论文或指定阅读，共 5,095 页**。
:::

## 怎样理解这份报告

- `来源映射`：这一讲应该补到哪些教程章；
- `逐页记录`：每页标题、文本摘要、字符量和低文本页已保存在 JSON；
- `正文深度`：当前章的可读文本单元（约 2 个汉字或 1 个英文词）、图解和交互组件；
- `扩写优先级`：来源页多但正文短、图解少的章节先处理。

逐页明细：`output/course-corpus/two-course-page-audit.json`。低文本页通常是图、表、过渡页或图片型幻灯片，扩写时必须回看原 PDF，不能据此判断为空白。

## 逐讲 Slides 审计

| 课程 | 讲次 | 主题 | Slides | 页数 | 低文本页 | 去向 |
|---|---:|---|---:|---:|---:|---|
| CS224N | L01 | NLP 的历史与课程地图 | 2 | 78 | 24 | [05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) |
| CS224N | L02 | 词向量：让词变成可计算的坐标 | 3 | 70 | 1 | [02 · 向量、表示与 Embedding](/beginner/02-vector) |
| CS224N | L03 | 神经网络与反向传播 | 4 | 115 | 28 | [04 · 损失、梯度与训练](/beginner/03-training)<br>[15 · 训练工程与 GPU](/beginner/26-training-engineering) |
| CS224N | L04 | 语言模型与循环神经网络 | 2 | 73 | 11 | [05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) |
| CS224N | L05 | Transformer | 2 | 88 | 12 | [06 · Attention 原理](/beginner/04-attention)<br>[07 · Transformer 架构](/beginner/05-transformer) |
| CS224N | L06 | 研究项目设计与实践建议 | 1 | 53 | 7 | [51 · 研究方法](/beginner/39-research-method) |
| CS224N | L07 | 预训练：规模、系统与数据 | 1 | 56 | 7 | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling)<br>[08 · BERT / Encoder-only](/beginner/14-bert) |
| CS224N | L08 | 后训练：SFT、RLHF 与 DPO | 1 | 64 | 7 | [30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[33 · LLM RL 系统、评测与安全](/beginner/48-rl-systems)<br>[23 · 后训练与强化学习](/beginner/28-alignment-rl) |
| CS224N | L09 | 高效适配：Prompt 与 PEFT | 1 | 67 | 9 | [20 · PEFT](/beginner/19-peft)<br>[21 · LoRA](/beginner/20-lora)<br>[18 · Prompt 与上下文学习](/beginner/17-prompting)<br>[19 · Prompt 进阶](/beginner/18-prompt-advanced) |
| CS224N | L10 | Agent、工具调用与 RAG | 1 | 72 | 5 | [39 · RAG 架构](/beginner/22-rag)<br>[40 · 检索与向量索引](/beginner/23-rag-retrieval)<br>[41 · RAG 生成与实践](/beginner/24-rag-generation)<br>[42 · Agent 与 Deep Research](/beginner/33-agents) |
| CS224N | L11 | 基准测试与评测 | 1 | 77 | 16 | [46 · 评测基础](/beginner/12-evaluation)<br>[47 · 高级评测与实验设计](/beginner/36-evaluation-research) |
| CS224N | L12 | 推理（一）：让模型展开思考 | 1 | 68 | 16 | [24 · 推理、验证器与测试时计算](/beginner/49-reasoning-test-time)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards) |
| CS224N | L13 | 推理（二）：验证与测试时计算 | 1 | 59 | 9 | [24 · 推理、验证器与测试时计算](/beginner/49-reasoning-test-time)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[12 · 生成、Prefill 与 KV Cache](/beginner/06-generation)<br>[38 · 大模型在线服务](/beginner/32-serving-systems) |
| CS224N | L14 | 分词与多语言 | 1 | 76 | 2 | [01 · Token 与分词](/beginner/01-token)<br>[03 · 多语言建模与 Token 公平性](/beginner/50-multilingual)<br>[45 · 大模型应用](/beginner/35-applications) |
| CS224N | L15 | 可解释性（官网链接待核对） | 1 | 86 | 25 | [48 · 模型可解释性](/beginner/52-interpretability) |
| CS224N | L16 | NLP 的社会影响与风险 | 1 | 49 | 16 | [49 · 安全、社会风险与攻击防护](/beginner/37-safety) |
| CS224N | L19 | 2026 年 NLP 开放问题 | 1 | 90 | 27 | [49 · 安全、社会风险与攻击防护](/beginner/37-safety) |
| CS224N | S01 | Python 复习课 | 1 | 66 | 3 | [04 · 损失、梯度与训练](/beginner/03-training)<br>[15 · 训练工程与 GPU](/beginner/26-training-engineering) |
| CS224N | S02 | Hugging Face Transformers 实作课 | 1 | 20 | 0 | [07 · Transformer 架构](/beginner/05-transformer)<br>[20 · PEFT](/beginner/19-peft) |
| CMU ANLP | L01 | 导论与高级 NLP 基础 | 1 | 62 | 9 | [00 · 模型、参数与训练](/beginner/00-model)<br>[05 · 语言模型演化](/beginner/10-language-models) |
| CMU ANLP | L02 | 学习表示：文字如何进入模型 | 1 | 68 | 12 | [02 · 向量、表示与 Embedding](/beginner/02-vector)<br>[00 · 模型、参数与训练](/beginner/00-model)<br>[05 · 语言模型演化](/beginner/10-language-models) |
| CMU ANLP | L03 | 自回归语言建模 | 1 | 60 | 9 | [05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm)<br>[00 · 模型、参数与训练](/beginner/00-model) |
| CMU ANLP | L04 | 架构一：循环神经网络 | 1 | 52 | 12 | [05 · 语言模型演化](/beginner/10-language-models)<br>[09 · Encoder–Decoder](/beginner/15-encoder-decoder)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm)<br>[11 · 架构全景](/beginner/13-architectures) |
| CMU ANLP | L05 | 架构二：Attention 与 Transformer | 1 | 68 | 12 | [06 · Attention 原理](/beginner/04-attention)<br>[07 · Transformer 架构](/beginner/05-transformer)<br>[11 · 架构全景](/beginner/13-architectures)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) |
| CMU ANLP | L06 | 学习一：预训练 | 1 | 57 | 12 | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling)<br>[08 · BERT / Encoder-only](/beginner/14-bert) |
| CMU ANLP | L07 | Scaling Laws 与上下文学习 | 1 | 51 | 10 | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling)<br>[18 · Prompt 与上下文学习](/beginner/17-prompting)<br>[19 · Prompt 进阶](/beginner/18-prompt-advanced) |
| CMU ANLP | L08 | 微调与知识蒸馏 | 1 | 40 | 4 | [34 · 知识蒸馏](/beginner/29-distillation)<br>[20 · PEFT](/beginner/19-peft)<br>[21 · LoRA](/beginner/20-lora)<br>[39 · RAG 架构](/beginner/22-rag) |
| CMU ANLP | L09 | 推理：解码算法 | 1 | 62 | 18 | [35 · 解码与采样](/beginner/11-decoding)<br>[12 · 生成、Prefill 与 KV Cache](/beginner/06-generation)<br>[24 · 推理、验证器与测试时计算](/beginner/49-reasoning-test-time)<br>[38 · 大模型在线服务](/beginner/32-serving-systems) |
| CMU ANLP | L10 | 建模一：检索与 RAG | 1 | 189 | 59 | [39 · RAG 架构](/beginner/22-rag)<br>[40 · 检索与向量索引](/beginner/23-rag-retrieval)<br>[41 · RAG 生成与实践](/beginner/24-rag-generation)<br>[22 · 模型编辑](/beginner/21-model-editing) |
| CMU ANLP | L11 | 建模二：多模态基础 | 1 | 47 | 23 | [43 · 多模态与具身智能](/beginner/34-multimodal) |
| CMU ANLP | L12 | 建模三：多模态生成 | 1 | 45 | 14 | [43 · 多模态与具身智能](/beginner/34-multimodal) |
| CMU ANLP | L13 | 评测技术 | 1 | 33 | 10 | [46 · 评测基础](/beginner/12-evaluation)<br>[47 · 高级评测与实验设计](/beginner/36-evaluation-research) |
| CMU ANLP | L14 | 研究技能与实验设计 | 1 | 60 | 11 | [51 · 研究方法](/beginner/39-research-method) |
| CMU ANLP | L15 | 建模四：扩散模型与 Flow | 1 | 43 | 6 | [44 · 扩散模型与 Flow Matching](/beginner/51-diffusion-flow)<br>[43 · 多模态与具身智能](/beginner/34-multimodal) |
| CMU ANLP | L16 | 强化学习一：基础 | 1 | 50 | 5 | [26 · MDP、回报与价值函数](/beginner/41-rl-mdp-value)<br>[27 · 策略梯度与 REINFORCE](/beginner/42-rl-policy-gradient)<br>[28 · Actor-Critic 与 GAE](/beginner/43-rl-actor-critic)<br>[29 · 重要性采样、TRPO 与 PPO](/beginner/44-rl-ppo) |
| CMU ANLP | L17 | 强化学习二：大模型应用 | 1 | 67 | 16 | [23 · 后训练与强化学习](/beginner/28-alignment-rl)<br>[30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[33 · LLM RL 系统、评测与安全](/beginner/48-rl-systems) |
| CMU ANLP | L18 | 基于语言模型的 Agent | 1 | 56 | 9 | [42 · Agent 与 Deep Research](/beginner/33-agents)<br>[32 · 离线 RL、探索与 Agent](/beginner/47-rl-agent)<br>[05 · 语言模型演化](/beginner/10-language-models)<br>[10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) |
| CMU ANLP | L19 | 量化 | 1 | 43 | 14 | [36 · 量化](/beginner/30-quantization) |
| CMU ANLP | L20 | 并行与分布式训练 | 1 | 60 | 16 | [16 · 分布式训练](/beginner/27-distributed-training) |
| CMU ANLP | L21 | 混合专家模型 MoE | 1 | 55 | 14 | [13 · MoE](/beginner/07-moe) |
| CMU ANLP | L22 | 扩展序列长度 | 1 | 54 | 16 | [37 · 高效 Attention 与长上下文](/beginner/31-efficient-attention)<br>[14 · 数据与 Scaling Laws](/beginner/25-data-scaling) |
| CMU ANLP | L23 | 测试时扩展 | 1 | 74 | 30 | [24 · 推理、验证器与测试时计算](/beginner/49-reasoning-test-time)<br>[31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards)<br>[35 · 解码与采样](/beginner/11-decoding)<br>[12 · 生成、Prefill 与 KV Cache](/beginner/06-generation) |

## 教程章深度缺口

| 优先级 | 教程章 | 对应 Slides 页 | 正文单元 | 图解 | 交互 | 推荐阅读段 |
|---|---|---:|---:|---:|---:|---|
| 常规 | [00 · 模型、参数与训练](/beginner/00-model) | 190 | 2263 | 8 | ConceptCheck、GeneralizationLab、TrainingLoopLab | 有 |
| 常规 | [01 · Token 与分词](/beginner/01-token) | 76 | 1473 | 6 | ConceptCheck、TokenLab | 有 |
| 常规 | [02 · 向量、表示与 Embedding](/beginner/02-vector) | 138 | 1959 | 8 | ConceptCheck、VectorSimilarityLab | 有 |
| 常规 | [03 · 多语言建模与 Token 公平性](/beginner/50-multilingual) | 76 | 2114 | 4 | TokenFairnessLab | 有 |
| 常规 | [04 · 损失、梯度与训练](/beginner/03-training) | 181 | 1348 | 7 | ConceptCheck | 有 |
| 常规 | [05 · 语言模型演化](/beginner/10-language-models) | 449 | 2764 | 5 | ConceptCheck、RecurrentGradientLab、SamplingLab | 有 |
| 常规 | [06 · Attention 原理](/beginner/04-attention) | 156 | 1992 | 10 | AttentionLab、ConceptCheck | 有 |
| 常规 | [07 · Transformer 架构](/beginner/05-transformer) | 176 | 1801 | 8 | ConceptCheck、TransformerArchitecture | 有 |
| 常规 | [08 · BERT / Encoder-only](/beginner/14-bert) | 113 | 1824 | 6 | ConceptCheck、MLMLab | 有 |
| 常规 | [09 · Encoder–Decoder](/beginner/15-encoder-decoder) | 52 | 2037 | 8 | ConceptCheck | 有 |
| 常规 | [10 · GPT、LLaMA、SSM](/beginner/16-decoder-ssm) | 387 | 2910 | 12 | ConceptCheck、ModernDecoderLab | 有 |
| 常规 | [11 · 架构全景](/beginner/13-architectures) | 120 | 1176 | 4 | ConceptCheck、TransformerArchitecture | 有 |
| 常规 | [12 · 生成、Prefill 与 KV Cache](/beginner/06-generation) | 195 | 2142 | 10 | ConceptCheck、KVCacheLab | 有 |
| 常规 | [13 · MoE](/beginner/07-moe) | 55 | 2084 | 6 | ConceptCheck、MoERouterLab | 有 |
| 常规 | [14 · 数据与 Scaling Laws](/beginner/25-data-scaling) | 218 | 2628 | 9 | ScalingLab | 有 |
| 常规 | [15 · 训练工程与 GPU](/beginner/26-training-engineering) | 181 | 1156 | 4 | — | 有 |
| 常规 | [16 · 分布式训练](/beginner/27-distributed-training) | 60 | 1249 | 10 | — | 有 |
| 常规 | [18 · Prompt 与上下文学习](/beginner/17-prompting) | 118 | 1403 | 9 | ConceptCheck | 有 |
| 常规 | [19 · Prompt 进阶](/beginner/18-prompt-advanced) | 118 | 1566 | 9 | ConceptCheck、ICLSensitivityLab | 有 |
| 常规 | [20 · PEFT](/beginner/19-peft) | 127 | 1100 | 5 | ConceptCheck | 有 |
| 常规 | [21 · LoRA](/beginner/20-lora) | 107 | 1612 | 8 | ConceptCheck、LoRALab | 有 |
| 常规 | [22 · 模型编辑](/beginner/21-model-editing) | 189 | 1280 | 8 | ConceptCheck | 有 |
| 常规 | [23 · 后训练与强化学习](/beginner/28-alignment-rl) | 131 | 2644 | 10 | ConceptCheck、PreferenceRLLab | 有 |
| 常规 | [24 · 推理、验证器与测试时计算](/beginner/49-reasoning-test-time) | 263 | 3771 | 6 | ConceptCheck、TestTimeScalingLab | 有 |
| 常规 | [26 · MDP、回报与价值函数](/beginner/41-rl-mdp-value) | 50 | 1419 | 5 | ConceptCheck、ReturnAdvantageLab | 有 |
| 常规 | [27 · 策略梯度与 REINFORCE](/beginner/42-rl-policy-gradient) | 50 | 1480 | 3 | ConceptCheck、ReturnAdvantageLab | 有 |
| 常规 | [28 · Actor-Critic 与 GAE](/beginner/43-rl-actor-critic) | 50 | 1286 | 5 | ConceptCheck、ReturnAdvantageLab | 有 |
| 常规 | [29 · 重要性采样、TRPO 与 PPO](/beginner/44-rl-ppo) | 50 | 2190 | 7 | ConceptCheck、PPOClipLab | 有 |
| 常规 | [30 · 奖励模型、RLHF 与 DPO](/beginner/45-rlhf-preference) | 131 | 2233 | 8 | ConceptCheck、PreferenceRLLab | 有 |
| 常规 | [31 · GRPO、可验证奖励与推理 RL](/beginner/46-verifiable-rewards) | 332 | 4024 | 6 | ConceptCheck、GRPOLab | 有 |
| 常规 | [32 · 离线 RL、探索与 Agent](/beginner/47-rl-agent) | 56 | 2290 | 10 | AgentShiftLab、ConceptCheck | 有 |
| 常规 | [33 · LLM RL 系统、评测与安全](/beginner/48-rl-systems) | 131 | 1961 | 7 | ConceptCheck、RLRolloutLab | 有 |
| 常规 | [34 · 知识蒸馏](/beginner/29-distillation) | 40 | 3078 | 10 | ConceptCheck、DistillationLab | 有 |
| 常规 | [35 · 解码与采样](/beginner/11-decoding) | 136 | 1616 | 6 | ConceptCheck、SamplingLab | 有 |
| 常规 | [36 · 量化](/beginner/30-quantization) | 43 | 1081 | 4 | — | 有 |
| 常规 | [37 · 高效 Attention 与长上下文](/beginner/31-efficient-attention) | 54 | 2885 | 7 | ConceptCheck、FlashAttentionLab | 有 |
| 常规 | [38 · 大模型在线服务](/beginner/32-serving-systems) | 121 | 2948 | 15 | ConceptCheck、VLLMSchedulerLab | 有 |
| 常规 | [39 · RAG 架构](/beginner/22-rag) | 301 | 1355 | 7 | ConceptCheck、RAGPipelineLab | 有 |
| 常规 | [40 · 检索与向量索引](/beginner/23-rag-retrieval) | 261 | 1581 | 6 | ConceptCheck、RAGPipelineLab | 有 |
| 常规 | [41 · RAG 生成与实践](/beginner/24-rag-generation) | 261 | 1652 | 6 | ConceptCheck、RAGGroundingLab | 有 |
| 常规 | [42 · Agent 与 Deep Research](/beginner/33-agents) | 128 | 1172 | 7 | — | 有 |
| 常规 | [43 · 多模态与具身智能](/beginner/34-multimodal) | 135 | 1208 | 8 | — | 有 |
| 常规 | [44 · 扩散模型与 Flow Matching](/beginner/51-diffusion-flow) | 43 | 2198 | 4 | DiffusionNoiseLab | 有 |
| 常规 | [45 · 大模型应用](/beginner/35-applications) | 76 | 1306 | 6 | — | 有 |
| 常规 | [46 · 评测基础](/beginner/12-evaluation) | 110 | 1654 | 8 | ConceptCheck、EvaluationThresholdLab | 有 |
| 常规 | [47 · 高级评测与实验设计](/beginner/36-evaluation-research) | 110 | 1149 | 5 | — | 有 |
| 常规 | [48 · 模型可解释性](/beginner/52-interpretability) | 86 | 2174 | 4 | ActivationPatchingLab | 有 |
| 常规 | [49 · 安全、社会风险与攻击防护](/beginner/37-safety) | 139 | 1192 | 6 | — | 有 |
| 常规 | [51 · 研究方法](/beginner/39-research-method) | 113 | 1174 | 6 | — | 有 |

## 后续使用规则

1. 每次扩写先打开 JSON 中该讲的所有逐页记录，再回看低文本页和关键图表原页；
2. 正文至少补齐问题、直觉、公式/算法、完整例子、失败模式和系统代价；
3. 章末推荐阅读区分必读与选读，并给出带问题的阅读指引；
4. 只有正文、图解、实验和引用都可从页面验证时，才把对应缺口视为完成。
