# Kimi K3 论文地图：读什么、读到什么程度

## 先说结论

不需要把 K3 的 150 篇参考文献逐篇读完。对一个有视觉 Transformer 基础、准备转 LLM 的工程师，建议：

- **P0：1 篇主论文**，贯穿全程，读三遍。
- **P1：12 篇核心材料**，只精读指定部分。
- **P2：10 篇方向选修**，根据求职方向选择 3–5 篇。
- 其余论文只在遇到具体公式、实现或历史问题时查阅。

阅读不是按发表时间，而是按认知依赖顺序进行。

## P0：主论文

| 编号 | 论文 | 你要解决的问题 | 阅读范围 |
|---|---|---|---|
| P0-1 | [Kimi K3: Open Frontier Intelligence, arXiv:2607.24653](https://arxiv.org/abs/2607.24653) | 把结构、训练、Agent RL、系统串成一张图 | 第一遍：摘要、§1、图2、表1、§8；第二遍：§2–4；第三遍：§5–6 与附录 |

不要第一次就啃 §5 的系统实现，也不要在 KDA 的 chunkwise 推导处停留数天。先知道每个模块解决哪种资源瓶颈。

## P1：核心材料

### A. 架构主干

| 顺序 | 论文/材料 | K3 中对应概念 | 推荐读法 |
|---:|---|---|---|
| 1 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | causal self-attention、残差、FFN | 你已有基础，只复习 §3.2–3.5；用 30 分钟确认张量形状与复杂度 |
| 2 | [DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model](https://arxiv.org/abs/2405.04434) | MLA、解耦 RoPE、DeepSeekMoE | 精读 MLA 与 MoE 架构图；重点回答“压缩的是 KV cache 还是模型参数” |
| 3 | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) | delta rule、gated recurrence、chunkwise parallelism | 精读方法部分；先掌握 recurrent form，再看 parallel form |
| 4 | [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | KDA、KDA–MLA 混合、UT transform、长上下文效率 | K3 结构最重要的前传论文，精读；复杂 kernel 推导可第二轮再读 |
| 5 | [Attention Residuals, arXiv:2603.15031](https://arxiv.org/abs/2603.15031) | Full/Block AttnRes、PreNorm dilution | 精读动机、公式、Block AttnRes 和消融；系统并行部分选读 |
| 6 | [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066) | shared experts、routed experts、细粒度专家 | 精读架构和负载均衡；把 dense FFN 与 MoE 写成同一套矩阵表达 |
| 7 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | auxiliary-loss-free balancing、MTP、FP8、Muon、大规模 MoE 配方 | 精读 §2 架构与训练；系统章节先看总图 |
| 8 | [LatentMoE: Toward Optimal Accuracy per FLOP and Parameter](https://arxiv.org/abs/2601.18089) | latent routed width、accuracy/FLOP/parameter 三方权衡 | 精读方法和 scaling ablation；理解 K3 为什么能 896 选 16 |

### B. 训练、推理与 Agent

| 顺序 | 论文/材料 | K3 中对应概念 | 推荐读法 |
|---:|---|---|---|
| 9 | [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) | scaling law、参数量/数据量/算力权衡 | 精读结论与拟合方法；不必复现全部统计模型 |
| 10 | [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) | Muon、MoE、Agentic data、工具调用基础 | 精读训练与 post-training；用于识别 K3 的继承项与新增项 |
| 11 | [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599) | long-CoT RL、partial rollout、长尾 rollout | 精读 RL 算法、长上下文训练基础设施和消融 |
| 12 | [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276) | 视觉 Agent、reasoning effort、长轨迹、多模态 RL | 精读视觉工具使用、Agent RL、partial rollout；它补足 K3 报告省略的算法上下文 |
| 13 | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) | 可验证奖励、冷启动、RL 诱发推理、蒸馏 | 读训练 pipeline 和主要消融；对照 SFT 与 RL 的角色 |
| 14 | [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | K3 的 Multi-Teacher On-Policy Distillation | 这是博客而非论文，但概念非常关键；读完后手推逐 token log-ratio reward |

> 说明：表中有 14 项，是因为 Transformer 和 DeepSeek-R1 对已有深度学习基础的读者可以快速读；真正需要反复研读的是 Kimi Linear、Attention Residuals、LatentMoE、K2.5 和 K3。

## P2：按方向选修

### 想做模型架构/长上下文

1. [Transformers are SSMs / Mamba-2](https://arxiv.org/abs/2405.21060)：理解 structured state space duality，帮助把 attention、linear attention、SSM 放在同一坐标系里。
2. [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)：delta rule 的直觉来源。
3. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)：并行 delta rule。
4. [Ring Attention](https://arxiv.org/abs/2310.01889)：全 attention 的 context parallelism，对比 KDA 固定大小状态的通信优势。
5. [YaRN](https://arxiv.org/abs/2309.00071)：RoPE context extension，对比 K3 的 NoPE 路线。

### 想做训练系统/推理系统

1. [Megatron-LM 大规模训练](https://arxiv.org/abs/2104.04473)：TP/PP/DP 的基本组合。
2. [ZeRO](https://arxiv.org/abs/1910.02054)：参数、梯度、优化器状态分片。
3. [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)：sequence/context parallelism。
4. [Mooncake](https://arxiv.org/abs/2407.00079)：KV-cache-centric serving 与传输。
5. [EAGLE-3](https://arxiv.org/abs/2503.01840)：K3 的 draft model 与 speculative decoding。
6. [Microscaling Data Formats](https://arxiv.org/abs/2310.10537)：MXFP4/MXFP8 的数值表示背景。

### 想做多模态/视觉 Agent

1. [Kimi-VL Technical Report](https://arxiv.org/abs/2504.07491)：MoonViT 系列的视觉通路。
2. [Kimi K2.5](https://arxiv.org/abs/2602.02276)：优先级高于大部分通用 VLM 综述。
3. [SigLIP](https://arxiv.org/abs/2303.15343)：理解 K3 为什么强调从 scratch 的 next-token visual encoder，而不是用 contrastive initialization。
4. [LLaVA](https://arxiv.org/abs/2304.08485)：理解“预训练视觉塔 + projector + LLM 对齐”的经典基线，从而看清 K3 的差异。

### 想做 Agent/RL

1. [ReAct](https://arxiv.org/abs/2210.03629)：reason–act–observe loop 的最小原型。
2. [Toolformer](https://arxiv.org/abs/2302.04761)：语言模型学习调用工具的早期范式。
3. [BrowseComp](https://arxiv.org/abs/2504.12516)：搜索 Agent 的可验证评测。
4. [OSWorld](https://arxiv.org/abs/2404.07972)：computer-use Agent 环境。
5. K3 论文引用的 MCP-Atlas、Tool Decathlon、Agents’ Last Exam：需要做 Agent 评测时再读，不属于理解模型本体的前置材料。

## 哪些经典论文不用再精读

你是视觉 Transformer 工程师，以下概念够用即可：ResNet、ViT、Bahdanau attention、RMSNorm、GLU/SwiGLU、量化基础。遇到公式时查原文，不值得为了“完整”打断主线。

## 推荐下载目录与命名

```text
papers/
├── 00_kimi_k3_2607.24653.pdf
├── 01_deepseek_v2_2405.04434.pdf
├── 02_gated_deltanet_2412.06464.pdf
├── 03_kimi_linear_2510.26692.pdf
├── 04_attention_residuals_2603.15031.pdf
├── 05_deepseek_moe_2401.06066.pdf
├── 06_deepseek_v3_2412.19437.pdf
├── 07_latent_moe_2601.18089.pdf
├── 08_chinchilla_2203.15556.pdf
├── 09_kimi_k2_2507.20534.pdf
├── 10_kimi_k1.5_2501.12599.pdf
├── 11_kimi_k2.5_2602.02276.pdf
└── 12_deepseek_r1_2501.12948.pdf
```

arXiv 的 PDF 地址统一是 `https://arxiv.org/pdf/<编号>`。建议先下载以上 13 个文件；不要一次收集几十篇，因为“拥有论文”很容易替代“形成模型”。

## 每篇论文的固定笔记模板

每篇只写一页，强制回答：

1. 它解决的瓶颈是质量、FLOPs、显存、通信、数据、奖励，还是延迟？
2. 输入/输出张量是什么，训练和推理分别存什么状态？
3. 与最接近的 baseline 只有哪一处关键变化？
4. 复杂度如何随序列长度、隐藏维、专家数、激活专家数变化？
5. 论文用什么消融证明收益来自该变化？
6. 哪些关键配方未披露，因而不能独立复现？
7. 它在 K3 中被原样采用、修改采用，还是仅作为背景？

