---
title: DeepSeek-V2： A Strong, Economical, and Efficient Mixture-of-Experts Language Model
description: MLA 的关键不是让 attention 变成低秩，而是把历史 token 的 K/V 压成一个可缓存 latent，并把能在推理时吸收到投影矩阵里的部分提前做代数重写。
---

# DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

<div class="paper-lesson-meta"><span>核心精读</span><span>52 页</span><span>arXiv 2405.04434</span></div>

<div class="lesson-lead">MLA 的关键不是让 attention 变成低秩，而是把历史 token 的 K/V 压成一个可缓存 latent，并把能在推理时吸收到投影矩阵里的部分提前做代数重写。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像把每本借过的书都留下整本副本改成只留一张高密度索引卡，需要时再按索引恢复要用的信息。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**标准 MHA 在 decode 时为每层、每个历史 token 保存全部 head 的 K/V，长上下文和大 batch 下首先被显存容量与带宽卡住。

**它在整条学习链中的位置：**MLA 与现代稀疏 LLM 的起点

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：先计算共享 latent $c_t^{KV}=W^{DKV}h_t$，再用上投影恢复各 head 的 content key/value；decode 只缓存 $c_t^{KV}$。
2. **再看核心变化**：无位置部分的 $W^{UK}$ 可吸收到 query 投影，$W^{UV}$ 可吸收到 output 投影，因此服务时不必真的物化完整 K/V。
3. **最后看输出**：RoPE 会破坏这种吸收，所以论文把位置通道拆为 decoupled RoPE query 和共享 RoPE key；缓存变为 latent 加一个较小的位置 key。
4. **系统如何执行**：query 也做低秩压缩，但主要节约训练 activation，而不是历史 KV cache。

## 论文拿什么证明

- 236B total/21B activated 模型在 8.1T token 上训练；论文报告相对 DeepSeek 67B 节省 42.5% 训练成本、减少 93.3% KV cache，并给出最高 5.76× generation throughput。
- 附录的小、大规模消融中 MLA 在保持较小 cache 的同时不弱于所比较的 MHA/GQA 变体。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 每四层保留一个 Gated MLA 作为全局交互锚点，但使用 NoPE 思路，把顺序信息更多交给 KDA；这比 V2 的 decoupled RoPE 更进一步。
- 理解 MLA 后，才能区分 K3 的两种省资源机制：MLA 压缩全局 attention 的历史缓存，KDA 用固定大小递归状态替代逐 token 缓存。

继续补背景：[第 3 章 · MLA](/guide/ch03) · [第 4 章 · MoE](/guide/ch04)

## 不要从论文中过度推出什么

- 93.3% cache 降幅取决于 head 数、latent 维度和上下文；5.76× 是特定部署设置的上限测量，不是任意 GPU 和 batch 的常数。
- 低秩投影并不等于 attention matrix 低秩，也不意味着计算复杂度从二次变成线性。

## 原文应该怎么读

**推荐范围：**精读 PDF p.6–11 的 MLA/DeepSeekMoE；再读 p.31–33 的 MLA 消融

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2405.04434" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2405.04434" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：112,455 字符 · 59 个标题</span></div>

### 原文章节地图

1. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
2. Contents
3. 1. Introduction
4. 2. Architecture
5. 3. Pre-Training
6. 5. Conclusion, Limitation, and Future Work
7. A. Contributions and Acknowledgments
8. B. DeepSeek-V2-Lite: A 16B Model Equipped with MLA and DeepSeekMoE
9. C. Full Formulas of MLA
10. D. Ablation of Attention Mechanisms
11. E. Discussion About Pre-Training Data Debiasing
12. F. Additional Evaluations on Math and Code

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

We present DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens. DeepSeek-V2 adopts innovative architectures including Multi-head Latent Attention (MLA) and DeepSeekMoE. MLA guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector, while DeepSeekMoE enables training strong models at an economical cost through sparse computation. Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times. We pretrain DeepSeek-V2 on a high-quality and multi-source corpus consisting of 8.1T tokens, and further perform Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to fully unlock its potential. Evaluation results show that, even with only 21B activated parameters, DeepSeek-V2 and its chat versions still achieve top-tier performance among open-source models. The model checkpoints are available at https://github.com/deepseek-ai/DeepSeek-V2. DeepSeek-V2 80 LLaMA 3 70B Mixtral 8x22B Command R+ Qwen1.5 72B DBRX Performance (MMLU) 75 DeepSeek 67B Qwen1.5 32B Grok-1 70 Mixtral 8x7B LLaMA 2 70B LLaMA 3 8B Command R 65 LLaMA 1 65B LLaMA 2 34B - LLaMA 1 Family - LLaMA 2 Family - LLaMA 3 Family Mistral 7B 60 Mixtral Family LLaMA 1 33B Command R Family 55 Qwen1.5 Family LLaMA 2 13B 0 20 40 60 80 100 Activated Parameters (Billions) (a) ###### Training Costs (K GPU Hours/T Tokens) DeepSeek 67B saving 42.5% of training costs DeepSeek-V2 0 50 100 150 200 250 300 ###### KV Cache for Generat

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>推导为什么 $W^{UK}$ 能吸收到 query，而带 RoPE 的 key 分支不能直接吸收；再写出 decode 时每个历史 token 真正缓存的张量。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
