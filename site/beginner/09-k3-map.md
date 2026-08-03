---
title: 第 52 课 Kimi K3 全景拼装
description: 用序列、深度、宽度、训练和系统五条线理解 K3
---

# 第 52 课：把 Kimi K3 一块一块装起来

<div class="lesson-lead">现在才第一次完整看 K3。不要把它记成模块清单，而要让每个模块回答一个明确的规模问题。</div>

::: info 本课不是宣传摘要，而是论文索引
主入口是 [Kimi K3 原论文精读页](/papers/kimi_k3)。每条技术线都绑定前置论文：序列看 [Kimi Linear](/papers/kimi_linear)，深度看 [Attention Residuals](/papers/attention_residuals)，宽度看 [DeepSeekMoE](/papers/deepseek_moe)与 [LatentMoE](/papers/latent_moe)，训练看 [Chinchilla](/papers/chinchilla)与 [Muon](/papers/muon_scalable)，后训练看 [Kimi k1.5](/papers/kimi_k1_5)、[DeepSeek-R1](/papers/deepseek_r1)与 [MOPD](/papers/mopd)。正文中的数字只作路线图，正式结论回到论文证据与实验口径核对。
:::

<figure class="teaching-figure">
  <img src="/illustrations/beginner-k3-panorama.webp" alt="由长序列道路、深层高塔、专家街区、视觉观测站和基础设施组成的 K3 模型城市">
  <figcaption>K3 像一座必须共同规划的城市：序列道路、深度高楼、专家分区、视觉入口和地下基础设施互相制约。</figcaption>
</figure>
<div class="visual-key"><div><b>地面道路</b>KDA 与 MLA 处理超长 token 序列。</div><div><b>高塔与街区</b>AttnRes 管深度，LatentMoE 管专家通道。</div><div><b>入口与地下</b>视觉 token 进入统一模型，训练和服务系统托住规模。</div></div>

## 1. 一句话版本，逐段拆开

> K3 是原生视觉—语言、自回归、稀疏 MoE 模型；总参数约 2.8T、每 token 激活约 104B；使用 3 层 KDA + 1 层 Gated MLA 的混合注意力处理最长 1M token，用 AttnRes 改造深度信息流，用 Stable LatentMoE 扩展宽度，再经 SFT、多领域多 effort RL 与多教师 on-policy distillation 获得推理和 Agent 能力。

如果这句话仍然密集，按下面五条线读。

<K3SystemDiagram />

<figure class="teaching-figure source-figure"><a href="/paper-figures/kimi-k3-figure-2.webp" target="_blank"><img src="/paper-figures/kimi-k3-figure-2.webp" alt="Kimi K3 技术报告 Figure 2，包含 Stable LatentMoE、KDA、混合主干、AttnRes 和视觉入口"></a><figcaption>Kimi K3 技术报告 Figure 2（PDF p.3）。阅读顺序：右下 MoonViT-V2 是视觉入口；右侧主干每组含 3×KDA 与 1×Gated MLA；红色长线是 AttnRes 的跨层读取；左下放大 KDA；左上放大 Stable LatentMoE。<a href="https://arxiv.org/pdf/2607.24653#page=3">打开原论文第 3 页</a>。</figcaption></figure>

## 2. 序列线：一个 token 怎样读取历史

### 痛点

标准 attention 的计算分数表随 T² 增长，KV Cache 随 T 增长。1M token 下，所有层都用全 attention 很贵。

### K3 的回答

```text
KDA → KDA → KDA → Gated MLA
  固定状态递推       压缩 KV 的全局检索
```

- KDA：大部分层用固定大小 fast-weight state，适合长历史；
- MLA：每四层一次显式全局 softmax，补充精确内容匹配；
- 末端 MLA：输出前再提供全局锚点。

为什么混合？KDA 有容量冲突，MLA 有长序列成本。3:1 是经验质量—效率折中，不是宇宙定律。

### KDA 的状态更新公式怎样读

论文对单个 attention head 写成：

$$
S_t=(I-\beta_t k_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}
+\beta_t k_tv_t^\top,
\qquad
\tilde o_t=S_t^\top q_t
$$

不要把矩阵式整行背下来，拆成四个动作：

1. $\operatorname{Diag}(\alpha_t)S_{t-1}$：按通道衰减旧状态，$\alpha_t$ 决定每类信息保留多少；
2. $I-\beta_tk_tk_t^\top$：沿当前 key 的方向修正旧记忆，避免只会不断叠加；
3. $+\beta_tk_tv_t^\top$：把当前 key-value 关系写进固定大小状态，$\beta_t$ 控制写入强度；
4. $S_t^\top q_t$：当前 query 从更新后的状态中读取输出。

标准 KV Cache 为每个历史 token 留档；KDA 把历史不断更新进固定形状的 $S_t$。因此它的状态大小不随序列长度 $T$ 线性增长，但固定容量也会带来信息覆盖与冲突问题。公式来源见[原论文 Eq. 1（PDF p.4）](https://arxiv.org/pdf/2607.24653#page=4)。

## 3. 深度线：当前层怎样找到早期表示

### 痛点

标准 residual 不断把历史更新相加，网络很深时，早期信息在越来越大的残差流中相对变弱。

### K3 的回答

Block AttnRes 让当前模块用学习到的 query，在 embedding 和历史 block 表示之间做 softmax 选择。

注意：这里 attention 发生在**层/块维度**，不是 token 维度。它不替代 KDA/MLA。

## 4. 宽度线：怎样拥有 2.8T 参数而不全算

### 痛点

普通 MoE 专家越多、Top-k 越大，权重访问、All-to-All 和激活尺度越容易失控。

### K3 的回答

Stable LatentMoE：

1. routed path 先从模型宽度 d 降到 latent 宽度 ℓ；
2. 每 token 从 896 个 routed experts 选 16 个；
3. 另有 2 个 shared experts；
4. 聚合后 RMSNorm 控制尺度；
5. SiTU-GLU 把分支乘积限制在有限范围；
6. Quantile Balancing 直接根据专家 margin 分布调 bias。

三种稳定化组件解决不同问题，不能互相替代。

只看 routed experts，单 token 的专家激活比例为：

$$
\frac{k}{E}=\frac{16}{896}=\frac{1}{56}\approx1.79\%
$$

这个 1.79% 只描述“896 个 routed experts 中执行 16 个”，不能直接说整台模型省了 56 倍：shared experts、Attention、投影、路由、通信和内存访问仍然存在。

## 5. 视觉线：图像怎样进入同一模型

MoonViT-V2 把图片和视频编码成视觉 token，经 projector 映射到语言模型隐藏空间，与文本 token 共同进行 next-token prediction。

K3 强调视觉塔从 scratch 训练，而不是从 SigLIP 对比学习初始化。报告给出的主要理由是联合训练梯度更稳定，并让视觉表示直接适配 next-token 目标。

“原生视觉”不只是能插入图片，还涉及共同训练目标、数据混合和统一 backbone。

## 6. 能力线：从 base model 到 Agent

```text
多模态预训练
  ↓
SFT 冷启动
  ↓
三领域 × 三 effort RL = 9 个教师
  ↓
Multi-Teacher On-Policy Distillation
  ↓
量化、draft model 与服务
```

三领域包括 general、general-agent、coding-agent。Effort 控制推理和工具预算。蒸馏把九个专长教师整合回一个可部署学生。

## 7. 系统线：为什么论文有大量基础设施章节

架构只有被系统支持才成立：

- KDA chunkwise kernel：训练时把递推变成矩阵并行；
- KDA Context Parallelism：百万 token 沿序列切分；
- MoonEP：让极端稀疏 MoE 的 token 负载接近完美平衡；
- 外置 KV Cache：长 RL rollout 状态不能全留在生成 GPU；
- 可暂停/恢复 microVM：Agent 环境跨训练更新继续执行；
- 混合 prefix cache：同时命中 MLA 页块与 KDA state checkpoint；
- fleet scheduling：短请求、长请求、不同 effort 按资源预算准入。

## 8. 五个数字，分别属于哪本账

| 数字 | 含义 | 不等于 |
|---|---|---|
| 2.78T | 总参数 | 每 token FLOPs |
| 104.2B | 激活参数 | 实际显存总量 |
| 896 选 16 | routed expert 稀疏度 56 | 所有模块都省 56× |
| 1M | 支持的最大上下文 | 必然能利用全部远程信息 |
| 约 2.5× | 相对 K2 的整体 scaling efficiency 主张 | 某一个模块单独带来的收益 |

## 9. 现在怎样读正式课程

你已经有一张全景图。接下来按问题进入：

- 想懂推理缓存：[第 2–3 章](/guide/ch02)；
- 想懂 MoE：[第 4 章](/guide/ch04) → [第 7 章](/guide/ch07)；
- 想懂 KDA：[第 5 章](/guide/ch05)；
- 想懂训练/RL：[第 10–12 章](/guide/ch10)；
- 想懂系统：[第 13 章](/guide/ch13)；
- 想逐篇读论文：[33 篇论文学习库](/papers/)。

## 最终闭卷任务

在一张 A4 纸上画五条横线：序列、深度、宽度、能力、系统。把每个 K3 模块放到正确横线，并为它写一个“如果没有它，哪种规模问题会出现”。

完成 17 章案例课后，用[第 53 课：K3 完整毕业项目](/beginner/53-k3-capstone)把架构、数据、训练、RL、Agent、服务、评测与安全做成一份可答辩作品。

<ConceptCheck question="AttnRes 与 KDA 的主要作用轴分别是什么？" :options='["都是序列轴", "AttnRes 是深度轴，KDA 是序列轴", "AttnRes 是宽度轴，KDA 是视觉轴"]' :answer="1" explanation="AttnRes 选择历史层/块表示；KDA 在 token 序列上维护递推状态。" />

完成后进入[第 0 章 · K3 到底是什么](/guide/ch00)，你会发现原本密集的术语已经各有位置。

<ChapterReadings lesson="09-k3-map" />
