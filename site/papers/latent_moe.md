---
title: LatentMoE： Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
description: LatentMoE 把 routed expert 的输入/输出维度从模型宽度 d 压到 latent 宽度 ℓ，再把省下的参数、权重带宽和通信预算换成更多专家与更高 top-k。
---

# LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

<div class="paper-lesson-meta"><span>核心精读</span><span>18 页</span><span>arXiv 2601.18089</span></div>

<div class="lesson-lead">LatentMoE 把 routed expert 的输入/输出维度从模型宽度 d 压到 latent 宽度 ℓ，再把省下的参数、权重带宽和通信预算换成更多专家与更高 top-k。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像专家会诊前先把厚病历压成标准摘要，减少每位专家要读和跨院传输的材料。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**MoE 在小 batch decode 时常被加载专家权重的 HBM 带宽限制，在大吞吐 expert parallel 时又被 all-to-all 限制；只看 FLOPs 会漏掉这两类真实瓶颈。

**它在整条学习链中的位置：**Stable LatentMoE 的直接结构来源

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：token 经共享 down-projection 从 d 到 ℓ，在 latent 空间路由、执行 expert 并聚合，最后用 up-projection 回到 d；shared experts 和 router 可继续留在原宽度。
2. **再看核心变化**：expert 参数读取和传输 token 的体积都近似按 d/ℓ 缩小，而中间非线性宽度 m 保持不变，因此每 token 的 nonlinear budget K·m 不必下降。
3. **最后看输出**：令 α=d/ℓ，可把总专家 N 与激活专家 K 同比例增加；在近似不增加专家侧带宽/通信预算时，组合稀疏空间显著扩大。
4. **系统如何执行**：论文强调同时优化 accuracy per FLOP 与 accuracy per parameter：前者偏计算，后者更能反映低延迟服务中的存储和带宽。

## 论文拿什么证明

- 作者进行最高 95B 参数、超过 1T token 的设计空间实验，并在 iso-FLOP/iso-parameter 设置下比较标准 MoE。
- 压缩率存在拐点：ℓ 低于任务所需 effective feature rank 后质量会塌陷，因此 latent 不是越窄越好。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 的 Stable LatentMoE 继承 latent routed experts，再加入归一化、SiTU-GLU 与 Quantile Balancing，重点解决 2.8T、896 experts 下的训练稳定性。

继续补背景：[第 7 章 · Stable LatentMoE](/guide/ch07)

## 不要从论文中过度推出什么

- roofline 推导使用特定 GPU、精度、并行度和 batch 假设；换硬件后转折点会变，但分析方法仍可迁移。
- 专家组合空间扩大只是表达能力代理；最终收益仍取决于 router 学习、数据覆盖和均衡机制。

## 原文应该怎么读

**推荐范围：**精读 PDF p.3–8；读 p.10–13 的 scaling 与 inference 结果

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2601.18089" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2601.18089" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：50,229 字符 · 24 个标题</span></div>

### 原文章节地图

1. LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
2. Venmugil Elango, Nidhi Bhatia, Roger Waleffe, Rasoul Shafipour, Tomer Asida, Abhinav Khattar, Nave Assaf, Maximilian Golub, Joey Guman, Tiyasa Mitra, Ritchie Zhao, Ritika Borkar, Ran Zilberstein, Mostofa Patwary, Mohammad Shoeybi, Bita Rouhani
3. 1. Introduction
4. 2. LatentMoE Core Design Principles
5. 3. LatentMoE Architecture
6. 4. Evaluation
7. 5. Related Work
8. 6. Conclusion



## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>给定 d=4096、ℓ=1024，计算理论压缩因子；说明为什么可以把 N、K 扩 4×，以及哪些非 expert 成本不会随之缩小。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
