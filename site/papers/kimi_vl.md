---
title: Kimi-VL Technical Report
description: Kimi-VL 用 MoonViT 编码不同分辨率图像，通过 token 压缩接入 MoE 语言模型，并同时训练感知、推理与视觉 Agent 能力。
---

# Kimi-VL Technical Report

<div class="paper-lesson-meta"><span>方向选读</span><span>24 页</span><span>arXiv 2504.07491</span></div>

<div class="lesson-lead">Kimi-VL 用 MoonViT 编码不同分辨率图像，通过 token 压缩接入 MoE 语言模型，并同时训练感知、推理与视觉 Agent 能力。</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>像把高分辨率图片切成视觉词块，再压缩成语言模型能够一起阅读的“视觉句子”。</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**固定方形缩放会损失文档小字和极端宽高比信息；视觉 token 太多又会挤占上下文并放大 attention 成本。

**它在整条学习链中的位置：**MoonViT 与原生分辨率视觉路线的前传

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

1. **先看输入**：图像按原生分辨率和动态尺寸进入视觉编码器，尽量保留细节与布局。
2. **再看核心变化**：视觉特征经过 projector 和 token 压缩映射到语言模型隐藏空间。
3. **最后看输出**：训练数据同时覆盖 OCR、图表、通用视觉问答、数学与 GUI/工具交互，而不只是 caption。

## 论文拿什么证明

- 报告给出 Kimi-VL-A3B 等模型在视觉理解、文档、视频和 agent benchmark 上的结果。
- 模型为 MoE，激活参数远小于总参数；视觉分支与语言分支的成本需要分开看。

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

- K3 的 MoonViT-V2 改为从 scratch 用 next-token prediction 训练，正是对 Kimi-VL/K2.5 视觉初始化路线的进一步调整。

继续补背景：[第 8 章 · 原生视觉](/guide/ch08) · [第 12 章 · Agent](/guide/ch12)

## 不要从论文中过度推出什么

- “原生分辨率”仍会受到像素上限、patching 与 token compression 的约束。
- 视觉 benchmark 提升不能仅归因于视觉塔，数据与后训练同样关键。

## 原文应该怎么读

**推荐范围：**读架构总图、MoonViT、视觉 token 压缩与 agent 数据；表格只选你关心的视觉任务

<div class="paper-source-row"><a href="https://arxiv.org/pdf/2504.07491" target="_blank">打开官方 PDF</a><a href="https://arxiv.org/pdf/2504.07491" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：119,279 字符 · 74 个标题</span></div>

### 原文章节地图

1. 35.4 Kimi-VL-A3B-Thinking
2. SMALL IMAGE 50px
3. 2.1 Model Architecture
4. 2.2 Muon Optimizer
5. 2.3 Pre-Training Stages
6. 2.4 Post-Training Stages
7. 2.5 Infrastructure
8. 3.1 Pre-Training Data
9. 3.2 Instruction Data
10. 3.3 Reasoning Data
11. 4.1 Comparison to the State-of-the-Art Models
12. 4.1.1 College-level Academic Problems

<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>

We present Kimi-VL, an efficient open-source Mixture-of-Experts (MoE) vision-language model (VLM) that offers advanced multimodal reasoning, long-context understanding, and strong agent capabilities—all while activating only 2.8B parameters in its language decoder (Kimi-VL-A3B). Kimi-VL demonstrates strong performance across challenging domains: as a general-purpose VLM, Kimi-VL excels in multi-turn agent tasks (e.g., OSWorld), matching flagship models. Furthermore, it exhibits remarkable capabilities across diverse challenging vision language tasks, including collegelevel image and video comprehension, OCR, mathematical reasoning, multi-image understanding. In comparative evaluations, it effectively competes with cutting-edge efficient VLMs such as GPT-4omini, Qwen2.5-VL-7B, and Gemma-3-12B-IT, while surpassing GPT-4o in several key domains. Kimi-VL also advances in processing long contexts and perceiving clearly. With a 128K extended context window, Kimi-VL can process diverse long inputs, achieving impressive scores of 64.5 on LongVideoBench and 35.1 on MMLongBench-Doc. Its native-resolution vision encoder, MoonViT, further allows it to see and understand ultra-high-resolution visual inputs, achieving 83.2 on InfoVQA and 34.5 on ScreenSpot-Pro, while maintaining lower computational cost for common tasks. Building upon Kimi-VL, we introduce an advanced long-thinking variant: Kimi-VL-Thinking-2506. Developed through long chain-of-thought (CoT) supervised fine-tuning (SFT) and reinforcement learning (RL), the latest model exhibits strong long-horizon reasoning capabilities (64.0 on MMMU, 46.3 on MMMU-Pro, 56.9 on MathVision, 80.1 on MathVista, 65.2 on VideoMMMU) while obtaining robust general abilities (84.4 on MMBench, 83.2 on V* and 52.8 on ScreenSpot-Pro). With only 

</details>

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>从一张 4K 文档图开始，画出像素、patch、视觉 token、projector、语言 token 的完整形状链。</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
