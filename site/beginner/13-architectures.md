---
title: 第 11 课 大模型架构总览
description: Encoder-only、Encoder–Decoder、Decoder-only 与状态空间模型
---

# 第 11 课　大模型架构总览：三类骨架怎样选

<div class="lesson-lead">BERT、T5、GPT 不是三个大小不同的同类产品。它们让 token “能看见谁”、把输入和输出放在哪里都不同，因此天然擅长的任务也不同。</div>

::: info 本课资料地图：先比较接口，再进入各自论文
- 三类预训练骨架：[CS224N · Pretraining Slides](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture07-pretraining.pdf)；
- 现代 Decoder 与超参数：[Stanford CS336 Lecture 3](https://stanford-cs336.github.io/spring2026/)；
- Attention 替代路线与 SSM/MoE：[Stanford CS336 Lecture 4](https://stanford-cs336.github.io/spring2026/)；
- 原始论文：[BERT](https://arxiv.org/pdf/1810.04805.pdf)、[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)、[Mamba](https://arxiv.org/pdf/2312.00752.pdf)。

第一次学习只回答“可见性、输入输出接口、历史状态”三个问题；BERT、Encoder–Decoder 与 Decoder/SSM 会在后面三课分别展开。
:::

<figure class="teaching-figure">
  <img src="/illustrations/foundations-architectures.webp" alt="双向阅读大厅、阅读写作双楼和因果生成高塔三类模型架构对照">
  <figcaption>左：一起读懂输入；中：先读输入再条件生成；右：把输入和输出接成一条因果序列。下方是固定状态递推的另一条路线。</figcaption>
</figure>
<div class="visual-key"><div><b>Encoder-only</b>所有输入位置互相看，擅长理解与表示。</div><div><b>Encoder–Decoder</b>读者与作者分工，擅长输入到输出的转换。</div><div><b>Decoder-only</b>只看左侧并不断续写，适合统一生成接口。</div></div>

## 1. 为什么“大数据 + 大模型”会改变能力

模型参数增加，提供更大的函数容量；数据增加，提供更多语言、事实和任务模式；计算增加，才有机会把两者真正训练起来。三者缺一不可：只有大模型而数据不足会过拟合，只有大数据而模型太小会欠拟合，预算不匹配则训练不充分。

### 1.1 能力增强与能力扩展

- **能力增强**：原本会的任务持续变好，例如语言建模 loss 降低、分类准确率提高；
- **能力扩展**：达到一定规模后，一些复杂任务从接近随机变为明显可用，例如 few-shot、代码和多步推理。

所谓“涌现”要谨慎解释。离散指标、评分阈值和对数坐标会让平滑进步看起来突然跳变；也有任务确实在规模后出现组合能力。判断时应看连续指标、多种模型规模与重复实验，不能只看一个点。

### 1.2 Scaling 不告诉你架构可以不选

扩大规模不会自动解决 Attention 长度成本、KV Cache、专家通信或训练不稳定。架构决定资源怎样流动，数据决定学到什么，后训练决定行为如何呈现，系统决定能力能否服务用户。

## 2. 先学会读 Attention mask

模型架构看似复杂，初学时先问一个问题：**当前 token 可以看见哪些位置？**

假设序列有 4 个位置，行表示“谁在看”，列表示“能看谁”：

```text
双向可见                    因果可见
    1  2  3  4                  1  2  3  4
1   ✓  ✓  ✓  ✓              1   ✓  ·  ·  ·
2   ✓  ✓  ✓  ✓              2   ✓  ✓  ·  ·
3   ✓  ✓  ✓  ✓              3   ✓  ✓  ✓  ·
4   ✓  ✓  ✓  ✓              4   ✓  ✓  ✓  ✓
```

双向可见适合“整句已经给你，请理解它”；因果可见适合“右边还没写，请继续生成”。

PyTorch 的布尔 mask 中，`True` 表示“这个位置不允许看”：

```python
import torch

T = 4
bidirectional_mask = torch.zeros(T, T, dtype=torch.bool)
causal_mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool), diagonal=1
)

print(bidirectional_mask.int())
print(causal_mask.int())
```

Encoder-only 通常用第一张全零 mask；自回归 Decoder 用第二张上三角 mask。Padding mask 是另一件事：它负责挡住补齐长度的 PAD token，不要与 causal mask 混为一谈。

## 3. Encoder-only：先把整段读懂

代表思路是 BERT。输入已经完整存在，每个位置可以看左右两边：

> 小王把银行卡交给了小李，因为**他**要去缴费。

理解“他”指谁，需要同时看前后文。双向 Attention 很自然。

### 3.1 怎样训练

BERT 的经典目标是 Masked Language Modeling：遮住一部分 token，让模型根据左右文恢复它们。

```text
原句：今天天气很好
输入：今天 [MASK] 很好
目标：恢复“天气”
```

### 3.2 怎样做任务

Encoder 最终得到每个 token 的上下文表示。再加一个小任务头：

- 文本分类：取整句表示，输出类别；
- 命名实体识别：每个 token 输出标签；
- 语义检索：把文本变成向量并比较相似度；
- 抽取式问答：指出答案在原文中的起止位置。

它不是不能生成，而是“从左到右长文本生成”不是最自然的训练接口。

## 4. Encoder–Decoder：一个读，一个写

代表思路是原始 Transformer、T5、BART。它有两部分：

1. Encoder 双向阅读完整输入；
2. Decoder 因果生成输出，并通过 Cross-Attention 查询 Encoder 的表示。

翻译例子：

```text
Encoder 输入：我喜欢机器学习。
Decoder 已写：I enjoy machine
Decoder 下一步：一边看已写英文，一边回头查询中文表示 → learning
```

这种“输入”和“输出”分开的结构适合条件生成：翻译、摘要、改写、语音识别等。

## 5. Decoder-only：把一切接成一条序列

GPT、LLaMA 以及许多现代通用大语言模型采用 Decoder-only。它把任务说明、输入、答案都串在一起：

```text
任务：把中文翻译成英文
输入：我喜欢机器学习
输出：I enjoy machine learning
```

训练目标统一为“预测下一个 token”。问答、翻译、代码、工具调用都能改写成续写问题。这种统一接口容易随着模型、数据和计算规模扩大，成为通用生成模型的主流选择。

### 5.1 名字为什么容易误导

Decoder-only 的“Decoder”来自 Encoder–Decoder 架构中的生成端。它并不是“只负责把密文解码”，也不是“没有理解能力”。在预测下一个 token 的训练中，它同样必须学语义、事实、格式和推理模式。

## 6. 三类架构一张表

| 问题 | Encoder-only | Encoder–Decoder | Decoder-only |
|---|---|---|---|
| 输入内部 | 双向可见 | Encoder 双向 | 因果可见 |
| 输出怎样产生 | 通常接任务头 | Decoder 逐 token | 逐 token 续写 |
| 输入输出是否分开 | 通常无独立输出序列 | 明确分开 | 常串为一条序列 |
| 典型强项 | 分类、抽取、向量表示 | 翻译、摘要、受控转换 | 开放生成、对话、代码、Agent |
| 常见代表 | BERT | T5、BART | GPT、LLaMA |

::: tip 选型口诀
要“读懂并标注”先想 Encoder；要“把 A 变成 B”先想 Encoder–Decoder；要“统一成提示后继续写”先想 Decoder-only。真实系统还要考虑模型质量、数据、延迟和生态，不能只按口诀决定。
:::

<TransformerArchitecture />

<figure class="teaching-figure source-figure"><a href="/lectures/images/transformer-architecture.png" target="_blank"><img src="/lectures/images/transformer-architecture.png" alt="Stanford CS336 的现代 causal Transformer 架构和单个 Pre-Norm Block 张量图"></a><figcaption>Stanford CS336 Lecture 1/3 的现代 Decoder-only 实现图。左边从 token/位置 embedding 进入多层 block 再回到词表概率，右边放大 Pre-Norm、Causal Self-Attention、FFN 与 residual。它是三类架构中的一种具体实现，不代表 BERT 或 T5 也使用完全相同的 mask 与接口。<a href="https://stanford-cs336.github.io/spring2026/">打开 Lecture 3 PDF</a>。</figcaption></figure>

## 7. 非 Transformer 路线：不保存全部历史行不行

标准 Attention 需要比较许多 token 对，长序列代价高。状态空间模型（SSM）尝试用固定大小状态递推：

$$h_t=A_th_{t-1}+B_tx_t,\qquad y_t=C_th_t$$

可以把 $h_t$ 想成“随序列更新的工作记忆”。训练时有些形式能改写为并行扫描或卷积；推理时只维护状态，不必保存所有 token 的 K/V。

### 7.1 Mamba、RWKV、TTT 在尝试什么

- **RWKV** 把类似 Attention 的加权聚合改写成可递推形式，兼顾训练并行和推理递归。
- **Mamba** 让状态更新参数随输入变化，重要内容选择性写入；同时强调适合硬件的扫描实现。
- **TTT** 在测试时用当前序列继续更新隐藏模型或快速权重，把“读上下文”部分转化为“在线学习”。

固定状态带来线性序列代价，但也可能形成信息瓶颈。它不是“全面取代 Transformer”的简单故事，而是计算、记忆、表达能力之间另一种交换。

## 8. K3 在家族图的哪里

K3 不是“纯 Transformer”与“纯 SSM”二选一：

- 某些层使用 MLA，保留内容寻址能力并压缩 KV；
- 大量层使用 KDA，以固定状态进行可擦写的线性记忆；
- 模型深处还有 AttnRes，让当前层查询历史层；
- FFN 侧使用稀疏 MoE 扩大参数容量。

所以现代架构常是**组件级组合**：序列轴、深度轴、通道轴分别选择适合的机制。

## 9. 四个容易混淆的说法

- **“BERT 看见未来，所以作弊。”** 它做理解任务时输入本来就完整；只有自回归生成训练才禁止看未来。
- **“Decoder-only 没有 Encoder，所以不能理解输入。”** 输入同样被多层因果 Attention 处理，只是可见性和接口不同。
- **“线性时间一定更快。”** 还取决于并行度、内存访问、内核实现、批量和硬件。
- **“架构决定全部能力。”** 数据质量、训练目标、规模、后训练和工具系统同样关键。

<ConceptCheck question="机器翻译为什么天然适合 Encoder–Decoder？" :options="['因为它不需要 Attention','因为输入可完整双向编码，输出再条件式逐词生成','因为它只训练一个分类标签']" :answer="1" explanation="Encoder 专门读完整源句，Decoder 同时看已生成目标词并通过 Cross-Attention 查询源句。" />

## 10. 动手画一次

请为“把一段客服邮件分类成 8 个工单类别”和“把会议录音转成摘要”各画一张输入输出图。先不选具体模型，只回答：

1. 输入是否完整可见？
2. 输出是一个标签，还是一串新 token？
3. 是否需要逐步生成？
4. 是否需要频繁查询原始输入？

> 本课对应原书第 2 章（PDF 第 40–103 页），将原书的模型家族重新组织为“可见性—接口—任务”的选择框架。

<ChapterReadings lesson="13-architectures" />
