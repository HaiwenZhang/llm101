---
title: 第 08 课 Encoder-only 与 BERT
description: 从双向可见、MLM 数据构造到分类、抽取与语义表示
---

# 第 08 课　Encoder-only 与 BERT：把整段读懂

<div class="lesson-lead">BERT 的核心不是“一个比 GPT 旧的模型名”，而是一套面向完整输入的表示学习接口：双向 Encoder 让每个位置同时利用左文与右文；MLM 从无标签文本制造监督；下游任务再从 `[CLS]`、每个 token 或一段 span 读取答案。</div>

::: info 本课资料地图
- 架构与图解：[台大 ADL · BERT](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250908_BERT.pdf)和 [BERT Variants](https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/240918_BERTVariants.pdf)；
- 预训练主线：[CS224N Pretraining 第 9–33 页](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture07-pretraining.pdf#page=9)与 [CMU ANLP L06 第 2–14 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-06-pretraining.pdf#page=2)；
- 原论文：[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/pdf/1810.04805.pdf)。
:::

## 1. 从静态词向量到上下文表示

静态 embedding 查表时，“苹果”无论出现在哪里都从同一个初始向量出发：

```text
我吃了一个苹果。       → 水果
苹果发布了新系统。     → 公司
```

BERT 仍然先查 token embedding，但多层双向 Self-Attention 会把周围词混进每个位置。最终得到的不是一个永远不变的“苹果向量”，而是：

$$
H=\operatorname{Encoder}(X),\qquad H\in\mathbb R^{B\times T\times d}
$$

同一个 token id 可以产生不同的 $H_{b,t,:}$。这就是“contextual representation”的精确含义。

<figure class="teaching-figure"><img src="/illustrations/foundations-architectures.webp" alt="双向 Encoder 与其他模型架构的阅读方向对照"><figcaption>本课聚焦图中的双向阅读大厅：输入已经完整到齐，每个位置都能向左、向右查信息。它适合表示与理解，但没有天然的逐 token 生成接口。</figcaption></figure>

## 2. Encoder-only 的可见性

对完整输入，普通 Encoder 不使用 causal mask。若位置 $i,j$ 都不是 padding，Attention logit 不因方向被屏蔽：

$$
M_{ij}=0
$$

这不叫“偷看答案”。分类、实体识别或检索任务一开始就拿到了整段文本，右侧 token 本来就是输入的一部分。真正需要屏蔽的是 padding 等无效位置。

### 三种 mask 不要混用

| 名称 | 作用在哪里 | 回答的问题 |
|---|---|---|
| causal mask | Attention logits | 当前 token 能否看未来位置 |
| padding/attention mask | Attention logits | `[PAD]` 是否参与上下文混合 |
| MLM loss mask | 交叉熵 labels | 哪些位置直接产生预训练监督 |

BERT 通常没有 causal mask；它有 padding mask；MLM 又另外只在少量位置计算 loss。**未产生 loss 的 token 仍会参与 Encoder 计算。**

## 3. BERT 输入的四个张量

经典 BERT 把三种 embedding 相加：

$$
x_i=e_i^{token}+e_i^{position}+e_i^{segment}
$$

- `input_ids [B,T]`：WordPiece token id；
- `token_type_ids [B,T]`：句子 A 或 B，查 segment embedding；
- `position_ids [B,T]`：绝对位置编号；
- `attention_mask [B,T]`：真实 token 为 1，padding 为 0。

特殊 token 的典型组织是：

```text
[CLS] 句子 A [SEP] 句子 B [SEP] [PAD] [PAD]
  A      A       A      B       B     PAD   PAD
```

`[CLS]` 只是一个可学习的特殊位置。它能否成为好的整句表示，取决于预训练与下游目标是否真的迫使信息汇聚到这里；符号本身没有魔法。`[SEP]` 用来分隔序列，segment id 则告诉模型归属。

## 4. MLM 是怎样从无标签文本制造标签的

给定原序列 $x$，先随机采样位置集合 $\mathcal M$，再用 corruption 过程得到 $\widetilde x$。Masked Language Modeling 目标是：

$$
\mathcal L_{MLM}
=\mathbb E_{\mathcal M\sim q(\mathcal M\mid x)}
\left[-\sum_{t\in\mathcal M}\log p_\theta(x_t\mid \widetilde x)\right]
$$

注意 target 是污染前的 $x_t$，模型输入是污染后的 $\widetilde x$。经典 BERT 随机选约 15% token 位置；在这些选中位置里：

- 80% 换成 `[MASK]`，相当于全序列位置的期望 12%；
- 10% 换成随机 token，期望 1.5%；
- 10% 保持原 token，期望 1.5%。

无论如何呈现给模型，**选中位置的 label 始终是原 token**。保持原词的 10% 仍计算 loss；随机词也不是新 target。

<figure class="teaching-figure concept-figure"><img src="/illustrations/bert-mlm-objective.svg" alt="BERT MLM 从 15% 位置采样、80/10/10 污染到稀疏交叉熵的完整流程"><figcaption>三张清单要分开保存：原序列提供 target，污染序列送入 Encoder，labels 决定哪些位置计算 loss。直接拿污染后的 token 当标签是实现错误。</figcaption></figure>

### 一批数据的监督密度

若 batch size 为 32、序列长 128，总共有 $32\times128=4096$ 个 token 位置。按 15% 采样，产生直接 MLM 监督的位置期望为：

$$
4096\times0.15=614.4
$$

其中期望约 491.5 个显示 `[MASK]`，61.4 个显示随机 token，61.4 个保持原词。实际必须是整数并会随随机采样波动。关键是：Encoder 仍为 4,096 个位置做完整前向，而直接 CE 监督只有约 614 个位置。这也是后来研究更密集预训练信号的原因之一。

<MLMLab />

## 5. 从 logits 到 MLM loss 的张量形状

Encoder 输出 $H\in\mathbb R^{B\times T\times d}$，MLM head 把每个位置映射到词表：

$$
Z=HW_{vocab}+b,\qquad Z\in\mathbb R^{B\times T\times V}
$$

训练代码常把未选中位置的 label 设为 `-100`：

```python
import torch
import torch.nn.functional as F

# logits: [B,T,V]；labels: [B,T]
logits = torch.randn(2, 6, 100)
labels = torch.full((2, 6), -100)
labels[0, 2] = 17   # 只监督这两格的原 token id
labels[1, 4] = 63

loss = F.cross_entropy(
    logits.reshape(-1, 100),
    labels.reshape(-1),
    ignore_index=-100,
)
```

`ignore_index=-100` 只影响 loss 汇总，不会让那一格从 Encoder 前向中消失。若错误地把所有位置都设成原 token label，模型会获得大量“看见自己再抄自己”的捷径，目标已经不是经典 MLM。

## 6. MLM 不是一个自然的从左到右生成概率

自回归模型用链式法则定义完整联合分布：

$$
p(x)=\prod_{t=1}^{T}p(x_t\mid x_{<t})
$$

MLM 学的是在不同污染上下文下恢复若干位置的条件概率。多个被遮位置通常由输出头在给定同一个污染输入时分别预测；它没有直接给出一个天然、规范化的从左到右 $p(x)$。因此：

- MLM 很适合学习双向表示；
- 可以反复遮罩、采样来生成，但过程不如 Decoder 自然高效；
- MLM loss 不能直接与自回归 NLL 当成完全同口径的序列概率比较。

把它理解为 denoising / pseudo-likelihood 风格目标更准确。

## 7. 为什么不是把 15% 全换成 `[MASK]`

下游输入通常没有 `[MASK]`，若预训练选中位置全部显示它，模型会过度依赖这个特殊符号。随机替换和保持原词的少量分支试图缩小“预训练看到的输入”与“微调看到的输入”之间的差异。

但 80/10/10 不是不可更改的自然规律：

- dynamic masking 可在不同 epoch 为同一文本重新采样位置；
- whole-word masking 避免只遮一个词的部分子词；
- span masking 连续遮一段，让恢复任务更依赖长上下文；
- ELECTRA 改为判断每个位置是否被生成器替换，让更多位置产生监督。

这些变化必须在相同数据量、计算量和模型规模下消融，否则无法判断收益来自目标还是更大训练预算。

## 8. NSP 做了什么，又为什么后来常被移除

经典 BERT 的输入由两个片段 A、B 组成，并增加 Next Sentence Prediction：判断 B 是 A 的真实后续，还是随机抽取的片段。其目的不是预测下一个 token，而是训练句间关系。

后来 RoBERTa 等工作移除 NSP，并通过更多数据、更长训练、动态遮罩等配方取得更好结果。这不能简化成“NSP 永远有害”，更准确的结论是：原始 BERT 实验里的目标、数据组织与训练步数纠缠在一起，必须靠控制变量实验拆分。ALBERT 等工作也尝试过不同的句间目标。

## 9. 预训练完成后，答案从哪里读出来

同一个 $H\in\mathbb R^{B\times T\times d}$ 可以接不同任务头：

<figure class="teaching-figure concept-figure"><img src="/illustrations/bert-task-heads.svg" alt="BERT 上下文表示连接整句分类、token 分类、抽取式问答与句对 Cross-Encoder 四种任务头"><figcaption>骨架相同，读出接口不同。先写清输出 shape，再决定从 `[CLS]`、每个 token 还是 start/end 位置计算监督。</figcaption></figure>

| 任务 | 读取表示 | 输出 shape | 典型 loss |
|---|---|---|---|
| 整句分类 | `H[:,0,:]` | `[B,C]` | 类别交叉熵 |
| Token 分类 | `H` 的每个有效位置 | `[B,T,C]` | 逐 token CE |
| 抽取式问答 | 每位置 start/end logit | 两个 `[B,T]` | 起点 CE + 终点 CE |
| 句对 Cross-Encoder | `[CLS] A [SEP] B [SEP]` 的聚合表示 | `[B,C]` 或 `[B,1]` | 分类或排序 loss |

### 子词标签怎样对齐

若 WordPiece 把一个词拆成三个子词，NER 标签不能凭空复制而不说明。常见方案是只监督第一个子词，其余设 `-100`；或按 BIO 规则扩展到所有子词。训练与评测必须使用同一对齐方案，否则 token-level 指标不可比。

### 抽取式问答不是自由生成

模型对原文每个位置打 start/end 分数，再选择合法区间。答案可回链原文，但若原文没有答案，需要专门定义 no-answer 位置/阈值；若答案要综合多处或改写，它的接口就不自然。

## 10. Cross-Encoder 与 Bi-Encoder 不要混淆

Cross-Encoder 把问题和候选文档拼在一次输入中，二者 token 能逐层交互，判断精细但每个文档都要重新前向。Bi-Encoder 则分别编码：

$$
q=f_\theta(\text{question}),\qquad d=g_\phi(\text{document}),\qquad s=q^\top d
$$

文档向量可以提前离线建立索引，适合百万级召回；但打分时缺少逐 token 交互。常见检索系统先用 Bi-Encoder 召回，再用 Cross-Encoder 重排。

原始 BERT 的 `[CLS]` 没有专门被训练成“余弦距离可直接表达语义相似度”的通用句向量。现代 embedding 模型通常还经过对比学习、检索数据或蒸馏，不能只取任意 BERT `[CLS]` 就假设效果好。

## 11. 冻结、Fine-tuning 与继续预训练

有三种常见适配路径：

1. **Linear probe**：冻结 Encoder，只训练小任务头。它能测试现有表示是否容易线性读出，也节省显存；
2. **Full fine-tuning**：任务头与 Encoder 一起更新，适配能力更强，但小数据时更容易不稳定或过拟合；
3. **Domain-adaptive pretraining**：先在领域无标签文本上继续 MLM，再用有标签任务微调，适合医学、法律等分布差异大的领域。

Full fine-tuning 时应明确学习率、warmup、batch、epoch、随机种子和 best checkpoint 选择规则。小数据上的一次运行波动可能很大，至少报告多随机种子均值与方差；类别不均衡时不能只看 accuracy。

## 12. BERT-base 与 BERT-large 数字怎样读

原始论文发布两种常见规格：

| 模型 | Encoder 层数 L | 隐藏维 d | Attention 头数 | 约参数量 |
|---|---:|---:|---:|---:|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

更大模型的收益不能脱离预训练数据、步数和下游配方讨论。模型名也不等于唯一 tokenizer：cased/uncased、语言覆盖和 WordPiece 词表会影响输入长度、专名与多语言效果。

## 13. 常见衍生路线解决的不是同一问题

| 路线 | 主要变化 | 它在改哪一层 |
|---|---|---|
| RoBERTa | 更多数据、更长训练、动态遮罩、移除 NSP | 数据与训练配方 |
| SpanBERT | 连续 span corruption | 预训练目标 |
| ALBERT | embedding 分解、跨层参数共享 | 参数效率 |
| DistilBERT | 教师蒸馏到更小学生 | 模型压缩 |
| ELECTRA | replaced-token detection | 监督密度与目标 |

这些模型不能只按发布时间背诵。问“它改了架构、任务、数据还是超参数？”更容易迁移到新论文。CMU L06 第 8 页也把预训练模型的四个主因归纳为 Architecture、Task、Data、Hyper-parameters。

## 14. 预训练学到“知识”，也可能记住数据

大规模预训练可学习句法、词义消歧、常识和一些事实关联；但训练 loss 只要求拟合数据分布，并不保证知识真实、公平或不会泄露。CS224N 课件第 20–22、54–56 页特别提醒：数据来源、版权、隐私、成员推断和有害偏见都是模型能力的一部分风险。

所以模型报告至少要回答：

- 训练数据从哪里来，是否有授权与可追溯记录？
- 做了怎样的过滤、去重与隐私处理？
- 不同语言、领域和群体覆盖是否失衡？
- 下游测试集是否可能出现在预训练语料中？
- 评测的是可泛化能力，还是背诵？

## 15. 调试一条 BERT 训练样本

若 loss 异常，按数据流逐层检查：

1. 保存 `original_input_ids`，确认 target 没被污染覆盖；
2. 打印 `masked_indices`，选中率是否接近预期；
3. 分别统计 `[MASK]` / random / keep 的比例；
4. 检查 `labels`：选中位置是原 id，其余为 `-100`；
5. 检查 padding mask：`[PAD]` 不应成为上下文；
6. 检查 MLM head 输出 `[B,T,V]` 与词表大小；
7. 单独算一个位置的 $-\log p(target)$，与框架 loss 对账；
8. 验证集 corruption 使用固定随机种子，避免每次评估样本难度变化过大。

## 16. 资料逐段精读路线

1. [CS224N 第 9–19 页](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture07-pretraining.pdf#page=9)：从静态词向量走到完整模型预训练；
2. [第 24–33 页](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture07-pretraining.pdf#page=24)：比较 Encoder、Encoder–Decoder、Decoder 的目标与用途；
3. [CMU L06 第 10–14 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-06-pretraining.pdf#page=10)：逐项复述 MLM objective、80/10/10 与分类适配；
4. [BERT 原论文 Figure 1](https://arxiv.org/pdf/1810.04805.pdf#page=3)：把预训练参数怎样流向不同任务头画出来；
5. [CS224N 第 54–56 页](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture07-pretraining.pdf#page=54)：把“学到语言规律”和“记住敏感数据”同时纳入结论。

## 本课闭卷复述

请从一条原句开始，不看笔记讲清：怎样采 15% 位置，80/10/10 分别改变模型输入什么，labels 为什么仍指向原词，为什么未选中位置仍参与 Attention，以及分类、NER、问答三种任务分别从哪个张量读答案。

<ConceptCheck question="某 token 被选为 MLM 监督位置，但落入 10% 的‘保持原词’分支。这个位置是否计算 loss？" :options='["计算，target 仍是原 token", "不计算，因为输入没有变化", "只计算 Attention loss"]' :answer="0" explanation="80/10/10 只决定选中位置怎样呈现给模型；是否计算 MLM loss 由最初的选中集合决定。" />

<ConceptCheck question="为什么把其他未选中位置的 label 设为 −100，不等于把它们从 Encoder 删除？" :options='["−100 只让交叉熵忽略这些位置，它们仍参与双向 Attention", "BERT 会把 −100 当成一个新 token", "因为只有 [CLS] 会参与 Attention"]' :answer="0" explanation="Attention mask 与 loss mask 是不同张量；未监督 token 仍为被遮位置提供上下文。" />

下一课：[T5、BART 与 Encoder–Decoder](/beginner/15-encoder-decoder)。

<ChapterReadings lesson="14-bert" />
