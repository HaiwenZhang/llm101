---
title: 第 09 课 Encoder-Decoder、T5 与 BART
description: 理解 Cross-Attention、条件生成、Text-to-Text 与去噪预训练
---

# 第 09 课　Encoder–Decoder、T5 与 BART

<div class="lesson-lead">Encoder 像读者，把完整输入整理成记忆；Decoder 像作者，一边看已经写出的内容，一边通过 Cross-Attention 回查原文。这是“把一种序列转换成另一种序列”的自然骨架。</div>

::: info 本课资料地图
- 历史起点：[CMU L04 · Recurrent Neural Networks](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-04-rnns.pdf)第 41–52 页与 [CS224N L04](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture04-rnnlm.pdf)第 51–58 页，从条件语言模型、固定 context bottleneck 推到逐步 Attention；
- 课程讲解：[CS224N · Pretraining Slides](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture07-pretraining.pdf)把 Decoder、Encoder、Encoder–Decoder 三种预训练方式放在一起比较；
- 架构原图：[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)；
- 训练细节：[CS224N Transformer Notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf)。

T5/BART 是后续预训练配方；Cross-Attention 与左右两塔的数据来源仍以原始 Transformer 为基准理解。
:::

<figure class="teaching-figure">
  <img src="/illustrations/transformer-reader-writer.webp" alt="Encoder 阅读完整输入，Decoder 查询编码记忆并逐 token 写出结果">
  <figcaption>左边先完成阅读；右边每写一个 token，都能查询左边全部输入表示，也只能看自己已经写出的前缀。</figcaption>
</figure>
<div class="visual-key"><div><b>Encoder Self-Attention</b>源文本内部双向理解。</div><div><b>Cross-Attention</b>Decoder 的 Query 查询 Encoder 的 Key/Value。</div><div><b>Decoder Self-Attention</b>目标文本只能看已经生成的左侧。</div></div>

## 1. 先把任务写成条件语言模型

源序列记作 $x=(x_1,\ldots,x_S)$，目标序列记作 $y=(y_1,\ldots,y_T)$。翻译、摘要和改写都不是建模独立的 $p(y)$，而是建模：

$$
p(y\mid x)=\prod_{t=1}^{T+1}p(y_t\mid y_{<t},x)
$$

最后一项仍要预测 `EOS`。式子表达了两个条件：

1. $y_{<t}$：已经写出的目标前缀，决定语法、风格和前后连贯；
2. $x$：完整源序列，决定应该翻译或摘要什么内容。

因此 Encoder–Decoder 中的 Decoder 本质上仍是语言模型，只是每一步都额外以 $x$ 为条件。[CS224N L04 第 56–58 页](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture04-rnnlm.pdf#page=56)把 NMT 明确写成 conditional language model，并给出同一分解。

训练集是一组平行样本 $(x^{(n)},y^{(n)})$。最大似然目标是：

$$
\mathcal L=-\sum_n\sum_t\log p_\theta
\left(y_t^{(n)}\mid y_{<t}^{(n)},x^{(n)}\right)
$$

这解释了为什么训练需要把目标右移一格，也解释了为什么仅有大量单语文本不能直接监督“这句中文对应哪句英文”：条件关系 $x\rightarrow y$ 必须来自平行数据、合成数据或别的弱监督过程。

## 2. 固定 context vector 为什么会成为瓶颈

最早的 RNN Encoder 逐词读入源句，只把最终隐藏状态交给 Decoder：

$$
h_i^{enc}=f_{enc}(h_{i-1}^{enc},E_x[x_i]),
\qquad c=h_S^{enc}
$$

Decoder 再把同一个 $c$ 用于所有目标时间步：

$$
s_t=f_{dec}(s_{t-1},E_y[y_{t-1}],c),
\qquad p(y_t)=\operatorname{softmax}(W_os_t)
$$

$c$ 可以作为 Decoder 初始状态、拼到每一步输入，或参与输出层；实现位置可以不同，结构限制相同：**所有源信息必须穿过一条固定宽度通道。**

设源句从 8 个 token 增长到 80 个，$c\in\mathbb R^d$ 的宽度并不会增长。模型必须在同样的 $d$ 个数字里同时保存人物、动作、否定、数字、词序与长距离指代。长句并非理论上绝对无法编码，而是优化和有限容量越来越困难：

- 早期 token 到最终状态要经过更多次递归，梯度更弱；
- 多个细节争抢同一状态容量，Decoder 无法回查原位置；
- Decoder 第 1 步和第 20 步收到的是同一个 $c$，不能按当前需要换一份摘要。

这就是 [CMU L04 第 44–48 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-04-rnns.pdf#page=44)和 [CS224N L04 第 57 页](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture04-rnnlm.pdf#page=57)所说的 conditioning bottleneck。

## 3. RNN Attention：每写一步，重新读一次源句

Attention 不再丢弃 Encoder 的中间状态，而是保留 $h_1^{enc},\ldots,h_S^{enc}$。Decoder 在第 $t$ 步做三件事：

$$
e_{t,i}=\operatorname{score}(s_{t-1},h_i^{enc})
$$

$$
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_{j=1}^{S}\exp(e_{t,j})},
\qquad
c_t=\sum_{i=1}^{S}\alpha_{t,i}h_i^{enc}
$$

`score` 回答“当前 Decoder 状态与第 $i$ 个源位置有多匹配”；softmax 让一整行权重非负且和为 1；加权和 $c_t$ 是这一解码步专用的源信息。下一步 $s_t$ 改变，分数、权重和 context 都会重新计算。

<figure class="teaching-figure concept-figure"><img src="/illustrations/encoder-decoder-attention-evolution.svg" alt="Encoder-Decoder 从单一固定 context vector 演化为每个解码步骤重新计算 Attention 权重"><figcaption>固定向量方案要求所有源信息穿过同一个瓶颈；Attention 保留每个源位置，并让 Decoder 每一步获得不同的加权 context。图中强调的是“重新读取”，不是只给旧模型多接一层。</figcaption></figure>

### 3.1 三种打分函数在学什么

| 名称 | 公式 | 参数与限制 | 直觉 |
|---|---|---|---|
| Dot product | $s^\top h_i$ | 无额外矩阵；两边维度要相同 | 直接比较方向是否一致 |
| Bilinear | $s^\top Wh_i$ | 学一张 $W$；可先变换 Key 空间 | 学“哪类 Decoder 特征匹配哪类源特征” |
| Additive / Bahdanau | $v^\top\tanh(W_ss+W_hh_i)$ | 参数更多；两边维度可不同 | 让小网络学习任意非线性匹配 |

Bahdanau Attention 的关键贡献不是某个公式永远最好，而是让对齐权重参与端到端训练，Decoder 无需先得到人工词对齐。[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)是这一转折的原论文。

### 3.2 手算一次 softmax 和 context

假设三个源位置的分数是 `[1, 2, 0]`：

$$
\operatorname{softmax}([1,2,0])
\approx[0.245,0.665,0.090]
$$

若三个 Encoder 状态为 $h_1=[1,0]$、$h_2=[0,1]$、$h_3=[1,1]$，那么：

$$
c_t=0.245[1,0]+0.665[0,1]+0.090[1,1]
=[0.335,0.755]
$$

它不是挑出一个词后丢掉其他词，而是软性混合全部 Value；权重大的位置贡献更多。若下一步分数变成 `[0, 1, 3]`，新的 context 会明显偏向 $h_3$。这就是“每一步重新读”的数值含义。

::: warning Attention 权重不是可靠的因果解释
高权重说明当前前向计算给这个位置较大系数，不自动证明该词是模型输出的唯一原因。多层变换、残差、多个 head 与 Value 内容都会影响结果。把 Attention 图当作对齐或软检索线索可以，把它当作完整因果解释则过度了。
:::

## 4. 三种 Attention 不要混

<figure class="teaching-figure concept-figure"><img src="/illustrations/cross-attention-sparse.webp" alt="Decoder 当前 Query 跨到 Encoder 的源文本记忆中选择信息并生成下一词"><figcaption>Cross-Attention 的“跨”指目标端 Query 跨到源端 memory 里检索；它不同于两边各自内部的 Self-Attention。</figcaption></figure>

设源句为中文，目标句为英文：

| 模块 | Q 来自 | K/V 来自 | 可见性 |
|---|---|---|---|
| Encoder Self-Attention | 中文位置 | 中文位置 | 中文内部全可见 |
| Decoder Self-Attention | 已写英文 | 已写英文 | 因果遮罩 |
| Cross-Attention | Decoder 当前表示 | Encoder 中文表示 | 可查全部源句 |

Cross-Attention 的作用不是把两边向量简单相加，而是让目标端每一步按内容检索源端信息。

公式与 Self-Attention 外形相同，但 Q/K/V 的来源不同：

$$
\operatorname{CrossAttn}(Q_d,K_e,V_e)
=\operatorname{softmax}\left(\frac{Q_dK_e^\top}{\sqrt{d_k}}\right)V_e
$$

- $Q_d\in\mathbb R^{T_d\times d_k}$：Decoder 当前 $T_d$ 个目标位置提出的问题；
- $K_e,V_e\in\mathbb R^{T_e\times d_k}$：Encoder 的 $T_e$ 个源位置提供索引与内容；
- 分数表形状是 $T_d\times T_e$：每个目标位置对每个源位置分配权重；
- Cross-Attention 通常不需要因果 mask，因为完整源句已经可见；Decoder Self-Attention 仍必须因果遮罩。

<figure class="teaching-figure source-figure"><a href="/paper-figures/attention-transformer-figure-1.webp" target="_blank"><img src="/paper-figures/attention-transformer-figure-1.webp" alt="Attention Is All You Need Figure 1，左侧 Encoder 与右侧 Decoder 由中间 Cross-Attention 相连"></a><figcaption>《Attention Is All You Need》Figure 1（PDF p.3）。Decoder 中间那层 Multi-Head Attention 的 K/V 来自左侧 Encoder 顶部，Q 来自 Decoder 下方；这是两塔之间唯一的内容检索桥。<a href="https://arxiv.org/pdf/1706.03762.pdf#page=3">打开原论文第 3 页</a>。</figcaption></figure>

### PyTorch：Q 来自目标端，K/V 来自源端

```python
import torch
import torch.nn as nn

B, T_source, T_target, hidden = 2, 7, 4, 32
encoder_memory = torch.randn(B, T_source, hidden)
decoder_state = torch.randn(B, T_target, hidden)

cross_attention = nn.MultiheadAttention(
    embed_dim=hidden, num_heads=4, batch_first=True
)
output, weights = cross_attention(
    query=decoder_state,
    key=encoder_memory,
    value=encoder_memory,
)
print(output.shape)   # [2, 4, 32]
print(weights.shape)  # [2, 4, 7]：目标位置 × 源位置
```

若把三项都传 `decoder_state`，就变成 Decoder Self-Attention；若三项都传 `encoder_memory`，就是 Encoder Self-Attention。Cross 的含义完全体现在来源跨越两塔。

## 5. 训练为什么仍可并行

训练时目标句已知：

```text
Decoder 输入：[BOS] I like machine learning
监督目标：     I like machine learning [EOS]
```

整句右移一位，因果 mask 阻止位置看到自己的答案。所有目标位置可以一次并行算 loss，这叫 Teacher Forcing。

推理时真实下一词未知，只能把刚生成 token 接回去，逐步生成。因此训练并行、推理串行不矛盾。

一次训练通常同时需要四种“可见性账”：

| 位置 | 能看什么 | 常用 mask |
|---|---|---|
| Encoder 的真实源 token | 全部真实源 token | 只屏蔽 source padding |
| Decoder 位置 $t$ | 目标位置 $\le t$ | causal mask + target padding mask |
| Cross-Attention | 全部真实 Encoder 位置 | source padding mask |
| Loss | 只统计真实目标与 EOS | 忽略 target padding，常也忽略纯提示位置 |

因果 mask 只管“不能偷看未来”，padding mask 只管“补齐 batch 的空位不是内容”。把两者混成一个词，会很难排查模型为何在 padding 上分配注意力或为何 loss 统计错了。

训练时第 $t$ 步总是看到真实 $y_{t-1}$；推理时它看到模型自己生成的 $\hat y_{t-1}$。如果前面错一次，后续条件分布就落到训练中较少见的前缀，这就是 exposure bias。Teacher Forcing 让训练高效稳定，却没有保证模型在自己的错误轨迹上也稳健。

## 6. 翻译的最小数据流

```text
源句：我 喜欢 机器 学习
  ↓ Encoder 双向表示
[h1, h2, h3, h4]

目标前缀：I enjoy machine
  ↓ Decoder Self-Attention
当前 Query
  ↓ Cross-Attention 查询 h1..h4
输出 learning 的概率最高
```

早期解码错误会成为后续输入，形成 exposure bias；Beam Search、数据增强和序列级训练等方法尝试缓解。

生成还要处理两个经常被“Beam Search 更强”掩盖的问题：

1. **长度偏置**：序列概率是许多小于 1 的数相乘，log 概率不断累加负数；若不作长度归一化，短序列经常占便宜；
2. **EOS 校准**：模型过早给 EOS 高概率会漏译，太晚给高概率会重复或拖长。Beam 只是在候选树里搜索，不会自动修复分布本身的校准错误。

翻译常用长度惩罚后的分数，例如 $\log p(y\mid x)/T^\alpha$。这不是概率论恒等式，而是任务层启发式；报告结果时必须说明 beam width、长度惩罚和最大长度。

## 7. T5：把所有任务写成 Text-to-Text

T5 的统一思想：输入和输出都用文本表达。

```text
translate English to German: That is good. → Das ist gut.
summarize: <article> → <summary>
cola sentence: The course is jumping well. → unacceptable
```

分类不再接固定类别头，也生成标签文本。统一接口便于多任务训练，但标签拼写、解码与格式也成为评测的一部分。

## 8. T5 的 span corruption

不是独立遮每个词，而是连续遮住若干片段，用 sentinel token 标记：

```text
原文：今天 下午 我们 去 公园 散步
输入：今天 <X> 去 公园 <Y>
目标：<X> 下午 我们 <Y> 散步
```

Encoder 读被破坏的文本，Decoder 生成被删片段。它让模型学习较长跨度的恢复，也让 Decoder 在预训练中真正参与生成。

不同缺失片段使用不同 sentinel，例如 `<extra_id_0>`、`<extra_id_1>`。目标只输出被删片段和 sentinel，而不是逐字复写整段原文，因此在相同输入长度下，Decoder 目标通常更短、训练更省。span 长度分布与 corruption rate 会改变任务难度：遮得太少近似抄写，遮得太多则源端证据不足。经典细节来自 [T5 原论文](https://arxiv.org/abs/1910.10683)。

## 9. BART：把被破坏的文本复原

BART 采用双向 Encoder 和自回归 Decoder，使用多种去噪任务：

- token masking；
- token deletion；
- 文本片段遮罩；
- 句子顺序打乱；
- 文档旋转等。

输入是“被弄乱的文章”，目标是恢复原文。因为预训练就是序列到序列去噪，它很适合摘要、生成和改写，也可用于理解任务。

BART 的训练目标仍是 $-\log p(x\mid \tilde x)$：$\tilde x$ 是经过噪声函数破坏的输入，$x$ 是原文。不同噪声迫使模型学习不同关系：token deletion 甚至不给出缺失位置，难于普通 mask；sentence permutation 强迫模型恢复篇章顺序；text infilling 用一个 mask 代表一整段，迫使 Decoder 估计缺失长度。原论文还发现，并非把所有噪声叠得越多越好，预训练任务必须与下游需求和计算成本一起比较。参见 [BART 原论文](https://arxiv.org/abs/1910.13461)。

## 10. T5 与 BART 的共同点和差异

| | T5 | BART |
|---|---|---|
| 骨架 | Encoder–Decoder | Encoder–Decoder |
| 核心统一 | 所有任务 Text-to-Text | 去噪序列到序列 |
| 典型破坏 | span corruption + sentinel | 多种噪声函数 |
| 使用重点 | 多任务统一迁移 | 生成、摘要、去噪 |

具体版本会调整数据、规模、位置编码和训练配方，表格只说明经典思想。

## 11. Encoder–Decoder 的成本直觉

需要分别保存 Encoder 和 Decoder 层；生成时 Encoder 只运行一次，Decoder 每步运行并查询 Encoder memory。若输入很长、输出很短，Encoder 读入成本占主导；若输出很长，Decoder 自回归成本占主导。

对源长度 $S$、目标长度 $T$，忽略隐藏维常数后，标准 Transformer 三部分 Attention 的分数矩阵规模约为：

$$
\underbrace{O(S^2)}_{\text{Encoder Self-Attention}}
+\underbrace{O(T^2)}_{\text{Decoder Self-Attention}}
+\underbrace{O(ST)}_{\text{Cross-Attention}}
$$

推理时 Encoder memory 只算一次，并可预先投影出 Cross-Attention 的 K/V；Decoder Self-Attention 则需要随着输出增长维护目标端 KV Cache。若 $S=10{,}000$、$T=100$，源端读入的 $S^2$ 往往更重；若 $S=50$、$T=5{,}000$，目标端自回归和 $T^2$ 读写更突出。只说“Encoder–Decoder 参数多一塔”还不足以判断真实延迟，必须同时写出输入/输出长度。

## 12. 为什么通用聊天模型更多是 Decoder-only

Decoder-only 把指令、资料、答案接为一条序列，接口统一，训练语料容易混合并随规模扩展。Encoder–Decoder 仍在翻译、摘要、语音、多模态转换等任务有清晰优势，不是“被淘汰”，而是通用聊天的生态选择不同。

二者也不是能力上的严格包含关系。Decoder-only 可以把源和目标拼成 `[source, separator, target]`，但源 token 通常受因果可见性限制，不能像双向 Encoder 那样彼此完整交互；Encoder–Decoder 能先双向整理源文本，却要维护两套层、两类缓存与 Cross-Attention 路径。选择应围绕训练数据形态、输入输出长度、是否需要强源端理解和部署栈，而不是按“新旧架构”排队。

<ConceptCheck question="Cross-Attention 中，Decoder 当前状态通常扮演什么？" :options="['Query，用来查询 Encoder 产生的 Key/Value','完整训练标签','只负责位置编码']" :answer="0" explanation="目标端提出当前需要的信息，源端提供可匹配的索引与内容。" />

## 13. 纸上练习

画出“把 500 字新闻压缩成 50 字摘要”的三条数据流：Encoder 输入、Decoder 已生成前缀、Cross-Attention 查询。标出哪些计算只做一次，哪些每生成一个 token 都重复。

> 本课对应原书第 2.4 节（PDF 第 66–72 页），详细展开了 T5、BART 与 Encoder–Decoder 的训练和生成过程。

<ChapterReadings lesson="15-encoder-decoder" />
