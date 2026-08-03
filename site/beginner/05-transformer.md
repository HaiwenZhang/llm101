---
title: 第 07 课 图解完整 Transformer
description: 从一个 Block 到 Encoder–Decoder，再对照现代大语言模型与 Kimi K3
---

# 第 07 课：从一个 Block 拼出完整 Transformer

<div class="lesson-lead">这一课先拆开一个 Transformer block，认清 Attention、FFN、Residual、Norm 的分工；再把许多 block 拼成 2017 年完整的 Encoder–Decoder Transformer；最后说明它怎样演变成现代 decoder-only 大语言模型。</div>

<div class="paper-lesson-meta"><span>零基础主课</span><span>建议 90 分钟</span><span>先修：第 06 课 Attention</span></div>

::: info 来源与改编说明
完整架构部分参考 Jay Alammar 的 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 的讲解次序，用中文重新组织并补充现代大语言模型与 K3 的对照；文字、结构图和例子均为本教程重新制作。

同时用原始研究材料校准技术细节：[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)负责 Encoder–Decoder、Multi-Head Attention 与位置编码，[Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)解释归一化来源，[Transformer Notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf)补充矩阵形状与训练细节，[Stanford CS336 Lecture 3](https://stanford-cs336.github.io/spring2026/)负责把 2017 架构接到现代 LLM 的超参数与组件选择。图解文章负责搭直觉，论文与 Slides 负责核对定义和证据，二者不能互相替代。
:::

<figure class="teaching-figure infographic-figure"><a href="/illustrations/transformer-one-page-infographic.webp" target="_blank"><img src="/illustrations/transformer-one-page-infographic.webp" alt="Transformer 从输入位置编码、自注意力、多头注意力到重复 Block 的中文总览图"></a><figcaption>一张图复习完整数据流。先只沿 1→4 的粗箭头看一遍，再回到正文逐段拆解；点击图片可单独放大。</figcaption></figure>

<figure class="teaching-figure source-figure"><a href="/lectures/images/transformer-architecture.png" target="_blank"><img src="/lectures/images/transformer-architecture.png" alt="Stanford CS336 Slides 中从 Transformer Block 连接到现代语言模型结构的原图"></a><figcaption>来源图：Stanford CS336 Lecture 1/3 的 Transformer architecture。本站中文图先帮助认路，这张原始 Slides 图用于核对各张量与模块的正式连接；可在 <a href="https://stanford-cs336.github.io/spring2026/">Lecture 3 PDF</a> 中继续深挖。</figcaption></figure>

## 0. 先看图：一台“先读、再写”的机器

<figure class="teaching-figure"><img src="/illustrations/transformer-reader-writer.webp" alt="左侧编码器阅读完整输入并整理记忆，右侧解码器查询记忆并逐个写出 token"><figcaption>左侧阅读者代表 Encoder，右侧写作者代表 Decoder。中间的彩色卡片桥代表 Encoder memory；两边的层叠底座代表重复堆叠的 Transformer blocks。图由本教程生成。</figcaption></figure>

<div class="visual-key"><div><b>左：Encoder</b>能看完整输入，为每个 token 写下结合全句后的理解。</div><div><b>中：Memory</b>不是一句话压成一个数，而是一串可被查询的上下文化向量。</div><div><b>右：Decoder</b>查看已经写出的前缀，同时查询原句记忆，每次只添加一个 token。</div></div>

先记住一句话：**Block 是零件，Encoder/Decoder 是用许多 block 组装出的机器。**

## 1. 一个现代 Transformer Block 的最小骨架

先从现代大语言模型常见的 decoder-only block 开始：

```text
输入 x
  │
  ├─ Norm → Causal Attention ─┐
  │                           + → x'
  │
  └─ Norm → FFN ──────────────┐
                              + → 输出
```

伪代码只有两行：

```python
x = x + attention(norm(x))
x = x + ffn(norm(x))
```

看图时不要把所有框一起背。只追问四个零件分别解决什么问题。

## 2. Attention：小组交换资料

每个 token 根据内容读取其他位置，回答：

> 为了更新我自己，现在应该从上下文取回什么？

Attention 主要在**序列维**混合信息。如果输入 $X$ 形状是 $T\times d$，它让 $T$ 个位置之间发生交互，输出形状仍是 $T\times d$。

### 2.1 Q、K、V 只是在分工

- **Query**：我现在想找什么；
- **Key**：我身上有什么标签，别人是否应该来找我；
- **Value**：如果别人关注我，我实际提供什么内容。

### 2.2 六步算出一次 Attention

1. 输入分别乘 $W_Q,W_K,W_V$，得到 Q、K、V；
2. Query 与每个 Key 做点积，得到匹配分数；
3. 分数除以 $\sqrt{d_k}$，避免数值过大；
4. Softmax 把分数变成总和为 1 的权重；
5. 每个权重乘对应的 Value；
6. 把加权后的 Value 相加，得到当前位置的新表示。

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

公式只是在说：**先用 Q 和 K 决定看谁，再对 V 加权求和决定拿回什么。** 完整数值例子见[第 06 课](/beginner/04-attention)。

### 2.3 为什么要 Multi-Head

只做一次 Attention，许多关系会挤在同一种匹配方式里。Multi-Head Attention 使用多组不同的 $W_Q,W_K,W_V$：

```text
同一批输入
 ├─ Head 1：从一种投影空间做匹配
 ├─ Head 2：从另一种投影空间做匹配
 └─ Head 3：再换一种观察方式
          ↓
       拼接 Concat
          ↓
     再乘输出矩阵 Wᴼ
```

原始模型使用 8 个头。若总隐藏维是 512，每个头的 Q/K/V 维度可取 64；8 个头拼回后又是 512 维。多头增加观察角度，但 block 输入输出宽度仍可保持一致。

::: warning 不要给每个头硬起名字
“一个头管指代、一个头管语法”只是入门类比。实际功能由训练形成，许多头共同承担关系，未必能被人干净命名。
:::

### 一个可运行的 Pre-Norm Decoder Block

下面把本节零件直接翻译成 PyTorch。输入和输出都是 `[batch, token, hidden]`，因此许多 block 才能首尾相接：

```python
import torch
import torch.nn as nn

class TinyDecoderBlock(nn.Module):
    def __init__(self, hidden=64, heads=4, expansion=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            hidden, heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, expansion * hidden),
            nn.GELU(),
            nn.Linear(expansion * hidden, hidden),
        )

    def forward(self, x):
        T = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, attn_mask=causal_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

x = torch.randn(2, 5, 64)           # 2 句、每句 5 token
print(TinyDecoderBlock()(x).shape)   # torch.Size([2, 5, 64])
```

这不是完整语言模型：前面还需要 tokenizer、embedding 和位置机制，后面还需要词表输出层；现代 LLM 也常换成 RMSNorm、SwiGLU、RoPE、GQA/MLA 等组件。这个小块的价值是把 **Norm → Attention → Residual → Norm → FFN → Residual** 跑通。

## 3. FFN：每个位置独立加工

Attention 之后，每个 token 单独通过同一套非线性网络：

$$
\operatorname{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2
$$

FFN 不直接混合不同 token；它在**通道维**扩展、激活、再投回隐藏维。

类比：Attention 是小组交换资料；FFN 是每个人回到座位，根据拿到的资料独立加工。

在许多 LLM 中，FFN 参数与计算占比很大，因此 MoE 通常替换的正是 FFN，而不是整个 Transformer。

## 4. Residual：别让每层从头重写

如果子层输出是 $f(x)$，Residual 计算：

$$
y=x+f(x)
$$

这意味着子层只需学习“在原表示上增加什么修正”，原信息可以沿直通路径继续流动。

把三层粗略展开：

$$
x_3=x_0+f_1(x_0)+f_2(x_1)+f_3(x_2)
$$

标准 residual 会把历史更新不断累加。网络很深时，早期表示的相对贡献可能被稀释；K3 的 Attention Residuals 就从这里出发，让当前层选择性读取历史深度。

## 5. Norm：控制送进子层的数值尺度

深层网络中数字尺度会不断变化。以 RMSNorm 为例：

1. 测量整个向量的均方根；
2. 除掉这个尺度；
3. 再乘可学习的逐维系数。

$$
\operatorname{RMSNorm}(x)=
\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot g
$$

不用背公式。Norm 不负责让 token 交流，也不负责增加语义容量；它主要控制数值尺度，让深层流水线更稳定。

### 5.1 PreNorm 与 PostNorm

现代大模型常见 **PreNorm**：

```python
x = x + sublayer(norm(x))
```

原始 2017 Transformer 使用 **PostNorm**：

$$
\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))
$$

它们使用相同零件，但放置顺序不同。看到经典图中的 “Add & Norm”，不要直接套成现代 PreNorm 伪代码。

## 6. 从 Block 到完整 Encoder–Decoder

现在把刚才的零件装成完整机器。原始 Transformer 为机器翻译设计，左右分成两条路径：

- **Encoder** 读完整源句，制作整句记忆；
- **Decoder** 根据已生成的译文前缀，查询记忆并预测下一个词。

<figure class="teaching-figure source-figure"><a href="/paper-figures/attention-transformer-figure-1.webp" target="_blank"><img src="/paper-figures/attention-transformer-figure-1.webp" alt="Attention Is All You Need 论文 Figure 1：左侧 Encoder 与右侧 Decoder 的完整架构"></a><figcaption>《Attention Is All You Need》Figure 1（PDF p.3）。左塔是 Encoder，右塔是 Decoder；右塔比左塔多一层读取 Encoder 输出的 Multi-Head Attention，最下层还带 Mask。请先用本站“阅读者—写作者”图建立直觉，再用这张论文原图核对模块。<a href="https://arxiv.org/pdf/1706.03762.pdf#page=3">打开原论文第 3 页</a>。</figcaption></figure>

<TransformerArchitecture />

建议按顺序点击上图四个标签：

1. 先看“整机全景”，只追输入到输出；
2. 再看“只看编码器”；
3. 然后看“只看解码器”，重点看 Cross-Attention；
4. 最后看“对比现代 LLM”，把两代结构分开。

## 7. 进入网络前：词向量还缺少顺序

Tokenizer 先把句子拆成 token，Embedding 表再把 token ID 换成向量。但只有词向量时，模型不知道：

```text
猫 追 老鼠
老鼠 追 猫
```

用了相同的三个 token，却不是同一件事。因此原始 Transformer 把**位置编码向量**加到词向量上：

$$
x_i=e_i+p_i
$$

- $e_i$：第 $i$ 个 token 的内容向量；
- $p_i$：第 $i$ 个位置的顺序向量；
- $x_i$：真正送进第一层的输入。

原始论文用不同频率的正弦、余弦函数产生位置编码。现在不必背公式，只需理解：**加法没有增加向量长度，而是在原向量里写入位置信号。**

::: tip 位置方法会继续演化
正弦位置编码是经典起点，不是永久规定。后来出现可学习位置向量、RoPE、相对位置方法；K3 的混合架构还使用 NoPE 层。读论文时先问“它怎样表达顺序”，不要默认答案永远相同。
:::

## 8. Encoder：先把原句读完整

原始论文把多个结构相同、参数各自独立的 Encoder block 叠起来。论文使用 $N=6$，但“六层”只是当时的配置，不是 Transformer 的定义。

一个 Encoder block 依次包含：

```text
输入
 ↓
双向 Self-Attention
 ↓  Residual + LayerNorm
逐位置 FFN
 ↓  Residual + LayerNorm
输出给下一层
```

### 8.1 为什么叫双向

编码器是在“读题”，不是在生成。每个位置可以查看左边和右边的完整源句，不需要 causal mask。

例如更新“它”的表示时，Self-Attention 可以同时检查前后文，把更多权重放到可能的指代对象上。

### 8.2 Encoder 输出不是一个向量

经过最后一个 Encoder block，我们得到与输入 token 数量相同的一串上下文化向量：

```text
[我]      → 向量 1 ┐
[喜欢]    → 向量 2 ├─ Encoder memory
[机器学习] → 向量 3 ┘
```

每个向量都已吸收整句信息，但仍保留自己的位置。这份向量序列就是 Decoder 可以反复查询的 memory。

## 9. Decoder：一边写，一边查原句

Decoder block 比 Encoder block 多一个 Attention 子层：

1. **Masked Self-Attention**：阅读已经生成的目标前缀；
2. **Cross-Attention**：查询 Encoder memory；
3. **FFN**：逐位置加工。

### 9.1 为什么一定要 Mask

训练时我们已经拿到整句正确译文，为了并行计算，会把目标序列一次送进 Decoder。但预测第 3 个词时，模型不能偷看第 4、5 个正确答案。

因此注意力分数矩阵右上方会被遮住：

```text
查询位置 \ 可读取位置    1     2     3     4
1                       ✓     ×     ×     ×
2                       ✓     ✓     ×     ×
3                       ✓     ✓     ✓     ×
4                       ✓     ✓     ✓     ✓
```

实现时通常把被遮住的位置加上一个极大的负数，Softmax 后权重接近 0。这就是 causal mask。

### 9.2 Cross-Attention “跨”了什么

注意力公式没有改变，改变的是 Q、K、V 的来源：

| 向量 | 来自哪里 | 人话问题 |
|---|---|---|
| Query | Decoder 当前状态 | 我为了继续翻译，现在要查什么？ |
| Key | Encoder 输出 | 原句各位置分别有什么标签？ |
| Value | Encoder 输出 | 我真正从原句各位置取回什么？ |

所以三种 Attention 必须分清：

- Encoder Self-Attention：Q/K/V 全来自 Encoder；
- Decoder Masked Self-Attention：Q/K/V 全来自 Decoder 前缀；
- Cross-Attention：Q 来自 Decoder，K/V 来自 Encoder。

## 10. Linear + Softmax：从隐藏向量回到词表

最后一个 Decoder block 输出的仍是隐藏向量，不是单词。Linear 层把它投影到词表大小：

$$
z=hW_{vocab}
$$

如果词表有 30,000 个 token，$z$ 就有 30,000 个 logits。Softmax 再把它们转成概率：

```text
interesting   0.61
fun           0.18
useful        0.07
...           ...
```

选出一个 token 后，把它接到输出前缀末尾，再运行下一步。遇到结束 token 或达到长度上限，生成停止。

::: warning 训练并行与生成串行不矛盾
训练时真实前缀已知，可以用 causal mask 并行计算许多位置的损失；真正生成时，第 $t+1$ 个 token 依赖前 $t$ 个 token，所以必须逐步进行。KV Cache 只避免重复计算历史，不会消除依赖。
:::

## 11. 用一个极小例子走完整条路

翻译“我 爱 猫”：

```text
① Encoder 输入
   [我] [爱] [猫] + 各自的位置编码

② Encoder Self-Attention
   每个位置读完整句，输出 3 个上下文化向量

③ Decoder 已有前缀
   [<开始>] [I] [love]

④ Masked Self-Attention
   “love”能看 <开始> 和 I，不能看尚未生成的词

⑤ Cross-Attention
   Decoder 的 Query 对 Encoder 的“猫”位置给出较高权重

⑥ Linear + Softmax
   cats: 0.72, dogs: 0.08, ...

⑦ 选出 cats，接到前缀，继续预测结束 token
```

如果这七步能不看答案复述出来，你就真正把整张经典架构图接通了。

## 12. 原始 Transformer 与现代 LLM 不要混画

现代 decoder-only 大语言模型通常把提示词和回答放进同一条 token 序列，统一通过 causal blocks。它没有独立 Encoder，也没有经典 Encoder–Decoder Cross-Attention。

| 问题 | 原始翻译 Transformer | Decoder-only LLM |
|---|---|---|
| 输入路径 | 源句、目标前缀两条 | 提示与回答一条 |
| 主体 | Encoder + Decoder | 只有 causal blocks |
| Cross-Attention | 有 | 通常没有 |
| 自注意力范围 | Encoder 双向；Decoder 因果 | 因果 |
| 生成方式 | 逐 token | 逐 token |
| 都保留的骨架 | Attention、FFN、Residual、Norm、输出头 | 同左 |

经典图仍值得学，因为它把零件职责分得最清楚。但阅读 GPT 或 K3 论文时，要把“完整 Transformer”和“Transformer block”区分开。

## 13. 这套骨架怎样连接到 Kimi K3

K3 没有抛弃 Transformer 的基本思想，而是在不同部位继续改造：

- **序列信息怎样混合**：组合 KDA 与 MLA，而不只使用标准 Softmax Attention；
- **FFN 怎样扩容**：使用稀疏 MoE，每个 token 只激活部分专家；
- **跨层信息怎样走**：Attention Residuals 选择性读取历史层；
- **位置怎样表达**：部分层使用 NoPE，把顺序信息交给混合架构中的其他机制；
- **视觉怎样进入**：先把图像变成视觉 token，再与文本统一建模。

学习顺序因此非常明确：先认识标准零件，再看论文替换了哪一件、为什么替换、代价是什么。

## 本课闭卷画图

合上本页，在纸上完成三张图：

1. 画一个现代 PreNorm block，标出 Attention、FFN、Residual、Norm；
2. 画左 Encoder、右 Decoder，以及 memory 到 Cross-Attention 的箭头；
3. 标出三种 Attention 的 Q/K/V 分别来自哪里。

<ConceptCheck question="MoE 在多数现代 LLM 中主要替换哪一部分？" :options='["Tokenizer", "FFN", "Causal mask"]' :answer="1" explanation="MoE 通常把 dense FFN 换成由 router 条件激活的多个专家 FFN。" />

<ConceptCheck question="Cross-Attention 中，Query、Key、Value 分别来自哪里？" :options='["Q 来自 Decoder，K/V 来自 Encoder", "Q/K/V 都来自 Encoder", "Q/K/V 都来自 Decoder"]' :answer="0" explanation="Decoder 用当前状态提出查询，Encoder 输出提供可匹配的 Key 和被读取的 Value。" />

<ConceptCheck question="为什么 Decoder 训练时仍需要 causal mask？" :options='["为了减少词表大小", "为了不让当前位置偷看后面的正确答案", "为了让 Encoder 只读左侧"]' :answer="1" explanation="训练时整句目标会并行送入网络，mask 保证第 t 个位置只能使用此前信息，与真实生成条件一致。" />

下一课：[模型怎样逐 token 生成，以及 KV Cache 为什么有用](/beginner/06-generation)。

<ChapterReadings lesson="05-transformer" />
