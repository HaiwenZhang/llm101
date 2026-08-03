---
title: Transformer 第一课
description: 从 token 到下一个 token，走完语言模型的一次前向过程
---

# Transformer 第一课：跟着一个 token 走完全程

<div class="lesson-lead">学完这一课，你应该能在白纸上画出：文本如何进入模型、token 之间怎样交换信息、模型如何产生下一个 token。</div>

## 0. 先看全流程

```text
文本
 ↓ tokenizer
token ID 序列
 ↓ embedding 查表
向量序列 X
 ↓ N 个 Transformer Block
上下文表示 H
 ↓ output projection
整个词表的 logits
 ↓ softmax / 采样
下一个 token
```

每生成一个 token，这条链就再走一次。现代系统会用缓存与并行优化避免重复工作，后面的课程会详细讲。

## 1. Token：模型眼里的文本积木

计算机不能直接对汉字或单词做矩阵乘法。Tokenizer 会把文本切成词表中的片段，并将每个片段映射到整数 ID。

例如，真实切分取决于词表：

```text
"机器学习真有趣"
→ ["机器", "学习", "真", "有趣"]
→ [3812, 927, 64, 10521]
```

Token 不一定是完整的词，也不一定是单个字符。常见词可能是一块，罕见词可能被拆成几块。

Tokenizer 会影响：

- 同一段文字占多少上下文；
- 中文、代码、数学等数据的压缩效率；
- 词表 embedding 和输出层的大小；
- 模型能否稳定处理罕见字符。

## 2. Embedding：用向量替代整数编号

Token ID `3812` 只是编号，大小本身没有语义。Embedding 表是一张巨大的参数表，每一行对应词表中的一个 token：

$$
E\in \mathbb{R}^{|V|\times d}
$$

`|V|` 是词表大小，`d` 是隐藏维。拿 token ID 去查对应行，得到 d 维向量。

经过多层网络后，“苹果”在“吃一个苹果”和“苹果发布手机”中的表示会因上下文而不同。Embedding 是起点，不是最终语义。

## 3. Attention：让 token 读取其他位置

考虑句子：

> 小猫追着毛线球，因为**它**滚远了。

处理“它”时，模型应该更多读取“毛线球”而不是“小猫”。Attention 提供了一种内容寻址机制。

对每个 token 的表示 `x` 做三套投影：

$$
q=xW_Q,\qquad k=xW_K,\qquad v=xW_V
$$

- `q`（Query）：当前位置想找什么；
- `k`（Key）：每个位置用什么标签被找到；
- `v`（Value）：这个位置真正提供什么内容。

Query 与所有 Key 做点积，得到相关性分数；经缩放、mask 和 softmax 后成为权重：

$$
A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)
$$

最后对 Value 加权求和：

$$
Y=AV
$$

### 为什么要除以 `√dₖ`

维度变大时，点积数值的方差也容易变大，softmax 会过早饱和。缩放让数值范围更稳定。

### Causal mask 做什么

生成模型在位置 t 只能读取位置 `≤t`。未来位置的分数被加上一个极大的负数，softmax 后概率近似 0。

训练时整段文本已知，所以所有位置仍能一次并行计算；mask 只保证信息不会从未来泄漏。这叫 teacher forcing。

## 4. Multi-head：不只用一种关系看句子

单个 attention head 只产生一套权重。Multi-head attention 将隐藏维拆成多个 head，让不同 head 有机会学习不同关系，例如：

- 代词指代；
- 临近搭配；
- 代码括号配对；
- 长距离主题信息。

这只是帮助理解的例子，不保证每个 head 都能被清晰命名。

各 head 计算后拼接，再做一次输出投影：

$$
\operatorname{MHA}(X)=\operatorname{Concat}(head_1,\ldots,head_H)W_O
$$

## 5. FFN：每个 token 独立加工

Attention 负责 token 之间交换信息；FFN 对每个位置独立做非线性变换。

简化写法：

$$
\operatorname{FFN}(x)=W_2\,\sigma(W_1x)
$$

同一套 FFN 参数应用于每个 token。现代模型常用 SwiGLU。你可以把 attention 想成“小组讨论”，FFN 想成“每个人根据讨论结果独立思考”。

## 6. Residual 与 Norm：让深层网络能训练

一个典型 PreNorm block：

$$
x\leftarrow x+\operatorname{Attention}(\operatorname{Norm}(x))
$$

$$
x\leftarrow x+\operatorname{FFN}(\operatorname{Norm}(x))
$$

Residual 的 `+x` 提供一条信息与梯度的直通路径。Norm 控制送入子层的数据尺度。K3 的 Attention Residuals 会进一步改变“沿深度怎样读取历史层”，但必须先理解这里的标准做法。

## 7. Logits：模型不是只输出一个词

最后一个隐藏向量通过输出投影，得到词表大小的一串 logits。假设词表只有五个 token：

| token | logit | softmax 概率 |
|---|---:|---:|
| 风筝 | 4.2 | 0.68 |
| 足球 | 3.4 | 0.30 |
| 钢琴 | 0.1 | 0.01 |
| 昨天 | -0.8 | 0.00 |
| 如果 | -1.2 | 0.00 |

系统可以总选最高概率（greedy decoding），也可以按概率采样。采样策略改变输出多样性，但不是模型结构的一部分。

## 8. 训练与推理为什么很不一样

### 训练

完整答案已知，把序列错开一位就能构造输入与标签：

```text
输入：我 喜欢 机器 学习
标签：喜欢 机器 学习 。
```

所有位置的损失可并行计算。反向传播还要保存或重算大量中间激活。

### 推理

下一个 token 未知，只能：生成一个 → 拼回前缀 → 再生成一个。每步都要读取模型权重与历史状态，延迟和内存带宽变得非常重要。

## 9. 一个最小 Transformer Block 的伪代码

```python
def block(x):
    # token 先通过 attention 交换信息
    x = x + causal_attention(rms_norm(x))

    # 每个 token 再独立通过 FFN 加工
    x = x + swiglu_ffn(rms_norm(x))
    return x
```

真实模型还会加入位置处理、缓存、并行、量化、MoE 或线性 attention，但骨架仍可从这几行展开。

## 本课自测

<ConceptCheck question="Attention 中真正被加权汇总的是哪一个对象？" :options='["Query", "Key", "Value"]' :answer="2" explanation="Query 与 Key 决定权重，权重再用于加权汇总 Value。" />

<ConceptCheck question="为什么 decoder-only LM 训练时能并行处理整段文本？" :options='["因为它训练时可以偷看未来", "因为目标序列已知，causal mask 在并行矩阵运算中阻止未来信息泄漏", "因为训练时不用 attention"]' :answer="1" explanation="Teacher forcing 提供完整前缀；逻辑上的因果约束由 mask 保证，并不要求逐 token 调用模型。" />

准备好了就进入正式课程：[第 0 章：Kimi K3 到底是什么](/guide/ch00)。
