---
title: 第 06 课 Attention 从零拆解
description: 用图书馆检索类比和小数字例子理解 Q、K、V
---

# 第 06 课：Attention 就是一次内容检索

<div class="lesson-lead">先把公式放下：当前位置提出一个查询，与所有位置的索引比较，再按匹配程度取回并混合内容。这就是 Q、K、V 的主线。</div>

::: info 本课资料地图
- 课件：[CS224N Transformer Slides](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture05-transformers.pdf)、[CMU Attention and Transformers](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-05-attention-transformers.pdf)与 [Stanford CS336 Lecture 3](https://stanford-cs336.github.io/spring2026/)；
- 原论文：[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)，重点看 PDF p.3 的 Figure 1 和 p.4 的 Scaled Dot-Product Attention；
- 延伸论文：[FlashAttention-2](/papers/flash_attention_2)改变计算组织但不改变 Attention 数学结果，[Kimi Linear](/papers/kimi_linear)与 [Gated DeltaNet](/papers/gated_deltanet)则改变长序列状态表示。
:::

<figure class="teaching-figure concept-figure"><img src="/illustrations/attention-four-steps-v2.webp" alt="Attention 分为提问、匹配、分配权重和汇总信息四步"><figcaption>低密度四步图：先只记数据流，再进入 Q、K、V 和矩阵公式。每个候选通常都会贡献一部分信息，并非硬复制某一个 token。</figcaption></figure>

## 1. 为什么 token 需要互相读取

看句子：

> 小猫追着毛线球，因为**它**滚远了。

“它”应该更多读取“毛线球”。如果每个 token 永远独立处理，它无法利用前文消除指代歧义。

### 1.1 Attention 最早先解决 Encoder 的单向瓶颈

早期机器翻译常让 Encoder RNN 把整句源语言压进最后一个隐藏状态，Decoder 再从这个固定向量逐词生成。句子越长，所有人名、语序和修饰关系越要争抢同一个瓶颈；远处信息的梯度还要穿过许多递归步，容易衰减。

Cross-Attention 改成：Decoder 每生成一个目标 token，就直接读取**所有** Encoder 隐藏状态。翻译“pie”时可以对源句中“entarté”分配较高权重，不必要求最后一个 Encoder 状态永久记住全部细节。

这也缩短了信息路径：在 RNN 中，相隔 $k$ 个位置的信息至少经过 $k$ 次状态更新；一层全局 self-attention 中，任意两个允许位置可直接发生一次匹配。路径短不等于免费——它用 $T^2$ 对位置交互换来了全局连接。

## 2. 图书馆类比

<figure class="teaching-figure"><img src="/illustrations/attention-library.webp" alt="将 Attention 比作图书馆检索：查询卡匹配索引卡，再取回书中内容"><figcaption>Query 像读者手里的需求卡；Key 像档案抽屉的索引；Value 是真正被取回并汇总的内容。图由本教程专门生成。</figcaption></figure>

<div class="visual-key"><div><b>Query：需求卡</b>当前 token 写下自己要寻找的信息。</div><div><b>Key：索引卡</b>每个候选位置提供可供匹配的标签。</div><div><b>Value：正文</b>匹配决定权重，真正被加权取回的是 Value。</div></div>

三个角色：

| 名称 | 人话 | 图书馆 |
|---|---|---|
| Query | 我现在想找什么 | 读者的查询卡 |
| Key | 我可以用什么标签被找到 | 索引卡 |
| Value | 如果找到我，真正拿走什么 | 书里的内容 |

同一个 token 通过三套不同权重得到 Q/K/V。不是三份不同文本。

### 2.1 Cross-Attention、Self-Attention 与 Causal Self-Attention

三者使用同一条数学公式，只是 Q、K、V 的来源和 mask 不同：

| 类型 | Query 来自 | Key / Value 来自 | 常见用途 |
|---|---|---|---|
| Cross-Attention | Decoder / 查询序列 | Encoder / 被检索序列 | 翻译、图文、多模态、RAG 融合 |
| Self-Attention | 同一序列 | 同一序列 | Encoder 双向表示 |
| Causal Self-Attention | 当前 Decoder 序列 | 同一 Decoder 序列 | GPT 下一 token 预测 |

Cross-Attention 的权重矩阵形状一般是 $T_q\times T_{kv}$，不一定是正方形；Self-Attention 才是 $T\times T$。说“Attention 一定产生 $T\times T$”只对等长 self-attention 成立。

### 2.2 为什么 Key 和 Value 不直接用同一向量

给输入表示 $X$ 三张可学习投影：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

$W_Q,W_K$ 学习“什么特征用于匹配”，$W_V$ 学习“匹配后传递什么内容”。图书馆的索引卡可以写“Transformer / 注意力”，书正文则包含完整推导；索引适合查找，正文适合读取，两者职责不同。若强制 K=V，模型仍能工作，但匹配空间与传输空间被绑定，表达能力更受限。

## 3. 用三个 token 手算一次

假设当前位置的 Query 是 `q=[1,0]`，三个历史 Key：

```text
k₁=[1,0]   “毛线球”
k₂=[0,1]   “小猫”
k₃=[0.5,0.5] “追着”
```

点积得分：

```text
q·k₁ = 1
q·k₂ = 0
q·k₃ = 0.5
```

Softmax 后假设约为 `[0.51, 0.19, 0.30]`。输出不是复制最高分 Value，而是加权混合：

$$
y=0.51v_1+0.19v_2+0.30v_3
$$

因此 attention 是软检索；所有允许的位置通常都有非零贡献。

### 3.1 把 Value 也带入数字，算到最后

令三个 Value 分别为：

$$
v_1=[2,0],\quad v_2=[0,2],\quad v_3=[1,1]
$$

则输出为：

$$
\begin{aligned}
y&=0.51[2,0]+0.19[0,2]+0.30[1,1]\\
&=[1.32,0.68]
\end{aligned}
$$

第一维更接近“毛线球”的内容，但仍混有其他位置。若最高权重从 0.51 变成 0.99，输出才近似硬查表。Attention 权重描述的是本层如何混合 Value；后续还有输出投影、残差、Norm 和 FFN，不能把一个热力格直接等同于最终答案原因。

### 3.2 Softmax 要逐行计算

对分数 $s=[1,0,0.5]$：

$$
\operatorname{softmax}(s)_j=
\frac{e^{s_j}}{e^1+e^0+e^{0.5}}
$$

得到约 `[0.506, 0.186, 0.307]`。矩阵 Attention 中，每一行属于一个 Query，所以 Softmax 沿 Key 轴做；若误沿 Query 轴归一化，虽然数字仍“每列和为 1”，语义已经变成多个 Query 竞争同一 Key。

## 4. 公式逐块翻译

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

不要整条背，按从左到右读：

1. `QKᵀ`：每个 Query 与每个 Key 做匹配，得到 `T × T` 分数表；
2. `/√dₖ`：控制点积尺度，避免 softmax 过早饱和；
3. `+M`：加入 mask，把不允许看的位置变成负无穷；
4. `softmax`：每行变成和为 1 的权重；
5. `×V`：按权重混合 Value。

<figure class="teaching-figure source-figure"><a href="/paper-figures/attention-figure-2.webp" target="_blank"><img src="/paper-figures/attention-figure-2.webp" alt="Attention Is All You Need 论文 Figure 2：左边为缩放点积注意力，右边为多头注意力"></a><figcaption>《Attention Is All You Need》Figure 2（PDF p.4）。左图从下往上正好对应公式：Q/K 点积 → Scale → 可选 Mask → Softmax → 与 V 相乘；右图显示多个头先独立投影和计算，再 Concat、Linear。<a href="https://arxiv.org/pdf/1706.03762.pdf#page=4">打开原论文第 4 页</a>。</figcaption></figure>

<figure class="teaching-figure concept-figure"><img src="/illustrations/attention-tensor-shapes.svg" alt="多头 Attention 从输入投影、拆头、分数矩阵、加权 Value 到拼接输出的张量形状"><figcaption>不要凭字母猜实现，沿形状走一遍。Head 维度像额外 batch 轴并行计算；h 个 <code>d_h</code> 维输出拼回 <code>d=h·d_h</code>，再经 <code>W_O</code> 混合。</figcaption></figure>

### 先把形状对齐，再看代码

单头、自注意力、暂时忽略 batch 时：

| 符号 | 形状 | 每一行/格表示什么 |
|---|---|---|
| $Q,K,V$ | $T\times d_k$ | $T$ 个 token，各有 $d_k$ 个特征 |
| $QK^\top$ | $T\times T$ | 第 $i$ 个 Query 对第 $j$ 个 Key 的分数 |
| `softmax(...)` | $T\times T$ | 每一行是一个 Query 的读取权重 |
| 输出 | $T\times d_k$ | 每个 Query 汇总后的 Value |

为什么除以 $\sqrt{d_k}$？若 Q、K 各维近似独立且方差相近，点积把 $d_k$ 项加起来，波动尺度会随 $\sqrt{d_k}$ 增长。缩放后，softmax 不容易过早变成接近 0/1 的尖峰，梯度更稳定。

更精确地说，若每维 $q_r,k_r$ 独立、均值 0、方差 1，则：

$$
q^\top k=\sum_{r=1}^{d_k}q_rk_r,qquad
\operatorname{Var}(q^\top k)=d_k
$$

标准差就是 $\sqrt{d_k}$。除掉它后，分数方差回到约 1。这是初始化附近的近似推导，不保证训练后各维永远独立；RMSNorm、初始化和 QK-Norm 等设计也会改变真实尺度。

### 数值稳定：先减每行最大值

数学上 $\mathrm{softmax}(s)=\mathrm{softmax}(s-c)$。实现常令 $c=\max_j s_j$：

```python
stable = scores - scores.max(dim=-1, keepdim=True).values
weights = stable.exp() / stable.exp().sum(dim=-1, keepdim=True)
```

这样最大指数是 $e^0=1$，避免大正数 `exp` 溢出。生产代码优先调用框架稳定实现或 fused kernel；手写这几行主要用于理解。

### PyTorch：不调用现成 Attention 的最小实现

```python
import math
import torch

def causal_attention(q, k, v):
    # q, k, v: [T, d_k]
    scores = q @ k.T / math.sqrt(q.shape[-1])  # [T, T]

    T = q.shape[0]
    future = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    scores = scores.masked_fill(future, float("-inf"))

    weights = torch.softmax(scores, dim=-1)    # 每行和为 1
    output = weights @ v                      # [T, d_k]
    return output, weights
```

这段代码与公式逐项对应。生产模型还会加入 batch、多个 head、Q/K/V 线性投影、dropout 与高效 kernel，但不会改变“匹配 → mask → 权重 → 汇总”的逻辑。

## 5. Causal mask 与 softmax 不是一回事

生成模型的位置 t 不能读取未来。Causal mask 先决定“哪些位置有资格”；softmax 再决定“在有资格的位置里怎样分配”。

<figure class="teaching-figure concept-figure"><img src="/illustrations/attention-mask-types.svg" alt="Causal mask 形成下三角，Padding mask 屏蔽所有 PAD 列"><figcaption>Causal mask 与 Padding mask 可以同时存在：前者按相对时间屏蔽未来，后者按样本长度屏蔽补齐位置。它们都在 Softmax 之前改 logits。</figcaption></figure>

### 5.1 三种常见 mask 的职责

| Mask | 屏蔽什么 | 是否随 Query 行变化 |
|---|---|---:|
| Causal | 未来 Key：$j>i$ | 是，下三角 |
| Padding | PAD Key | 通常各行相同，但各 batch 样本不同 |
| Local / Sliding window | 距离当前过远的位置 | 是，只保留窗口 |

常见实现把不允许位置加上 `-inf`，Softmax 后概率变 0。低精度实现有时用 dtype 的最小有限值；若一整行全部被屏蔽，`-inf - (-inf)` 会导致 NaN，必须保证至少一个可见 Key 或显式处理空行。

### 5.2 Attention mask 与 loss mask 不是一件事

Attention mask 控制前向时“能读谁”；loss mask 控制训练时“哪些输出位置计分”。Chat SFT 可以允许 assistant 读取 user token，却只对 assistant token 计算 loss。把两种 mask 共用一张布尔表，往往会产生看似训练正常、实际目标错位的 bug。

<AttentionLab />

把 Query 移到句子中间，右侧位置显示 MASK；无论怎样调温度，它们都不会重新获得权重。

## 6. Multi-head 为什么存在

一套 Q/K/V 投影只能产生一种匹配空间。Multi-head 把隐藏维拆成多组，让不同 head 可以学习不同关系：

- 局部搭配；
- 代词指代；
- 代码括号；
- 文档主题；
- 位置与格式。

这是直觉，不保证每个 head 都能被干净地命名。

### 6.1 多头公式与输出投影

$$
\mathrm{head}_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
\mathrm{MHA}(Q,K,V)=
\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W_O
$$

若总隐藏维 $d=1024$、$h=16$，常见每头维度 $d_h=64$。16 个头并非各自输出 1024 维，而是各输出 64 维，拼接后仍是 1024 维。$W_O$ 让不同头的信息重新混合回残差流。

### 6.2 Head 数增加为什么不必线性增加参数

固定 $d=h d_h$ 时，Q/K/V 三张总投影与输出投影仍各约 $d\times d$，普通 MHA 合计约 $4d^2$。增加 head 数会让每头变窄，而不是凭空复制完整矩阵。GQA/MQA 才会通过减少 KV heads 真实缩小 K/V 投影与 KV Cache；详见[参数量与 Scaling](/beginner/25-data-scaling#parameter-count)。

### 6.3 Head 可视化不能自动证明模型“在想什么”

某个 head 对代词高亮名词，是有用现象，但不证明该 head 是唯一因果机制：

- 不同 head 可能冗余；
- Value 和输出投影会改变信息；
- 后续层可能覆盖或抵消；
- Attention 权重高，不等于该位置对最终 logit 的因果贡献大。

要做机制解释，应配合 head ablation、activation patching、梯度或因果干预，而不只展示热力图。

## 7. 标准 attention 的两本成本账

### 计算账

`QKᵀ` 的分数表有 `T²` 个格子。序列翻倍，格子约变四倍。

更完整的单层数量级是：

$$
\underbrace{O(Td^2)}_{Q,K,V,O\text{ 投影}}
+\underbrace{O(T^2d)}_{QK^\top\text{ 与 }AV}
$$

短序列、很宽模型时，$Td^2$ 投影可能占大头；超长序列时，$T^2d$ 交互逐渐主导。因此“Attention 是二次复杂度”正确但不完整，不能由此推出任何长度下它都占最多 FLOPs。

例如 $T$ 从 4K 增到 32K，注意力分数格子数增 64 倍；若只把模型宽度 $d$ 翻倍，分数矩阵格子数不变，但每格点积和投影都更贵。

### 状态账

自回归生成时，每层要为历史 token 保存 K/V，大小随 T 线性增长。

标准 MHA 的 KV Cache 元素量近似：

$$
2\times L\times B\times T\times h_{kv}\times d_h
$$

前面的 2 是 K 和 V。GQA 让 $h_{kv}<h$，所以不改变 Query head 数也能直接缩小 Cache。权重矩阵属于模型参数，KV Cache 属于当前请求状态，二者不能用同一个“模型大小”口径相加。

后续论文分成不同路线：

- FlashAttention：不改结果，减少中间数据搬运；
- MLA：保留全局 softmax，压缩历史 K/V；
- KDA/线性 attention：改成固定状态递推；
- Ring/Ulysses：把序列分到多卡。

### 7.1 训练可并行，生成仍逐 token 顺序

Causal mask 允许训练时把整段正确序列一次送入模型，并行计算每个位置的下一 token loss；mask 防止位置偷看右侧答案。部署生成时，第 $t+1$ 个 token 的输入依赖刚刚生成的第 $t$ 个 token，因此 Decoder 仍有时间方向的串行依赖。KV Cache 避免重复计算历史 K/V，却不能让未知未来 token 提前出现。

## 8. Attention 常见实现错误排查表

| 症状 | 首先检查 |
|---|---|
| 每行权重和不是 1 | Softmax 轴是否为 Key 轴 `dim=-1` |
| 训练 loss 异常低 | Causal mask 是否方向反了、是否偷看目标 |
| PAD 越多结果越变 | Padding mask 是否屏蔽 Key 列，loss 是否忽略 PAD |
| 出现 NaN | 是否整行全 mask、缩放/精度/稳定 Softmax 是否正确 |
| 多头 reshape 后语义错 | `[B,T,h,d_h]` 转 `[B,h,T,d_h]` 是否 transpose |
| Cache 推理与全序列不一致 | RoPE 位置、K/V 追加顺序、mask 与 cache offset |
| 显存远超预期 | 是否显式保留完整 $T\times T$ 权重、是否需要返回 attention map |

最小单元测试应验证：权重逐行和为 1；mask 位置权重为 0；改变被 mask 的 Value 不影响输出；缓存逐 token 输出与一次性前向在容差内一致。

## 9. 本章阅读路线

1. [CS224N L05 Slides](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture05-transformers.pdf) 第 21–40 页：从 RNN bottleneck 看 Cross-Attention，再过渡到 Self-Attention；
2. [CS224N Transformer Notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf) 第 4–11 页：逐式核对 Q/K/V、位置、mask、多头与 batch 轴形状；
3. [CMU ANLP L05](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-05-attention-transformers.pdf) 第 6–13、24–31、64–66 页：比较 Cross/Self-Attention、打分函数、矩阵实现与复杂度；
4. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 第 3.2 节：亲手标注 Figure 2 每条箭头的形状，再读 Table 1 的每层复杂度与最大路径长度。

## 本课闭卷复述

不用 Q、K、V 三个字母，解释 attention 的五个步骤。然后再把每一步对回公式。

<ConceptCheck question="Attention 最终直接加权汇总的是哪个对象？" :options='["Query", "Key", "Value"]' :answer="2" explanation="Query 与 Key 产生匹配权重，权重再用于汇总 Value。" />

<ConceptCheck question="一批 Decoder 序列既有未来 token 又有 PAD，正确顺序是什么？" :options='["先 Softmax，再把两种 mask 后乘到概率上", "把 causal 与 padding mask 加到 score logits，再沿 Key 轴 Softmax", "只用 loss mask 就足够"]' :answer="1" explanation="Mask 必须在 Softmax 前把非法位置变为负无穷；loss mask 只控制计分位置，不能阻止信息读取。" />

下一课：[一个 Transformer block 里发生什么](/beginner/05-transformer)。

<ChapterReadings lesson="04-attention" />
