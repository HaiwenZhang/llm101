---
title: 第 10 课 Decoder-only、GPT、LLaMA 与非 Transformer
description: 从因果语言模型到 RoPE、现代化组件、SSM、Mamba 与 TTT
---

# 第 10 课　Decoder-only、GPT、LLaMA 与非 Transformer

<div class="lesson-lead">Decoder-only 用一个目标统一预训练与生成：根据左侧 token 预测下一个 token。规模化让它成为通用大模型主流；SSM、Mamba、TTT 则在追问：超长序列是否必须保存并比较全部历史？</div>

::: info 本课资料地图
- 从 Self-Attention 搭出 Decoder：[CS224N L05 Slides](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture05-transformers.pdf)第 39–63 页与[配套 Notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf)；
- 现代 Transformer++ 组件：[CMU ANLP L05](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-05-attention-transformers.pdf)第 45–56 页；
- 长序列与 SSM 两种视图：[CMU ANLP L22](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-22-longcontext.pdf)第 34–52 页；
- 现代 Decoder 配方：[Stanford CS336 Lecture 3](https://stanford-cs336.github.io/spring2026/)；
- Attention 替代与 SSM：[Stanford CS336 Lecture 4](https://stanford-cs336.github.io/spring2026/)；
- 原论文：[Mamba](https://arxiv.org/pdf/2312.00752.pdf)、[RoPE 官方 PDF](https://arxiv.org/pdf/2104.09864)、[Kimi Linear 官方 PDF](https://arxiv.org/pdf/2510.26692)。

本课不把 SSM 当成“更快 Transformer”。先核对状态怎样更新，再分别比较训练并行、推理状态、精确回忆和硬件实现。
:::

<figure class="teaching-figure">
  <img src="/illustrations/foundations-architectures.webp" alt="因果生成高塔与下方固定状态递推工坊">
  <figcaption>本课先看右侧因果高塔，再看下方递推工坊：两条路线的关键差别是历史以 token 档案还是固定状态保留。</figcaption>
</figure>

## 1. Decoder-only 的最小训练样本

```text
输入位置：  [BOS]  今  天  天  气
预测目标：     今   天  天  气  好
```

每个位置只看左侧，训练损失为：

$$\mathcal L=-\sum_t\log P(x_t\mid x_{<t})$$

一篇文章天然产生很多监督信号；无需为每个知识点人工标签。规模扩大后，模型在预测下一个 token 的压力下学习语法、事实、格式、代码和一部分推理模式。

### 1.1 右移一格、因果遮罩和 loss mask 各管一件事

设一条序列是 `[BOS, 今, 天, 好, EOS]`。送进模型的输入与监督目标错开一格：

```text
input_ids: [BOS, 今, 天, 好]
labels:    [今,  天, 好, EOS]
```

一次前向可得到 `logits[B, T, V]`，每个位置有 $V$ 个词表分数。训练并行不等于可以偷看答案，因为 Attention 分数矩阵在上三角加 $-\infty$：

$$
M_{ij}=\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases},\qquad
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
$$

这三种机制不要混：

| 机制 | 它阻止什么 | 若做错会怎样 |
|---|---|---|
| 标签右移 | 让位置 $t$ 预测下一格 | 模型学成复述当前 token |
| Causal mask | 阻止隐藏状态读取未来 | 训练 loss 虚低，推理时能力崩溃 |
| Loss mask | 不在 padding、纯提示或无监督位置计分 | 大量无意义位置污染梯度 |

[CS224N L05 第 48–52 页](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture05-transformers.pdf#page=48)不是直接给出完整 Decoder 图，而是先指出 Self-Attention 的三个障碍：没有顺序、没有逐位置非线性、会偷看未来；位置表示、FFN 和 causal mask 正是逐项修复。

### 1.2 训练并行与生成串行为什么同时成立

训练时真实序列已经存在，所有行的 Query 可以一次矩阵乘法算出；mask 只是把每行右侧列盖住。生成时第 $t+1$ 个 token 尚不存在，必须等第 $t$ 个 token 选出后才能形成新前缀。因此：

- **训练**：序列维可并行，但要保存激活并反向传播；
- **Prefill**：已有提示整体并行，顺便建立每层 KV Cache；
- **Decode**：每步只有一个新 Query，必须读取历史 K/V 并串行追加。

“Transformer 可并行”主要说训练和 Prefill，不能推导出自回归 Decode 也能一次吐出整段文本。

## 2. GPT 路线在演化什么

不用死记版本数字，先抓住 Decoder-only 路线究竟在扩大什么：

1. **规模**：参数、数据、计算扩大；
2. **数据与任务**：从通用预训练到多任务、代码和高质量数据；
3. **上下文与架构**：位置表示、Attention、FFN、归一化不断调整；
4. **后训练**：指令微调、偏好学习、RL 和工具使用把 base model 变成助手。

“GPT”不是一个固定网络细节，而是一条以自回归 Transformer 为核心不断系统化的路线。

| 阶段 | 主要变化 | 不能误读成什么 |
|---|---|---|
| 早期 GPT | 证明通用预训练后可迁移 | “下一词预测天然服从指令” |
| GPT-2 式扩大 | 更大模型与更多网页文本呈现零样本迁移 | “模型已可靠掌握事实” |
| GPT-3 式 Scaling | Few-shot / in-context learning 随规模增强 | “上下文学习等于更新权重” |
| Instruct / Chat | SFT、偏好学习、RL、工具接口与安全策略 | “换了一个全新 Transformer 骨架” |

从 base model 到助手，目标函数、数据分布与交互协议都在改变。把所有能力归因于“参数变大”会漏掉后训练；把所有提升归因于 RL 又会漏掉 base model 已经通过预训练获得的表示和知识。

## 3. LLaMA 路线常见的现代化组件

“现代 Decoder”不是把五个名词摆在一起，而是把数据流改成一套更易扩展的 Block。典型 Pre-Norm 写法是：

$$
u_\ell=x_\ell+\operatorname{Attn}(\operatorname{RMSNorm}(x_\ell))
$$

$$
x_{\ell+1}=u_\ell+\operatorname{FFN}(\operatorname{RMSNorm}(u_\ell))
$$

残差流 $x$ 是贯穿深度的主干；Attention 负责跨 token 交换信息，FFN 负责每个 token 内部变换特征。RoPE 进入 Q/K，GQA 改变 K/V 头数量，RMSNorm 和 SwiGLU 分别改变归一化与逐位置计算。

<figure class="teaching-figure concept-figure"><img src="/illustrations/modern-decoder-block-flow.svg" alt="现代 Pre-Norm Decoder Block 中残差流、RMSNorm、Causal Attention、RoPE、GQA 与 SwiGLU 的数据流"><figcaption>沿深色横线追踪残差主干：每个子层只计算一个增量再加回去。Pre-Norm、RoPE、GQA、SwiGLU 作用位置不同，不能把它们理解成四个可随意互换的激活函数。</figcaption></figure>

### 3.1 RMSNorm

只按均方根缩放，不减均值，结构更简洁：

$$\operatorname{RMSNorm}(x)=\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot g$$

若 $x=[3,4]$ 且暂时令 $g=[1,1]$、忽略 $\epsilon$，RMS 是 $\sqrt{(9+16)/2}=\sqrt{12.5}\approx3.536$，输出约为 `[0.849, 1.131]`。它控制的是整体尺度，不保证输出均值为 0。

LayerNorm 会先减去均值再除标准差；RMSNorm 省略中心化，只保留每维可学习缩放 $g$。两者都按**同一个 token 的隐藏维**计算统计，不会把不同 token 或不同 batch 样本混在一起。不能因为公式短就说 RMSNorm “没有参数”：$g\in\mathbb R^d$ 仍是可训练向量。原论文见 [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467.pdf)。

### 3.2 SwiGLU

FFN 加入门控分支：

$$
\operatorname{SwiGLU}(x)
=\left(\operatorname{SiLU}(xW_{gate})\odot xW_{up}\right)W_{down}
$$

`up` 产生候选特征，`gate` 决定哪些维度通过，`down` 再投回 $d$ 维残差流。普通 FFN 有两张大矩阵，参数约 $2dd_{ff}$；SwiGLU 有三张，约 $3dd_{ff}$。若仍令 $d_{ff}=4d$，参数会从 $8d^2$ 增至 $12d^2$，比较不公平。因此常把 SwiGLU 中间宽度取到约 $\frac{8}{3}d$，使 $3d\cdot\frac{8}{3}d=8d^2$，再按硬件倍数对齐。

SiLU$(z)=z\sigma(z)$ 平滑且允许小幅负值；乘法门让同一 token 的特征彼此调制。它仍是**逐位置**网络：位置之间的信息交换已由 Attention 完成。CMU L05 [第 45–46 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-05-attention-transformers.pdf#page=45)把 ReLU、SiLU 和 SwiGLU 的三矩阵结构连续画出。

### 3.3 RoPE

按位置旋转 Q/K 向量，使点积自然带有相对位置信息。核心直觉是两支向量的相对旋转角度与位置距离有关。它不自动保证无限长度外推，超长上下文还需训练和缩放策略。

把每两个通道看成一个二维平面，位置 $m$ 让 Query 乘旋转矩阵 $R_m$，位置 $n$ 让 Key 乘 $R_n$：

$$
(R_mq)^\top(R_nk)=q^\top R_m^\top R_nk=q^\top R_{n-m}k
$$

点积中的绝对旋转抵消，只留下相对距离 $n-m$。不同二维通道使用不同频率：高频通道很快转一圈，敏感于局部差异；低频通道转得慢，承载更长尺度的信息。

<figure class="teaching-figure concept-figure"><img src="/illustrations/rope-relative-rotation.svg" alt="RoPE 将位置 m 和 n 编码为 Query 与 Key 的二维旋转，点积只留下相对角度 n-m"><figcaption>RoPE 的关键不是“给 token 加一个正弦向量”，而是直接旋转 Q/K，使 Attention 点积成为内容与相对距离的联合函数。</figcaption></figure>

训练最长 8K 并不意味着旋转公式在 128K 仍可靠：新角度组合可能超出训练分布，频率周期还可能产生混叠。NTK-aware scaling、YaRN、位置插值或增大 base 都是在改变长距离频率分布，通常还要继续用长文本训练。参见 [RoFormer](https://arxiv.org/pdf/2104.09864.pdf)与 [CMU L22 第 22–26 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-22-longcontext.pdf#page=22)。

### 3.4 GQA

多个 Query 头共享较少的 K/V 头，减少 KV Cache。它在 MHA 的质量与 MQA 的缓存效率之间折中。

设 Query head 数为 $h$，KV head 数为 $h_{kv}$，每头维度为 $d_h$：

- MHA：$h_{kv}=h$，每个 Query head 有独立 K/V；
- GQA：$1<h_{kv}<h$，每组 $h/h_{kv}$ 个 Query heads 共用 K/V；
- MQA：$h_{kv}=1$，全部 Query heads 共用一组 K/V。

单 batch、$L$ 层、上下文 $T$、每元素 $b$ bytes 时，KV Cache 近似为：

$$
\boxed{2LTh_{kv}d_hb}
$$

前面的 2 分别是 K 和 V。以 $L=32,T=8192,h=32,h_{kv}=8,d_h=128$、BF16 为例，GQA 缓存约 1 GiB；相同配置的 MHA 约 4 GiB。节省来自 K/V 变少，不是 Query 也变少。原论文见 [GQA](https://arxiv.org/pdf/2305.13245.pdf)。

<ModernDecoderLab />

### 3.5 Pre-Norm 为什么更容易训练深网络

Post-Norm 把归一化放在残差相加之后：

$$x_{\ell+1}=\operatorname{Norm}(x_\ell+F(x_\ell))$$

Pre-Norm 则先归一化分支输入：

$$x_{\ell+1}=x_\ell+F(\operatorname{Norm}(x_\ell))$$

第二式对 $x_\ell$ 的导数显式含有恒等项 $I$：

$$
\frac{\partial x_{\ell+1}}{\partial x_\ell}
=I+\frac{\partial F}{\partial\operatorname{Norm}(x_\ell)}
\frac{\partial\operatorname{Norm}(x_\ell)}{\partial x_\ell}
$$

即使变换分支的梯度很小，仍有一条直接加法路径。这不代表 Pre-Norm 在所有指标上绝对更好，但它通常让初始化和深层优化更稳，减少对极长 warmup 的依赖。[CMU L05 第 51 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-05-attention-transformers.pdf#page=51)与 [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745.pdf)给出该比较。

具体 LLaMA 版本的层数、维度、词表和上下文不同，不应把上述组件当成所有版本完全一致的清单。CMU Slides 第 56 页引用的“约 10× 更高效”是特定 scaling 曲线、数据和配方下的经验结果，不是单个组件在任意模型上固定带来 10× 加速。架构改动必须用等参数、等 token 或等 FLOPs 的消融来归因。

## 4. Instruct 模型不是换了骨架

Base model 主要学续写分布；Instruct/Chat model 通常在同一底座上进一步接受 SFT、偏好学习或 RL。聊天模板会把 system/user/assistant 角色编码为特殊 token。忽略官方模板可能明显破坏效果。

## 5. 标准 Attention 的长序列账本

<figure class="teaching-figure concept-figure"><img src="/illustrations/attention-vs-ssm-sparse.webp" alt="Attention 保存完整历史供当前查询回看，SSM 将历史压入固定状态"><figcaption>左边用更大的历史档案换精确内容寻址；右边用固定状态换线性递推，也同时接受容量瓶颈。</figcaption></figure>

训练时长度 $T$ 的分数矩阵约有 $T^2$ 个元素；生成时通过 KV Cache 避免重算历史，但每层仍保存每个历史 token 的 K/V，并在新一步读取它们。

更精确地分开两次矩阵乘法。对单层、单 batch、隐藏维 $d$：

$$
QK^\top:\quad [T,d]\times[d,T]\rightarrow[T,T],\qquad O(T^2d)
$$

$$
AV:\quad [T,T]\times[T,d]\rightarrow[T,d],\qquad O(T^2d)
$$

多头把 $d$ 切成 $h\times d_h$，总 FLOPs 的数量级没有凭空乘 $h$，但朴素实现的 Attention 权重有 `[B,h,T,T]`，显存和内存访问非常昂贵。FlashAttention 通过分块与 online softmax 避免把完整 $T^2$ 矩阵写回 HBM；它仍计算精确 Attention，计算复杂度没有变成线性。

Decode 的账又不同。第 $t$ 步只有一个 Query，每层对 $t$ 个历史 Key 打分，算术约 $O(td)$；但还必须从显存读取历史 KV。batch 小时，GPU 往往不是算不动乘法，而是等待权重与缓存搬运。GQA、量化 KV、分页缓存和连续批处理主要在优化这本**带宽与容量账**。

这催生两类优化：

- 压缩或稀疏化 Attention/KV，如 GQA、MLA、滑窗；
- 改用固定状态递推，如 SSM、线性 Attention、KDA。

## 6. 状态空间模型 SSM

连续系统可写成：

$$h'(t)=Ah(t)+Bx(t),\qquad y(t)=Ch(t)$$

离散后得到递推：

$$h_t=\bar Ah_{t-1}+\bar Bx_t,\qquad y_t=Ch_t$$

$h_t$ 是固定大小状态。推理时读一个 token、更新一次状态，不需要保存全部历史 token 的 K/V。

### 6.1 连续系统怎样变成 token 递推

假设 token 之间间隔为 $\Delta$，在一个间隔内输入保持不变（zero-order hold）。连续解给出：

$$
\bar A=e^{\Delta A},\qquad
\bar B=A^{-1}(e^{\Delta A}-I)B
$$

于是：

$$
h_t=\bar Ah_{t-1}+\bar Bx_t,\qquad y_t=Ch_t+Dx_t
$$

$\Delta$ 不只是采样频率：它控制一次更新跨过多少“连续时间”。当 $A$ 的某些方向为负时，$e^{\Delta A}$ 会产生不同速度的衰减；慢模式保存更久，快模式迅速响应局部变化。实现中不能在 $A$ 不可逆时机械调用 $A^{-1}$，通常使用数值稳定的联合矩阵指数或针对结构化/对角 $A$ 的公式。[CMU L22 第 37–41 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-22-longcontext.pdf#page=37)把连续信号、离散化和递推视图依次展开。

### 6.2 一个标量状态能记多久

先看最简单的一维模型：

$$
h_t=ah_{t-1}+bx_t
$$

展开后，$k$ 步前输入的贡献是 $a^kbx_{t-k}$。若 $a=0.9$，10 步后只剩 $0.9^{10}\approx0.349$，50 步后约 $0.005$；若 $a$ 接近 1，记得更久，却也更难快速忘掉无关信息。

真实 SSM 用多个状态维度和不同时间尺度叠加，不是只有一个固定指数；但这个例子揭示了核心交换：固定状态必须决定哪些信息保留、以什么衰减速度保留。它不像 Attention 那样在需要时重新打开任意历史 token。

### 6.3 为什么同一个 SSM 又像 RNN，又像卷积

对线性时不变（LTI）参数，递推可以完全展开：

$$
y_t=C\bar Bx_t+C\bar A\bar Bx_{t-1}
+C\bar A^2\bar Bx_{t-2}+\cdots
$$

定义卷积核：

$$
K=[C\bar B,C\bar A\bar B,C\bar A^2\bar B,\ldots]
$$

就得到 $y=x*K$。同一模型因此有两种计算计划：训练时整段卷积/并行扫描，推理时递推固定状态。等价的是数学函数，不是运行时成本。

<figure class="teaching-figure concept-figure"><img src="/illustrations/ssm-recurrence-convolution.svg" alt="同一个线性时不变 SSM 的递推视图和卷积视图，分别用于固定状态推理与并行训练"><figcaption>LTI SSM 可以把递推展开为一条由 $C\bar A^k\bar B$ 组成的固定卷积核。Mamba 的参数随输入改变后，这条固定核不再存在，必须改用 selective scan。</figcaption></figure>

<figure class="teaching-figure source-figure"><a href="/paper-figures/mamba-figure-1.webp" target="_blank"><img src="/paper-figures/mamba-figure-1.webp" alt="Mamba 论文 Figure 1，输入 xt 通过选择机制生成 Bt Ct 与步长 Delta t，并更新固定状态 ht"></a><figcaption>Mamba 论文 Figure 1（PDF p.3）。橙色横线是从 $h_{t-1}$ 到 $h_t$ 的状态通道；蓝色选择机制让 $B_t,C_t,\Delta_t$ 随当前输入 $x_t$ 改变；右下 GPU 金字塔强调算法必须照顾 SRAM/HBM 层次。图同时表达“选择性”和“硬件感知”，缺一都不是完整 Mamba 故事。<a href="https://arxiv.org/pdf/2312.00752.pdf#page=3">打开原论文第 3 页</a>。</figcaption></figure>

### PyTorch：先实现最普通的对角递推

```python
import torch

def diagonal_ssm(x, a, b, c):
    # x: [T, D]；a/b/c: [D]
    state = torch.zeros_like(a)
    outputs = []
    for token in x:
        state = a * state + b * token
        outputs.append(c * state)
    return torch.stack(outputs)

x = torch.randn(6, 8)
a = torch.sigmoid(torch.randn(8))  # 0~1，控制旧状态保留
b = torch.randn(8)                 # 控制新输入写入
c = torch.randn(8)                 # 控制状态怎样读出
print(diagonal_ssm(x, a, b, c).shape)  # [6, 8]
```

这是解释 $h_t=\bar Ah_{t-1}+\bar Bx_t$ 的教学递推，不是 Mamba 复现：Mamba 让部分参数随输入变化、扩大内部状态，并用 selective scan 与内存层次共设计来获得速度。

### 6.4 并行 scan 为什么可行

每一步仿射更新可记成二元组 $(A_t,b_t)$，表示 $h_t=A_th_{t-1}+b_t$。相邻两步可以合成为：

$$
(A_2,b_2)\circ(A_1,b_1)
=(A_2A_1,A_2b_1+b_2)
$$

这个组合满足结合律，所以可以像前缀和一样用树形 scan 在 $O(\log T)$ 并行深度内计算各位置状态，尽管单条生成轨迹仍要逐 token 更新。算法复杂度低也不自动等于墙钟更快：若内核频繁把大状态写回 HBM，实际吞吐仍可能输给高度优化的 Attention。

## 7. Mamba：让状态更新看输入

固定 $A,B,C$ 对所有 token 使用同一套写入、衰减和读取规则，难以完成“只记住被标记的彩色 token”这类 selective copying。Mamba / S6 让：

$$
B_t=s_B(x_t),\qquad C_t=s_C(x_t),\qquad
\Delta_t=\operatorname{softplus}(s_\Delta(x_t))
$$

- $B_t$：当前输入怎样写入状态；
- $C_t$：当前状态怎样读出；
- $\Delta_t$：这一步推进多快，从而影响遗忘强度；
- $A$：通常仍采用便于稳定与高效计算的结构化形式。

于是同一个词在不同上下文中可以被强写入、弱写入或快速跳过。代价是参数随时间变化，不能再预先构造一条固定卷积核。

### 7.1 Selective scan 不只是“把 for 循环放上 GPU”

Mamba 的硬件感知实现避免把每个位置的大状态完整物化到 HBM：

1. 从慢速 HBM 读取输入和参数；
2. 在更快的 SRAM 中完成离散化和 scan；
3. 只写回必要输出；
4. 反向时重算部分中间状态，以更多 FLOPs 换更少内存流量。

因此 Mamba 的故事由两部分组成：**选择性状态更新**解决表达能力，**融合 selective scan**解决硬件效率。只有线性复杂度公式而没有合适内核，不能保证实际速度。[CMU L22 第 44–50 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-22-longcontext.pdf#page=44)逐页从 S4、S6 参数变化推到 parallel scan；[Mamba Figure 1](https://arxiv.org/pdf/2312.00752.pdf#page=3)同时画出选择机制与 GPU 内存层次。

### 7.2 固定状态的能力边界

选择性提高了“写什么”的能力，却没有让状态容量随上下文增长。两个历史片段压到同一状态后，后面无法像 Attention 那样按内容地址重新区分。评估时至少分开：

- **语言建模 loss**：平均预测是否更好；
- **selective copying / induction**：能否按模式复制指定信息；
- **needle retrieval**：能否从很远处精确取回一次性细节；
- **长度外推**：训练短、测试长时是否稳定；
- **真实吞吐**：Prefill、Decode、batch 和硬件分别测量。

某一项领先不能自动推出其他项领先。

## 8. RWKV 与 TTT 的方向感

- RWKV 将类似 Attention 的加权聚合组织成可递推更新，训练可并行、推理按状态递归；
- TTT 在测试序列上执行小型学习更新，把隐藏状态本身看成一个可训练模型；
- K3 的 KDA 则用带门和 delta rule 的 fast-weight state，并周期性插入 MLA 全局检索。

它们都在尝试“不保留完整历史也能利用长上下文”，但状态定义、更新规则和训练方式不同。

| 路线 | 历史保存在什么里 | 每步更新 | 擅长换取什么 | 主要风险 |
|---|---|---|---|---|
| Softmax Attention | 每个历史 token 的 K/V | 追加 K/V，再内容寻址 | 精确、灵活检索 | KV 随长度增长，打分成本高 |
| Linear Attention / KDA | 固定大小 fast-weight 矩阵 | 外积写入、门控/Delta Rule 修正 | 线性递推、并行扫描 | 状态冲突与容量瓶颈 |
| SSM / Mamba | 固定状态向量/矩阵 | 结构化衰减、输入选择性写读 | 长序列吞吐、常数状态 | 精确回忆与任务依赖 |
| RWKV | 可递推加权统计 | 类 Attention 的时间混合 | 训练并行、Decode 常数状态 | 实现与长期选择性 |
| TTT | 一个小模型的权重 | 测试时做学习更新 | 用在线学习扩大状态表达 | 更新成本、稳定性、遗忘 |

K3 采用 KDA 与 MLA 混合，正说明工程选择不一定是“Attention 或递推二选一”：局部大部分层用便宜状态，高价值位置周期性保留全局内容寻址，可以让两类机制互补。

## 9. 选择架构要看五本账

| 账本 | 问题 |
|---|---|
| 质量 | 目标任务和长度上是否准确 |
| 计算 | Prefill/Decode 各要多少 FLOPs |
| 内存 | 参数、激活、KV/状态各占多少 |
| 并行 | 训练和批量服务是否利用硬件 |
| 生态 | 内核、量化、框架和部署是否成熟 |

“理论 $O(T)$”只回答其中一个问题。

把选择写成可验证问题：

1. **上下文任务是什么**：模糊主题建模还是逐字 needle retrieval？
2. **训练长度和部署长度是多少**：外推不是只改一个配置项；
3. **Prefill 与 Decode 哪个占主导**：二者瓶颈不同；
4. **状态/缓存是否随 batch 放大**：单请求省内存不等于高并发一定省；
5. **是否有成熟内核**：复杂度论文与生产吞吐之间隔着编译、融合、量化和调度；
6. **比较预算是否一致**：至少控制参数、训练 token 或 FLOPs 中的一项，并报告其余差异。

<ConceptCheck question="SSM 推理时最典型的历史表示是什么？" :options="['保存每一层所有 token 的完整 QKV 表','维护固定大小、随 token 更新的状态','完全不使用历史信息']" :answer="1" explanation="固定状态带来线性递推，但也形成容量瓶颈，需要设计选择性更新。" />

> 本课对应原书第 2.5–2.6 节（PDF 第 73–99 页），并连接 GPT/LLaMA 组件、SSM、Mamba、RWKV、TTT 与 K3 的混合路线。

<ChapterReadings lesson="16-decoder-ssm" />
