---
title: 第 21 课 LoRA 深入：秩、目标层、合并与实验
description: 用完整参数账本和手算理解 LoRA、QLoRA 与适配器部署
---

# 第 21 课　LoRA 深入：秩、目标层、合并与实验

<div class="lesson-lead">上一课看见了 PEFT 全景；本课只追一条 LoRA 支路，从矩阵尺寸、初始化、缩放、目标层一路走到训练、合并和多适配器部署。</div>

::: info 本课资料地图：公式、实现与系统账本一起看
- 原论文与 Figure 1：[LoRA](https://arxiv.org/pdf/2106.09685.pdf)；
- 4-bit 底座与分页优化器：[QLoRA](https://arxiv.org/pdf/2305.14314.pdf)；
- 逐步训练课件：[台大 ADL · LoRA Training](https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/w5-LoRA.pdf)；
- 大模型微调系统视角：[LLM Systems · Efficient Fine-tuning](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-18-peft-1555d9a7770e87fb10e2e95bf46ef12d.pdf)。

正文代码是从论文公式直接翻译出的教学模块，不依赖 PEFT 框架，便于看清哪些参数冻结、哪些参数获得梯度。
:::

<figure class="teaching-figure">
  <img src="/illustrations/foundations-peft-lora.webp" alt="冻结大矩阵旁边的两个低秩薄矩阵构成可训练修正">
  <figcaption>原矩阵负责底座能力；A 把输入压到低秩空间，B 再展开。训练时只有这条窄支路更新。</figcaption>
</figure>

<figure class="teaching-figure infographic-figure"><a href="/illustrations/lora-one-page-infographic.webp" target="_blank"><img src="/illustrations/lora-one-page-infographic.webp" alt="LoRA 从冻结底座、低秩支路、训练参数到合并部署的中文总览图"></a><figcaption>先用总览图建立参数账本，再在正文手算矩阵尺寸。点击图片可放大查看。</figcaption></figure>

## 1. 先把尺寸写清楚

线性层 $W\in\mathbb R^{d_{out}\times d_{in}}$，输入 $x\in\mathbb R^{d_{in}}$：

$$y=Wx+\frac{\alpha}{r}BAx$$

其中：

$$A\in\mathbb R^{r\times d_{in}},\qquad B\in\mathbb R^{d_{out}\times r}$$

数据流：

```text
x(d_in) → A → r 维 → B → d_out 维 → 加到 Wx
```

<figure class="teaching-figure source-figure"><a href="/paper-figures/lora-figure-1.webp" target="_blank"><img src="/paper-figures/lora-figure-1.webp" alt="LoRA 论文 Figure 1，冻结预训练权重并并联可训练的低秩 A B 矩阵"></a><figcaption>LoRA 论文 Figure 1（PDF p.1）。蓝色 $W$ 是冻结的预训练矩阵；橙色支路先经 $A$ 降到秩 $r$，再经初始为零的 $B$ 回到输出维度；两路结果相加得到 $h$。图中的 $B=0$ 保证刚挂载适配器时输出不变。<a href="https://arxiv.org/pdf/2106.09685.pdf#page=1">打开原论文第 1 页</a>。</figcaption></figure>

## 2. 参数量公式

原矩阵参数：

$$N_{full}=d_{out}d_{in}$$

LoRA 参数：

$$N_{lora}=r(d_{in}+d_{out})$$

若输入输出均 4096、$r=16$：

```text
原矩阵：4096 × 4096 = 16,777,216
LoRA：16 × 4096 + 4096 × 16 = 131,072
比例：约 0.78%
```

模型中有很多矩阵，最终比例取决于对哪些层挂 LoRA。

更一般地，若目标矩阵集合为 $\mathcal S$，第 $j$ 个矩阵 shape 是 $d_{out}^{(j)}\times d_{in}^{(j)}$：

$$
P_{LoRA}=\sum_{j\in\mathcal S}r_j\left(d_{in}^{(j)}+d_{out}^{(j)}\right)
$$

不要用“层数 × $2rd$”估整个模型，除非所有目标矩阵确实都是 $d\times d$ 且每层只挂一个。GQA 的 K/V 投影输出维可能小于 $d$，fused QKV 又可能存成一个大矩阵，最可靠的方法是读取 checkpoint 的实际 weight shape 后逐项求和。

<LoRALab />

## 3. 为什么常让初始增量为零

常见初始化让一块矩阵随机、另一块为零，使初始 $BA=0$。这样训练开始时模型输出与原底座一致，不会一挂适配器就突然改变。随后梯度逐步打开修正路径。

### PyTorch：从公式写出一个最小 LoRA 线性层

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.requires_grad_(False)  # W 与 bias 都冻结

        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        base_output = self.base(x)             # x W^T
        low_rank_output = F.linear(F.linear(x, self.A), self.B)
        return base_output + self.scale * low_rank_output

    @torch.no_grad()
    def merged_weight(self):
        return self.base.weight + self.scale * (self.B @ self.A)

layer = LoRALinear(4096, 4096, rank=8, alpha=16)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
print(trainable)  # 65,536；底座参数不计入
```

PyTorch 的 `nn.Linear` 内部使用 $xW^\top$，因此 `A`、`B` 的储存形状仍按公式设置，`F.linear` 会自动处理转置。训练第一步时 $B=0$，所以 $BA=0$；但 $B$ 会先获得梯度，随后非零的 $B$ 才让梯度继续传到 $A$。

<figure class="teaching-figure concept-figure"><img src="/illustrations/lora-zero-init-gradient.svg" alt="LoRA 在 A 随机、B 为零初始化时，首步 B 获得梯度，更新后 A 才开始获得梯度的流程"><figcaption>零初始化保证函数起点不变，但不会让支路永远没有梯度。首步只有 B 打开，之后 A 与 B 才共同学习。</figcaption></figure>

### 为什么低秩可能够用

LoRA 假设任务适配需要的权重更新 $\Delta W$ 具有较低的“有效秩”，所以用 $BA$ 近似，而不是自由学习全部 $d_{out}d_{in}$ 个数。这是经验性归纳偏置，不是每个任务的定理：若任务改变很大、数据复杂或目标层选错，低 rank 会形成容量瓶颈。

`rank(B@A) ≤ r`，但“参数少”不等于“表达只有 r 个固定技能”。A、B 会联合学习输入与输出方向，不同层也有不同的低秩子空间。

## 4. r、alpha、dropout 分别是什么

- $r$：低秩通道宽度；越大容量和参数越多；
- $\alpha$：增量缩放，常通过 $\alpha/r$ 归一；
- LoRA dropout：训练时随机丢一部分支路输入，帮助正则化。

它们互相影响，不应只单独扫 r。数据量小、任务简单时高秩可能过拟合；任务变化大时低秩可能欠拟合。

经典缩放常写成 $\alpha/r$。如果只增大 $r$ 而固定 $\alpha$，单个方向的增量尺度也会变化，所以“rank 消融”可能同时改变容量与优化尺度。部分实现提供 rank-stabilized LoRA（常见缩放为 $\alpha/\sqrt r$）；无论使用哪种约定，都要把 `r`、`alpha`、scaling rule 和初始化一起记录。

LoRA dropout 只在训练态起作用。做 merge 数值对账时要先切换 `eval()`，否则 dropout 会让未合并分支每次输出不同，无法与 $W+BA$ 精确比较。

## 5. Target Modules 怎样选

一个 Transformer 层常有：

```text
Attention: q_proj, k_proj, v_proj, o_proj
FFN: gate_proj, up_proj, down_proj
```

选择越多，可训练容量越大，显存和检查点也越大。先用小验证集比较：只 Q/V、全部 Attention、Attention+FFN。不要只看训练 loss，要看目标任务和通用回归。

选择目标层其实是在下注“任务变化主要需要改哪里”：

- Q/K：改变关注位置与匹配关系；
- V/O：改变被读取内容及输出混合；
- FFN：改变逐 token 的非线性特征变换；
- embedding / lm_head：直接改变输入/输出词表映射，参数与过拟合风险另算。

这些只是结构直觉，不能从模块名直接推断真实因果作用。可控实验应保持训练 token、学习率搜索预算和总 step 不变，比较 Q/V、全 Attention、Attention+FFN；若参数数目差很多，还要增加“相同可训练参数预算”的对照。

## 6. 训练时显存去哪了

即使底座冻结，仍要：

- 保存量化或高精度底座以做前向/反向输入梯度；
- 保存部分激活；
- 保存 LoRA 参数、梯度和优化器状态；
- 存放临时矩阵、KV 或数据批次。

Gradient Checkpointing 用额外重算换激活显存；梯度累积用多次小 batch 模拟大 batch。两者不会减少模型参数本身。

### 三块显存要分开报

1. **冻结底座权重**：仍要驻留或按层加载，量化主要压这一块；
2. **可训练状态**：LoRA 参数、梯度、Adam 的一阶/二阶矩，必要时还有 FP32 master weights；
3. **激活与临时张量**：随 batch、序列长、隐藏维、Attention 实现和 checkpointing 变化。

一个常见教学口径是每个可训练参数约 `12 B = FP16/BF16 权重 2 + 梯度 2 + FP32 Adam m/v 8`；若再保留 FP32 master weight 则约 16 B。框架、优化器和分片方式会改变这个数字，必须写口径。冻结 W 不保存 W 的梯度/Adam 状态，但为了给 LoRA 计算梯度，网络仍需传播隐藏状态梯度，激活显存不会消失。

## 7. QLoRA 的三层精度

概念上可分：

1. 底座权重以低位量化格式存储；
2. 计算时按算子需要反量化到计算 dtype；
3. LoRA 参数和优化器以较高精度训练。

这样把最大块——冻结底座——压小，同时保持可训练增量的精度。量化格式、double quantization、paged optimizer 等属于具体方案，使用时以框架和硬件支持为准。

QLoRA 的关键组件应分别理解：

- **NF4**：针对近似正态分布权重设计的 4-bit 存储码；
- **double quantization**：连量化 scale/常数也进一步压缩；
- **compute dtype**：矩阵计算时不是直接用 4-bit 做所有累加，通常反量化到 BF16/FP16 等计算 dtype；
- **paged optimizer**：借助统一内存等机制缓解长序列或 checkpointing 带来的瞬时显存峰值。

所以“4-bit QLoRA”不等于训练过程每个张量都是 4 bit，也不意味着理论权重大小就是整卡峰值。量化误差、kernel 支持与吞吐必须单独测。

## 8. 合并与不合并

### 保留独立适配器

优点：一个底座加载多个任务，快速切换；易撤销和版本管理。缺点：推理多一条分支，批次内不同适配器调度更复杂。

### 合并到权重

$$W_{merged}=W+\frac{\alpha}{r}BA$$

优点：推理图与普通模型相同。缺点：每个任务需要一份合并模型；量化底座合并时还要注意精度和再次量化误差。

在无 dropout、同一 dtype 且没有量化误差时，分支与合并在数学上等价：

$$
Wx+sBAx=(W+sBA)x
$$

工程验收可随机采样输入，比较 merge 前后最大绝对误差和任务指标。若底座先反量化、加增量、再量化，误差不再只来自浮点结合律；要用部署 dtype 重新做回归。

## 9. 多个 LoRA 能直接相加吗

形式上可相加多个增量，但任务方向可能冲突，缩放也不同。适配器融合、路由或权重插值需要验证。两个单独优秀的 LoRA 合并后不一定同样优秀。

在线多租户常选择不 merge：一份底座配许多小 adapter。此时系统还要处理 adapter 缓存、版本匹配、批内不同 adapter 的 kernel 调度与冷加载尾延迟。文件小只解决存储，不自动解决高并发切换。

## 10. 数据比秩更先决定上限

训练集应包含：

- 正常样本；
- 容易混淆的边界；
- 应拒答或信息不足样本；
- 多种措辞和长度；
- 与部署模板一致的聊天格式。

低质量、重复、互相矛盾的数据会让更大 r 更快地学坏。

## 11. 最小实验矩阵

| 实验 | r | 目标层 | 数据 | 要回答的问题 |
|---|---:|---|---|---|
| A | 8 | Q/V | 固定 | 最小方案够不够 |
| B | 16 | Q/V | 固定 | 增加秩是否有效 |
| C | 8 | 全 Attention | 固定 | 扩大层覆盖是否更有效 |
| D | 8 | Q/V | 去重清洗 | 数据质量影响多大 |

固定随机种子和验证集，记录质量、训练显存、时间、检查点大小与通用能力回退。

更严格的秩实验应同时包含：

| 控制项 | 为什么 |
|---|---|
| 固定数据顺序、训练 token 与 warmup | 避免优化预算不同 |
| 为每个 r 搜索合理 LR/alpha | 避免把缩放不合适误判为容量不足 |
| 同时报 train/dev loss | 区分欠拟合与过拟合 |
| 报 3 个以上随机种子 | 小数据 PEFT 波动可能很大 |
| 通用能力回归集 | 目标任务提升可能伴随遗忘 |
| 峰值显存、tokens/s、checkpoint MiB | 参数效率不等于系统效率 |

## 12. 部署检查

- tokenizer 与聊天模板必须与训练一致；
- base model 版本必须精确匹配；
- adapter config 中 r、alpha、target modules 不能丢；
- 合并前后做数值和任务回归；
- 量化部署要单独评测；
- 保存训练数据版本与许可证信息。

另外要检查监督边界：SFT 时通常只对 assistant answer token 计算 loss，system/user token 可能作为上下文但被设为 `-100`。若聊天模板错位或 EOS 缺失，LoRA 会非常高效地学到错误格式。

<ConceptCheck question="LoRA 的 rank r 增大通常直接带来什么？" :options="['低秩支路容量和参数量增加','基础模型参数全部删除','上下文长度自动无限']" :answer="0" explanation="更高 r 提供更多更新方向，但也增加参数、显存和过拟合风险。" />

<ConceptCheck question="A 随机、B=0 初始化时，第一次 backward 最准确的描述是什么？" :options='["B 通常先获得非零梯度，A 的梯度此时为零", "A、B 永远都没有梯度", "冻结底座 W 也会被 Adam 更新"]' :answer="0" explanation="∂L/∂B 含 A，因此通常非零；∂L/∂A 含 Bᵀ，B=0 时首步为零。B 更新后 A 才开始收到梯度。" />

> 本课对应原书第 4.4–4.5 节（PDF 第 174–191 页），从公式扩展到目标层、训练账本、合并和实验设计。

## 13. 课件与论文精读路线

1. [CS224N L09 第 45–54 页](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture09-peft.pdf#page=45)：从 full fine-tuning 的 $\Delta\phi$ 走到低秩 $BA$ 与 QLoRA；
2. [LoRA 原论文 Figure 1 与第 4 节](https://arxiv.org/pdf/2106.09685.pdf#page=1)：核对冻结路径、初始化、缩放、目标模块和消融；
3. [QLoRA 原论文](https://arxiv.org/pdf/2305.14314.pdf)：分别摘出 NF4、double quantization 与 paged optimizer 解决的显存问题；
4. 对一个真实 checkpoint 打印所有 target weight shape，用本章求和公式与框架报告的 `trainable params` 对账；
5. 用同一验证集比较未合并、FP16 合并、量化后再合并三个版本的数值误差、质量与延迟。

<ChapterReadings lesson="20-lora" />
