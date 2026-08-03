---
title: 第 36 课 量化与低精度计算
description: 从数值刻度理解 FP16、BF16、INT8、INT4、PTQ、QAT 与量化误差
---

# 第 36 课　量化与低精度计算

<div class="lesson-lead">量化像把一把连续刻度尺换成有限格子的尺。格子越少，模型越省内存、矩阵乘可能越快；但原来的数必须被舍入，异常值和敏感层会把误差放大。</div>

<figure class="teaching-figure"><img src="/illustrations/mla-cache-compare-sparse.webp" alt="高维状态与压缩状态的存储对比"><figcaption>量化和 MLA 都在减少状态成本，但对象不同：量化改变数值精度，MLA 改变需要保存的表示结构。</figcaption></figure>

::: info 名校课程来源
本课沿 [CMU Quantization](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-19-quantization.pdf) 的“数字怎样存储 → 线性映射 → 异常值 → 训练/推理权衡”展开，并结合[台大 Inference Recitation](https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/w6-LLMInferenceEval.pdf)与 LLM Systems 的 [Quantization I](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-16-quantization-80e192f2e967b00c68b29faa9d9e71de.pdf)、[Quantization II](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-17-quantization2-bec91c67e6870c9c398fcc4a22f0b446.pdf)。原论文重点核对 [LLM.int8()](https://arxiv.org/pdf/2208.07339.pdf)、[GPTQ](https://arxiv.org/pdf/2210.17323.pdf) 与 [QLoRA](https://arxiv.org/pdf/2305.14314.pdf)。
:::

## 1. 先算一笔显存账

70B 参数若用 FP16，每个参数 2 bytes，仅权重约 140GB；INT8 约 70GB；INT4 约 35GB。实际运行还要加 KV Cache、激活、临时 buffer 和框架开销，所以“能装下权重”不等于“能服务”。

## 2. 线性量化的最小公式

把浮点 $x$ 映射到整数 $q$：

$$
q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max})
$$

$s$ 是 scale，$z$ 是 zero-point。使用时近似恢复 $\hat x=s(q-z)$。如果一个组里既有大量小值又有极端大值，scale 会被异常值拉大，小值只能挤在少数格子里。

### PyTorch：手写对称 INT8 fake quantization

```python
import torch

def fake_quantize_int8(x):
    max_abs = x.abs().amax().clamp_min(1e-8)
    scale = max_abs / 127
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    x_hat = q.float() * scale
    return q, x_hat, scale

weight = torch.tensor([-3.2, -0.7, 0.0, 1.1, 2.8])
q, restored, scale = fake_quantize_int8(weight)
print(q, restored, (restored - weight).abs())
```

`q` 是真正的整数编码，`restored` 是反量化后用于观察误差的浮点近似。这只是 per-tensor 对称量化；实际高性能 kernel 会在矩阵乘中融合反量化，并常按 channel 或 group 保存不同 scale。

<figure class="teaching-figure source-figure"><a href="/lectures/images/awq-schema.png" target="_blank"><img src="/lectures/images/awq-schema.png" alt="Stanford CS336 展示 AWQ 依据激活重要性保护少量显著权重的量化流程"></a><figcaption>Stanford CS336 Lecture 10 的 AWQ 示意图。量化误差不是所有权重同样重要：校准激活帮助识别敏感通道，再选择缩放和量化方案。它仍属于训练后权重量化，校准集分布若偏离真实请求，保护的方向也会偏。<a href="/lectures/?trace=var/traces/lecture_10.json">打开可执行 Slides</a>。</figcaption></figure>

## 3. 粒度决定元数据与误差

- **Per-tensor**：整个张量一套 scale，最省元数据但最粗；
- **Per-channel**：每个输出通道一套，通常更准；
- **Group-wise**：每 32/64/128 个权重一组，是 INT4 常见折中；
- **Per-token activation**：针对输入动态算 scale，代价更高。

## 4. PTQ 与 QAT

PTQ 在训练后用校准数据确定 scale，部署快、成本低；QAT 在训练中模拟舍入，让参数学会适应量化噪声，通常更准但要重新训练。权重量化、激活量化和 KV Cache 量化要分开评测，因为它们的误差传播不同。

## 5. 常见方法在解决什么

- SmoothQuant 把激活异常值的一部分难度迁到权重；
- GPTQ / AWQ 依据校准样本或激活重要性选择更小误差的权重量化；
- FP8 保留浮点指数，更适合训练和现代加速器矩阵乘；
- KV Cache 量化直接降低长上下文和高并发的缓存成本。

## 6. 量化评测不能只看困惑度

至少分开测知识、数学、代码、长上下文、生成稳定性和工具调用；同时报告首 token 延迟、每 token 延迟、吞吐、峰值显存和实际硬件。若硬件没有高效 INT4 kernel，文件变小也未必变快。

## 7. 浮点数先看“范围”和“精细度”

浮点通常由符号、指数和尾数组成：

$$
x=(-1)^s\times(1.\text{mantissa})\times 2^{e-\text{bias}}
$$

- 指数位决定能表示多大/多小的范围；
- 尾数位决定同一数量级附近能分多细；
- FP16 尾数更多但指数范围较窄；
- BF16 保留与 FP32 类似的指数范围，尾数更少，训练时更不易溢出但精度更粗。

“16 bit”不是一种格式。训练、激活和权重各自需要的范围不同，选择要看数值分布与硬件支持。

## 8. 对称与非对称量化手算

### 对称量化

把区间 $[-a,a]$ 映射到 INT8 的 $[-127,127]$：

$$
s=\frac{a}{127},\qquad q=\operatorname{round}(x/s)
$$

零点固定为 0，矩阵乘实现简单，适合以 0 为中心的权重。

### 非对称量化

对范围 $[x_{min},x_{max}]$：

$$
s=\frac{x_{max}-x_{min}}{q_{max}-q_{min}},
\qquad z=\operatorname{round}(q_{min}-x_{min}/s)
$$

适合 ReLU 后大多为正的激活，可利用全部整数格子；多了 zero-point 处理。

## 9. Group size 为什么是质量/速度旋钮

INT4 常把连续 $g$ 个权重共享 scale：

```text
[64 个权重] + 1 个 scale
```

- $g$ 小：scale 更贴合局部分布，误差小；元数据和反量化操作更多；
- $g$ 大：压缩率与 kernel 简单度更好；异常值影响更多普通值。

所以“INT4 模型大小”还取决于 scale、zero-point、索引和打包对齐，不能只用参数量 × 0.5 byte。

## 10. 为什么异常值特别麻烦

一组数为：

```text
[-0.08, 0.03, 0.10, -0.04, 7.20]
```

若所有值共享 scale，7.20 决定范围，其余小值可能都舍入到 0 或相邻两格。Transformer 激活中的异常通道可能跨许多层稳定存在，因此简单 per-tensor INT8 会明显掉点。

三条解决路线：

- 更细粒度 scale，让异常值只影响小组；
- 保留异常通道为高精度，普通通道低精度；
- 通过缩放把激活难度迁移到更容易量化的权重（SmoothQuant 思路）。

## 11. Weight-only、W8A8 与 KV 量化的区别

| 方案 | 低精度对象 | 主要节省 | 典型难点 |
|---|---|---|---|
| W4A16 | 权重 INT4，激活 FP16/BF16 | 权重显存与带宽 | 每次需反量化，算子是否融合 |
| W8A8 | 权重与激活 INT8 | 权重 + GEMM 吞吐 | 激活异常值与校准 |
| FP8 | 权重/激活浮点 8 bit | 训练/推理吞吐 | 格式、scale 与硬件 |
| KV INT8/INT4 | K/V Cache | 长上下文与并发显存 | 误差随层和生成累积 |

Weight-only 很适合内存带宽受限的单 batch Decode；高 batch 或 Prefill 更可能受矩阵计算限制，需要真正的低精度 GEMM。

## 12. PTQ 校准集应该覆盖什么

校准集无需像训练集一样巨大，但必须代表部署分布：

- 不同语言与文本长度；
- 系统 Prompt、工具 schema 和代码；
- 普通聊天与长上下文；
- 容易产生异常激活的数学/代码；
- 真实 batch 与 attention mask。

只用 128 条短英文句子校准，再部署中文长代码 Agent，scale 很可能不匹配。

## 13. GPTQ、AWQ、SmoothQuant 各在优化什么

- **GPTQ**：按层近似二阶误差，逐步量化权重并补偿剩余权重；
- **AWQ**：利用激活统计保护少量重要权重通道；
- **SmoothQuant**：重缩放权重与激活，把激活异常难题转移一部分到权重；
- **QLoRA**：以 4-bit 基座节省存储，在其上训练 LoRA，并不等于把所有训练计算都变成 4-bit。

名称不能替代设置。必须记录 bit、group size、对称性、校准数据、哪些层例外与实际 kernel。

## 14. 端到端延迟为什么不会按 bit 线性下降

一次请求还包含：tokenization、CPU 调度、权重/scale 读取、反量化、Attention、KV Cache、采样和网络返回。若 INT4 GEMM 先解包到 FP16，或 GPU 对该形状没有高效 kernel，理论 4 倍压缩不会变成 4 倍加速。

因此至少分别测：

```text
模型加载大小
峰值显存
Prefill tokens/s
Decode tokens/s
TTFT / TPOT / P99
功耗与每百万 token 成本
```

## 15. 一次量化回归怎样定位损失

1. 先只量化权重，固定 KV 与激活；
2. 按层恢复高精度，找敏感层；
3. 比较 per-tensor、per-channel、group-wise；
4. 检查短/长上下文差异；
5. 查看 token 概率漂移，而不只看最终答案；
6. 再单独加入 KV/activation 量化。

Embedding、输出头、第一/最后层有时更敏感，不能假定所有矩阵同样适合低 bit。

## 本课自测

1. INT4 权重为什么不一定带来 4 倍端到端加速？
2. Per-channel 为什么通常比 per-tensor 准？
3. 权重量化与 KV Cache 量化各影响哪部分内存？

下一课看不近似结果也能大幅提速的 Attention 方法：[FlashAttention 与长上下文](/beginner/31-efficient-attention)。

<ChapterReadings lesson="30-quantization" />
