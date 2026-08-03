---
title: 第 15 课 自动微分、优化器、框架与 GPU
description: 从计算图到 GPU kernel，理解大模型一步训练怎样真正执行
---

# 第 15 课　自动微分、优化器、框架与 GPU

<div class="lesson-lead">“反向传播”是数学规则，“自动微分”是把规则落到计算图上的程序，“GPU kernel”才是矩阵真正执行的地方。三层混在一起，就很难判断训练为什么慢或为什么爆显存。</div>

<figure class="teaching-figure"><img src="/illustrations/training-loop-sparse.webp" alt="模型前向、损失、反向与参数更新循环"><figcaption>一次训练 step：前向保存必要状态，反向计算梯度，优化器更新参数，然后才进入下一批数据。</figcaption></figure>

::: info 名校课程来源
本课承接 [CS224N Backpropagation](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture03-neuralnets.pdf) 和[台大 Neural Network Basics](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250901_NNBasics.pdf)，再用 [CS336 Lecture 2 可执行 Slides](/lectures/?trace=var/traces/lecture_02.json)、[Lecture 5 GPUs](https://stanford-cs336.github.io/spring2026/) 与 [Lecture 6 Kernels](/lectures/?trace=var/traces/lecture_06.json) 把[第 04 课的数学计算图](/beginner/03-training)落到张量、混合精度、GPU kernel、编译与 profiler。
:::

## 1. 从一个乘法节点开始

若 $y=wx$、损失为 $L(y)$，链式法则给出：

$$
\frac{\partial L}{\partial w}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial w}=\frac{\partial L}{\partial y}x
$$

大模型只是把数十亿个这样的局部运算连成计算图。自动微分框架记录“谁由谁算出”，反向时从损失沿图倒着传播局部梯度。

### PyTorch：看清一次 backward 与 optimizer step

```python
import torch

x = torch.tensor(3.0)
target = torch.tensor(12.0)
w = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.05)

for step in range(4):
    optimizer.zero_grad(set_to_none=True)
    prediction = w * x
    loss = (prediction - target).square()
    loss.backward()               # 只计算梯度，不会自动改 w
    print(step, loss.item(), w.grad.item())
    optimizer.step()              # 这里才使用梯度更新 w
```

`loss.backward()` 从标量损失沿计算图反传，把结果累加到 `w.grad`；`optimizer.step()` 才改变参数。若忘记 `zero_grad()`，下一轮梯度会叠加，这在梯度累积时是功能，在普通训练中往往是 bug。

## 2. 前向到底保存了什么

反向计算某些梯度需要前向激活。例如线性层 $Y=XW$ 要算 $\partial L/\partial W$，需要保留 $X$。层数、batch、序列长度增大后，激活可能比参数还占显存。

三种常见办法：

- **Activation checkpointing**：只保存少数边界，反向时重算中间激活；
- **混合精度**：大部分矩阵乘用 BF16/FP16，关键归约保留更高精度；
- **算子融合**：把多个小操作合成一个 kernel，减少显存往返。

## 3. 优化器维护几本账

以 Adam 为例，每个参数常有参数值、梯度、一阶动量 $m$ 和二阶动量 $v$：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

因此“7B 参数 × 2 bytes”远不是训练显存总量。还要加梯度、优化器状态、激活和临时 buffer。Muon 等优化器改变的是更新几何和状态处理，不能只看名字判断更快。

## 4. GPU 为什么适合 Transformer

GPU 有大量并行计算单元，擅长规则、密集的矩阵乘。CPU 像少数全能工程师，GPU 像成千上万名执行同一工序的工人。真正性能取决于两件事：

- **计算受限**：Tensor Core 一直有矩阵可算；
- **内存带宽受限**：大量时间在 HBM 与片上 SRAM 间搬数据。

Attention softmax、LayerNorm、激活函数等若不断读写 HBM，算术量不大也可能很慢。FlashAttention 的关键正是减少搬运，详见[第 37 课](/beginner/31-efficient-attention)。

<figure class="teaching-figure source-figure"><a href="/lectures/images/triton-softmax.png" target="_blank"><img src="/lectures/images/triton-softmax.png" alt="Stanford CS336 用 Triton 解释 Softmax kernel 如何按行加载并在片上完成归约"></a><figcaption>Stanford CS336 Lecture 6 的 Triton Softmax 图。普通实现会让减最大值、指数、求和、除法各自读写 HBM；融合 kernel 把一行搬到片上后完成全部步骤，只把结果写回一次。公式没变，减少的是 kernel launch 与数据搬运。<a href="/lectures/?trace=var/traces/lecture_06.json">打开可执行 Slides</a>。</figcaption></figure>

## 5. 深度学习框架做了什么

PyTorch 等框架负责张量、计算图、自动微分、算子调度、设备管理和分布式通信。编译器进一步尝试图优化、kernel fusion、常量折叠和形状特化。即时编译并非“把 Python 变快”这么简单，而是把稳定的张量计算图变成更少、更合适的底层 kernel。

## 6. 诊断训练慢的顺序

1. 看 GPU 利用率和每 step 时间是否稳定；
2. 分开测数据加载、前向、反向、优化器和通信；
3. 看显存峰值发生在激活、梯度还是优化器；
4. 用 profiler 定位小 kernel、同步点和 HBM 往返；
5. 最后才换更复杂的并行或编译方案。

## 7. 张量 shape 是第一种调试语言

线性层：

$$
X[B,T,d]\;W[d,m]\rightarrow Y[B,T,m]
$$

Attention：

```text
Q,K,V: [B, heads, T, head_dim]
score: [B, heads, T, T]
output: [B, T, d]
```

看到 OOM 先列每个大张量的 shape、dtype 与生命周期。一个意外 broadcast 可能把 `[B,T,1]` 与 `[B,1,T]` 变成 `[B,T,T]`，数学能运行却瞬间占满显存。

## 8. FLOPs 与显存带宽是两种不同上限

Arithmetic intensity 近似为：

$$
\frac{\text{执行 FLOPs}}{\text{从显存搬运 bytes}}
$$

大矩阵乘复用输入多，强度高，常受 Tensor Core 计算上限；逐元素激活、LayerNorm 和 Decode 小矩阵复用少，常受带宽限制。

优化要匹配瓶颈：计算受限时减少 FLOPs/用低精度；带宽受限时融合算子、减少中间写回或增加 batch 提复用。

## 9. Kernel launch 为什么会成为问题

GPU kernel 每次启动有固定调度成本。若计算图拆成大量微小操作：

```text
bias add → activation → dropout → residual add → norm
```

GPU 可能频繁启动和读写 HBM。Fusion 把它们合进一个 kernel，在寄存器/SRAM 中传中间值。收益取决于 shape、编译器与数值要求，不是所有算子都能安全融合。

## 10. FP16、BF16、FP32 在一次训练中怎样共存

典型混合精度：

- 权重参与 GEMM 的副本：BF16/FP16；
- Tensor Core 累加：常用 FP32 或更高精度累加；
- 优化器主权重与动量：FP32；
- LayerNorm、softmax 某些归约：高精度；
- 通信：可按策略压缩。

FP16 范围窄，常需 loss scaling：放大 loss/梯度避免下溢，检测溢出后跳过 step 并缩小 scale。BF16 指数范围大，通常更稳。

## 11. Activation checkpointing 应该切在哪里

把连续层划成若干段，只保存段输入，反向时重跑段内前向。检查点太密，省显存少；太疏，重算多、单段临时峰值也大。

对带 dropout 的模型，重计算必须恢复相同随机状态；否则 backward 对应的前向与原前向不同。框架通常帮你管理，但自定义 kernel 需注意。

## 12. Optimizer step 也可能是瓶颈

AdamW 要读参数、梯度、两个动量并写回，算术不多但搬运量大。模型大、micro-batch 小时，optimizer step 占比会明显。

Fused optimizer 把多次逐元素遍历合并；8-bit optimizer 压缩状态；ZeRO 分片状态。三者分别优化 kernel、存储精度与设备复制，不能混为一项。

## 13. Eager、Graph compile 与自定义 kernel

| 层级 | 优点 | 风险 |
|---|---|---|
| Eager | 易调试、动态灵活 | Python/launch 开销，融合少 |
| 图编译 | 自动 fusion、布局与代码生成 | 动态 shape 重编译、graph break |
| 自定义 Triton/CUDA | 针对热点最强控制 | 开发、验证与维护成本高 |

先用 profiler 证明热点，再写自定义 kernel。优化一个只占 2% 时间的算子，即使快 10 倍，端到端最多也只提升约 2%。

## 14. 数据管线怎样让 GPU 饿死

CPU 解压、PDF/图像解码、tokenization、网络存储或 Python collate 太慢，GPU 会在 step 间空等。改进：

- 离线 tokenize/pack；
- 多 worker 与预取；
- pinned memory、异步 host-to-device；
- 连续读取 shard，避免大量小文件；
- 记录数据等待时间，而不只看 GPU utilization 平均值。

## 15. Profiler 时间线怎样读

一次 step 标记：data → H2D → forward → backward → all-reduce → optimizer。观察：

- kernel 间大空隙：CPU/同步/launch；
- H2D 与计算不重叠：预取或 pinned memory；
- 许多小 kernel：fusion/编译机会；
- collective 后 GPU 空闲：通信未重叠；
- 同一 shape 反复编译：动态维或 graph break。

优化后用相同 warmup、batch、长度分布和硬件重测，避免缓存或编译首次成本造成假提升。

## 16. 数值正确性是性能优化的底线

低精度和自定义 kernel 验证：

1. 与 FP32/reference 在小 shape 比输出与梯度；
2. 覆盖极端值、mask、非连续内存与奇数长度；
3. 检查 NaN/Inf 与误差随长度累积；
4. 跑短训练比较 loss 曲线；
5. 再做端到端能力回归。

“训练没崩”不代表 kernel 正确，微小系统误差可能在长训练后变成能力差异。

## 本课自测

- 自动微分与反向传播有什么区别？
- 为什么 checkpointing 省显存却增加计算？
- 参数只占 14GB 时，训练为什么仍可能需要上百 GB？

下一课把单卡训练拆到多台机器：[分布式训练与通信](/beginner/27-distributed-training)。

<ChapterReadings lesson="26-training-engineering" />
