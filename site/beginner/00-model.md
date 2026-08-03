---
title: 第 00 课 模型、参数与训练
description: 从任务、分数、概率、参数、梯度到泛化，建立机器学习最基本的完整框架
---

# 第 00 课　模型、参数与训练：模型到底是什么

<div class="lesson-lead">一句话：模型是一个“输入数字 → 按参数做计算 → 输出数字”的函数；训练是自动寻找更合适参数的过程。</div>

::: info 本课资料地图：先听课，再读论文
- 主线逐页整合 [CMU ANLP L01 · Introduction & Fundamentals](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-01-intro.pdf)、[L02 · Learned Representations](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-02-representations.pdf)与 [L03 · Autoregressive Language Modeling](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-03-lm.pdf)，覆盖建模—学习—推断、张量/表示、交叉熵/反向传播和泛化实验。
- 补充入口：[CMU LLM Applications · Origins of LLMs](https://storage.googleapis.com/cmu-llms/2026/2026-01-13-llm-history.pdf)负责历史直觉，[CS224N · Pretraining](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture07-pretraining.pdf)负责现代训练全景，[Stanford CS336 Lecture 1](/lectures/?trace=var/traces/lecture_01.json)则从“亲手造一个语言模型”解释课程全栈与实现目标。
- 奠基论文：[Bengio et al., 2003 · A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)展示“词向量 + 神经网络 + 下一词概率”的早期完整形态。
- 规模论文：[Chinchilla](/papers/chinchilla)提醒我们：参数更大不等于训练得更好，数据量和计算预算必须匹配。

第一次学习只读正文；完成本课后再打开论文，看摘要、模型图和结论，不要求读懂全部实验。
:::

<figure class="teaching-figure">
  <img src="/illustrations/beginner-model-machine.webp" alt="不同输入进入参数机器，产生多个候选输出，并由错误反馈调整机器">
  <figcaption>模型先把输入变成候选概率；训练时，错误信号沿虚线返回，逐步调整内部参数。</figcaption>
</figure>
<div class="visual-key"><div><b>左边</b>数据可以是文字、图像或其他数字表示。</div><div><b>中间</b>参数决定每一步计算怎样发生。</div><div><b>右边与回路</b>输出概率；误差只在训练时用来更新参数。</div></div>

<figure class="teaching-figure concept-figure"><img src="/illustrations/model-learning-inference-map.svg" alt="建模、学习与推断以及 Train Validation Test 三种数据职责的总图"><figcaption>CMU L01 的总框架：模型/参数化规定“怎么算分”，学习规定“怎样用监督改参数”，推断规定“参数固定后怎样选输出”。Train、Validation 和 Test 又分别承担更新、选择和最终验收，不能混用。</figcaption></figure>

## 1. 先从普通函数开始

自动售货机也可以看成一个函数：

```text
输入：商品编号 + 金额
内部规则：检查金额、库存、找零
输出：商品 + 余额
```

语言模型同样有输入、内部规则和输出：

```text
输入：一串 token 编号
内部规则：数千次矩阵运算，规则由参数决定
输出：下一个 token 的一组概率
```

差别在于：售货机规则由工程师逐条写死；神经网络的大部分规则通过数据训练得到。

### 1.1 任务先写成输入集合到输出集合

课程用 $x\in\mathcal X$ 表示输入，用 $y\in\mathcal Y$ 表示输出。先不要被符号吓到，它只是说清“允许输入什么、希望输出什么”：

| 任务 | 输入 $x$ | 输出 $y$ |
|---|---|---|
| 情感分类 | 一段影评 | 正面 / 中性 / 负面 |
| 翻译 | 英文句子 | 中文句子 |
| 检索 | 搜索问题 | 排序后的文档列表 |
| 图像描述 | 图片像素 | 一段文字 |
| Agent | 当前环境状态与历史 | 下一步动作 |

分类的候选输出只有几个；生成一句话的候选是所有可能 token 序列，数量几乎无法枚举；Agent 的输出又会改变环境，产生下一个输入。任务形式不同，但“输入—参数化计算—输出”仍是共同骨架。

### 1.2 模型不是完整产品

一个可以使用的系统通常还包含 Tokenizer、Prompt 模板、检索器、工具、解码器、缓存、安全规则和监控。模型只负责其中可学习的函数。把规则引擎、检索或工具带来的提升全部说成“模型学会了”，会让实验归因失真。

建立 NLP 系统也不只有神经网络一种方式：

| 方法 | 监督从哪里来 | 主要优点 | 主要限制 |
|---|---|---|---|
| 手写规则 | 专家直接写逻辑 | 可解释、少数据 | 长尾规则爆炸，难覆盖隐喻与语言变化 |
| Prompt / few-shot | 人写说明与少量例子 | 上手快，不改权重 | 对措辞敏感，性能难保证 |
| 监督学习 | 输入—目标对 | 信号直接、训练稳定 | 标注贵，只覆盖示范分布 |
| 强化学习 | 环境、动作与奖励 | 能探索数据外策略 | 奖励设计和信用分配困难 |

选择方法前要先盘点能获得哪种数据，而不是看到新算法名就套用。

## 2. 参数不是知识条目

假设最简单的模型：

$$
y=w\times x
$$

这里只有一个参数 `w`。如果 `w=3`，输入 `x=2`，输出就是 `6`。

神经网络的参数也是数字，只是数量可能达到数十亿或万亿。它们不是一个个“法国首都是巴黎”的数据库格子，而是共同改变大量中间表示与输出概率。

### 2.1 参数如何从“分数”变成“选择”

最简单的分类器可以先提取特征 $f(x)$，再用参数 $w$ 打分：

$$
s_\theta(x)=w^\top f(x)
$$

正分可判为正面，负分判为负面。多分类时，对每个候选 $y$ 给一个 logit $s_\theta(x,y)$，再用 Softmax 变成总和为 1 的概率：

$$
p_\theta(y\mid x)=
\frac{\exp(s_\theta(x,y))}{\sum_{y'}\exp(s_\theta(x,y'))}
$$

这里要分清三层：

1. **logit / score**：还没归一化的任意实数；
2. **probability**：Softmax 后的相对概率；
3. **decision**：取最大概率、按概率采样，或用搜索比较完整序列。

同一套模型概率，换一种决策规则就可能产生不同输出。生成温度、top-p、Beam Search 属于推断设置，不是训练后参数。

### 2.2 张量只是带形状的数字盒子

神经网络代码反复出现 scalar、vector、matrix 和 tensor：

| 名称 | 示例形状 | 直觉 |
|---|---|---|
| 标量 scalar | `[]` | 一个 loss 或学习率 |
| 向量 vector | `[d]` | 一个 token 的 $d$ 维表示 |
| 矩阵 matrix | `[V,d]` | $V$ 个 token 的 Embedding 表 |
| 张量 tensor | `[B,T,d]` | $B$ 条序列、每条 $T$ 个位置、每位置 $d$ 维 |

形状比变量名更可靠。矩阵 `[V,d]` 乘 one-hot `[V]` 会取出 `[d]` 向量；批量 Embedding 则把 token IDs `[B,T]` 变为 `[B,T,d]`。后续读公式时，先在每个符号旁写形状，很多“高级公式”会退化成能否相乘的尺寸检查。

### 三个词不要混

| 对象 | 它是什么 | 类比 |
|---|---|---|
| 参数 parameter | 训练后长期保留的模型权重 | 食谱里的固定配方 |
| 激活 activation | 本次计算产生的中间结果 | 正在案板上的半成品 |
| 状态 / cache | 为后续步骤保留的历史信息 | 做到一半留下的备忘录 |

K3 论文后面会反复讨论参数、激活和 KV Cache。它们占显存的原因不同，不能统称“模型内存”。

训练时还会多出两类对象：**梯度**记录 loss 对每个参数的局部敏感度，**优化器状态**记录动量或二阶统计。以 BF16 权重 + BF16 梯度 + FP32 Adam 两个 moment 的粗略口径，光这些长期训练状态就可能约 12 bytes/参数，还未包括激活、临时张量和通信缓冲。推理通常不需要梯度和 Adam 状态，所以“能推理”不等于“能训练”。

## 3. “模型会学习”究竟是什么意思

<figure class="teaching-figure concept-figure"><img src="/illustrations/training-loop-sparse.webp" alt="前向计算、损失、反向传播和参数更新组成训练循环"><figcaption>训练不是模型自己“顿悟”：先前向预测，再用损失量出差距，梯度把修改方向传回去，最后更新参数并重复。</figcaption></figure>

训练开始时参数大多是随机的，输出也很差。一次训练循环：

1. 给模型输入；
2. 计算预测；
3. 用损失函数测量预测与目标有多远；
4. 计算每个参数对错误的影响方向；
5. 把参数移动一小步；
6. 对海量样本重复。

这不是模型主动决定“我要理解语言”，而是大量参数在预测任务上被逐步调整。语言规律、事实关联和程序结构，是为了降低预测错误而形成的内部表示。

### 3.1 Loss 不是老师说的一句话，而是一个可微标量

模型必须把“哪里错了”压成一个数。分类常用交叉熵：正确类别概率 $p$ 越接近 1，$-\log p$ 越接近 0；若模型给正确类别极小概率，loss 会很大。语言模型在每个目标 token 上计算同一种损失，再对有效位置求和或平均。

Loss 的定义就是训练目标。只训练下一 token，不会自动保证事实正确；只奖励最终答案，不会自动保证过程安全；只对 assistant token 计 loss，也与把 system/user 消息一起计 loss 不同。看模型报告时，先找“loss 精确在哪些位置、对什么目标计算”。

### 3.2 计算图为什么能自动求梯度

前向计算可画成有向无环图：节点保存张量，边表示函数依赖。例如：

```text
x ─┐
   ├─ 乘法 z = wx ─→ 预测 ŷ ─→ loss L
w ─┘
```

反向传播从 $\partial L/\partial L=1$ 开始，逆着图把“上游梯度 × 本地导数”相乘。对 $L=(wx-y^*)^2$：

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial w}
=2(\hat y-y^*)x
$$

自动微分没有猜答案，它只是保存前向图并系统执行链式法则。`backward()` 计算梯度；SGD、AdamW 或 Muon 才决定如何把梯度变成参数更新。

## 4. 一个参数也能训练

<TrainingLoopLab />

请连续点击“训练一步”，观察三件事：

- 参数 `w` 从 1 接近 5；
- 预测从 2 接近 10；
- loss 越来越小。

真实语言模型的不同点主要是：输入更复杂、参数更多、损失同时在许多 token 上计算、训练分布在大量 GPU 上。

### 把动画写成 12 行 PyTorch

动画里的目标仍是：输入 $x=2$ 时，希望输出 $y^*=10$。预测 $\hat y=wx$，平方损失为：

$$
L=(\hat y-y^*)^2=(wx-y^*)^2
$$

它对参数 $w$ 的梯度是 $2(wx-y^*)x$。梯度为负，表示增大 $w$ 会让损失下降；梯度为正，则应该减小 $w$。

```python
import torch

x = torch.tensor(2.0)
target = torch.tensor(10.0)
w = torch.tensor(1.0, requires_grad=True)

for step in range(8):
    prediction = w * x                 # 前向：ŷ = wx
    loss = (prediction - target) ** 2  # 衡量差距
    loss.backward()                    # 自动计算 w.grad

    with torch.no_grad():              # 更新参数本身不需要记录计算图
        w -= 0.1 * w.grad
    w.grad.zero_()                     # 清掉本轮梯度

print(w.item(), (w * x).item())
```

这里的 `requires_grad=True` 表示“请追踪所有会影响 `w` 的计算”；`backward()` 只负责求梯度，真正改参数的是 `w -= ...`。大型训练把手写更新替换成 AdamW、Muon 等优化器，但“前向 → 损失 → 反向 → 更新”不变。

## 5. 大语言模型“大”在哪里

“大”至少有四种不同尺度：

| 尺度 | 例子 | 变大后的主要影响 |
|---|---|---|
| 参数量 | 7B、70B、2.8T | 存储、计算、容量 |
| 训练 token | 1T、10T、32T | 数据成本、训练时间 |
| 上下文长度 | 8K、128K、1M | Attention、KV Cache、检索难度 |
| 生成轨迹 | 200 token、几万步 Agent | 延迟、环境状态、奖励稀疏 |

K3 的设计不是只处理参数量，而是四种尺度同时变大。

“7B”也不是人为贴上的型号：它来自词表矩阵、每层 Attention、FFN 和 Norm 的逐项求和。到 [第 14 课参数、数据与 Scaling](/beginner/25-data-scaling#parameter-count)，你会亲手从架构旋钮算出总参数，并把结果继续换算成显存和训练 FLOPs。

<figure class="teaching-figure source-figure"><a href="/lectures/images/compute-memory.png" target="_blank"><img src="/lectures/images/compute-memory.png" alt="Stanford CS336 用上下两块区域和中间窄通道表示计算单元与显存之间的数据传输"></a><figcaption>Stanford CS336 Lecture 1/2 的 Compute-Memory 示意图。上方有很多计算单元，下方是大容量显存，中间窄通道代表有限带宽：参数很多时，模型可能不是“算不动”，而是来不及把权重和激活搬到计算单元。<a href="/lectures/?trace=var/traces/lecture_02.json">打开 Lecture 2 可执行 Slides</a>。</figcaption></figure>

“大”也不等于“在每个请求中激活全部能力”。Dense 模型每个 token 使用所有层的主要权重；MoE 模型的总参数可以很大，但每个 token 只路由到少量专家。训练 token 数决定参数看过多少数据，上下文长度决定单次前向能看多远，生成轨迹决定请求持续多久。四个规模量必须分别报告。

## 6. 推理不是逻辑推理的专用词

工程里 `inference` 指训练完成后使用模型做预测。它包括：

- 输入提示词，生成回答；
- 对图片做理解；
- 让 Agent 调用工具；
- 计算 embedding。

“推理模型”的 reasoning 则强调多步思考能力。中文里都叫推理，语境不同。

## 7. Train、Validation 与 Test 为什么必须分开

设真实世界数据来自未知分布 $p_{data}$，手头数据只是它的一批有限样本。模型可能记住训练集细节，却没有学到可迁移规律。因此常分为：

| 数据 | 能不能参与梯度 | 用途 |
|---|---:|---|
| Train | 能 | 拟合参数 |
| Validation / Dev | 不能 | 选学习率、模型大小、训练步数与 checkpoint |
| Test | 不能，且开发中不反复查看 | 最后估计一次泛化能力 |

若你每天根据 Test 分数改模型，Test 已经通过人的决策间接参与训练，不再是无偏验收。应另建隐藏集或重新收集数据。

<GeneralizationLab />

### 7.1 过拟合不是“训练 loss 太低”

典型症状是 Train loss 继续下降而 Validation loss 开始上升。原因可能是容量过大、数据太少/太噪、训练太久或重复样本过多。常见手段包括更多有效数据、正则化、dropout、数据增强、减小模型、早停与更合理的任务划分。

还要区分 **distribution shift**：如果验证集与部署流量来自不同语言、时间或用户群，即使没有经典过拟合，线上也可能变差。泛化不是一个总分，要按语言、领域、长度、时间和安全风险分桶。

## 8. Batch、Padding 与 Mask：一次不只训练一条样本

GPU 擅长并行矩阵运算，因此常把 $B$ 条序列组成 batch。序列长度不同时，短样本用 PAD 补齐成矩形 `[B,T]`：

```text
样本 A: [我, 爱, 学习, EOS]
样本 B: [天气, 好, EOS, PAD]
```

Padding 只是对齐占位，不是有意义 token。Attention mask 阻止模型读取 PAD 或未来位置，loss mask 则阻止 PAD 贡献训练损失。两种 mask 解决不同问题；“看不到”和“不计分”不能混为一谈。

Batch 越大，梯度平均通常更稳定、硬件利用率可能更高，但显存增加，且学习率、warmup 和泛化行为可能改变。梯度累积可以用多个小 micro-batch 模拟较大的更新 batch，却不能完全消除通信和随机性差异。

## 9. 初始化、学习率与 Warmup 决定能不能开始学

参数通常不是全零初始化：若同层神经元完全相同，它们会收到相同梯度，无法分工。Xavier 等方法按输入/输出宽度缩放随机权重，目标是让信号在层间传播时不要迅速爆炸或消失。

学习率 $\eta$ 决定一步走多远：

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

过大可能震荡或发散，过小则训练缓慢。Warmup 在早期逐步升高学习率，避免随机初始化、未稳定归一化和大 batch 梯度在一开始造成破坏；后续 cosine decay 等调度再逐步减小步长。优化器、学习率、batch 和初始化是一套耦合配方，不能只复制其中一个数字。

## 10. 语言模型怎样落入同一个框架

语言模型不是“直接输出一句话的黑箱”，而是对所有 token 序列定义概率。利用概率链式法则：

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}
p_\theta(x_t\mid x_{<t})
$$

指数大的完整序列空间被拆成 $T$ 次词表分类。训练时用真实前缀并最大化每个目标 token 的概率；生成时把自己刚采样的 token 追加到上下文，再预测下一步。完整推导、Bigram/N-gram 与神经语言模型演化放在[第 05 课](/beginner/10-language-models)。

概率连乘很快下溢，所以实现中使用 log 空间：

$$
\log p(x_{1:T})=\sum_t\log p(x_t\mid x_{<t})
$$

平均负 log-likelihood 可指数化为 perplexity。它衡量模型对测试 token 的平均惊讶程度，却不直接等价于事实性、帮助性或任务成功率。

## 11. 一次最小可信机器学习实验

1. 写清输入、输出、允许动作和成功指标；
2. 冻结 Train/Validation/Test 划分并查去重、时间泄漏；
3. 定义参数化模型和精确 loss/mask；
4. 先跑能过拟合几十条样本的小实验，验证管线确实能学；
5. 记录随机种子、数据版本、初始化、优化器、学习率与 batch；
6. 看 Train/Validation 曲线，按 Validation 选 checkpoint；
7. 只在最终方案上打开 Test，并按子群与成本报告；
8. 保存模型、Tokenizer、配置与日志，使结果可复现。

若第 4 步都做不到，先查 label、mask、shape、梯度和数据管线，不要直接增加 GPU。若训练集很好、验证集差，才进入过拟合与分布偏移诊断。

## 12. 本章阅读路线

1. [CMU ANLP L01](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-01-intro.pdf) 第 7–40 页：跟着分类例子区分 scoring function、parameterization、learning 与 inference；
2. [CMU ANLP L02](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-02-representations.pdf) 第 17–42、44–63 页：依次看 Embedding、非线性、交叉熵、计算图与反向传播；
3. [CMU ANLP L03](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-03-lm.pdf) 第 5–26、48–58 页：理解序列概率、最大似然、log 空间、perplexity、数据划分与过拟合；
4. 最后读 [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) 的模型图与第 3 节，把“Embedding → 隐藏层 → 词表概率”对应到本课三件事。

## 本课闭卷复述

不使用“AI”“智能”两个词，向朋友解释：模型、参数、训练、推理分别是什么。

<ConceptCheck question="KV Cache 更接近哪一类对象？" :options='["训练后长期固定的模型参数", "为当前或后续生成保留的历史状态", "训练数据文件"]' :answer="1" explanation="KV Cache 由本次输入计算得到，会随请求和生成位置变化；它不是模型权重。" />

<ConceptCheck question="你反复查看 Test 集结果并据此修改学习率，最准确的描述是什么？" :options='["这是标准训练步骤", "Test 集已参与开发决策，不再是独立最终验收", "只要没有对 Test 做 backward 就完全没问题"]' :answer="1" explanation="信息会通过人的超参数选择进入模型；是否调用 backward 不是数据泄漏的唯一判断标准。" />

下一课：[文字怎样变成 token](/beginner/01-token)。

<ChapterReadings lesson="00-model" />
