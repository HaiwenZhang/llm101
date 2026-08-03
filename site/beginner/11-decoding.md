---
title: 第 35 课 从概率到文字：解码与采样
description: 详细理解 Greedy、Beam Search、Top-k、Top-p、Temperature 与停止条件
---

# 第 35 课　从概率到文字：解码与采样

<div class="lesson-lead">语言模型每一步只交出一张“候选 token 概率表”。解码器怎样从表里选词，会直接改变答案是稳定、重复、保守还是多样。模型参数没变，文字风格却可能完全不同。</div>

::: info 本课资料地图
- 基础课件：[台大 ADL · NLG Decoding](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/251027_NLG.pdf)负责 Greedy、Beam、Sampling，[CMU Decoding Algorithms](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-09-decoding.pdf)补充更一般的推理时算法；
- 研究综述：[From Decoding to Meta-Generation](https://arxiv.org/pdf/2406.16838.pdf)把单路径解码、多候选、修正、验证和搜索放入同一框架；
- 系统论文：[Speculative Decoding](https://arxiv.org/pdf/2211.17192.pdf)讨论怎样在不改变目标分布的前提下加速采样；[Stanford CS336 Lecture 10 可执行 Slides](/lectures/?trace=var/traces/lecture_10.json)把它放回 Prefill、Decode 与服务资源账中。它与“多花计算提高答案质量”的测试时扩展不是一回事。
:::

<figure class="teaching-figure">
  <img src="/illustrations/foundations-language-models.webp" alt="语言模型输出候选词概率树并沿不同分支继续生成">
  <figcaption>最右侧的分叉不是模型在脑中准备好的完整答案，而是当前一步的候选。每选一次，后续分布都会重新计算。</figcaption>
</figure>

<figure class="teaching-figure source-figure"><a href="/lectures/images/speculative-sampling-algorithm.png" target="_blank"><img src="/lectures/images/speculative-sampling-algorithm.png" alt="Stanford CS336 Slides 中推测采样由小模型提出草稿、大模型并行验证的算法图"></a><figcaption>来源图：Stanford CS336 Lecture 10 的 Speculative Sampling。小模型一次提出多个草稿 token，大模型并行验证；它优化生成速度，不是另一种提高答案创意度的采样参数。<a href="/lectures/?trace=var/traces/lecture_10.json">打开本地可执行 Slides</a>。</figcaption></figure>

## 1. 先做一个三步手算

提示是“长颈鹿”，第一步候选如下：

| 第一步 | 概率 |
|---|---:|
| 是 | 0.30 |
| 脖子 | 0.20 |
| 看着 | 0.15 |

如果选“是”，下一步“草食”的条件概率为 0.10，路径概率是：

$$0.30\times0.10=0.03$$

如果第一步选“脖子”，下一步“长”的概率为 0.50：

$$0.20\times0.50=0.10$$

所以每一步都选眼前最大值，不保证整条序列概率最大。这是 Greedy Search 的局部最优问题。

### 1.1 先声明你想优化什么

同一个 $p_\theta(y\mid x)$ 可以对应完全不同的推理目标：

$$
y_{MAP}=\arg\max_y p_\theta(y\mid x)
$$

这是找一个最高概率序列；祖先采样则是 $y\sim p_\theta(y\mid x)$，目标是按分布抽一条；Best-of-N 又先采多个候选，再用奖励或验证器选择。它们不是同一算法换几个超参数。

<figure class="teaching-figure concept-figure"><img src="/illustrations/decoding-four-objectives.svg" alt="MAP、祖先采样、截断温度采样和 Best-of-N 四类解码目标的对比"><figcaption>先分清目标，再选算法。Greedy/Beam 近似找 mode；Temperature/Top-p 改造采样分布；Best-of-N 还引入候选覆盖与选择器误差。</figcaption></figure>

## 2. Greedy：每次只走最宽的路

$$w_{t+1}=\arg\max_w P(w\mid w_{\le t})$$

优点是确定、速度快、容易复现；缺点是早期一步选错就没有回头路，也容易进入重复模式。适合输出空间明确、低随机性重要的任务，但“最可能”不等于“事实正确”。

## 3. Beam Search：同时保留几条路

若 beam size 为 2：

1. 第一步保留概率最高的 2 个前缀；
2. 分别展开每个前缀的下一 token；
3. 在所有新路径中再保留累计得分最高的 2 条；
4. 直到出现结束 token 或达到长度限制。

实际比较常使用对数概率，避免很多小数相乘下溢：

$$\log P(w_{1:T})=\sum_{t=1}^{T}\log P(w_t\mid w_{<t})$$

若 beam width 为 $B$、词表为 $V$，每一步概念上会从 $B|V|$ 个扩展中保留 B 条。`B=1` 退化成 Greedy；只有保留整个指数级搜索空间才可能保证精确 MAP，实际做不到。实现还要分开维护“尚可继续展开的前缀”和“已经生成 EOS 的完成序列”，否则完成项会被错误地继续扩展或过早丢弃。

### 3.1 为什么需要长度惩罚

每多生成一步就多加一个非正的 log 概率，短句天然占便宜。Beam Search 常加入长度归一化或惩罚，否则可能过早结束。它适合翻译、语音识别等有明确输入约束的任务；开放聊天里往往显得平庸。

一种教学写法是：

$$
s(y)=\frac{\log p_\theta(y\mid x)}{|y|^\alpha}
$$

$\alpha=0$ 时不归一，短序列优势最强；增大 $\alpha$ 会提高长序列相对得分。框架可能使用不同的 `(5+length)/6` 形式或把参数符号定义反过来，迁移配置时必须核对源码。

### 3.2 最高概率为什么可能不是“典型好文本”

抛 100 次略偏的硬币，单个最高概率序列可能是全正面，但这并不代表通常会观察到的样子。语言也有大量近义表述：同一个正确含义的概率会分散到许多表面形式，而一个短、重复的字符串可能集中较多概率。课件因此区分 **likely** 与 **maximally likely**；模型概率高也不是事实验证器。

## 4. Temperature：先改变概率形状

模型输出的原始分数叫 logits。温度后：

$$P_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$$

假设 logits 为 `[4, 2, 1]`：

- $T<1$：差距被放大，第一名更占优势；
- $T=1$：保持原始 softmax；
- $T>1$：分布变平，弱候选更容易被抽到。

温度很接近 0 时，行为接近贪心。它控制的是分布尖锐度，不是“模型智商”或“事实核验强度”。

## 5. Top-k：固定保留前 k 名

Top-k 先把词表中除前 k 名外的候选设为不可选，再对保留项重新归一化。

问题在于固定 k 不看分布形状：

- 分布很尖：第一名 0.90，第二名 0.02；Top-5 仍可能把离谱词带进来。
- 分布很平：前 20 名都合理；Top-5 又会无端删掉多样性。

## 6. Top-p：保留累计概率够用的最小集合

先按概率从高到低排序，累加到阈值 $p$ 为止。例如：

```text
A 0.42  累计 0.42
B 0.28  累计 0.70
C 0.13  累计 0.83
D 0.09  累计 0.92  ← p=0.9 时到这里
E 0.04  不保留
```

因为候选数量会随分布变化，Top-p 又叫 nucleus sampling。它不是“只保留概率超过 p 的词”。

<SamplingLab />

### PyTorch：Temperature + Top-k 的最小采样器

```python
import torch

def sample_top_k(logits, temperature=1.0, top_k=50):
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    scaled = logits / temperature
    k = min(top_k, scaled.numel())
    values, indices = torch.topk(scaled, k)
    probabilities = torch.softmax(values, dim=-1)
    local_id = torch.multinomial(probabilities, num_samples=1)
    return indices[local_id].item()
```

不要对 softmax 后的概率再除温度；温度作用在 logits 上，再重新 softmax。Top-p 还需把概率降序排列、按累计和保留最小前缀，逻辑与上面的手算表一致。

## 7. 三个旋钮如何一起工作

一个便于理解的顺序是：

```text
logits → Temperature 改分布 → Top-k 截断 → Top-p 截断 → 重新归一化 → 采样
```

不同推理框架的具体过滤顺序可能不同，所以迁移参数时要看实现。调参应一次只改一个变量，并用固定问题集比较。

### 7.1 祖先采样与截断采样的分布不同

若每一步直接从原始模型条件分布采样：

$$
y_t\sim p_\theta(y_t\mid y_{<t},x)
$$

根据链式法则，完整序列就是来自 $p_\theta(y\mid x)$ 的样本。加入温度、Top-k 或 Top-p 后，实际采样的是变换后的 $q(y\mid x)$，不再是原始模型分布的精确样本。这不是错误，而是主动用偏差换取更合适的质量—多样性权衡。

### 7.2 Heavy tail 为什么会在长序列累积

假设每一步“坏 token”的总概率质量只有 $\varepsilon$，并做简化独立近似，长度 T 内一次坏 token 都没抽到的概率是：

$$
P(\text{no bad token})=(1-\varepsilon)^T
$$

| $\varepsilon$ | T | 全程不抽到坏 token |
|---:|---:|---:|
| 0.01 | 128 | 0.276 |
| 0.05 | 128 | 0.0014 |
| 0.01 | 1,024 | 约 0.000034 |

真实错误并不独立，但这个数量级说明：词表尾部每项概率很小，合起来仍可能很大；序列越长，偶然进入低质量区域的机会越多。Top-k/Top-p 试图削掉尾部，却也可能删除罕见但正确的术语。

## 8. 不只三个旋钮：停止与重复同样重要

### 8.1 停止条件

- 生成 EOS（结束 token）；
- 命中 stop string；
- 达到最大 token 数；
- 结构化输出解析完成；
- Agent 达到工具调用预算或验证通过。

停止字符串可能跨 token 边界，不能简单假设一个字符串等于一个 token。

### 8.2 重复惩罚

常见机制会降低已经出现过 token 的 logit：

- presence penalty：出现过就惩罚一次；
- frequency penalty：出现次数越多惩罚越大；
- no-repeat n-gram：禁止重复某长度片段。

惩罚太强会导致专有名词无法重复、代码变量名被改坏。先判断重复源于提示、模型还是采样，不要把所有问题都交给惩罚项。

### 8.3 约束解码：把无效 token 直接设为不可选

对 JSON、正则、语法树或有限标签，可以根据当前解析状态构造允许集合 $A_t$：

$$
z'_i=\begin{cases}z_i,&i\in A_t\\-\infty,&i\notin A_t\end{cases}
$$

再对 $z'$ 做 Softmax。它能保证语法层面的合法前缀，却不能保证字段事实正确、业务规则正确或 JSON 中的 SQL 安全。tokenizer 还会让一个字符跨多个 token，约束器必须工作在 token 与解析状态的真实映射上。

### 8.4 EOS、stop string 与 max tokens 是三种结束

EOS 是模型词表中的特殊 token；stop string 是服务端在解码文本中匹配的字符串；`max_new_tokens` 是硬预算。Stop string 可能跨 token 边界，也可能已经被生成但在返回时被裁掉。严格日志要保存 finish reason，区分自然完成、命中 stop、长度截断和安全中止。

## 9. 场景化起点，不是万能参数表

| 场景 | 倾向 | 仍需验证 |
|---|---|---|
| 事实抽取、结构化 JSON | 低随机性 | 格式验证、证据、重试策略 |
| 代码补全 | 较保守或采多份再测 | 编译与单元测试比温度更关键 |
| 创意写作 | 允许较高多样性 | 连贯性、禁区、用户风格 |
| 数学推理 | 可采多个候选再验证 | 最终答案检查器 |
| 翻译 | Greedy/Beam 或低随机性 | 长度归一化、术语一致性 |

### 9.1 速度优化与质量搜索不要混为一谈

Speculative Decoding 用小 draft model 提前提出 token，再由目标模型并行验证。正确实现的接受/修正步骤可以保持目标模型的采样分布，因此主要优化延迟；Best-of-N、Tree Search 或 verifier-guided search 则多花计算改变最终选择，目标是质量。二者都可能“多生成候选”，但验收标准完全不同：前者检查分布等价与加速比，后者检查质量—计算曲线和选择器偏差。

### 9.2 评测必须同时看四类指标

| 维度 | 例子 | 单独使用的陷阱 |
|---|---|---|
| 任务质量 | accuracy、BLEU/COMET、单测通过 | 一个均值看不见失败类型 |
| 校准/概率 | NLL、Brier、ECE | 概率好不等于文本满足业务 |
| 多样性 | distinct-n、self-BLEU、答案熵 | 胡言乱语也可能“很不重复” |
| 系统成本 | TTFT、ITL、tokens/s、总 token | 快不等于正确 |

比较解码器时要固定 checkpoint、prompt、测试集和最大输出预算；否则 beam 更长或 sampling 多生成的 token 会混入质量差异。

## 10. 复现为什么仍可能失败

即使温度为 0，不同硬件内核、浮点精度、批处理顺序、模型版本和服务端实现也可能产生微小差异，随后在自回归过程中放大。严格实验要记录模型版本、tokenizer、全部采样参数、随机种子与推理框架。

<ConceptCheck question="Top-p=0.9 的准确含义是什么？" :options="['只保留单个概率大于 0.9 的 token','保留从高到低累计概率达到 0.9 的最小候选集合','随机删除 10% 的模型参数']" :answer="1" explanation="Top-p 根据当前分布动态决定候选数量，保留集合后还要重新归一化。" />

<ConceptCheck question="使用 Temperature 与 Top-p 后逐 token 采样，完整序列仍是原始 pθ 的精确样本吗？" :options='["通常不是，采样来自变换后的 q", "是，因为任何采样都不改变分布", "只有 Beam Search 才是随机样本"]' :answer="0" explanation="温度重塑 logits，Top-p 又截断并重新归一化，所以实际条件分布已经变成 q。" />

## 11. 闭卷练习

给定概率 `[0.55, 0.20, 0.12, 0.08, 0.05]`：

1. Top-k=3 会保留谁？
2. Top-p=0.8 会保留几个？
3. 为什么高温不会自动增加事实正确率？
4. 为什么 Beam Search 需要长度处理？

> 本课对应原书第 1.4 节（PDF 第 25–30 页），把概率最大化与随机采样拆成了完整的生成决策流程。

## 12. 课件与论文精读路线

1. [CMU L09 第 5–15 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-09-decoding.pdf#page=5)：从自回归分解写出 MAP，再证明 Greedy 不保证全局最优；
2. [第 16–30 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-09-decoding.pdf#page=16)：逐页追踪 Beam、短序列、重复、atypicality 与概率分散；
3. [第 32–48 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-09-decoding.pdf#page=32)：祖先采样、heavy tail、Top-k、Top-p 与 temperature；
4. [第 50–60 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-09-decoding.pdf#page=50)：把 KV Cache、Roofline、Speculative Decoding 与多请求调度分开；
5. [From Decoding to Meta-Generation](https://arxiv.org/pdf/2406.16838.pdf)：用“生成—修正—验证—搜索”框架重新归类本章算法。

<ChapterReadings lesson="11-decoding" />
