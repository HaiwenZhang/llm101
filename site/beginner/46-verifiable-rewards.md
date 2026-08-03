---
title: 第 31 课 GRPO、可验证奖励与推理 RL
description: 从 REINFORCE、PPO 到 GRPO 与 DAPO，完整理解可验证奖励怎样训练推理模型
---

# 第 31 课　GRPO、可验证奖励与推理 RL

<div class="lesson-lead">这一课不把 GRPO 当作需要背诵的公式。我们从一次真实训练迭代出发：模型为什么要对同一道题生成一组回答？正确与错误怎样变成优势？旧策略和参考策略分别做什么？为什么全对、全错、超长回答和有漏洞的验证器都会让训练失败？</div>

::: info 本课怎样整合名校课程与论文
主线逐页跟随 [Stanford CS224N L08：Post-training](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture08-posttraining.pdf)、[L12：Reasoning I](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture12-reasoning-part1.pdf)、[L13：Reasoning II](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture13-reasoning-part2.pdf)，以及 [CMU ANLP L17：Reinforcement Learning II](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-17-rl-llms.pdf) 和 [L23：Test-Time Scaling](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-23-inference-scaling.pdf)。算法细节核对 [DeepSeek-R1](https://arxiv.org/pdf/2501.12948.pdf)、[DAPO](https://arxiv.org/pdf/2503.14476.pdf) 与 [DeepSeekMath](https://arxiv.org/pdf/2402.03300)。
:::

## 0. 学完以后，你要能回答什么

不要急着看公式。先记住本课要解决的七个问题：

1. 一个语言模型的“动作”到底是整段回答，还是每一个 token？
2. 只有最终答案对错时，梯度怎样回到成千上万个 token？
3. baseline 为什么能降方差，却不改变期望梯度？
4. PPO 的 `old policy` 与 KL 的 `reference policy` 为什么不是一回事？
5. GRPO 为什么能去掉 value model，又为什么必须为同一道题多次采样？
6. 全对组和全错组为什么都没有组内学习信号？
7. DAPO 的四个修改分别在修复哪个可观测的训练故障？

最终你应该能看懂一份 GRPO 训练日志，而不是只会说“它用组平均代替 Critic”。

## 1. 先从一轮训练看全局

假设题目是：

> 求方程的整数解，只输出最后的整数。

训练系统不会只让模型答一次，而会让同一份旧策略采样 (G) 条回答。验证器检查每条最终答案，再把同题回答放在一起比较。

<figure class="teaching-figure"><img src="/illustrations/grpo-training-dataflow.svg" alt="GRPO 从同一道题采样一组回答，经验证器和组内比较后更新当前策略"><figcaption>一轮 GRPO 的核心数据流。注意有两个不同的比较：回答之间比较奖励得到优势；新旧策略之间比较 token 概率得到更新比率。</figcaption></figure>

一轮最小训练循环可以写成：

```text
1. 冻结本轮旧策略 π_old
2. 对每个题目 q，从 π_old 采样 G 条回答 o_1 ... o_G
3. 验证每条回答，得到 R_1 ... R_G
4. 在同一道题内部计算相对优势 A_1 ... A_G
5. 用 π_θ / π_old 的概率比和 PPO 裁剪更新当前策略 π_θ
6. 可选：用 π_ref 的 KL 惩罚约束行为漂移
7. 进入下一轮，重新生成回答，而不是永久复用旧轨迹
```

它同时包含三个计算世界：

| 世界 | 对象 | 回答的问题 |
|---|---|---|
| 生成 | prompt、completion、token | 模型探索了哪些解法？ |
| 评价 | verifier、reward、advantage | 哪些回答相对更值得学习？ |
| 优化 | log-prob、ratio、clip、KL | 应把参数向这个方向推多远？ |

只看其中一层，很容易产生误解：有正确奖励不代表一定有可靠信用分配；有漂亮的 PPO loss 也不代表验证器没有漏洞。

## 2. 语言生成到底是哪一种 MDP

在普通强化学习里，我们用状态 (s_t)、动作 (a_t)、奖励 (r_t) 描述决策。语言模型可以有两种视角。

### 2.1 整段回答作为一个动作

把 prompt (q) 当状态，把完整回答 (o=(o_1,ldots,o_T)) 当动作：

$$
o\sim\pi_\theta(\cdot\mid q),\qquad R=V(q,o)
$$

这里的 (V) 是 verifier，不是 value network。这个视角适合解释“同一道题采样 (G) 个候选”，但会隐藏 token 级概率。

### 2.2 每一个 token 作为一个动作

更细的视角是：

$$
s_t=(q,o_{<t}),\qquad a_t=o_t,
\qquad \pi_\theta(o\mid q)=\prod_{t=1}^{T}\pi_\theta(o_t\mid q,o_{<t})
$$

取对数后，整段回答的 log probability 变成逐 token 求和：

$$
\log\pi_\theta(o\mid q)=\sum_{t=1}^{T}\log\pi_\theta(o_t\mid q,o_{<t})
$$

因此，即使 verifier 只在结尾给一个 0/1 奖励，优化器仍能对每个生成 token 的 log probability 求导。

::: warning 能求导，不等于信用分配精确
若最终答案正确，结果奖励通常会把整条轨迹都当作正样本；但回答中可能有无用绕路、错误中间步骤甚至碰巧猜对。结果奖励告诉模型“这条轨迹整体值得增加概率”，却没有可靠指出哪一步真正贡献了成功。
:::

## 3. 从 REINFORCE 推到优势函数

我们的目标是最大化回答的期望奖励：

$$
J(\theta)=\mathbb E_{o\sim\pi_\theta(\cdot\mid q)}[R(q,o)]
$$

直接对离散采样求导不方便。利用 log-derivative trick：

$$
\begin{aligned}
\nabla_\theta J
&=\sum_o R(q,o)\nabla_\theta\pi_\theta(o\mid q)\\
&=\sum_o \pi_\theta(o\mid q)R(q,o)\nabla_\theta\log\pi_\theta(o\mid q)\\
&=\mathbb E[R(q,o)\nabla_\theta\log\pi_\theta(o\mid q)]
\end{aligned}
$$

这就是最基本的 REINFORCE：高奖励轨迹增加概率，低奖励轨迹减少概率。

### 3.1 为什么可以减掉 baseline

从奖励中减去一个只依赖题目、不依赖本次动作的 baseline (b(q))：

$$
\mathbb E[(R-b(q))\nabla_\theta\log\pi_\theta(o\mid q)]
$$

它不会改变期望梯度，因为：

$$
\begin{aligned}
\mathbb E[b(q)\nabla_\theta\log\pi_\theta(o\mid q)]
&=b(q)\sum_o\pi_\theta(o\mid q)\nabla_\theta\log\pi_\theta(o\mid q)\\
&=b(q)\nabla_\theta\sum_o\pi_\theta(o\mid q)\\
&=b(q)\nabla_\theta 1=0
\end{aligned}
$$

减 baseline 的目的不是“改奖励”，而是降低方差。于是定义优势：

$$
A(q,o)=R(q,o)-b(q)
$$

- (A>0)：这条回答比题目的基准水平更好；
- (A<0)：比基准更差；
- (A=0)：没有相对更新信号。

PPO 常用 value model 估计状态价值，并借助 GAE 估计逐步优势。GRPO 的关键变化是：不再训练这个 Critic，而用同题样本的组统计量建立 baseline。

## 4. PPO 为什么需要“旧策略”与裁剪

如果采到一条成功轨迹后反复大幅提高它的概率，策略可能在一次更新中跳得太远，后续训练失稳。PPO 记录采样时旧策略的概率，并计算逐 token 比率：

$$
r_{i,t}(\theta)=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\text{old}}(o_{i,t}\mid q,o_{i,<t})}
$$

- (r=1)：当前策略与采样时相同；
- (r=1.2)：当前策略把该 token 的概率提高了 20%；
- (r=0.8)：概率降低了 20%。

PPO-Clip 的单 token 目标是：

$$
L_{i,t}=\min\left(
r_{i,t}\hat A_i,
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_i
\right)
$$

### 4.1 正优势和负优势的裁剪方向不同

| 优势 | 优化器想做什么 | 什么时候被裁剪 |
|---|---|---|
| (hat A>0) | 提高这枚 token 的概率，即增大 (r) | (r>1+\epsilon) 后继续上涨不再获益 |
| (hat A<0) | 降低这枚 token 的概率，即减小 (r) | (r<1-\epsilon) 后继续下跌不再获益 |

`min` 看似对两种符号都一样，乘上负优势后却会改变大小顺序。最稳妥的理解方式不是背公式，而是分别代入正数和负数。

举例，(epsilon=0.2)：

| (A) | (r) | (rA) | 裁剪项 | PPO 取值 | 含义 |
|---:|---:|---:|---:|---:|---|
| +1 | 1.30 | 1.30 | 1.20 | 1.20 | 成功 token 已提高太多 |
| +1 | 0.70 | 0.70 | 0.80 | 0.70 | 降低好 token 不受保护性奖励 |
| −1 | 0.70 | −0.70 | −0.80 | −0.80 | 失败 token 已降低太多 |
| −1 | 1.30 | −1.30 | −1.20 | −1.30 | 提高坏 token 会受到完整惩罚 |

裁剪不是严格的 KL 距离上界，也不能保证每次更新安全；它只是让目标在越过局部信任区间后变平。

## 5. old policy、reference policy 与 verifier 不要混

<figure class="teaching-figure"><img src="/illustrations/ppo-grpo-policy-roles.svg" alt="GRPO 中当前策略、旧策略、参考策略和验证器的职责区别"><figcaption>旧策略用于“这批数据由谁生成”的重要性校正，参考策略用于“行为不要偏离锚点太远”的正则，验证器只负责把回答映射成奖励。</figcaption></figure>

| 对象 | 是否训练 | 典型更新时间 | 主要用途 |
|---|---:|---|---|
| 当前策略 (pi_\theta) | 是 | 每个 optimizer step | 接收梯度，成为最终模型 |
| 旧策略 (pi_{old}) | 否 | 每轮 rollout 前从当前策略复制 | 生成数据并构造概率比 |
| 参考策略 (pi_{ref}) | 否 | 常冻结较久，也可周期更新 | 通过 KL 约束行为漂移 |
| verifier / reward | 通常否 | 规则或独立训练 | 判断任务完成质量 |

GRPO 原始形式常带直接 KL 惩罚：

$$
L=L_{\text{clip}}-\beta D_{KL}(\pi_\theta\Vert\pi_{ref})
$$

但“使用组内优势”与“是否加入 KL”是两个独立设计选择。DAPO 在其长链数学推理配方中移除了 KL；DeepSeek-R1 报告的某些阶段则使用 KL 系数并周期更新 reference。不能把“无 KL”当作 GRPO 的定义。

::: tip 一句辨别
`old` 回答“我的训练数据来自哪个策略”；`reference` 回答“我不希望行为偏离哪个锚点”。有些代码为了省显存会复用权重或改变更新频率，但概念上仍应分开。
:::

## 6. GRPO 怎样用一组回答估计优势

对同一道题 (q)，旧策略采样 (G) 条回答，得到奖励 (R_1,ldots,R_G)。最常见的组内标准化优势是：

$$
\hat A_i=
\frac{R_i-\mu_R}{\sigma_R+\varepsilon},\qquad
\mu_R=\frac1G\sum_{j=1}^{G}R_j
$$

它把不同难度题目的奖励放在各自组内比较。简单题和难题不能混用一个 batch 均值，否则同样的“答对”会因为其他题而得到不同基准。

### 6.1 手算一个 CMU 课件里的例子

同一道题采样四条回答，奖励为：

$$
[0,0,0,1]
$$

组均值是 (0.25)。如果只减均值，不除标准差：

$$
[-0.25,-0.25,-0.25,+0.75]
$$

总体标准差为：

$$
\sigma=\sqrt{0.25(1-0.25)}\approx0.433
$$

标准化后约为：

$$
[-0.577,-0.577,-0.577,+1.732]
$$

若实现采用不同的标准差约定或额外缩放，数值会不同；必须检查代码使用的是总体标准差、样本标准差，还是根本不做标准差归一化。算法名字相同，不代表归一化细节相同。

### 6.2 二元奖励下的一个快捷公式

若一组中成功比例为 (hat p=k/G)，总体标准差就是：

$$
\sigma=\sqrt{\hat p(1-\hat p)}
$$

成功与失败样本的标准化优势分别是：

$$
A_{success}=\sqrt{\frac{1-\hat p}{\hat p}},\qquad
A_{failure}=-\sqrt{\frac{\hat p}{1-\hat p}}
$$

这揭示了一个重要现象：组里只有一次成功时，成功轨迹会得到很大的正优势；组里只有一次失败时，失败轨迹会得到很大的负优势。稀有样本被放大既能加速学习，也会放大 verifier 的误判。

## 7. 完整 GRPO 目标逐项拆开

一种常见的逐 token 写法是：

$$
\begin{aligned}
J_{GRPO}(\theta)=\mathbb E\Bigg[
\frac1G\sum_{i=1}^{G}\frac1{|o_i|}\sum_{t=1}^{|o_i|}
&\min\big(r_{i,t}\hat A_i,
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big)\\
&-\beta D_{KL}(\pi_\theta\Vert\pi_{ref})
\Bigg]
\end{aligned}
$$

逐项解释：

1. (mathbb E)：跨题目、采样回答和训练 batch 取平均；
2. (1/G)：同一题的 (G) 条回答平均；
3. (1/|o_i|)：先在每条回答内部按 token 平均；
4. (r_{i,t})：校正当前策略相对采样策略的概率变化；
5. (hat A_i)：同一条回答的结果优势通常复制给所有 token；
6. `clip`：限制一次更新从旧策略走得太远；
7. KL：可选的参考行为约束。

### 7.1 “去掉 Critic”省掉了什么

- 不需要训练与保存 value model；
- 不需要让 Critic 拟合长推理轨迹的逐状态回报；
- 降低一部分显存、同步和训练复杂度。

### 7.2 它增加了什么

- 每个 prompt 必须生成多条 completion；
- rollout token 成本近似乘以组大小 (G)；
- 组内样本相关时，有效多样性低于名义组大小；
- 全对或全错时，组内 baseline 完全失去区分力；
- 只用结果优势时，逐 token 信用分配仍然粗糙。

所以 GRPO 是用更多在线采样换掉价值网络，而不是“免费简化 PPO”。

## 8. 全对和全错：为什么标准差加 epsilon 也救不了

若奖励是 `[0,0,0,0]` 或 `[1,1,1,1]`：

$$
R_i-\mu_R=0,\qquad \sigma_R=0
$$

给分母加 (arepsilon) 只能避免数值除零，结果仍是 0。它不能凭空创造“哪条轨迹更好”的信息。

若单条回答独立成功概率为 (p)，组大小为 (G)，一组同时包含成功和失败的概率是：

$$
P(\text{informative})=1-p^G-(1-p)^G
$$

| 单条成功率 (p) | (G=4) | (G=8) | (G=16) |
|---:|---:|---:|---:|
| 0.01 | 3.9% | 7.7% | 14.9% |
| 0.10 | 34.4% | 57.0% | 81.5% |
| 0.50 | 87.5% | 99.2% | 100.0% |
| 0.90 | 34.4% | 57.0% | 81.5% |
| 0.99 | 3.9% | 7.7% | 14.9% |

这张表解释了课程学习的核心：题目最好处在模型“有时能做对、有时会做错”的能力边界。单纯扩大 (G) 能缓解退化，却会线性增加生成成本，而且样本相关时收益更小。

<GRPOLab />

## 9. 可验证奖励到底验证了什么

可验证任务的优势是反馈便宜、稳定、可重复：

| 任务 | 可执行验证器 | 它真正证明了什么 | 没有证明什么 |
|---|---|---|---|
| 数学 | 数值或符号等价 | 最终表达式与标准答案等价 | 中间证明有效、没有猜中 |
| 代码 | 编译器、隐藏单测、属性测试 | 在给定环境和测试上行为正确 | 没有后门、对所有输入都正确 |
| 工具调用 | 环境最终状态、API 回执 | 某个可观测目标已完成 | 路径合规、权限和成本合理 |
| 结构输出 | JSON Schema、正则 | 格式可解析 | 内容真实、有用或安全 |

DeepSeek-R1-Zero 的规则奖励主要由正确性奖励和格式奖励构成：

$$
R_{rule}=R_{accuracy}+R_{format}
$$

这不是说格式与正确性同等重要，而是说明奖励函数会塑造模型行为。若格式分太容易拿，模型会先优化标签；若 parser 有漏洞，模型会优化漏洞。

### 9.1 设计数学验证器时

- 先把回答规范化：空格、单位、分数、小数、集合顺序；
- 区分字符串相等与数学等价；
- 对无法可靠解析的答案拒绝打正分，而不是猜测；
- 用对抗样本测试 `NaN`、溢出、注入文本和多答案；
- 记录 verifier 版本，避免奖励定义悄悄变化。

### 9.2 设计代码验证器时

- 测试放在隔离沙箱，限制网络、文件和进程权限；
- 隐藏测试与训练 prompt 分离；
- 除示例测试外加入边界、随机和属性测试；
- 分开记录编译失败、超时、运行错误和错误答案；
- 不把“读取测试文件并硬编码答案”当作任务成功。

### 9.3 verifier 也会被优化

强化学习不理解设计者“本来想要什么”，只会提高被计分行为的概率。训练越久、探索越强，越可能找到奖励代理的边界。因此应把 reward hacking 看成预期的优化结果，而不是模型突然“变坏”。

## 10. Outcome Reward 与 Process Reward

### 10.1 结果奖励 ORM

只在回答结束后评分。优点是客观、便宜、易扩展；缺点是信用分配粗。

若最终奖励 (R_i) 被复制给回答的全部 token：

$$
A_{i,1}=A_{i,2}=\cdots=A_{i,T}=A_i
$$

模型会整体提高成功轨迹的概率，但不能直接知道“第 37 步换元”比“第 140 步重复检查”更关键。

### 10.2 过程奖励 PRM

把推理拆成若干步骤，对每一步预测正确性。它能提供更密的反馈，也能用于 Best-of-N 排序或搜索，但困难在于：

1. 通用推理的“一个步骤”没有统一边界；
2. 某一步局部正确不代表后续一定可完成；
3. 自动标注器本身会错，人工标注又昂贵；
4. 训练中的策略会主动寻找 PRM 的盲点；
5. PRM 需要持续重训，形成额外系统闭环。

DeepSeek-R1 报告明确记录了其 PRM 训练尝试没有带来足以覆盖复杂度的收益；这不证明 PRM 永远无效。它在重排与引导搜索中仍可能有价值，关键是限定结论的任务、模型和用法。

### 10.3 不要把“过程分高”误当成真实证明

PRM 给的是模型或规则对步骤的预测，不是形式证明。对高风险数学、代码或工具任务，真正的强信号仍来自可执行环境、形式检查器、隐藏测试和独立审计。

## 11. 为什么长回答会改变损失权重

设一条短回答有 100 个 token，一条长回答有 1000 个 token。

### 11.1 先按回答平均：sample-level reduction

$$
L_{sample}=\frac1G\sum_i\left(\frac1{|o_i|}\sum_t\ell_{i,t}\right)
$$

每条回答总权重相同，所以长回答里的每个 token 权重只有短回答的十分之一。

### 11.2 全部 token 一起平均：token-level reduction

$$
L_{token}=\frac{\sum_i\sum_t\ell_{i,t}}{\sum_i|o_i|}
$$

每个有效 token 权重相同，于是长回答总计影响更大。

| 归约方式 | 每条回答总权重 | 每个 token 权重 | 可能偏差 |
|---|---|---|---|
| sample-level | 相同 | 长回答更小 | 学不到长轨迹里的有效模式，也罚不够长轨迹的重复 |
| token-level | 与长度成正比 | 相同 | 长回答可能主导 batch 梯度 |

DAPO 报告称 token-level policy gradient loss 对其长 CoT 训练更稳定，并改善熵与长度行为。这是特定实验观察，不是普适定理；选择归约方式就是在选择长度权重。

## 12. DAPO：四个故障，四项修正

DAPO 的价值不在于再造一个算法名字，而在于把朴素 GRPO 在大规模长推理训练中的故障逐项暴露出来。

<figure class="teaching-figure"><img src="/illustrations/dapo-four-fixes.svg" alt="DAPO 的 Clip-Higher、动态采样、token 级损失和超长奖励塑形"><figcaption>DAPO 的四项技术分别针对探索塌缩、无效组、长度权重和截断奖励噪声。每项修复也会引入新的分布或稳定性风险。</figcaption></figure>

### 12.1 Clip-Higher：上下裁剪不必对称

标准 PPO 常用 ([1-\epsilon,1+\epsilon])。DAPO 使用不同的 (epsilon_{low}) 和 (epsilon_{high})：

$$
\operatorname{clip}(r,1-\epsilon_{low},1+\epsilon_{high})
$$

直觉：对一个旧概率只有 0.01 的探索 token，20% 上涨只到 0.012；对旧概率 0.9 的常见 token，理论上 20% 上涨已经接近概率上限。对称的相对比率并不意味着相同的绝对探索空间。提高上裁剪为低概率好 token 留出更多增长空间。

但上界越宽，越可能出现过大更新。因此应该同时监控：

- 生成熵与 token 概率分布；
- 被上裁剪/下裁剪 token 的比例；
- KL、梯度范数和训练崩溃；
- 低概率 token 是否真的对应有用探索，而非噪声。

### 12.2 Dynamic Sampling：只把混合组送进更新

DAPO 在一批 prompt 中继续过采样，过滤全对和全错组，直到收集到足够的有效 prompt。这样能保持 batch 中有梯度的题目数量。

它提升样本效率，却会改变训练分布：

- 极难题更常因全错被过滤；
- 极易题更常因全对被过滤；
- 能力边界附近的题被过度表示；
- 若不同领域成功率不同，领域比例也会漂移。

因此日志必须同时记录原始 prompt 分布和过滤后分布，不能只报告 optimizer batch。

### 12.3 Token-Level Policy Gradient Loss

不先在每条回答内部平均，而是按 batch 中所有有效 token 归约。它修复长回答每 token 权重过小的问题，同时意味着长回答贡献更多总梯度。配方必须配合长度统计和过长处理一起看。

### 12.4 Overlong Reward Shaping

达到最大生成长度后被截断，可能有两种完全不同的原因：

- 模型陷入循环、废话或无限检查；
- 推理方向正确，但预算刚好不够完成答案。

若统一给截断样本硬负分，会把第二种情况也当错误。DAPO 先尝试屏蔽截断样本的损失，并提出在最大长度前的缓冲区逐渐增加长度惩罚：

$$
R_{length}(y)=
\begin{cases}
0,& |y|\le L_{max}-L_{cache}\\
\dfrac{(L_{max}-L_{cache})-|y|}{L_{cache}},& L_{max}-L_{cache}<|y|\le L_{max}\\
-1,& |y|>L_{max}
\end{cases}
$$

软惩罚降低奖励突变，但也可能让模型为了躲避长度罚而过早给答案。要同时画准确率—长度联合分布，而非只追求平均输出变短。

## 13. 冷启动、SFT 与纯 RL 的真实关系

“R1-Zero 不先做 SFT”常被简化成“推理不需要示范”。更准确的理解是：

- 基础模型已经从大规模预训练语料见过数学、代码和推理文本；
- RL 只有在采样中偶尔产生可验证成功时才有正信号；
- 强基础模型提供了最初的成功概率与可探索行为；
- 纯 RL 可能产生语言混合、可读性差和格式问题；
- 完整 DeepSeek-R1 又加入冷启动数据、多阶段 SFT、拒绝采样和第二阶段 RL。

### 13.1 SFT 对探索既可能帮助，也可能限制

SFT 会把概率质量集中到示范附近：

- 好处：提高格式正确率和初始任务成功率，让 verifier 更常给出非零信号；
- 风险：降低轨迹多样性，让模型较少探索示范之外的方法。

所以问题不是“SFT 还是 RL 二选一”，而是不同阶段需要多少可控模仿和多少在线探索。

### 13.2 小模型不一定适合从零做大规模 RL

DeepSeek-R1 报告的附录指出，其较小基础模型在纯 RL 中容易随长度增长进入重复，并未在 AIME 上得到有意义提升；对小模型，蒸馏强模型的高质量轨迹往往更经济。这个结论提醒我们：RL 不能创造基础模型从未具备、也从不采到的行为。

## 14. 从论文报告读懂一套训练配方

看到“我们使用 GRPO”时，至少还要问以下问题：

| 维度 | 必须核对的实现细节 |
|---|---|
| reward | 二元还是连续？规则、模型还是环境？格式分怎样组合？ |
| group | 每题采样多少条？温度多少？样本是否相关？ |
| advantage | 只减均值还是除标准差？全同组怎样处理？ |
| loss | PPO clip 还是其他目标？上下界是否对称？ |
| reduction | 先按回答平均，还是按所有 token 平均？ |
| KL | 是否使用 reference？系数多少？reference 多久更新？ |
| rollout | 最大长度、采样引擎、策略版本和离策略程度如何？ |
| filtering | 是否动态采样？过滤后题目分布怎样变化？ |
| truncation | 截断样本屏蔽、硬罚还是软罚？ |
| evaluation | 报告 pass@1、pass@k 还是 majority？预算是否相同？ |

GRPO、PPO、DAPO 更像“设计空间里的家族名”。真正决定行为的是上述组合。

## 15. 在线 rollout 的工程闭环

一套可复现的训练系统至少有四个服务：

1. **rollout worker**：用明确版本的旧策略批量生成回答；
2. **verifier worker**：在隔离环境计算奖励与失败类型；
3. **trainer**：读取 token、old log-prob、reward、mask，计算优势与 loss；
4. **orchestrator**：控制模型版本、采样温度、过滤、checkpoint 和评估。

每条轨迹至少记录：

```text
prompt_id
prompt_version
policy_checkpoint
reference_checkpoint
token_ids / attention_mask
old_logprobs
generation_temperature
verifier_version
reward_components
termination_reason
response_length
```

### 15.1 为什么策略版本差会危险

rollout 生成很慢，trainer 更新很快。如果一条轨迹由很旧的策略产生：

$$
r_{i,t}=\frac{\pi_\theta}{\pi_{old}}
$$

会大量远离 1，更多 token 被裁剪，样本价值下降，重要性权重方差增大。应监控 rollout staleness、ratio 分布和 clip fraction，而不是只看总吞吐。

### 15.2 训练吞吐不等于 GPU 一直满载

长回答有严重的尾延迟：同一批中最慢的序列可能拖住所有 worker。动态采样又会生成被过滤的组。因此系统指标应分开记录：

- 生成 token/s 与训练 token/s；
- 有效组率与被过滤 rollout token；
- P50/P95 回答长度和尾延迟；
- verifier 吞吐、超时与沙箱失败；
- 每个有效优化 token 的端到端成本。

## 16. 训练监控：哪些曲线要一起看

单看 reward 上升非常危险。至少需要下面这组联合仪表：

### 学习是否真的发生

- 独立验证集 pass@1；
- 同预算 pass@k / self-consistency；
- 训练题与新题型的差距；
- 按难度、领域、答案格式分层的准确率。

### 优化是否稳定

- KL、ratio 的分位数、clip fraction；
- policy entropy、梯度范数、NaN/Inf；
- 正负优势比例与优势尺度；
- 当前/旧/reference checkpoint 的版本距离。

### rollout 是否有效

- 全对组率、全错组率、混合组率；
- 动态采样平均重试组数；
- 平均/P95 输出长度、截断率、重复率；
- 每个有效 prompt 消耗的 rollout token。

### 奖励是否被攻击

- 各 reward component 的单独曲线；
- 格式分上涨但正确率不涨；
- 公开测试通过、隐藏测试失败；
- verifier 置信度与人工复核不一致；
- 新出现的异常模板、答案字符串和环境调用。

## 17. 常见失败案例与排查顺序

### 症状 A：reward 很快上涨，独立准确率不涨

1. 人工查看最高奖励样本；
2. 把格式奖励和正确奖励拆开；
3. 更换隐藏测试或 verifier 版本复核；
4. 检查答案泄露与环境越权；
5. 暂停继续扩大 RL 步数。

### 症状 B：几乎所有组都是全错

1. 测基础模型 pass@k，确认是否存在可采样成功；
2. 降低题目难度或增加结构提示；
3. 先用教师轨迹/SFT/蒸馏建立冷启动；
4. 提高组大小前先核算 token 成本；
5. 检查 verifier 是否把等价正确答案误判成错。

### 症状 C：几乎所有组都是全对

1. 减少这类题目的组大小；
2. 提升难度或加入更强隐藏测试；
3. 保留少量简单题防遗忘；
4. 检查是否发生训练集记忆或答案泄露。

### 症状 D：回答越来越长，reward 也上涨

1. 画 reward—length 与 accuracy—length 联合图；
2. 区分有用推理、重复检查和截断；
3. 核对 sample/token-level reduction；
4. 在固定 token 预算下重新比较；
5. 测试软长度塑形是否让模型过早猜答案。

### 症状 E：生成熵快速塌缩

1. 看上裁剪 token 的旧概率分布；
2. 核对采样温度和随机种子；
3. 增加探索不能只看平均熵，还要看答案与策略多样性；
4. 测试非对称 clip，并单独做稳定性消融。

## 18. 把 GRPO 写成可检查的伪代码

```python
for step in range(num_steps):
    old_policy.load_state_dict(policy.state_dict())
    prompts = sample_prompts(prompt_dataset)

    # 每题 G 条；必须保存采样时逐 token log-prob
    groups = rollout(old_policy, prompts, group_size=G)
    rewards = verifier(groups.completions)

    # 归一化只发生在同一 prompt 内部
    advantages = group_normalize(rewards)
    valid = group_has_both_success_and_failure(rewards)

    for minibatch in make_minibatches(groups, valid):
        new_logp = policy.log_probs(minibatch.tokens)
        ratio = exp(new_logp - minibatch.old_logp)

        unclipped = ratio * minibatch.advantage
        clipped = clamp(ratio, 1 - eps_low, 1 + eps_high) \
                  * minibatch.advantage
        policy_gain = minimum(unclipped, clipped)

        # 选择 sample-level 或 token-level mask/reduction
        loss = -masked_reduce(policy_gain, minibatch.loss_mask)
        if use_reference_kl:
            loss += beta * reference_kl(policy, reference, minibatch)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
```

伪代码故意没有隐藏关键选择：`group_normalize`、`valid`、`masked_reduce`、reference 更新频率和截断 mask 都会改变算法行为。

## 19. 与 Kimi K3 案例怎样连接

阅读任何前沿模型报告中的“reasoning RL”时，用本课的框架追问：

- 奖励来自规则验证器、偏好模型还是交互环境？
- 一条训练样本是单回答、同题多回答还是多步 Agent 轨迹？
- 优势是组内相对、value/GAE，还是其他估计？
- 是否在线重新 rollout？策略版本差如何控制？
- 长轨迹按样本还是 token 加权？
- 工具调用成功是否同时满足权限、安全和成本约束？

这样读 [K3 第 10 章：Pre-training、SFT 与 RL](/guide/ch10) 时，就不会把“用了 RL”当作完整解释，而能识别报告披露了哪些环节、哪些仍未知。

## 20. 本课自测

<ConceptCheck question="同一道题的 8 条回答全部通过验证器时，给标准差加 epsilon 能恢复 GRPO 的组内学习信号吗？" :options='["能，epsilon 会自动把最好的回答找出来", "不能，所有奖励减去组均值仍为 0", "能，因为 PPO 会自动训练一个 Critic"]' :answer="1" explanation="epsilon 只防止除零，不能创造组内差异；需要更难题、更多探索、其他奖励或动态采样。" />

<ConceptCheck question="GRPO 中 old policy 与 reference policy 最准确的区别是什么？" :options='["old 用于采样与概率比，reference 用于 KL 行为锚定", "两者永远是同一个模型且作用完全相同", "reference 接收梯度，old 负责打奖励"]' :answer="0" explanation="old 解释数据由谁生成；reference 定义不希望偏离的行为锚点。实现可复用快照，但两个概念职责不同。" />

<ConceptCheck question="为什么 token-level loss reduction 不是无条件更好？" :options='["它会让长回答贡献更多总梯度，可能由长轨迹主导更新", "它无法对 token 求导", "它一定删除全部长度信息"]' :answer="0" explanation="每 token 等权意味着更长回答拥有更多 token、也拥有更高总权重；是否合适要结合长度质量与任务验证。" />

<details><summary>练习 1：奖励 [1, 0, 0, 0] 的均值、总体标准差和标准化优势是多少？</summary>

均值 (0.25)，总体标准差 (sqrt{0.25\times0.75}\approx0.433)。成功样本优势约 (+1.732)，三个失败样本各约 (-0.577)。四个优势之和为 0。
</details>

<details><summary>练习 2：若单条成功率 p=0.1、组大小 G=8，随机一组有正负样本的概率是多少？</summary>

(1-0.1^8-0.9^8\approx0.5695)，约 57%。独立假设下，为得到一个有效组平均要采样 (1/0.5695\approx1.76) 组。真实样本有关联，成本可能更高。
</details>

<details><summary>练习 3：为什么“最终答案正确”不能证明中间推理正确？</summary>

结果 verifier 只观察最终可验证条件。轨迹可能包含无效步骤、遗漏论证或偶然猜中；结果奖励适合训练整体行为，却不是逐步证明。需要形式检查、过程验证或人工审计承担更强断言。
</details>

## 21. 推荐阅读路线

不要一次啃完所有论文。按下面顺序读，能把公式和系统对应起来。

### 第一遍：建立 PPO / RLHF 基础

1. [CS224N L08 Post-training slides](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture08-posttraining.pdf)：重点看 REINFORCE 推导、PPO clipped objective、奖励模型与 DPO 的关系。
2. [CMU ANLP L17 RL Applications slides](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-17-rl-llms.pdf)：重点看整段动作与 token 动作、组奖励例子，以及 reward / advantage / loss 三层分类。
3. [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf)：理解经典 SFT—reward model—PPO 管线。

### 第二遍：理解 GRPO 与推理 RL

1. [DeepSeekMath](https://arxiv.org/pdf/2402.03300)：看 GRPO 最初的动机、组内优势和数学奖励。
2. [DeepSeek-R1](https://arxiv.org/pdf/2501.12948.pdf)：先读第 2 节 GRPO 与 reward design，再读附录中的 PPO 对比、reward hacking、失败尝试和蒸馏比较。
3. [CS224N L12 Reasoning I slides](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture12-reasoning-part1.pdf)：把训练时推理 RL 和推理时计算区分开。

### 第三遍：理解为什么朴素 GRPO 会失败

1. [DAPO](https://arxiv.org/pdf/2503.14476.pdf)：先读第 2 节基线，再逐项读 3.1–3.4；每项都写下“故障观测—修改—代价”。
2. [DeepSeek-R1 附录 G](https://arxiv.org/pdf/2501.12948.pdf#page=63)：关注基础模型能力、verifier 可靠性、PRM 与 MCTS 的失败边界。
3. [CS224N L13 Reasoning II slides](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture13-reasoning-part2.pdf)：继续连接 verifier、搜索、并行生成和系统吞吐。

### 阅读论文时做这张五列表

| 作者报告的改动 | 它修复的观测故障 | 公式改变 | 额外成本 | 结论适用范围 |
|---|---|---|---|---|
| 例如 Dynamic Sampling | 全同组导致零优势 | 增加有效组约束 | 更多 rollout | 长 CoT 数学、指定模型与数据 |

这能防止把一篇论文的工程配方误记成普遍定律。

下一课：[离线 RL、探索与 Agent](/beginner/47-rl-agent)。

<ChapterReadings lesson="46-verifiable-rewards" />
