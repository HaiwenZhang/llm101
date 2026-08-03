---
title: 第 13 课 MoE 从零拆解
description: 从 Dense FFN、Top-k Router 到容量、负载均衡与专家并行
---

# 第 13 课：MoE 为什么能“参数很多，每次只算一点”

<div class="lesson-lead">Mixture of Experts（MoE）不是把整个 Transformer 随机切成几份，而是通常把每层的 Dense FFN 换成许多候选 FFN。Router 针对每个 token 只选择其中少数专家。要真正读懂 MoE，必须同时算清三本账：模型拥有多少参数、一次激活多少计算、token 为找专家搬了多少数据。</div>

::: info 本课怎样使用资料
- 主线跟随 [CMU ANLP L21](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-21-moe.pdf)：第 2–16 页建立结构，第 17–23 页讨论专家粒度与 shared expert，第 25–36 页讨论路由，第 38–43 页讨论稳定训练，第 46–49 页讨论专家并行与 upcycling，第 50–55 页看真实模型；
- 系统侧补充 [LLM Systems MoE](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-19-MoE-1fd3b9f72ba69c8d3d71e01186674c5e.pdf)；
- 论文主线是 [Switch Transformer](/papers/switch_transformers)、[DeepSeekMoE](/papers/deepseek_moe)、[OLMoE 原论文 PDF](https://arxiv.org/pdf/2409.02060.pdf)与 [LatentMoE](/papers/latent_moe)。
:::

## 1. 先定位：MoE 通常替换的是 FFN

一层 Decoder 可以先粗略写成：

$$
\widetilde h=\operatorname{Attention}(h)+h,
\qquad
h'=\operatorname{FFN}(\widetilde h)+\widetilde h
$$

Dense Transformer 中，批次里的所有 token 都经过同一个 FFN。MoE 把第二行改成：

$$
h'=\operatorname{MoE}(\widetilde h)+\widetilde h
$$

Attention 仍负责 token 与 token 交换信息；专家通常逐 token 地做非线性特征变换。**MoE 不是 Attention 的替代品。**有些架构只把隔几层的 FFN 换成 MoE，也有些几乎每层都放；这是架构设计变量，不是定义的一部分。

<figure class="teaching-figure"><img src="/illustrations/moe-workshop.webp" alt="MoE 专家工坊：Router 将每个输入只送到少量专家，再汇总输出"><figcaption>把它想成工坊：所有工位都属于模型，但一个包裹只送到少数工位。这个比喻只说明条件计算；真实实现还要进行张量分桶和跨设备通信。</figcaption></figure>

### 从一个 Dense FFN 到 E 个专家

假设一个 gated FFN（例如 SwiGLU）含三个大矩阵：

$$
E_i(x)=W_{down}^{(i)}\left(\operatorname{SiLU}(W_{gate}^{(i)}x)\odot W_{up}^{(i)}x\right)
$$

若隐藏维为 $d$、专家中间维为 $d_{ff}$，忽略 bias，一个专家约有：

$$
P_{expert}\approx 3d\,d_{ff}
$$

经典两矩阵 FFN 则约为 $2d\,d_{ff}$。因此比较论文参数量时，先确认它用的是哪种 FFN，不能机械套一个系数。

## 2. Router 到底算了什么

对第 $t$ 个 token 表示 $h_t\in\mathbb R^d$，最简单的线性 Router 有参数 $W_r\in\mathbb R^{d\times E}$：

$$
s_t=h_tW_r\in\mathbb R^E,
\qquad
p_{t,i}=\frac{e^{s_{t,i}}}{\sum_{j=1}^{E}e^{s_{t,j}}}
$$

$s_{t,i}$ 是 logit，$p_{t,i}$ 才是专家概率。接着取概率最大的 $k$ 位专家：

$$
S_t=\operatorname{TopK}(p_t,k)
$$

许多实现会在选中的集合内重新归一化：

$$
g_{t,i}=\frac{p_{t,i}}{\sum_{j\in S_t}p_{t,j}},\ i\in S_t,
\qquad
y_t=\sum_{i\in S_t}g_{t,i}E_i(h_t)
$$

也有实现保留原概率、不做门内归一化。阅读代码时要查清这一点，因为它会改变输出尺度和梯度。

### 手算一个 Top-2

若 Router 给四位专家的概率为 `[0.55, 0.09, 0.33, 0.03]`，Top-2 选 E1 与 E3。门内权重是：

$$
g_1=\frac{0.55}{0.55+0.33}=0.625,
\qquad
g_3=0.375
$$

若两位专家输出被简化为标量 $E_1(x)=10$、$E_3(x)=4$，则 $y=0.625\times10+0.375\times4=7.75$。真实输出是长度为 $d$ 的向量，门控标量对整条向量逐元素缩放。

<figure class="teaching-figure concept-figure"><img src="/illustrations/moe-routing-capacity.svg" alt="MoE 路由从 token、softmax 概率、Top-2、专家分桶到容量溢出的五步流程"><figcaption>Top-k 与容量是两道不同的门。Router 的高分表达“想去这位专家”，并不保证该专家还有系统容量。</figcaption></figure>

## 3. Top-k 为什么是训练难点

Softmax 是连续可导的，Top-k 的“专家编号”却是离散选择：分数发生很小变化时，入选专家可能突然从 E2 变为 E3。工程上通常把被选专家的 gate 路径正常求导，但不会对“另一个专家为什么没被选中”得到普通的离散索引梯度。

这会形成一个正反馈：

```text
某专家早期分数略高
→ 收到更多 token 与梯度
→ 更擅长当前数据
→ Router 更愿意继续选它
→ 其他专家越来越没有训练机会
```

所以 Router 不能只优化语言建模损失，还常要配合负载均衡目标、噪声、容量策略或路由偏置。路由层也常用 FP32 计算：Router 参数并不多，稍高精度的代价小，却能降低接近 Top-k 边界时因舍入造成的专家抖动。

## 4. 一批 token 会怎样把专家挤爆

设一个路由批次有 $N$ 个 token、$E$ 位专家、每 token 选 $k$ 位。总 assignment 数是 $Nk$，完全均匀时每专家收到 $Nk/E$ 个。容量因子 `CF` 把物理容量定义为：

$$
C=\left\lceil \operatorname{CF}\cdot\frac{Nk}{E}\right\rceil
$$

例如 $N=8,E=4,k=2,\operatorname{CF}=1.0$，每专家容量是 4。即使四位专家总负载为 `[4,3,5,4]`、总座位数并不少，E3 仍有一个 assignment 溢出，因为别的专家空位不能自动替代 E3 的函数。

常见处理方式有三类：

| 策略 | 溢出时做什么 | 代价与风险 |
|---|---|---|
| token dropping | 丢掉超容量专家分支，残差路径仍可能保留 | 快且容量固定，但 token 少得到一次专家变换 |
| reroute | 尝试次优专家 | 计算更完整，但语义选择和负载会改变 |
| dropless | 不丢，按真实负载动态计算 | 数值更完整，但慢专家、动态 shape 与显存峰值更难控制 |

提高 CF 会降低溢出概率，却会增加缓冲区、padding 或显存预留；它不会从学习层面修复专家塌缩。

<MoERouterLab />

## 5. 负载均衡损失在约束哪两个量

对专家 $i$，定义：

- $f_i$：实际分给该专家的硬 assignment 比例；
- $P_i$：整个批次给该专家的平均软概率质量。

一种常见辅助项可写成：

$$
L_{aux}=\alpha E\sum_{i=1}^{E} f_iP_i
$$

若硬负载与软概率都均匀，$f_i=P_i=1/E$，不含系数 $\alpha$ 的归一化量正好为 1；若二者都塌到同一专家，该量接近 $E$。同时看 $f_i$ 与 $P_i$ 很重要：前者反映真正的系统工作量，后者给 Router 一条连续的优化信号。

但辅助损失也有副作用。$alpha$ 太弱，专家仍可能塌缩；太强，Router 可能为了“人数好看”把 token 送给不合适的专家，语言建模主目标反而变差。因此报告一个 MoE 时至少应同时给主损失、辅助损失、每专家 token 数和最大/平均负载，而不是只写“使用 load balancing”。

### Router z-loss 约束的是另一件事

Router logits 整体变得很大时，Softmax 可能极端饱和并带来数值不稳定。z-loss 常惩罚归一化常数的对数：

$$
L_z=\frac{1}{N}\sum_t\left(\log\sum_i e^{s_{t,i}}\right)^2
$$

$L_z$ 不是负载均衡损失：它主要防止 logits 的绝对尺度无控制地增长；$L_{aux}$ 关注 token 在专家之间的分布。两者要分开记录。

## 6. 三本账：总参数、激活计算、系统通信

设非专家部分参数为 $P_{dense}$，有 $E$ 个 routed experts、$S$ 个 always-on shared experts，每 token 选择 $k$ 个 routed experts。忽略 Router 小项时：

$$
P_{total}\approx P_{dense}+(E+S)P_{expert}
$$

$$
P_{active/token}\approx P_{dense}+(k+S)P_{expert}
$$

Router 自身约有 $dE$ 个参数，通常远小于全部 FFN 专家。以单层 $d=4096,d_{ff}=11008$ 的 SwiGLU 专家为例：

$$
P_{expert}=3\times4096\times11008=135{,}266{,}304\approx135.3\text{M}
$$

若 $E=64,k=2$，这一层的 routed experts 总计约 8.66B 参数，但一个 token 只触碰约 270.5M routed-expert 参数。这个计算**只算一层的专家部分**，没有包含 Attention、Norm、Embedding、shared experts 和其他层。

::: warning Activated parameters 不是 FLOPs
“触碰了多少参数”适合做量级直觉，但精确 FLOPs 还取决于序列/批次形状、矩阵乘法约定、shared experts、Router、是否重计算以及 kernel 利用率；推理延迟还包括权重读取和网络通信。
:::

<figure class="teaching-figure concept-figure"><img src="/illustrations/moe-expert-parallel-flow.svg" alt="MoE 专家并行中 pack、All-to-All、grouped GEMM、返回与合并的跨卡流程"><figcaption>专家并行的真实数据流。第一次 All-to-All 把 token 送到专家，第二次把结果送回原卡；负载最重的专家会让其他设备等待。</figcaption></figure>

### 为什么需要 grouped GEMM

逐 token 调用许多小 FFN 会产生大量小 kernel，GPU 很难吃满。实现通常先按 `expert_id` 对 token 排序，让同一专家的输入连续，再批量执行 grouped GEMM。算完后根据 inverse permutation 恢复原 token 顺序，并按 gate 合并。

系统时间可粗拆为：

```text
Router + Top-k
+ pack / sort
+ 第一次 All-to-All（dispatch）
+ grouped GEMM（experts）
+ 第二次 All-to-All（combine）
+ unpermute + gate 聚合
```

因此专家数增多但每专家 token 太少时，虽然理论激活 FLOPs 没变，GEMM 尺寸可能变碎；Top-k 增大时，每个 token 的通信副本也变多。必须真实测 All-to-All 字节数、网络时间和每专家 batch size。

## 7. 三种路由视角不是一回事

CMU 课件第 25–36 页把路由分成几类：

| 视角 | 谁做选择 | 容量特性 | 主要问题 |
|---|---|---|---|
| token-choice | 每个 token 选自己的 Top-k 专家 | 每专家负载不固定 | 最自然，但可能拥堵 |
| expert-choice | 每位专家从一批 token 中选最想要的若干个 | 容量天然固定 | 在线逐 token 生成时难以等待一个完整批次再全局选择 |
| global assignment | 联合优化整批 token→expert 匹配 | 可显式加约束 | 计算、近似与分布式实现更复杂 |

Hash routing 则直接用 token 身份映射专家，完全避开可学习 Router，却不一定能根据上下文区分同一个 token 的不同含义。最佳路由没有只靠结构就能决定的统一答案，必须结合训练目标、在线生成约束和硬件拓扑。

## 8. 专家越细，组合数为什么暴涨

从 $E$ 位专家里选 $k$ 位，不考虑顺序时有 $\binom Ek$ 种集合。例如：

$$
\binom 82=28,
\qquad
\binom{16}4=1820
$$

更细粒度的专家可以用更多组合表达不同 token 需求。这是 DeepSeekMoE 强调 fine-grained expert segmentation 的直觉。但组合数不等于模型一定学到同样多种有意义技能：gate 权重、专家共适应、数据分布和负载约束都会减少实际使用的组合。

### Shared expert 的作用与边界

Shared experts 对每个 token 都执行，目标是吸收通用变换，让 routed experts 更专门化。若有 $S$ 个 shared experts，它们必须加到每 token 激活账里，并非免费。CMU 课件也强调：不同研究对 shared experts 的收益并不完全一致，因此应把它作为要消融的设计，而不是“加了必然更好”的定理。

可复现实验至少对照：

| 保持不变 | 只改变 | 同时观察 |
|---|---|---|
| 总训练 token、优化器、总/激活参数预算 | 是否使用 shared experts | 验证损失、专家相似度、负载、吞吐 |
| 激活 FLOPs 与并行策略 | 专家数量与粒度 | 模型质量、All-to-All 时间、每专家 batch |
| 数据顺序与 Router 初始化 | 辅助损失系数 | 主损失、最大/平均负载、溢出率 |

## 9. 专家并行怎样和其他并行共存

Expert Parallelism（EP）把不同专家放在不同 GPU；Data Parallelism（DP）复制模型处理不同数据；Tensor Parallelism（TP）把一个大矩阵切到多卡；Pipeline Parallelism（PP）按层分阶段。真实大模型可能同时组合它们。

EP 的关键不是“每张卡只存一位专家”这句口号，而是：

1. 每个 rank 先拥有本地 token；
2. Router 计算目标专家与 gate；
3. pack 后按目标 rank 做 All-to-All；
4. 本地 grouped GEMM 运行一个或多个专家；
5. 第二次 All-to-All 返回结果；
6. unpermute 后恢复 `[B,T,d]`。

设备侧应监控：每 rank 发送/接收字节、通信与计算重叠率、最慢 rank 时间、专家 token 直方图、padding/容量浪费、溢出率、grouped GEMM 的实际形状。只给理论 FLOPs 无法解释尾延迟。

## 10. 从头训练、Upcycling 与专家“专业化”

从头训练 MoE 能让 Router 与专家共同形成分工，但代价高。Upcycling 会从已经训练好的 Dense checkpoint 出发，把 FFN 权重复制或拆分为多个专家，再继续训练；它节省前期成本，却可能让专家起点过于相似，需要后续数据与路由打破对称性。

研究有时会展示某些专家偏好代码、数学或特定语言，但“专家 7 就是数学专家”通常过度简化。一个可靠的专业化分析至少要区分：

- Router 选择频率：哪些 token 更常去它；
- 因果贡献：屏蔽或替换该专家是否真的损害相应能力；
- 跨层差异：不同层的专家可能承担完全不同的抽象；
- 稳定性：换随机种子、数据切片后该现象是否仍存在。

路由热力图给的是相关性，不自动给出因果解释。

## 11. LatentMoE 为什么先降维

当隐藏维 $d$ 很大时，跨卡发送 token 表示和存储专家权重都很贵。LatentMoE 的核心路径可概括为：

```text
d 维 token → 降到 ℓ 维 → 在 latent 空间路由与专家计算 → 聚合 → 升回 d 维
```

若 $\ell=d/4$，专家侧激活通信量可获得约 4 倍的量级缩减，于是有机会用相同预算容纳更多、更细专家。但这是结构权衡，不是无损压缩：$\ell$ 太小会形成信息瓶颈，降/升维投影本身也有参数与计算。

## 12. 读一份 MoE 报告时检查什么

不要只抄“总参数 / 激活参数”。至少追问十件事：

1. 哪些层是 MoE，哪些仍是 Dense FFN？
2. 专家用两矩阵 FFN 还是 gated FFN？每专家的 $d_{ff}$ 多大？
3. routed experts、shared experts、Top-k 各是多少？
4. Top-k gate 是否重新归一化？
5. token-choice、expert-choice 还是别的路由？
6. 容量因子多少？溢出是 drop、reroute 还是 dropless？
7. 辅助负载损失、routing bias、z-loss 分别怎样设置？
8. Router 用什么精度？是否加噪声或 jitter？
9. EP/TP/DP/PP 怎样组合，All-to-All 占多少时间？
10. 总参数、激活参数、训练 FLOPs、吞吐与质量是否在同一口径下比较？

这十问能把“一个很大的数字”还原成可复现的架构与系统设计。

## 13. 资料逐段精读路线

第一次学习不必从论文公式硬啃，可以按下面顺序：

1. [CMU L21 第 2–16 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-21-moe.pdf#page=2)：只回答 MoE 替换哪一层、稀疏在哪里；
2. [第 25–36 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-21-moe.pdf#page=25)：比较 token-choice、expert-choice、hash 与全局分配；
3. [第 38–43 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-21-moe.pdf#page=38)：逐项写出 $f_i$、$P_i$、z-loss 和 FP32 routing 解决的问题；
4. [第 46–49 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-21-moe.pdf#page=46)：对照本页专家并行图，追踪一次 token 往返；
5. [DeepSeekMoE PDF](https://arxiv.org/pdf/2401.06066.pdf)与 [OLMoE PDF](https://arxiv.org/pdf/2409.02060.pdf)：把专家配置、训练损失、负载策略和系统结果填进第 12 节十问。

## 本课闭卷复述

请不用“省计算”三个字糊弄过去，而是完整说出：Dense FFN 怎样变成专家集合；Router logits 怎样变成 Top-k gate；为什么一批 token 会容量溢出；$f_i$ 与 $P_i$ 分别是什么；为什么激活参数少仍可能被 All-to-All 拖慢。

<ConceptCheck question="N=1,024 个 token、E=8 位专家、每 token 选 k=2、容量因子 CF=1.25。每专家容量 C 是多少？" :options='["256", "320", "2,560"]' :answer="1" explanation="平均 assignment 数为 Nk/E=1,024×2÷8=256；乘容量因子 1.25 后为 320。容量按 assignment 而不是原始 token 数计算。" />

<ConceptCheck question="Router 已经严重偏向 E1。把容量因子从 1.0 提到 2.0，最准确的说法是什么？" :options='["E1 会自动学会少接 token", "溢出可能减少，但路由偏斜没有被修复", "总参数会减半"]' :answer="1" explanation="更大的容量只是多留缓冲与计算空间，不能替代负载均衡目标或路由策略；它还可能提高显存和计算峰值。" />

下一课：[Pre-training、SFT、RL 与 Agent](/beginner/08-post-training)。

<ChapterReadings lesson="07-moe" />
