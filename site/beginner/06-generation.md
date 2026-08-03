---
title: 第 12 课 生成、Prefill 与 KV Cache
description: 从一次聊天请求拆解训练和在线推理
---

# 第 12 课：模型怎样一个 token 一个 token 地说话

<div class="lesson-lead">生成分两段：Prefill 并行读完整提示词，Decode 再逐 token 生成。KV Cache 避免每一步都重新计算全部历史 Key/Value。</div>

::: info 本课资料地图
- 两门主课：[CMU ANLP L09 · Decoding Algorithms](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-09-decoding.pdf)第 49–60 页讲 KV Cache、瓶颈、推测解码与多请求优化；[CS224N Reasoning II](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture13-reasoning-part2.pdf)与 [CMU Test-Time Scaling](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-23-inference-scaling.pdf)补充多候选推理怎样放大生成预算。
- 系统课件：[LLM Decoding](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-09-decoding-7735f8be9186c8840ed83128173a0c8f.pdf)、[PagedAttention Serving](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-22-vLLM_woosuk_kwon-1f34697dbb1a1fb5b798daf6eff14b67.pdf)、[DistServe](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-24-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf)与 [Stanford CS336 Lecture 10 可执行 Slides](/lectures/?trace=var/traces/lecture_10.json)；
- 系统论文：[PagedAttention / vLLM](/papers/pagedattention_vllm)解释 KV Cache 为什么需要分页，[Mooncake](/papers/mooncake)解释缓存如何成为分布式服务的中心资源；
- K3 对照：[Kimi K3](/papers/kimi_k3)展示 MLA、KDA 固定状态和混合 prefix cache 如何共同服务 1M 上下文。
:::

## 1. 训练和生成为什么形状不同

训练时完整答案已知，所有位置可并行预测下一位。生成时未来 token 不存在：

```text
“北京是中国的” → 生成“首”
“北京是中国的首” → 生成“都”
“北京是中国的首都” → 生成“。”
```

后三步互相依赖，不能提前同时知道。

### 1.1 自回归生成算法到底循环什么

给定 Prompt $x$，第 $t$ 步模型输出词表 logits：

$$
z_t=f_\theta(x,y_{<t}),\qquad
p_t=\operatorname{softmax}(z_t)
$$

解码器从 $p_t$ 取 argmax、采样或搜索得到 $y_t$，把它追加回上下文，再运行下一步。循环在以下条件之一停止：生成 EOS、命中 stop sequence、达到最大 token 数、超时、被用户取消，或工具协议要求暂时交出控制权。

因此一次请求必须记录 `finish_reason`。回答突然结束可能是模型选择 EOS，也可能只是 `max_tokens` 截断；两者对质量诊断完全不同。

### 1.2 Teacher forcing 为什么能让训练并行

训练时答案 `首 都 。` 已经存在，把它右移后一次送入：

```text
输入位置: 北京 是 中国 的 | 首 | 都
监督目标: 是   中国 的   首 | 都 | 。
```

每个位置只允许看自己左侧，但所有目标 token 都已知，所以 GPU 可以一次计算整张 causal attention 矩阵和所有位置 loss。生成时模型必须吃自己刚选的 token，不能用金标准前缀替代，这也是训练—部署前缀分布不同的来源。

## 2. 一次请求分成两个阶段

<figure class="teaching-figure concept-figure"><img src="/illustrations/prefill-decode-sparse.webp" alt="Prefill 并行处理提示并建立 KV Cache，Decode 逐个生成并追加缓存"><figcaption>Prefill 的工作形状像“整批读题”；Decode 像“读档案、写一个词、追加一张卡”，直到结束。</figcaption></figure>

<figure class="teaching-figure source-figure"><a href="/lectures/images/prefill-decode.png" target="_blank"><img src="/lectures/images/prefill-decode.png" alt="Stanford CS336 Slides 对 Prefill 与 Decode 两阶段工作负载的原始示意图"></a><figcaption>来源图：Stanford CS336 Lecture 10《Inference》。它把 Prefill 与 Decode 放在同一时间线上，适合在读完本站低密度图后核对系统术语；<a href="/lectures/?trace=var/traces/lecture_10.json">打开本地可执行 Slides</a>。</figcaption></figure>

<figure class="teaching-figure concept-figure"><img src="/illustrations/generation-three-shapes.svg" alt="训练、Prefill 与 Decode 的输入形状、状态和性能指标对比"><figcaption>三个阶段运行同一套模型权重，却不是同一种负载。训练有 backward；Prefill 用大矩阵并行建立 Cache；Decode 每步只来一列，却反复读取权重和历史状态。</figcaption></figure>

### Prefill：先读题

Prompt 有 T 个 token。模型一次并行处理它们，建立每层历史状态并产生第一个新 token。

- 矩阵较大；
- 并行度高；
- 长 prompt 的 attention 计算重；
- 指标常看 TTFT（首 token 延迟）。

### Decode：逐字回答

每一步只输入最新 token，读取历史状态，再产生一个 token。

- 矩阵很窄；
- 反复读取模型权重和 cache；
- 常受显存带宽影响；
- 指标常看 TPOT/TBT（token 间延迟）。

所以“训练吞吐高”不能推出“在线生成快”。

### 2.1 Prefill 结束时已经做了什么

Prompt 长度为 $T_p$ 时，每一层都会产生并保存 $K_{1:T_p},V_{1:T_p}$；最后一个位置的隐藏状态经 LM Head 产生首个输出 token 的 logits。因此 TTFT 同时包含排队、Tokenize、Prompt 数据传输、Prefill 前向和第一个 token 的选取，不能把纯 GPU kernel 时间当完整用户延迟。

### 2.2 Decode 一步的矩阵形状

单层、单个新 token 时：

$$
q_t\in\mathbb R^{B\times H_q\times1\times d_h}
$$

$$
K_{1:t},V_{1:t}\in
\mathbb R^{B\times H_{kv}\times t\times d_h}
$$

新 Query 只有一行，却要与长度 $t$ 的历史 Key 匹配。Cache 让旧 K/V 无需重投影，但历史仍要被读取，Attention 仍随当前上下文增长。

## 3. 没有 KV Cache 会发生什么

生成第 1001 个 token 时，标准 attention 需要前 1000 个位置的 K/V。如果每步都重新从头计算：

```text
第 1 步重算 1 个历史 token
第 2 步重算 2 个
...
第 1000 步重算 1000 个
```

大量历史投影被重复计算。

KV Cache 的做法：每产生一个 token，就把它在每层的 K/V 追加保存；下一步只算新 token 的 Q/K/V，历史 K/V 直接读取。

### 3.1 Cache 省掉什么，没省掉什么

设 Prompt 后还要生成 $N$ 个 token：

- **没有 Cache**：每一步都对全部前缀重新做 Q/K/V 投影和各层前向；同一个旧 token 被反复处理；
- **有 Cache**：旧 token 的 K/V 投影只算一次，新步只投影新 token 并追加；
- **仍然存在**：新 Query 必须读取并匹配全部允许的历史 Key，并加权历史 Value。

因此 KV Cache 不是“把 Decode 变成常数复杂度”。标准全局 Attention 每步交互仍约 $O(td_h)$，生成 $N$ 步的累计交互仍有二次项；它主要消除更昂贵的历史层计算和投影重复。

### 3.2 Cache 是派生状态，不是可以随意复用的文本缓存

同一段文字只有在以下配置都一致时才可复用其 KV：模型权重/checkpoint、Tokenizer、RoPE/位置编号、Adapter/LoRA、精度与量化方式、Attention 架构以及前缀 token IDs。字符串看起来相同但 system 模板或特殊 token 不同，Cache 就不等价。

<figure class="teaching-figure"><img src="/illustrations/generation-kv-archive.webp" alt="书写者逐个生成 token，并从保存历史 Key 和 Value 的档案中读取后追加一格"><figcaption>上方是只能一步一步延长的生成时间线；下方档案柜保存历史 K/V。新一步复用旧档案，只追加本位置的新 K/V。图由本教程生成。</figcaption></figure>

<div class="visual-key"><div><b>上：Token 时间线</b>后一个 token 依赖已经生成的前缀，所以必须串行推进。</div><div><b>下：历史档案</b>每格保存一个历史位置在各层的 Key 与 Value。</div><div><b>虚线新格</b>当前步只新增一格，不再重算此前全部投影。</div></div>

## 4. Cache 大小怎样算

标准 MHA 近似：

$$
M_{KV}\approx B\times T\times L\times H_{kv}\times d_h\times 2\times bytes
$$

逐项翻译：

- `B`：并发请求；
- `T`：历史长度；
- `L`：层数；
- `Hkv`：KV head 数；
- `dh`：每个 head 维度；
- `2`：Key 和 Value 两份；
- `bytes`：每个数占多少字节。

<KVCacheLab />

拖动上下文和并发，你会看到两者翻倍，cache 也大约翻倍。

### 4.1 用单位一步一步核对

以 `B=4, T=8192, L=32, Hkv=8, dh=128, BF16=2 bytes` 为例：

$$
4\times8192\times32\times8\times128\times2\times2
=4{,}294{,}967{,}296\text{ bytes}=4\text{ GiB}
$$

这只是 KV 张量本身。实际引擎还需要 block table、allocator 元数据、对齐、临时 attention workspace、输出 logits 和通信缓冲。显存预算必须给碎片与峰值留余量。

### 4.2 “并发 B”最好换成总驻留 token

生产请求长度各不相同，把最大长度 $T$ 乘并发数会过度保守；只用平均长度又会低估长尾。更稳健的调度口径是所有活跃请求当前已缓存 token 总数：

$$
N_{resident}=\sum_{r=1}^{B}T_r
$$

KV 显存再用 $N_{resident}\times L\times H_{kv}\times d_h\times2\times bytes$ 估算。Admission control 应按可容纳 token blocks 判断是否接收新请求，而不只是数请求个数。

### PyTorch：把形状换算成显存

```python
import torch

B, T, L = 4, 8192, 32       # 并发、上下文、层数
H_kv, d_h = 8, 128          # KV heads、每头维度
bytes_per_number = torch.tensor([], dtype=torch.float16).element_size()

elements = B * T * L * H_kv * d_h * 2  # 最后的 2 是 K 与 V
gib = elements * bytes_per_number / 1024**3
print(f"KV Cache ≈ {gib:.2f} GiB")      # 4.00 GiB
```

一次 decode 追加的新缓存，形状可写成 `[B, H_kv, 1, d_h]`；历史缓存是 `[B, H_kv, T, d_h]`。教学代码可以用 `torch.cat` 追加：

```python
new_k = torch.randn(B, H_kv, 1, d_h, dtype=torch.float16)
past_k = torch.randn(B, H_kv, T, d_h, dtype=torch.float16)
past_k = torch.cat([past_k, new_k], dim=2)
```

真实推理引擎通常预分配或分页管理内存，不会每步真的 `cat` 整块张量；否则复制旧缓存本身就很贵。这个例子只用于看懂维度。

## 5. MHA、GQA、MLA 分别改哪里

| 方法 | 历史 token 保存什么 | 主直觉 |
|---|---|---|
| MHA | 每个 head 的 K/V | 最直接、cache 大 |
| GQA | 少量共享 KV heads | 多个 query heads 共用 K/V |
| MLA | 低维 latent + 位置分量 | 需要时再恢复，不缓存完整 K/V |

MLA 没有把 attention 变成线性复杂度；它仍做全局 softmax，主要压缩 cache。

GQA 的重要点是 Query heads 与 KV heads 分离。若 $H_q=64,H_{kv}=8$，每 8 个 Query heads 共享一组 K/V；Cache 约缩为 MHA 的 $8/64=1/8$，Q 与输出投影并未同比缩小。模型质量是否保持要由训练和评测验证，不能只看内存公式。

## 6. KDA 又是另一条路线

KDA 不为每个历史 token 永久保留 K/V，而是递推更新固定大小状态。类比：

- MLA：为每条历史保留压缩档案；
- KDA：不断把历史整理进一块固定大小白板。

固定状态省长序列 cache，但会有记忆冲突和容量上限，所以 K3 用 3 层 KDA + 1 层 MLA 混合。

## 7. Prefix Cache：相同开头不要重复读

许多请求共享 system prompt、工具声明或长文档前缀。Prefix cache 保存这段前缀的运行状态，命中后从分叉点继续。

<figure class="teaching-figure concept-figure"><img src="/illustrations/kv-cache-pages-prefix.svg" alt="逻辑请求 token blocks 映射到物理 KV 页并共享相同前缀"><figcaption>PagedAttention 类方法把 KV 切成固定大小 block，通过 block table 映射。请求 A/B 可以引用相同前缀页，之后分别追加；结束时按引用计数释放，而不是整段连续搬家。</figcaption></figure>

真实系统还要解决：

- 前缀怎样哈希；
- cache 怎样分页；
- 不同模型版本是否兼容；
- KDA 固定状态在什么位置做检查点；
- 命中收益是否大于传输成本。

### 7.1 为什么分页优于每请求预留最大连续空间

若每条请求一开始就预留 128K 连续 Cache，多数短回答会浪费大量显存；请求增长还可能因找不到更大连续块而搬迁。分页把逻辑序列切成固定 block，只在需要时分配物理页，外部碎片大幅减少，最后一页仍可能有少量内部碎片。

分页不是零成本：每步要查 block table，kernel 必须支持不连续页，淘汰与换入还会产生管理和传输开销。Block 太大会浪费尾页，太小则元数据与寻址更多。

### 7.2 Prefix Cache 有隔离与泄漏风险

跨用户共享只应发生在明确公共、版本一致的前缀。私有文档、个性化 system prompt 或租户密钥不能因为 hash 命中就跨边界复用。缓存 key 应包含租户/权限域与模型配置，并有过期、审计和删除机制。

## 8. 延迟指标必须定义分母

| 指标 | 常见定义 | 它主要受什么影响 |
|---|---|---|
| Queue time | 到达至开始执行 | 调度、负载、优先级 |
| TTFT | 到达至首 token 返回 | 排队 + Prefill + 首步 |
| ITL / TBT | 相邻输出 token 的间隔 | Decode batch、带宽、同步 |
| TPOT | 首 token 后的平均每 token 时间 | 总 Decode 时间 / 输出间隔数 |
| E2E latency | 到达至请求完成 | Prompt、输出长度和全部阶段 |
| Throughput | 单位时间完成的 token 或请求 | batching、利用率、长度分布 |

不同论文对 TPOT 是否包含首 token、是平均还是 P50/P99 可能不同。必须同时报告 Prompt/Output 长度分布、并发、硬件和 SLO；只报“tokens/s”无法判断是单请求速度还是服务器总吞吐。

## 9. 为什么 Prefill 偏计算、Decode 常偏带宽

CMU L09 用 Roofline 近似：

$$
t\approx\max\left(
\frac{\text{FLOPs}}{\text{Device FLOP/s}},
\frac{\text{Bytes moved}}{\text{Memory bandwidth}}
\right)
$$

Prefill 把许多 token 拼成大矩阵，一次读取权重可参与大量乘加，算术强度较高；小 batch Decode 每步只有一个/少量 token，却要重新读大部分权重和 KV，常由内存带宽决定。增大 Decode batch 能让一次权重读取服务更多请求，提高吞吐，但单请求等待和尾延迟可能上升。

所以量化有两种收益：减小模型/Cache 数据搬运，也可能让硬件低精度单元获得更高计算吞吐。最终是否加速取决于 kernel、解量化开销与瓶颈位置，不是 bit 数自动按比例变快。

## 10. Continuous Batching：请求在不同时间加入和离开

静态 batch 要等整批最长请求结束，短请求完成后 GPU 槽位空置。Continuous batching 在 token step 边界把完成请求移出、把新请求加入，使活跃 batch 随时间变化。

调度器至少要同时考虑：

- Prefill 是否会阻塞正在 Decode 的低延迟请求；
- 每轮能容纳多少 token blocks；
- 长 Prompt 是否分块 prefill；
- 优先级、公平性与最大等待时间；
- 请求取消后何时回收 Cache；
- P99 TTFT 与总吞吐怎样权衡。

Prefill/Decode 分离部署可以让两种形状使用不同 GPU 池，但会增加 KV 传输、路由和故障恢复复杂度。它不是所有负载都更快，应比较传输成本与 SLO。

## 11. 推测解码省的是串行步，不是换一个答案分布

小 Draft 模型先便宜地产生若干候选 token，大 Target 模型一次并行验证多个位置。精确 speculative decoding 用接受/拒绝校正确保最终样本仍服从 Target 分布；若直接接受 Draft 结果或只做启发式验证，输出分布就可能改变。

收益取决于 Draft 便宜程度、接受率、一次草拟长度和验证 kernel。Draft 太弱会频繁拒绝，太大又失去成本优势。它与 KV Cache 正交：Cache 复用历史状态，推测解码减少必须串行调用 Target 的轮数。

## 12. 生成服务故障定位

| 症状 | 优先检查 |
|---|---|
| TTFT 高、ITL 正常 | 排队、长 Prompt、Prefill batch、prefix 命中 |
| TTFT 正常、ITL 高 | Decode batch、权重/KV 带宽、同步通信 |
| 显存随时间只涨不降 | 完成/取消请求 Cache 未释放、引用计数泄漏 |
| Cache 命中后答案变化 | token IDs、位置 offset、模型/Adapter 版本不一致 |
| P50 快、P99 很慢 | 长尾长度、队首阻塞、抢占与大 Prefill 干扰 |
| 高并发吞吐下降 | block 不足、频繁换入换出、batch 调度或通信饱和 |
| 输出被截断 | EOS、stop sequence、max token、超时的 finish reason |

端到端 trace 应给每个请求同一 correlation ID，串起排队、Tokenize、Prefill、每轮 Decode、Cache 分配/命中、网络流式返回和结束原因。

## 13. 本章阅读路线

1. [CMU ANLP L09](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-09-decoding.pdf) 第 5–8 页先复习逐 token 选择，第 50–60 页重点读 KV Cache、Roofline、推测解码与共享前缀；
2. [Stanford CS336 Inference Slides](/lectures/?trace=var/traces/lecture_10.json) 按 Prefill → Decode → memory/computation → batching 顺序，把每张图标上输入形状；
3. [PagedAttention / vLLM](/papers/pagedattention_vllm) 重点读 block table、内部/外部碎片与 continuous batching 实验，区分算法与系统实现；
4. [Mooncake](/papers/mooncake) 关注 KV-centric disaggregation、传输拓扑、缓存命中与 SLO，而不只看总吞吐；
5. 再回到 [测试时计算](/beginner/49-reasoning-test-time)，计算 best-of-$N$、树搜索和 verifier 会把输出 token、Cache 与延迟放大多少。

## 本课闭卷复述

画一条请求时间线，标出 Prefill、Decode、KV Cache 的写入与读取。再解释为什么长 prompt 和长输出可能卡在不同资源。

<ConceptCheck question="KV Cache 的主要目标是什么？" :options='["减少重复计算历史 token 的 K/V 投影", "减少模型总参数", "让训练不需要反向传播"]' :answer="0" explanation="Cache 用空间换时间，保存历史 K/V 供后续 decode 复用。" />

<ConceptCheck question="使用 KV Cache 后，标准全局 Attention 的 Decode 是否变成与上下文长度无关的常数工作？" :options='["是，历史完全不用读取", "不是；旧 K/V 不再重投影，但新 Query 仍要读取并匹配全部允许历史", "只有模型参数会随上下文增长"]' :answer="1" explanation="Cache 消除历史层计算重复，却没有消除 Query 对历史 K/V 的线性读取与匹配。" />

下一课：[MoE 为什么能拥有很多参数却只激活少量](/beginner/07-moe)。

<ChapterReadings lesson="06-generation" />
