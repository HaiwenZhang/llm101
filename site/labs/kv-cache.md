---
title: KV Cache 计算器
description: 比较 MHA、GQA 与 MLA 的缓存规模
---

# KV Cache 计算器

## 先记住这条近似式

标准 MHA 每层、每个历史 token 都缓存所有 head 的 Key 和 Value：

$$
M_{KV}\approx B\times T\times L\times H\times d_h\times 2\times b
$$

`B` 是并发请求，`T` 是上下文长度，`L` 是层数，`H` 是 KV head 数，`d_h` 是 head 维，`2` 代表 K 与 V，`b` 是每个数占的字节。

<KVCacheLab />

## 建议做的四组实验

1. 固定其他参数，把上下文从 32K 拖到 1M；
2. 固定上下文，把 batch 从 1 拖到 32；
3. 让 GQA 的 KV head 从 64 降到 8，再降到 1；
4. 比较 MHA 与 MLA 近似结果，但不要把差值解释成整台服务器的总显存。

实际服务还要存模型权重、运行时 workspace、临时激活，并考虑分页、碎片、量化与 prefix 共享。
