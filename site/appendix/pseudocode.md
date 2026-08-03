---
title: 最小实现伪代码
description: Kimi K3 自学教程附录
---

# 附录 C　最小实现伪代码

这些代码用于验证数学，不是高性能 kernel。

### C.1 单头 recurrent KDA 核心

```python
def kda_step(state, q, k, v, alpha, beta):
    # state: [d_k, d_v]
    # q, k: [d_k], v: [d_v], alpha: [d_k], beta: scalar
    k = k / (k.norm() + 1e-6)
    q = q / (q.norm() + 1e-6)

    decayed = alpha[:, None] * state
    old_value = decayed.T @ k
    state = decayed + beta * k[:, None] * (v - old_value)[None, :]
    output = state.T @ q
    return state, output
```

对照 K3 公式：`decayed + β k(v-decayedᵀk)ᵀ` 展开后就是 `(I-βkkᵀ)Diag(α)S + βkvᵀ`。

### C.2 Full AttnRes

```python
def full_attn_res(history, layer_query, rms_norm):
    # history: list of [batch, tokens, d]
    values = torch.stack(history, dim=-2)       # [B, T, depth, d]
    keys = rms_norm(values)
    logits = torch.einsum("btld,d->btl", keys, layer_query)
    weights = logits.softmax(dim=-1)
    return torch.einsum("btl,btld->btd", weights, values)
```

layer query 固定，但 keys 依赖每个 batch/token，所以 weights 仍随内容变化。

### C.3 教学版 Top-k MoE

```python
def moe(x, router, experts, k=2, bias=None):
    raw = torch.sigmoid(router(x))
    dispatch_score = raw if bias is None else raw + bias
    ids = dispatch_score.topk(k, dim=-1).indices
    selected_raw = raw.gather(-1, ids)
    probs = selected_raw / selected_raw.sum(-1, keepdim=True)

    out = torch.zeros_like(x)
    for slot in range(k):
        expert_id = ids[:, slot]
        # 教学实现逐专家 mask；真实系统会 permute + grouped GEMM + all-to-all
        for j, expert in enumerate(experts):
            mask = expert_id == j
            out[mask] += probs[mask, slot, None] * expert(x[mask])
    return out, ids
```

### C.4 Quantile bias update

```python
@torch.no_grad()
def quantile_balance_bias(raw_score, old_bias, k):
    # raw_score: [m, n]
    biased = raw_score + old_bias
    top_k1 = biased.topk(k + 1, dim=-1).values
    cutoff = top_k1[:, k]                 # [m]
    margin = raw_score - cutoff[:, None]  # [m, n]
    n = raw_score.shape[-1]
    new_bias = -torch.quantile(margin, 1 - k / n, dim=0)
    return new_bias - new_bias.mean()
```

大规模实现必须用跨 rank histogram 估计，不能 gather 所有 margin。

---
