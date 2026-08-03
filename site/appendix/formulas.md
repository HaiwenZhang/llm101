---
title: 公式速查
description: Kimi K3 自学教程附录
---

# 附录 B　公式速查

### B.1 自回归目标

$$
\mathcal L=-\sum_t\log p(x_t|x_{<t}).
$$

### B.2 KV cache

$$
M_{KV}\approx 2LBT H_{kv}d_h\cdot bytes.
$$

### B.3 MoE

$$
y=\sum_{shared}E_j(x)+\sum_{i\in TopK}p_iE_i(x).
$$

### B.4 Delta update

$$
S_t=S_{t-1}+\beta k_t(v_t-S_{t-1}^Tk_t)^T.
$$

### B.5 KDA

$$
S_t=(I-\beta kk^T)\operatorname{Diag}(\alpha)S_{t-1}+\beta kv^T,
\quad o_t=S_t^Tq_t.
$$

### B.6 K3 bounded decay

$$
g=g_{min}\sigma(e^Az),\quad g_{min}=-5,\quad \alpha=e^g.
$$

### B.7 AttnRes

$$
\alpha_{i\to l}=\operatorname{softmax}_i(w_l^T\operatorname{RMSNorm}(h_i)),
\quad h_l=\sum_i\alpha_{i\to l}h_i.
$$

### B.8 LatentMoE

$$
z=W_\downarrow x,\quad
u=\sum_{i\in TopK}p_iE_i(z),\quad
y=\sum E^{shared}(x)+W_\uparrow RMSNorm(u).
$$

### B.9 SiTU-GLU

$$
[\beta_1\tanh(g/\beta_1)\sigma(g)]\odot[\beta_2\tanh(u/\beta_2)].
$$

### B.10 Quantile Balancing

$$
b_j^{t+1}=-Q_{1-k/n}(s_{:,j}-\alpha^t),
\quad b\leftarrow b-mean(b).
$$

### B.11 Effort budget

$$
T(y)>\tau b_0(x)\Rightarrow R=-1.
$$

### B.12 OPD reward

$$
r=clip\left(sg\left[\log\frac{\pi_T(y_t|state)}{\pi_S(y_t|state)}\right],-R_{max},R_{max}\right).
$$

### B.13 Speculative acceptance loss

$$
L_{LK}=-\log\sum_x\min(p(x),q(x)).
$$

### B.14 KDA segment composition

$$
(M_2,S_2)\circ(M_1,S_1)=(M_2M_1,M_2S_1+S_2).
$$

---
