---
title: 核心术语表
description: Kimi K3 自学教程附录
---

# 附录 A　核心术语表

| 英文 | 中文/解释 |
|---|---|
| activated parameters | 单 token 前向真正经过的参数规模，不含未选专家 |
| agent harness | 包装模型、工具、prompt、memory、循环和预算的执行脚手架 |
| All-to-All | 各 rank 互相发送不同数据，MoE dispatch/combine 常用 |
| AttnRes | 对历史层/块表示做内容依赖的深度 attention |
| autoregressive | 当前输出依赖此前输入/输出，逐步生成 |
| auxiliary-loss-free balancing | 不用额外负载损失干扰主梯度，以 dispatch bias 等方式平衡 |
| Block AttnRes | 以 block representation 代替所有 layer outputs 的 AttnRes |
| causal mask | 禁止 token 读取未来位置的下三角 mask |
| context parallelism | 沿序列维切分计算/状态 |
| cooldown | 预训练后期降学习率并调整数据/上下文的阶段 |
| decode | 已有 cache 后逐 token 生成阶段 |
| delta rule | 按当前 key 的预测误差擦写 fast-weight memory |
| draft model | speculative decoding 中提出候选 token 的小模型 |
| effort | 推理/Agent 允许使用的 thinking/token/tool budget 档位 |
| expert parallelism | 将不同 MoE experts 分布到不同设备 |
| FFN | Transformer 的逐 token 通道混合网络 |
| Gated MLA | 带输入依赖 output gate 的 MLA |
| GRM | 生成式奖励模型/Agent judge，根据 rubric 评估开放输出 |
| KCP | KDA Context Parallelism，用 affine fragments 做 prefix scan |
| KDA | Kimi Delta Attention，逐通道 decay 的 gated delta recurrence |
| KV cache | 为 autoregressive decode 保存历史 attention key/value |
| latent width | routed path 中低于 model hidden width 的表示维度 |
| linear attention | 可用固定状态递推/核分解计算的 attention 家族 |
| MLA | Multi-head Latent Attention，低秩压缩 K/V cache |
| MoE | Mixture of Experts，条件激活部分专家的稀疏层 |
| MOPD | Multi-Teacher On-Policy Distillation，多教师在 student 状态上蒸馏 |
| MTP | Multi-Token Prediction，额外预测未来多个 token 的训练模块 |
| MXFP4/8 | microscaling 低精度格式，分块共享 scale |
| NoPE | query/key 不加显式位置编码 |
| on-policy | 数据状态分布由当前/近当前 policy 自己采样 |
| partial rollout | 完成一定比例轨迹就更新，剩余轨迹暂停后续跑 |
| pipeline bubble | pipeline stage 因依赖而空闲的时间段 |
| prefill | 并行处理完整输入 prompt、建立 cache 的阶段 |
| prefix cache | 跨请求复用相同前缀的模型运行状态 |
| QAT | Quantization-Aware Training，训练时模拟目标量化误差 |
| Quantile Balancing | 用 router margin quantile 直接设下一步专家 bias |
| recurrent state | 递推模型跨 token 保留的固定/有限状态 |
| reward hacking | 利用 verifier/judge 漏洞得高分但不完成真实目标 |
| routed expert | 每 token 由 router 条件选择的专家 |
| shared expert | 每个 token 固定执行的通用专家 |
| SiTU-GLU | 两 branch 用 scaled tanh softcap 的有界 gated FFN activation |
| speculative decoding | draft 提案、target 并行验证的无损加速方法 |
| state staleness | rollout 由旧 policy 生成，训练时 policy 已更新 |
| teacher forcing | 训练时用真实历史 token 并行预测各下一个 token |
| token mixing | 跨序列位置的信息交互 |
| TTFT | Time To First Token，首 token 延迟 |
| verifier | 依据答案/最终环境状态给分的独立检查器 |

---
