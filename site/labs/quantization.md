---
title: 量化与权重显存实验
description: 比较 FP16、INT8 与 INT4 权重大小及最少显卡数
---

# 量化：低 bit 到底省了什么

## 先做预测

1. 7B 模型的 FP16 权重是否正好是 14 GiB？注意 GB 与 GiB 的区别。
2. INT4 权重理论上是 FP16 的几分之一？
3. 权重刚好装进一张卡，是否代表长上下文服务一定能运行？

<QuantizationLab />

本实验只算权重字节数。实际量化还包括 scale、zero-point、分组方式、异常值处理和校准数据；端到端显存还必须加入 KV Cache、激活与 workspace。

继续学习：[第 36 课量化与低精度计算](/beginner/30-quantization)。
