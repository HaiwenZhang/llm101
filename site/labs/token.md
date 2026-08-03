---
title: 文字进入模型实验
description: 逐步观察文字、Token、ID 与 Embedding 的区别
---

# 文字怎样进入模型

## 先做预测

1. Token ID 是不是数值越大，语义就越重要？
2. “机器学习”一定会是一个 Token 吗？
3. 模型真正做矩阵运算的是文字、ID 还是向量？

<TokenLab />

这个实验先把四件事拆开：原始文字、切分结果、词表编号、Embedding 向量。编号只是查表地址，只有查到的向量才进入后续计算。

继续学习：[第 01 课 Token 与分词](/beginner/01-token) → [BPE 逐轮合并实验](/labs/bpe)。
