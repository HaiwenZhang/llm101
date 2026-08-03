---
title: 词向量方向与余弦相似度
description: 通过二维教学投影理解点积、向量长度与余弦相似度
---

# 词向量方向与余弦相似度

## 先做预测

1. “猫”和“狗”的余弦相似度应该更接近 1、0 还是 -1？
2. 把“狗”换成“汽车”，点积和余弦会同时下降吗？
3. 两个向量长度都翻倍，余弦相似度会不会改变？

<VectorSimilarityLab />

## 对回公式

$$
\cos(\mathbf a,\mathbf b)=
\frac{\mathbf a^\top\mathbf b}{\|\mathbf a\|_2\|\mathbf b\|_2}
$$

分子是点积，既受方向也受长度影响；分母把长度除掉，所以余弦主要比较方向。真实 Embedding 有成百上千维，本实验只保留二维来建立几何直觉。

::: warning 教学投影的边界
图中坐标由教程手工设计，不是某个真实模型导出的 Embedding。真实向量的单独一维通常没有稳定中文含义，降维图也可能扭曲距离。
:::

回到[第 02 课：向量与 Word Embeddings](/beginner/02-vector)，继续学习共现、Word2Vec、负采样与 GloVe。
