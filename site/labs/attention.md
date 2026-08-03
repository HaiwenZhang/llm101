---
title: Attention 温度实验
description: 直观看懂 softmax、温度与 causal mask
---

# Attention 温度实验

## 先做三个预测

1. 温度从 1 降到 0.2，最高分位置的权重会变大还是变小？
2. 把当前 Query 移到句子中间，右侧 token 会发生什么？
3. 温度能不能让被 causal mask 遮住的未来 token 重新获得权重？

<AttentionLab />

## 解释

Softmax 对分数的相对差异敏感。除以较小温度，相当于把分数差距放大；最大值会拿走更多概率。除以较大温度则压平差距。

Causal mask 的作用更早：未来位置的分数被改成负无穷，softmax 后严格为 0。调整温度只在允许读取的位置之间重新分配，无法越过 mask。

## 延伸思考

Attention 的“温度”与生成采样温度数学形式相似，但发生在不同地方：前者改变层内信息混合，后者改变最终词表 logits 的采样分布。真实模型通常不会让用户直接调整每层 attention 温度。
