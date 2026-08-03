---
title: 第 17 课 从预训练到 Agent
description: 分清 Pre-training、SFT、偏好学习、RL、蒸馏和 Agent Harness
---

# 第 17 课：模型能力不是一次训练完成的

<div class="lesson-lead">Pre-training 学数据规律，SFT 把可用行为放进高概率区域，RL 优化采样结果，蒸馏整合能力；Agent 则是模型与工具、环境、记忆和验证器组成的系统。</div>

::: info 本课资料地图
- 课件：[CS224N Post-training](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture08-posttraining.pdf)、[台大 ADL Post-training](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/250922_PostTraining.pdf)与 [Stanford CS336 Lecture 15](https://stanford-cs336.github.io/spring2026/)负责流程，[Lecture 16](https://stanford-cs336.github.io/spring2026/)负责可验证奖励强化学习；
- 人类偏好：[InstructGPT](/papers/instructgpt)把 SFT、奖励模型、PPO 串成经典 RLHF，[DPO](/papers/dpo)把成对偏好改写成直接分类式目标；
- 推理能力：[DeepSeek-R1](/papers/deepseek_r1)展示可验证奖励与推理 RL，[Kimi k1.5](/papers/kimi_k1_5)研究长思维链与搜索；
- 系统能力：[MOPD](/papers/mopd)把多个教师整合进学生，[ReAct](/papers/react)把推理与环境行动交替组织。
:::

<figure class="teaching-figure">
  <img src="/illustrations/beginner-learning-journey.webp" alt="从大规模阅读、示范课堂、反馈训练到工具实践的四阶段学习校园">
  <figcaption>把模型生命周期想成一所学校：先广泛阅读，再看示范，随后按反馈练习，最后进入真实工具环境完成任务。</figcaption>
</figure>
<div class="visual-key"><div><b>预训练</b>从海量数据学习一般规律。</div><div><b>SFT 与 RL</b>示范可用行为，再按结果反馈优化。</div><div><b>Agent 系统</b>模型进入有工具、环境和验证器的闭环。</div></div>

## 1. 先看生命周期

<figure class="teaching-figure concept-figure"><img src="/illustrations/posttraining-objectives-sparse.webp" alt="预训练、SFT、RL 与 Agent 环境的四种不同学习目标"><figcaption>四站使用的监督信号不同：下一个 token、理想示范、结果奖励、真实工具与验证。它们不是同一训练重复四遍。</figcaption></figure>

<LearningStagesDiagram />

```text
海量文本/图像/代码
      ↓ next-token pre-training
Base model：会续写、有知识，但不一定听指令
      ↓ SFT
Assistant：学会回答格式、工具示范与基本行为
      ↓ preference / RL
Policy：更偏向满足偏好或完成可验证目标
      ↓ distillation / quantization / serving adaptation
可部署模型
```

这些阶段不是“同一训练多跑几遍”，数据来源、目标函数和失败模式不同。

## 2. Pre-training：压缩条件分布

直接目标：

$$
\mathcal L_{NTP}=-\sum_t\log p(x_t\mid x_{<t})
$$

它擅长从大规模数据获得广泛表示与生成能力，但不直接保证：

- 遵循用户意图；
- 多步推理正确；
- 不编造；
- 工具调用改变正确世界状态。

## 3. SFT：看老师示范

SFT 数据是一组“输入 → 理想回答/轨迹”。模型仍做 token-level cross entropy，只是数据更像我们希望的助手行为。

SFT 适合教：

- 回答结构；
- 安全边界；
- 工具调用格式；
- 基本 ReAct 轨迹；
- 特定风格。

但纯模仿只覆盖示范访问过的状态。模型一旦自己走错一步，后续状态可能脱离训练分布。

## 4. 偏好学习：A 比 B 好

同一 prompt 产生两个回答，人类或规则判断 A 胜过 B。奖励模型或 DPO 从成对比较学习。

偏好非常适合开放式质量，如写作、帮助性、简洁度；但它受标注指南、评审模型与文化偏好影响。

<figure class="teaching-figure source-figure"><a href="/paper-figures/dpo-figure-1.webp" target="_blank"><img src="/paper-figures/dpo-figure-1.webp" alt="DPO 论文 Figure 1，对比传统 RLHF 的奖励模型加在线强化学习流程与 DPO 的直接偏好优化流程"></a><figcaption>DPO 论文 Figure 1（PDF p.2）。左边 RLHF 先拟合奖励模型，再反复采样和强化学习；右边 DPO 直接用偏好对训练最终语言模型。图展示流程差异，下面的公式则解释 DPO 如何仍然约束策略相对参考模型的偏移。<a href="https://arxiv.org/pdf/2305.18290.pdf#page=2">打开原论文第 2 页</a>。</figcaption></figure>

### DPO 公式怎样读

对同一问题，记偏好回答为 $y_w$，较差回答为 $y_l$。DPO 比较“当前策略相对参考模型，把哪个回答提高得更多”：

$$
\mathcal L_{\mathrm{DPO}}
=-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]\right)
$$

- $\pi_\theta$：正在训练的模型；
- $\pi_{\mathrm{ref}}$：冻结的参考模型，防止策略无约束漂移；
- $\beta$：控制偏好推动强度；
- `winner − loser` 越大，`-log sigmoid` 越小。

实际实现先把回答中各 token 的 log probability 相加，再代入：

```python
import torch
import torch.nn.functional as F

def dpo_loss(logp_w, logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    preferred_gain = logp_w - ref_logp_w
    rejected_gain = logp_l - ref_logp_l
    margin = beta * (preferred_gain - rejected_gain)
    return -F.logsigmoid(margin).mean()
```

DPO 不需要在线 rollout 和显式奖励模型，但仍需要高质量偏好对、参考模型概率与稳定训练。它也不等于“没有 RL 思想”：公式本身来自带 KL 约束的偏好优化推导。

## 5. RL：让模型自己走，再按结果更新

基本循环：

```text
模型采样完整轨迹
  ↓
环境 / verifier 给奖励
  ↓
估计哪些 token/动作提高了结果
  ↓
更新策略
```

可验证奖励例子：

- 数学最终答案；
- 代码单元测试；
- 网页最终状态；
- 游戏得分；
- 搜索任务引用证据是否支持答案。

长轨迹难点在于：一次错误可能很早发生，最终才知道失败；rollout 还占大量算力与 KV Cache。

## 6. Reasoning effort 是预算，不是另一套知识

K3 按 low/high/max 控制思考 token、工具调用与执行预算。更高 effort 通常允许：

- 更多搜索；
- 更多候选与验证；
- 更长规划与纠错；
- 更高推理成本。

比较模型时如果 effort、工具和最大步数不同，分数就不是严格同条件。

## 7. 蒸馏：把教师能力合进学生

普通离线蒸馏让教师先生成，学生模仿教师轨迹。On-policy distillation 反过来：

1. 学生自己生成；
2. 教师在学生真实访问的状态上给 token 分布或奖励；
3. 学生学习怎样从自己的状态回到更好的行为。

K3 用多个领域 × 多 effort 教师，再整合成一个模型，避免上线九套完整模型。

## 8. Agent 不只是“多想几步”

最小闭环：

```text
目标 → 模型思考 → 调用工具 → 环境返回观察
                    ↑                 ↓
                    └──── 继续决策 ───┘
```

完整 Agent 系统包含：

| 组件 | 负责什么 |
|---|---|
| Model | 产生文本、决策与工具调用 |
| Harness | 拼 prompt、维护循环、处理 tool schema |
| Tools | 搜索、代码、浏览器、数据库等动作 |
| Environment | 动作真正改变的世界状态 |
| Memory | 保留跨步或跨会话信息 |
| Verifier | 独立检查最终是否完成目标 |
| Budget | 限制 token、工具次数、时间和成本 |

模型分数不能与 harness 能力完全分开。

## 9. Reward hacking

如果奖励只检查“页面里出现成功两个字”，Agent 可能直接写入成功，而不真正完成任务。解决方法包括：

- verifier 隔离；
- hidden tests；
- 检查最终环境状态；
- 限制提交次数；
- 对异常捷径加惩罚。

## 本课闭卷复述

为“训练一个会修复 GitHub bug 的 Agent”分别设计 SFT 数据、RL 环境、奖励和 verifier。

<ConceptCheck question="下面哪项最准确地描述 Agent？" :options='["会输出很长思维链的模型", "模型与工具、环境、循环、记忆、预算和验证器组成的系统", "只做搜索的脚本"]' :answer="1" explanation="模型是核心决策器，但 Agent 行为取决于完整 harness 与环境。" />

下一课：[把所有概念装回 Kimi K3](/beginner/09-k3-map)。

<ChapterReadings lesson="08-post-training" />
