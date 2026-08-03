---
title: Berkeley Deep RL 逐讲资料
description: 从模仿学习、策略梯度和 Actor-Critic，到 LLM RL、模型式 RL、离线 RL 与探索的系统强化学习课程 的逐讲 Slides、讲义与论文阅读官方索引
---

# Berkeley Deep RL 逐讲资料

> **课程**：UC Berkeley CS 185/285: Deep Reinforcement Learning (Spring 2026)  
> **学校**：University of California, Berkeley  
> **官方主页**：[https://rail.eecs.berkeley.edu/deeprlcourse/](https://rail.eecs.berkeley.edu/deeprlcourse/)  
> **抓取与校验日期**：2026-08-03

::: tip 这是来源与深挖页，不是主学习顺序
本页共索引 **43 份官方 Slides / PDF**，合计 **1,062 页 / 66.7 MB**。PDF 统一链接到课程官网、论文官网或 arXiv。
:::

初学请先沿[大模型系统课](/beginner/)学习；需要核对某一知识点来自哪些课程时，使用[名校课程知识覆盖表](/curriculum/sources)。

Berkeley 25 讲主线从模仿学习与 MDP 开始，逐步进入策略梯度、Actor-Critic、PPO、LLM RL、模型式与离线 RL。讨论课、作业和两个默认项目已挂到最相关的讲次下。
共索引 **43 份 PDF / 1,062 页**。课程主页列出的 `Course Project Assignment` 当前返回 404，来源页会保留原链接和失败状态，不用其他文件冒充。

::: tip 面向大语言模型的学习顺序
不必先学完整机器人控制课程。初学者可按本站 [25–33 强化学习专题](/beginner/40-rl-language-model)学习；需要推导或原始例子时再回到本页对应 Slides。
:::

## 建议怎么学

1. 先读每一讲的“本讲抓什么”，确认自己要回答的问题。
2. 第一遍快速翻 Slides，只看标题、图和结论；不在公式处停太久。
3. 第二遍结合 Notes 或 Recitation，把不懂的概念补齐。
4. 最后从论文阅读里挑 1–2 篇精读，并回到本站对应的零基础章节复习。

[查看课程资料机器可读清单（JSON）](/course-materials/manifest.json)

---

## L01 · 导论：强化学习解决什么问题

**日期**：Spring 2026  
**英文主题**：Introduction

**本讲抓什么**：先建立智能体、环境、观测、动作和奖励的总地图，知道 RL 与监督学习的边界。

### Slides 与讲义

- **Slides** · [Lecture 1: Introduction（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-1.pdf) · 38 页 · 3.18 MB

---

## L02 · 行为克隆（一）：从专家示范学习

**日期**：Spring 2026  
**英文主题**：Behavioral Cloning

**本讲抓什么**：把模仿学习看成状态到动作的监督学习，并识别分布偏移为什么会让小错误滚成大错误。

### Slides 与讲义

- **Slides** · [Lecture 2: Behavioral Cloning（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-2.pdf) · 39 页 · 2.42 MB
- **补充讲义** · [Homework 1: Imitation Learning（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw1.pdf) · 4 页 · 0.21 MB
- **Recitation Slides** · [Section 1: Imitation Learning（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-1.pdf) · 62 页 · 0.77 MB

---

## L03 · 行为克隆（二）：数据聚合与模仿学习

**日期**：Spring 2026  
**英文主题**：Behavioral Cloning Part 2

**本讲抓什么**：用 DAgger 等方法修复训练分布与执行分布不一致，为离线数据和在线交互建立直觉。

### Slides 与讲义

- **Slides** · [Lecture 3: Behavioral Cloning Part 2（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-3.pdf) · 32 页 · 2.8 MB

---

## L04 · 强化学习基础：MDP、回报与价值

**日期**：Spring 2026  
**英文主题**：RL Basics

**本讲抓什么**：理解 MDP、轨迹概率、折扣回报、价值函数与 Q 函数；这是后续 LLM RL 的共同语言。

### Slides 与讲义

- **Slides** · [Lecture 4: RL Basics（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf) · 41 页 · 1.69 MB
- **Recitation Slides** · [Section 2.1: RL Basics（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-2-1.pdf) · 4 页 · 0.16 MB
- **Recitation Slides** · [Section 2.2: PyTorch / RL Practice（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-2-2.pdf) · 5 页 · 0.18 MB

---

## L05 · 策略梯度：直接提高好动作的概率

**日期**：Spring 2026  
**英文主题**：Policy Gradients

**本讲抓什么**：推导 REINFORCE，理解 reward-to-go、baseline 与高方差问题。

### Slides 与讲义

- **Slides** · [Lecture 5: Policy Gradients（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) · 23 页 · 1.37 MB
- **补充讲义** · [Homework 2: Policy Gradients（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf) · 10 页 · 0.24 MB
- **Recitation Slides** · [Section 3: Policy Gradients and Actor Critic（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-3.pdf) · 8 页 · 0.25 MB

---

## L06 · Actor-Critic：用价值网络指导策略

**日期**：Spring 2026  
**英文主题**：Actor Critic

**本讲抓什么**：让 Actor 负责行动、Critic 负责估值，并用 TD 误差降低策略梯度方差。

### Slides 与讲义

- **Slides** · [Lecture 6: Actor Critic（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf) · 33 页 · 2.33 MB

---

## L07 · 价值型强化学习

**日期**：Spring 2026  
**英文主题**：Value-Based RL

**本讲抓什么**：理解贝尔曼方程、动态规划与 Q-learning，学会区分价值型和策略型方法。

### Slides 与讲义

- **Slides** · [Lecture 7: Value-Based RL（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-7.pdf) · 20 页 · 1.31 MB
- **补充讲义** · [Homework 3: Q-Learning and Actor Critic（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf) · 9 页 · 0.22 MB
- **Recitation Slides** · [Section 4: Value-Based RL（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-4.pdf) · 5 页 · 0.2 MB

---

## L08 · Q-learning 实践

**日期**：Spring 2026  
**英文主题**：Q-learning in Practice

**本讲抓什么**：理解经验回放、目标网络、过估计与离线数据分布偏移。

### Slides 与讲义

- **Slides** · [Lecture 8: Q-learning in Practice（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-8.pdf) · 36 页 · 2.44 MB

---

## L09 · 高级策略梯度（一）：重要性采样

**日期**：Spring 2026  
**英文主题**：Advanced Policy Gradients Part 1

**本讲抓什么**：用概率比率复用旧策略数据，连接 on-policy、off-policy 与 PPO。

### Slides 与讲义

- **Slides** · [Lecture 9: Advanced Policy Gradients Part 1（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-9.pdf) · 16 页 · 1.09 MB
- **Recitation Slides** · [Section 5: Advanced Policy Gradients（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-5.pdf) · 14 页 · 0.27 MB

---

## L10 · 高级策略梯度（二）：约束更新幅度

**日期**：Spring 2026  
**英文主题**：Advanced Policy Gradients Part 2

**本讲抓什么**：理解自然梯度、TRPO、PPO 与 KL 约束为什么能让训练更稳定。

### Slides 与讲义

- **Slides** · [Lecture 10: Advanced Policy Gradients Part 2（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf) · 33 页 · 2.14 MB

---

## L11 · 变分推断基础

**日期**：Spring 2026  
**英文主题**：Variational Inference

**本讲抓什么**：建立概率推断、ELBO 与潜变量直觉，为控制即推断和最大熵 RL 铺路。

### Slides 与讲义

- **Slides** · [Lecture 11: Variational Inference（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-11.pdf) · 18 页 · 1.24 MB
- **Recitation Slides** · [Section 6: Variational Inference（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-6.pdf) · 8 页 · 0.41 MB

---

## L12 · 强化学习中的变分推断

**日期**：Spring 2026  
**英文主题**：VI in RL

**本讲抓什么**：把策略优化写成概率推断问题，理解熵奖励和随机策略的意义。

### Slides 与讲义

- **Slides** · [Lecture 12: VI in RL（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-12.pdf) · 37 页 · 2.5 MB

---

## L13 · 控制即推断

**日期**：Spring 2026  
**英文主题**：Control as Inference

**本讲抓什么**：把高奖励轨迹视为更可能的轨迹，连接最大熵 RL、偏好建模和语言模型采样。

### Slides 与讲义

- **Slides** · [Lecture 13: Control as Inference（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-13.pdf) · 34 页 · 2.59 MB

---

## L14 · 大语言模型强化学习

**日期**：Spring 2026  
**英文主题**：LLM RL

**本讲抓什么**：把 prompt、token、回答和奖励正式放进 MDP，学习 LLM 策略梯度、PPO 与验证式奖励。

### Slides 与讲义

- **Slides** · [Lecture 14: LLM RL（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-14.pdf) · 51 页 · 3.81 MB
- **补充讲义** · [Homework 4: LLM RL（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf) · 15 页 · 0.26 MB
- **Recitation Slides** · [Section 7: IRL and LLM RL（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-7.pdf) · 4 页 · 0.18 MB
- **补充讲义** · [Default Final Project: LLM RL（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/misc/llm_rl_default_final_project.pdf) · 25 页 · 0.28 MB

---

## L15 · 模型式强化学习（一）

**日期**：Spring 2026  
**英文主题**：Model-Based RL Part 1

**本讲抓什么**：学习环境动力学模型、规划和模型误差，理解世界模型怎样帮助智能体。

### Slides 与讲义

- **Slides** · [Lecture 15: Model-Based RL Part 1（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-15.pdf) · 22 页 · 1.09 MB
- **Recitation Slides** · [Section 8: Model-Based RL（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-8.pdf) · 4 页 · 0.29 MB

---

## L16 · 模型式强化学习（二）

**日期**：Spring 2026  
**英文主题**：Model-Based RL Part 2

**本讲抓什么**：进一步理解模型预测控制、数据收集与规划—学习闭环。

### Slides 与讲义

- **Slides** · [Lecture 16: Model-Based RL Part 2（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-16.pdf) · 38 页 · 2.27 MB

---

## L17 · 离线强化学习（一）

**日期**：Spring 2026  
**英文主题**：Offline RL Part 1

**本讲抓什么**：只用固定数据集学习策略，理解分布外动作和保守估值。

### Slides 与讲义

- **Slides** · [Lecture 17: Offline RL Part 1（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-17.pdf) · 25 页 · 1.52 MB
- **补充讲义** · [Homework 5: Offline RL（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5.pdf) · 11 页 · 0.68 MB
- **Recitation Slides** · [Section 9: Offline RL（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-9.pdf) · 5 页 · 0.2 MB
- **补充讲义** · [Default Final Project: Offline-to-Online RL（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/misc/offline_to_online_rl_default_final_project.pdf) · 21 页 · 0.36 MB

---

## L18 · 离线强化学习（二）

**日期**：Spring 2026  
**英文主题**：Offline RL Part 2

**本讲抓什么**：比较行为约束、保守 Q 学习和离线到在线微调。

### Slides 与讲义

- **Slides** · [Lecture 18: Offline RL Part 2（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-18.pdf) · 35 页 · 2.3 MB

---

## L19 · 探索：怎样发现更好的行为

**日期**：Spring 2026  
**英文主题**：Exploration

**本讲抓什么**：理解随机探索、内在奖励与信息增益，以及语言智能体如何避免只重复已知解法。

### Slides 与讲义

- **Slides** · [Lecture 19: Exploration（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-19.pdf) · 23 页 · 1.14 MB

---

## L20 · 强化学习理论

**日期**：Spring 2026  
**英文主题**：RL Theory

**本讲抓什么**：用样本复杂度、性能差距与误差传播理解算法能保证什么、不能保证什么。

### Slides 与讲义

- **Slides** · [Lecture 20: RL Theory（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-20.pdf) · 28 页 · 1.74 MB

---

## L21 · 期中复习（一）

**日期**：Spring 2026  
**英文主题**：Midterm Review Part 1

**本讲抓什么**：把 MDP、价值函数、策略梯度和 Actor-Critic 串成一张知识图。

### Slides 与讲义

- **Slides** · [Lecture 21: Midterm Review Part 1（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-21.pdf) · 34 页 · 2.99 MB

---

## L22 · 期中复习（二）

**日期**：Spring 2026  
**英文主题**：Midterm Review Part 2

**本讲抓什么**：复盘变分推断、PPO、模型式与离线 RL 的联系。

### Slides 与讲义

- **Slides** · [Lecture 22: Midterm Review Part 2（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-22.pdf) · 73 页 · 5.46 MB

---

## L23 · 高级探索

**日期**：Spring 2026  
**英文主题**：Advanced Exploration

**本讲抓什么**：学习乐观估计、后验采样与基于模型的探索。

### Slides 与讲义

- **Slides** · [Lecture 23: Advanced Exploration（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-23.pdf) · 18 页 · 1.29 MB

---

## L24 · 多任务强化学习

**日期**：Spring 2026  
**英文主题**：Multi-task RL

**本讲抓什么**：理解条件策略、迁移、元学习和多任务数据怎样改善泛化。

### Slides 与讲义

- **Slides** · [Lecture 24: Multi-task RL（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-24.pdf) · 53 页 · 6.06 MB

---

## L25 · 挑战与开放问题

**日期**：Spring 2026  
**英文主题**：Challenges and Open Problems

**本讲抓什么**：识别奖励设计、稳定性、泛化、长期信用分配和真实世界安全等未解难题。

### Slides 与讲义

- **Slides** · [Lecture 25: Challenges and Open Problems（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-25.pdf) · 44 页 · 4.65 MB
- **补充讲义 · 官网链接失效** · [Course Project Assignment](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/project_assignment.pdf) — 课程主页保留链接，但当前无法下载；课程主页保留此链接，但抓取时返回 HTTP 404
- **补充讲义** · [Final Project Outline（官方 PDF）](https://rail.eecs.berkeley.edu/deeprlcourse/static/misc/final_project_outline.pdf) · 4 页 · 0.14 MB

---

## 版权与更新说明

本站不随 GitHub Pages 重新分发课程 PDF；条目优先链接课程官网、出版方或 arXiv。版权归原作者、课程团队及出版方所有；引用时请使用 PDF 内的正式作者与出版信息。机器清单保留官方 URL、页数、大小与 SHA-256，方便后续复核。
