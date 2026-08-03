---
title: 第 19 课 Prompt 进阶：示例、分解、验证与应用
description: 系统理解 Few-shot、CoT、自一致性、任务分解、数据合成与 Text-to-SQL
---

# 第 19 课　Prompt 进阶：示例、分解、验证与应用

<div class="lesson-lead">复杂 Prompt 的目标不是让模型“想得更神秘”，而是把任务拆成可观察、可验证、可重试的阶段。示例负责定义边界，分解负责降低单步难度，验证器负责发现错误。</div>

::: info 本课资料地图：不要把方法名当魔法词
- Few-shot、CoT 与 Prompt 适用边界：[CS224N · Efficient Adaptation Slides](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture09-peft.pdf)；
- ICL 的 task retrieval、标签偏置、顺序敏感性与 prompt chains：[CMU ANLP L07 第 28–50 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-07-pretraining2-prompting.pdf#page=28)；
- 工程化 Prompt 设计与敏感性：[CMU LLM Applications · The Science of Prompting](https://storage.googleapis.com/cmu-llms/2026/2026-01-20-prompting.pdf)；
- 原论文：[Chain-of-Thought Prompting](https://arxiv.org/pdf/2201.11903.pdf)、[ReAct](https://arxiv.org/pdf/2210.03629)。

论文图展示的是特定模型、任务和实验设置下的结果，不等于任何模型只加一句“逐步思考”都会变正确。正文会把“生成中间步骤”与“验证中间步骤”分开。
:::

<figure class="teaching-figure">
  <img src="/illustrations/guide-agent-loop.webp" alt="目标经过规划、工具、观察、验证和记忆形成循环">
  <figcaption>当任务超出一次回答，Prompt 会变成系统流程的一部分：计划不是结果，工具输出不是事实保证，验证通过才结束。</figcaption>
</figure>

## 1. Few-shot 示例到底在教什么

示例同时传递四类信息：

1. 输入分布长什么样；
2. 标签或答案的含义；
3. 推理/转换步骤的形式；
4. 输出格式和语气。

如果类别名是 `A/B/C`，自然语言定义不够，少量边界示例尤其重要。

### 1.1 示例选择

- 随机示例：简单但不针对当前问题；
- 语义近邻：从示例库检索与当前输入相似的案例；
- 覆盖式选择：刻意覆盖不同类别、长度和边界；
- 难例优先：选择模型最容易混淆的反例。

近邻不一定最好：过于相似可能全来自同一类别，导致标签偏斜。可以先保证类别覆盖，再按相似度选。

### 1.2 示例顺序

模型可能更受最后几个示例影响。测试时应交换顺序，看结果是否稳定；若顺序一变答案就变，说明规则边界还没有写清。

### 1.3 标签覆盖、标签平衡与 pretraining bias

这是三个不同变量：

- **coverage**：允许的每个标签是否至少出现过；
- **balance**：各标签出现次数是否接近；
- **pretraining bias**：模型在示例之前就偏好某些标签词或任务格式。

例如类别语义被故意翻转：`positive → C, neutral → A, negative → B`。模型既要从示例检索“这是情感分类任务”，又要压过预训练里 `positive` 与“正面”天然相连的偏好，学会临时映射。CMU L07 第 32–35 页把这种现象概括为 task retrieval 与 unlearning pretraining bias。

只给输入、不给输出的 unsupervised ICL 有时也能帮助模型识别输入领域；使用模型自己产生的输出作为示例则属于 reinforced/self-generated ICL 一类做法。它们可能提供任务线索，也可能复制模型原有错误，不能和人工正确示范等价。

### 1.4 分类前先检查标签词本身

标签写成 `A/B/C`、`正面/负面` 或完整句子，会改变 token 数与先验概率。对第 $c$ 个标签 verbalizer $v_c$，可比较：

$$
s_c=\sum_{t=1}^{|v_c|}\log p_\theta(v_{c,t}\mid P,x,v_{c,<t})
$$

多 token 标签天然累计更多负 log probability，必要时报告长度归一化结果；还应在“内容为空或无信息”的对照输入上测标签先验，避免把模型偏爱某个词误当任务能力。

<ICLSensitivityLab />

## 2. CoT 适合什么问题

链式分解通常更适合：多步算术、符号操作、多跳知识、约束规划。对简单事实抽取，它只会增加延迟和犯错步骤。

工程上优先要求**可验证的中间产物**：

```text
输出：
1. 已知事实（每条带来源编号）
2. 计算式或子问题
3. 检查结果
4. 最终答案
```

不要把冗长自然语言解释当作推理正确的证据。最终应由计算器、代码执行、检索引用或业务规则检查。

<figure class="teaching-figure source-figure"><a href="/paper-figures/cot-figure-1.webp" target="_blank"><img src="/paper-figures/cot-figure-1.webp" alt="Chain-of-Thought 论文 Figure 1 对比标准提示和带推理示例的提示"></a><figcaption>《Chain-of-Thought Prompting》Figure 1（PDF p.1）。左侧示例只给答案，模型在新题上算错；右侧示例同时给中间计算，新题也生成相似的分解并答对。它证明“示例中提供过程”可能改变输出路径，但单个成功案例不能替代整套评测。<a href="https://arxiv.org/pdf/2201.11903.pdf#page=1">打开原论文第 1 页</a>。</figcaption></figure>

## 3. “按部就班”怎样真正落地

<figure class="teaching-figure concept-figure"><img src="/illustrations/prompt-plan-verify-sparse.webp" alt="复杂目标拆成子任务，调用工具取得观察，并由验证器检查与重试"><figcaption>复杂 Prompt 的价值是把失败定位到具体子任务；验证器不通过时，只重试失败步骤，而不是整段重新猜。</figcaption></figure>

<figure class="teaching-figure concept-figure"><img src="/illustrations/prompt-chain-validation.svg" alt="输入规范化、计划、检索工具、生成草稿与验证门组成的 Prompt Chain，失败只回退到对应步骤"><figcaption>一个可靠 chain 不只是连续调用模型，而是每步有 schema、验证门和有限重试。没有新增证据时反复自我反思，常只是重复同一种错误。</figcaption></figure>

把“为我制定一次日本旅行”拆成：

```text
阶段 A：澄清城市、日期、预算、签证与体力限制
阶段 B：检索交通、开放时间和预约要求
阶段 C：生成日程草案
阶段 D：检查时间冲突、闭馆、路程和预算
阶段 E：输出带来源与不确定性的最终计划
```

每阶段有不同输入、工具和验收条件。一个长 Prompt 一次完成所有阶段，错误难定位。

### 3.1 一条 chain 的接口账

对每个阶段 $j$，至少定义五项：

```text
input_schema_j
allowed_evidence_j
output_schema_j
validator_j
retry_budget_j
```

验证失败时返回结构化错误，例如 `missing_citation: claim_3`，而不是只说“再检查一下”。重试要有新信息、缩小后的上下文或确定性修复规则；否则循环不会凭空增加知识。总成本近似为所有调用的输入/输出 token、工具延迟和重试次数之和，拆得更细并不自动更便宜。

## 4. “三思后行”：先计划，再执行

Plan-and-Solve、Least-to-Most 等方法的共同直觉是：先建立子问题或计划，再逐项解决。计划需要检查：

- 子问题是否覆盖目标；
- 是否存在循环依赖；
- 哪些步骤需要外部工具；
- 哪些信息尚缺；
- 失败时从哪一步重试。

计划不是承诺，观察到新信息后应允许修订。

## 5. “集思广益”：多个候选怎样合并

### 5.1 Self-Consistency

对同一问题采样多条独立解法，再对最终答案投票。它适合有唯一答案且多条合理路径的问题；成本随采样数增加，大家一起犯同一种错时也无效。

若第 $m$ 条采样路径得到最终答案 $a_m$，多数投票写成：

$$
\hat a=\arg\max_a\sum_{m=1}^{M}\mathbf 1[a_m=a]
$$

- $M$：采样路径数，越大通常越贵；
- $\mathbf 1[\cdot]$：条件成立记 1，否则记 0；
- $\hat a$：票数最多的答案，而不是“最长的推理”；
- 路径必须有差异，若温度太低或提示完全相同，可能只重复同一个错误。

```python
from collections import Counter

def self_consistency(sample_answer, runs=7):
    answers = [sample_answer() for _ in range(runs)]
    winner, votes = Counter(answers).most_common(1)[0]
    confidence = votes / runs
    return winner, confidence, answers
```

`sample_answer()` 应返回已经规范化的最终答案，例如整数或选项字母。开放式长文本不能直接按字符串投票，应先做等价答案归一化或用可审计的判定器聚类。

多数投票能否提高正确率还取决于错误相关性。若 10 条路径只是同一错误的措辞变体，有效候选数远小于 10。实验至少报告：`pass@M`、多数票正确率、候选去重率、答案熵、选择器/验证器准确率与总 token；不能只展示票数最多的一次成功。

### 5.2 Tree of Thoughts

把中间状态看作树节点：生成多个候选步骤，评分、剪枝、继续搜索。它适合组合规划，但分支数会快速爆炸，评分器质量成为瓶颈。

### 5.3 多角色讨论

让“支持方/反对方/审计方”分别输出观点，能增加覆盖面，但角色名称本身不产生独立知识。若都使用同一模型和同一资料，错误高度相关。更重要的是使用不同证据、工具或检查规则。

## 6. 归纳式追问：把模糊需求逐渐变清

用户说“帮我分析销量下降”，系统可以按信息价值追问：

1. 下降的时间、区域和产品是什么；
2. 对比基线是同比、环比还是目标；
3. 是否有价格、渠道、库存、活动数据；
4. 最终希望解释原因还是给行动建议。

不要一次问 20 个问题。先问最能改变分析路径的 2–4 项。

## 7. Prompt 在 Agent 中的位置

Agent 至少包含：目标、状态、工具、动作、观察、记忆、验证、预算。Prompt 可以描述策略，但权限必须由系统控制：

```text
模型输出“删除文件” ≠ 文件已经删除
工具层检查权限与参数 → 执行 → 返回真实观察
```

把工具返回内容视为不可信数据，防止网页或文档中的提示注入获得系统指令优先级。

## 8. 离散 Prompt、Soft Prompt 与 Prefix Tuning 的边界

日常 Prompt 是可读 token，通常没有训练。CS224N L09 第 55–58 页还介绍了可训练的连续提示：

| 方法 | 训练什么 | 放在哪里 | 人能否直接读 |
|---|---|---|---|
| Discrete prompt / ICL | 不训练，人工或检索得到 token | 输入 token 序列 | 能 |
| Prompt tuning | 少量 virtual-token embeddings | 输入 embedding 前缀 | 不能直接解释为词 |
| Prefix tuning | 各层可用的连续 prefix 状态/参数 | 常作用于层内 Attention | 不能 |

Soft prompt 虽然名字里有 Prompt，但它有数据、loss、backward 和 optimizer step，因此属于参数高效适配。它冻结主模型，不等于完全没有训练；任务切换时也要加载对应的 virtual token 参数。课件还指出 prompt tuning 的效果与模型规模有关，不能假设小模型上也自动追平 full fine-tuning。

## 9. 数据合成：模型生成训练样本

典型流程：

```mermaid
flowchart LR
  A["定义任务与标签"] --> B["生成多样样本"]
  B --> C["规则过滤"]
  C --> D["教师或人工审查"]
  D --> E["去重与分布检查"]
  E --> F["小规模训练"]
  F --> G["真实数据评测"]
```

风险包括：教师模型偏差被复制、样本模板化、训练测试泄漏、稀有类别缺失。合成数据只能补充，不能用合成数据自己证明有效。

## 10. Text-to-SQL：为什么必须给 Schema

只说“查上月华东销量”不够。Prompt 应提供：

- 允许访问的表和字段；
- 主外键关系；
- 日期和金额口径；
- 方言（PostgreSQL/MySQL 等）；
- 只读限制；
- 代表性查询示例。

生成后要做语法解析、表列白名单、只读验证、查询成本限制，并在隔离环境执行。SQL 看起来正确不等于业务口径正确。

一个最小验证顺序是：

```text
解析 SQL AST
→ 禁止 INSERT/UPDATE/DELETE/DDL 与多语句
→ 表/列白名单
→ 注入租户与行级权限
→ EXPLAIN 检查成本
→ 只读事务 + timeout + row limit
→ 对结果做业务口径断言
```

这些约束必须由数据库代理/程序实现，不能只写在 Prompt 里期待模型自觉遵守。

## 11. 一个 Prompt 实验表

| 版本 | 唯一改动 | 任务正确率 | 格式通过率 | 平均 token | 主要失败 |
|---|---|---:|---:|---:|---|
| V0 | 基线 |  |  |  |  |
| V1 | 加类别边界例子 |  |  |  |  |
| V2 | 加输出 schema |  |  |  |  |
| V3 | 加证据检查 |  |  |  |  |

一次只改一个因素，否则无法知道提升来自哪里。

### 更完整的实验矩阵

对 Few-shot 方法建议至少做：

| 维度 | 取值示例 | 目的 |
|---|---|---|
| K | 0 / 1 / 4 / 8 | 区分示例收益与上下文成本 |
| selection | random / nearest / coverage | 区分相似度与标签覆盖 |
| order | 多个固定排列 | 测顺序敏感性 |
| label words | A/B vs 语义标签 | 测 verbalizer prior |
| decoding | greedy / 固定采样参数 | 区分 Prompt 与采样方差 |
| model scale/checkpoint | 至少两个 | 检查能力是否依赖规模 |

每个格子在相同测试集运行，报告均值、方差、格式失败、token、TTFT 与错误类型。若先看测试结果再挑最佳 Prompt，测试集已经参与开发，应另留最终 holdout。

<ConceptCheck question="多角色讨论为什么不自动等于独立证据？" :options="['因为角色名字不会改变模型所掌握的资料和相关错误','因为模型不能生成多个回答','因为任何讨论都必须用最高温度']" :answer="0" explanation="真正提高可靠性通常要引入不同来源、工具、验证器或独立模型，而非只换角色称呼。" />

<ConceptCheck question="Prompt tuning 为什么不属于普通的‘在输入框里改措辞’？" :options='["它通过 loss 和梯度学习连续 virtual-token 参数", "它会永久修改 tokenizer 词表", "它完全不需要训练数据"]' :answer="0" explanation="Prompt tuning 冻结主模型，但仍优化一小组连续参数；它是 PEFT，不是纯 ICL。" />

> 本课对应原书第 3.2–3.5 节（PDF 第 114–157 页），详细展开示例选择、CoT、计划、多候选、Agent、数据合成与 Text-to-SQL。

## 12. 课件与论文精读路线

1. [CMU L07 第 28–35 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-07-pretraining2-prompting.pdf#page=28)：few-shot、input-only ICL、标签映射、顺序与覆盖；
2. [第 37–48 页](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-07-pretraining2-prompting.pdf#page=37)：chat template、CoT、问题分解与 prompt chain；
3. [CS224N L09 第 18–34 页](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture09-peft.pdf#page=18)：zero/one/few-shot、规模效应、CoT 和敏感性；
4. [第 55–66 页](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture09-peft.pdf#page=55)：从 input perspective 看 prefix/prompt tuning，再与 adapter/LoRA 的 function/parameter perspective 对照；
5. [Chain-of-Thought 原论文 Figure 1](https://arxiv.org/pdf/2201.11903.pdf#page=1)：把“生成了步骤”与“步骤经外部验证”分开评价。

<ChapterReadings lesson="18-prompt-advanced" />
