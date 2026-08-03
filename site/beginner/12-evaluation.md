---
title: 第 46 课 大模型到底怎样评测
description: 从困惑度、BLEU、ROUGE、BERTScore 到 LLM Judge 与系统评测
---

# 第 46 课　大模型到底怎样评测

<div class="lesson-lead">没有一个“总分”能代表模型好坏。困惑度看语言预测，BLEU/ROUGE 看参考重合，语义指标看意思接近，LLM Judge 看复杂标准；真实系统还要看事实、成本、延迟和任务成功。</div>

::: info 本课资料地图
- 课件：[CS224N Benchmarking and Evaluation](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture11-evaluation.pdf)、[台大 ADL NLG Evaluation](https://www.csie.ntu.edu.tw/~miulab/f114-adl/doc/251027_NLGEval.pdf)、[CMU Evaluation Techniques](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-13-evaluation.pdf)和 [Stanford CS336 Lecture 12 可执行 Slides](/lectures/?trace=var/traces/lecture_12.json)；
- 基准论文：[MMLU](https://arxiv.org/pdf/2009.03300.pdf)展示多任务知识测试，[HELM](https://arxiv.org/pdf/2211.09110.pdf)强调场景、指标和标准化必须一起报告；
- 统计论文：[Adding Error Bars to Evals](https://arxiv.org/pdf/2411.00640.pdf)解释为什么排行榜上几个百分点的差距可能不足以下结论。
:::

<figure class="teaching-figure">
  <img src="/illustrations/guide-evaluation.webp" alt="评测天平同时比较质量、成本、污染、基线与不确定性">
  <figcaption>评测像验收一座桥：不仅看承重分，还要核对测试条件、成本、污染和测量误差。</figcaption>
</figure>

<figure class="teaching-figure source-figure"><a href="/lectures/images/helm-capabilities-leaderboard.png" target="_blank"><img src="/lectures/images/helm-capabilities-leaderboard.png" alt="Stanford CS336 Slides 引用 HELM 风格的多能力评测榜单图"></a><figcaption>来源图：Stanford CS336 Lecture 12《Evaluation》中的 HELM 能力榜单。它提醒我们同一个模型会在不同能力列上表现不同，不能把单列分数当成“总智能”。<a href="/lectures/?trace=var/traces/lecture_12.json">打开本地可执行 Slides</a>，或阅读 <a href="https://arxiv.org/pdf/2211.09110.pdf">HELM 原论文</a>。</figcaption></figure>
<div class="visual-key"><div><b>先定任务</b>谁使用、在哪种输入上、什么算成功。</div><div><b>再选指标</b>指标必须与任务失败代价对齐。</div><div><b>最后审口径</b>数据、提示、解码、成本和置信区间一起报告。</div></div>

<figure class="teaching-figure concept-figure"><img src="/illustrations/evaluation-from-task-to-decision.svg" alt="从用户决策、任务单位、数据、推理配置、原始输出、评分到带置信区间上线决策的七步评测链"><figcaption>不要从“排行榜常用什么指标”开始。先写用户、任务单位和失败代价，再冻结数据与推理配置，最后才汇总分数和误差条。</figcaption></figure>

## 0. 评测对象到底是一题、一轮还是整个任务

同一个 Agent 可以单步答案正确，却在 20 步任务中因一次工具参数错误而失败。先定义 unit of analysis：

| 单位 | 例子 | 不能回答什么 |
|---|---|---|
| token | NLL / PPL | 完整回答是否有用 |
| response | 分类、问答、引用 | 多轮任务是否最终完成 |
| trajectory | Agent 工具调用链 | 用户是否长期满意 |
| user/session | 完成率、放弃率 | 具体哪一步错了 |

还要区分 **metric** 与 **decision**：模型输出一个风险分数，阈值把它变成“拦截/放行”；阈值变化时模型参数不变，Precision、Recall 和业务成本仍会变化。

## 1. 内在评测：不做具体任务也能测吗

困惑度（Perplexity, PPL）衡量模型对测试文本的意外程度。平均负对数似然为 $L$ 时：

$$\mathrm{PPL}=e^L$$

如果 PPL=10，可粗略理解为模型每一步像在 10 个等可能候选间犹豫；这只是直觉，不代表真实候选恰好 10 个。

### 1.1 一个两词手算

正确 token 的概率分别为 0.5、0.25：

$$\mathrm{PPL}=(0.5\times0.25)^{-1/2}=\sqrt{8}\approx2.83$$

若模型给正确 token 更高概率，PPL 会降低。

### PyTorch：从 logits 计算 PPL

```python
import torch
import torch.nn.functional as F

# 例：2 个位置、4 个候选 token
logits = torch.tensor([
    [2.0, 0.5, -1.0, 0.0],
    [0.2, 1.8, 0.3, -0.5],
])
targets = torch.tensor([0, 1])

mean_nll = F.cross_entropy(logits, targets)
perplexity = torch.exp(mean_nll)
print(mean_nll.item(), perplexity.item())
```

`cross_entropy` 内部完成 `log_softmax + 取目标 token 的负对数概率 + 求平均`。评测长文本时还要正确处理 padding、滑动窗口和不同样本长度，不能把 pad token 也计入平均。

### 1.2 PPL 的三条边界

1. 测试文本应与目标分布相关，新闻 PPL 不能代表代码能力；
2. 不同 tokenizer 会改变 token 数与概率分解，通常不可直接横比；
3. PPL 低表示更会预测该分布，不自动保证事实、安全、指令遵循或推理正确。

另外必须写清平均口径。先对全语料 token 总 NLL 求平均，是 micro/token-weighted 口径；先算每篇文档平均再平均，会让短文档权重变大。滑动窗口还要确保重叠上下文 token 只作为条件，不重复计入 target loss。

## 2. BLEU：生成内容有多少是“准的”

BLEU 最初用于机器翻译，核心是多阶 n-gram precision，并加入长度惩罚。例子：

```text
参考：large language models
生成：big language models
```

unigram 中有 `language`、`models` 两个匹配；bigram 有 `language models` 一个匹配。BLEU 同时看不同长度片段。

### 2.1 为什么要裁剪计数

若参考只有一个“好”，生成“好 好 好 好”，不能算 100% 精确。Modified Precision 会按参考中最大出现次数裁剪匹配计数。

### 2.2 局限

“这家店不贵”和“这家店价格合理”语义接近但词面重合低；反过来，“允许退款”和“不允许退款”词面高度重合却意思相反。BLEU 适合语料级翻译趋势，不适合单独评判开放回答。

BLEU 的几何平均使任一高阶 n-gram precision 为 0 时总分可能归零，因此短句常需 smoothing。Corpus BLEU 与逐句 BLEU 再平均不是同一个量；必须报告 tokenizer、大小写、参考数量与实现版本。

## 3. ROUGE：参考内容覆盖了多少

ROUGE 更偏 Recall，常用于摘要：

- ROUGE-N：参考 n-gram 被覆盖多少；
- ROUGE-L：依据最长公共子序列衡量顺序一致的覆盖；
- 其他变体考虑跳跃二元组或加权连续片段。

如果摘要把参考事实都抄进来却加入大量错误，Recall 仍可能很高。所以应同时看 precision、事实一致性和长度。

ROUGE 也受分词、词干化和参考摘要数量影响。生成更长通常更容易覆盖参考内容，所以必须同时报告输出长度；否则模型可能靠“多写”提高 recall。

## 4. BERTScore：允许换一种说法

BERTScore 分别为候选和参考的 token 生成上下文向量，再做最大相似度匹配，计算语义 precision、recall、F1。它比词面指标更能识别同义改写，但仍受编码模型、语言、领域和参考答案质量影响。

## 5. LLM-as-a-Judge：让模型按量表评分

G-EVAL 一类方法把任务描述、评分标准、评测步骤、输入和候选输出交给生成模型，让它给分。优势是能评估连贯、帮助性、风格等复杂维度，也可在缺少唯一参考答案时使用。

### 5.1 评委也会偏

- 位置偏差：更偏爱排在前面的答案；
- 长度偏差：把更长误当更完整；
- 自我偏好：偏爱相似模型的写法；
- 提示敏感：量表措辞改变分数；
- 事实盲点：评委自己也不知道正确答案；
- 非确定性：重复评审结果可能波动。

缓解方式：交换答案顺序、隐藏模型身份、细化评分 rubric、要求引用证据、使用多评委或人工校准，并报告一致性。

### 5.2 Pairwise Judge 与 pointwise 分数

Pointwise 让 Judge 独立给 1–5 分，容易出现不同评委量表漂移；pairwise 让它比较 A/B，通常更容易，但会有位置偏差和大量平局。至少做 `A,B` 与 `B,A` 双顺序；若翻转则标为不稳定，不应强行判胜。

Judge 上线前要和人工 gold set 对齐，报告准确率/相关性与分切片偏差。Judge 的自然语言理由可供审计，但理由流畅不等于标签正确；最终计算应基于结构化分数。

## 6. 准确率也不是总够用

二分类混淆矩阵：

|  | 实际正 | 实际负 |
|---|---:|---:|
| 预测正 | TP | FP |
| 预测负 | FN | TN |

$$\mathrm{Precision}=\frac{TP}{TP+FP},\quad \mathrm{Recall}=\frac{TP}{TP+FN}$$

医疗筛查更怕漏诊，重 Recall；垃圾邮件过滤若误删重要邮件代价高，重 Precision。指标选择必须反映错误代价。

$$
F_1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
$$

F1 不包含 TN，类别极不平衡时比 accuracy 更关注正类，但它默认 Precision 与 Recall 同等重要，也不等于业务成本最小。若漏报代价更高，可用 $F_\beta$ 或直接定义 cost matrix。

<EvaluationThresholdLab />

### 6.1 Macro、micro、weighted 为什么会给不同结论

多分类中：

- macro：每个类别先算指标再等权平均，稀有类权重高；
- micro：汇总所有 TP/FP/FN 再计算，大类主导；
- weighted macro：按类别样本数加权，介于二者。

假设 90 个常见类样本全对、10 个稀有类全错，accuracy/micro 可达 90%，但 macro recall 只有 `(100%+0%)/2=50%`。报告总体分数时必须同时给每类 support 与关键切片。

### 6.2 阈值在 validation 选，test 只报告一次

反复看 test set 挑阈值、Prompt 或 checkpoint，就是把测试集用于开发。正确流程是 train 学参数，validation 选超参数/阈值，test 在方案冻结后做最终估计；若继续据 test 改系统，就要准备新的 holdout。

## 7. 生成系统应分层评测

<figure class="teaching-figure concept-figure"><img src="/illustrations/evaluation-lenses-sparse.webp" alt="同一模型答案分别接受词面、语义、事实和成本延迟四种检查"><figcaption>每把尺只看一个侧面。开放回答若只报一个分数，往往会把事实、成本或用户任务失败藏起来。</figcaption></figure>

### 模型层

语言预测、知识、推理、代码、长上下文、安全。

### 任务层

给定真实输入，是否完成用户目标；输出格式是否可解析；事实是否有证据。

### 系统层

检索是否找到文档、工具调用是否成功、Agent 是否在预算内完成、失败能否恢复。

### 运营层

首 token 延迟、总延迟、吞吐、token 成本、GPU 利用率、用户放弃率。

一个模型 benchmark 提升 3 分，但延迟翻倍、工具成功率下降，产品未必更好。

系统成功率常是多环节乘积的结果。若检索成功率 0.9、工具执行 0.95、生成在正确证据下成功 0.9，简化独立近似下端到端仅约 `0.9×0.95×0.9=0.77`。真实环节相关，但这个账提醒我们：只把生成模型提高 1 分，未必是最大瓶颈。

## 8. 污染与“刷题”

如果测试题出现在训练数据中，分数可能测到记忆而非泛化。检查方法包括：搜索重合片段、使用训练截止时间后的题、构造私有变体、比较改写敏感度。动态 benchmark 也会随着公开而逐渐被污染。

污染检查还要区分精确重合、近似改写、同模板不同数字与答案泄漏。仅搜索完整句子会漏掉 paraphrase；仅看训练截止日期也不能排除网页早期版本。私有测试集应限制访问、记录查询次数，并保留最后一次未参与开发的 audit set。

## 9. 置信区间：别把小差距当胜负

100 道题中 A 对 72、B 对 74，不足以轻易断言 B 更强。应使用 bootstrap 或合适统计检验估计不确定性；报告样本量、题目来源和多次运行波动。Agent 与随机采样任务尤其需要多次重复。

### 9.1 为什么优先做 paired comparison

A/B 在同一批题上评测时，每道题都有差值 $d_i=s_i^A-s_i^B$。对这些成对差值 bootstrap，比把两组分数当独立样本更能利用“题目难度相同”的信息。置信区间若跨 0，就不能按预设显著性声称 A 更好；但“不显著”也不证明两者完全相等，需要预先定义非劣界或最小有意义差异。

随机生成还多一层 rollout 方差。可对每题采多次，再用分层/cluster bootstrap 保持同题样本聚在一起；不要把同一道题的 20 次 rollout 假装成 20 道独立题。

## 10. 一张可执行评测卡

```text
目标用户与场景：
成功标准：
最严重的三类失败：
数据来源与时间：
模型、提示、采样参数：
主要指标 / 辅助指标：
人工抽检规则：
延迟和成本口径：
样本量、置信区间：
污染检查：
```

把卡片再补上：tokenizer/chat template、few-shot 示例与顺序、max output、stop 条件、解析失败如何计分、Judge 版本/顺序、硬件与推理框架、样本级原始输出存放位置。没有这些信息，评测结果无法复跑。

<ConceptCheck question="为什么两个模型的 PPL 通常不能跨 tokenizer 直接比较？" :options="['因为 tokenizer 会改变序列的 token 切分和概率分解','因为 PPL 只能用于图像','因为 PPL 越大一定越好']" :answer="0" explanation="同一句话被切成不同数量和边界的 token 后，平均负对数似然的单位已经不同。" />

<ConceptCheck question="在同一固定模型分数上提高分类阈值，通常首先发生什么？" :options='["预测正例变少，FP 常减少而 FN 常增加", "模型参数自动重新训练", "样本真实标签改变"]' :answer="0" explanation="阈值改变决策边界而非模型分数；Precision/Recall 的方向仍要由实际数据验证。" />

## 11. 本课练习

为“基于公司制度回答员工问题”设计评测：至少包含检索召回、答案事实性、引用支持率、应拒答问题、权限泄露、延迟和成本。说明每个指标对应哪类真实风险。

> 本课对应原书第 1.5 节（PDF 第 30–39 页），扩展为从语言模型内在指标到真实系统验收的完整评测框架。

## 12. 课件与论文精读路线

1. [CS224N L11 Slides](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture11-evaluation.pdf)：按页区分 benchmark construction、能力覆盖、污染、LLM Judge 与社会/系统维度；
2. [CMU ANLP L13 Slides](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-13-evaluation.pdf)：从自动指标、人评、模型评委走到统计不确定性；
3. [Adding Error Bars to Evals](https://arxiv.org/pdf/2411.00640.pdf)：用样本级 paired bootstrap 重算一个 A/B 差异的区间；
4. [HELM](https://arxiv.org/pdf/2211.09110.pdf)：选一个 scenario，列出 accuracy 之外的 calibration、robustness、fairness、efficiency；
5. 为自己的系统建立 `item_id → raw output → parsed result → score → error type → cost/latency` 表，任何总分都能回钻到样本。

<ChapterReadings lesson="12-evaluation" />
