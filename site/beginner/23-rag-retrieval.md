---
title: 第 40 课 RAG 检索系统深入
description: 从切块、稀疏与稠密检索到向量索引、查询增强和重排
---

# 第 40 课　RAG 检索系统深入：先找全，再找准

<div class="lesson-lead">RAG 最常见的失败不是模型“不会写答案”，而是正确证据从未进入上下文。本课把检索拆成五层：文档块、查询、召回器、近似索引、重排器。</div>

::: info 本课资料地图：检索必须独立于生成评测
- 表示学习与 Embedding：[CMU LLM Applications · Representations and Embeddings](https://storage.googleapis.com/cmu-llms/2026/2026-01-27-embeddings.pdf)；
- 关键词、向量、索引与检索系统：[CMU LLM Applications · Retrieval I](https://storage.googleapis.com/cmu-llms/2026/2026-01-29-retrieval.pdf)；
- RAG 检索论文与实践：[CMU ANLP · Retrieval and RAG](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-10-rag.pdf)；
- 向量索引论文：[HNSW](https://arxiv.org/pdf/1603.09320.pdf)。

本课把 Recall@k、MRR、nDCG 与最终答案指标分开：生成器答错不一定是检索错，检索没找到证据也不能怪生成器。
:::

<figure class="teaching-figure">
  <img src="/illustrations/foundations-rag.webp" alt="问题进入知识库召回多个文档，再经筛选和重排形成证据">
  <figcaption>检索不是一次动作：先用便宜方法召回较多候选，再用较贵方法重排少量候选，最后才把证据交给生成模型。</figcaption>
</figure>

<figure class="teaching-figure concept-figure"><img src="/illustrations/rag-offline-online.svg" alt="RAG 从离线解析、切块、元数据、表示与索引，到在线查询、过滤、混合召回、重排和证据包的双流水线"><figcaption>离线建库与在线查询要分别版本化。最终答案错时，沿“gold chunk 是否存在 → 是否召回 → 是否进入上下文预算”逆向排查。</figcaption></figure>

## 1. 先定义检索单位：Document 还是 Chunk

整份 80 页制度通常太长。检索返回“哪一份文档”还不够，生成模型需要具体条款。

### 1.1 切块的四种方式

| 方式 | 做法 | 优点 | 风险 |
|---|---|---|---|
| 固定长度 | 每 N token 一块 | 简单、稳定 | 切断句子和表格 |
| 递归分隔 | 标题→段落→句子→字符 | 尽量保留结构 | 仍要调长度 |
| 语义切块 | 相邻语义变化大时切开 | 主题更集中 | 计算更贵、边界不稳定 |
| 结构切块 | 按条款、章节、表格行 | 引用和权限清楚 | 需要定制解析器 |

实践通常组合：先按结构分，再对超长段落递归切。

Chunk 大小同时控制两个相反问题：太小会丢掉定义、例外和表头之间的联系；太大则把多个主题平均进一个 embedding，精确片段也会被无关文本稀释。一个常用折中是 **child chunk 负责匹配，parent chunk 负责提供上下文**：先检索短子块，再把它所属的完整条款或父段落送入生成器。

切块评测不应只看平均 token 数，还要测黄金答案的必要证据能否完整落在某块或可重组的一组块中。表格要保留表头与行的关系，扫描 PDF 还要记录 OCR 置信度。

### 1.2 Chunk overlap 解决什么

重叠让跨边界信息同时出现在相邻块，代价是索引变大、结果重复。若结构切块已经保留完整条款，不必机械设置很大 overlap。

## 2. Metadata 是隐藏的检索能力

每块至少保存：

```text
document_id, title, section, page
version, effective_date, source_url
department, language, access_control
content_hash, parser_version
```

向量相似度无法可靠判断“只看 2025 版”或“用户是否有权限”。这些应在检索前后用元数据过滤。

ACL 最好在候选产生前就参与过滤，减少越权内容进入重排器、生成 Prompt 或日志。若底层 ANN 不支持复杂过滤，可以按租户/权限建分区索引，或先得到允许 ID 集再做交集；不能仅在最终展示时把来源链接隐藏。

## 3. 稀疏检索：词出现得是否有区分度

### 3.1 TF-IDF 直觉

- TF：词在当前文档出现得多，可能重要；
- IDF：在所有文档都常见的词区分度低，稀有词更有用。

$$\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)\log\frac{N}{df(t)}$$

### 3.2 BM25 为什么常是强基线

BM25 对词频增长做饱和处理，并校正文档长度。专有名词、错误码、合同编号和精确短语通常表现很好。缺点是同义改写匹配弱。

一种常见 BM25 项得分写成：

$$
\operatorname{score}(q,d)=\sum_{t\in q}\operatorname{IDF}(t)
\frac{tf(t,d)(k_1+1)}{tf(t,d)+k_1\left(1-b+b\frac{|d|}{\operatorname{avgdl}}\right)}
$$

$k_1$ 控制词频饱和速度，$b$ 控制文档长度校正。BM25 分数不是跨查询校准后的概率，不能直接和 cosine 分数相加而不做归一化或 rank fusion。

## 4. 稠密检索：把语义映成向量

双塔模型分别编码查询与文档：

$$q=f_{query}(Q),\qquad d_i=f_{doc}(D_i)$$

常用相似度：

$$s(q,d)=q^\top d\quad\text{或}\quad\frac{q^\top d}{\|q\|\|d\|}$$

因为文档向量可提前计算，查询时只编码一次问题。它能找到同义表达，但可能把主题相似、事实不支持的文档排高。

若 $q,d$ 都做 L2 归一化，则点积等于 cosine；若只归一一边或完全不归一，向量模长也会进入排名。Embedding 模型可能要求查询前缀（如 `query:`）与文档前缀，遗漏会造成明显分布错位。索引中的向量、在线 query encoder、tokenizer 与归一化配置必须共享精确版本号。

### 4.1 训练正例和负例

检索器要学会“问题与支持答案的文档靠近”。随机负例太容易，模型学不到细边界；hard negatives 是看起来很像但不能回答的文档，例如旧版制度、同产品不同型号。

### PyTorch：用批内负例训练最小双塔检索器

```python
import torch
import torch.nn.functional as F

# 教学中省略文本编码器，假设它已输出 B 条查询与 B 条正例文档向量
B, hidden = 4, 128
query_vectors = F.normalize(torch.randn(B, hidden), dim=-1)
document_vectors = F.normalize(torch.randn(B, hidden), dim=-1)

temperature = 0.05
scores = query_vectors @ document_vectors.T / temperature  # [B, B]
labels = torch.arange(B)  # 第 i 个问题的正例是第 i 篇文档
loss = F.cross_entropy(scores, labels)

top_documents = scores.topk(k=2, dim=-1).indices
print(scores.shape, loss.item(), top_documents)
```

对角线是正例，其余 $B-1$ 篇文档自动成为负例，损失会拉高正确配对分数、压低错误配对分数。真实训练应使用文本 Encoder、跨设备负例与 hard negatives；若同一批里有多篇实际上都正确，却只标一篇为正例，会制造假负例。

对应的 InfoNCE 风格单向损失可写成：

$$
L=-\frac1B\sum_{i=1}^{B}\log
\frac{\exp(s(q_i,d_i^+)/\tau)}{\sum_{j=1}^{B}\exp(s(q_i,d_j)/\tau)}
$$

温度 $\tau$ 越小，分数差异被放大。跨设备 in-batch negatives 增加负例数，也会增加假负例机会；训练数据必须允许一问多正例或做去假负例处理。

## 5. 混合检索

<figure class="teaching-figure concept-figure"><img src="/illustrations/rag-retrieval-funnel-sparse.webp" alt="BM25 与向量检索并行召回候选，经过重排漏斗得到少量关键证据"><figcaption>第一层便宜而宽，目标是不漏；第二层更贵而窄，目标是把真正能回答的证据放进有限上下文。</figcaption></figure>

把 BM25 与向量检索的排名融合：

- 加权归一化分数；
- Reciprocal Rank Fusion，按名次倒数合并；
- 先关键词过滤再向量排序；
- 多路召回并集后统一重排。

混合检索对专有名词和同义表达同时存在的企业资料尤其实用。

Reciprocal Rank Fusion 不要求两路分数同尺度：

$$
\operatorname{RRF}(d)=\sum_{m=1}^{M}\frac{1}{c+\operatorname{rank}_m(d)}
$$

$c$ 是平滑常数，缺席某一路的文档不贡献该项。RRF 只用名次，稳健但会丢掉“第一名远高于第二名”的分差信息；加权 score fusion 则需要先做可靠校准。

## 6. 大规模向量怎样快速搜索

逐个比较百万向量太慢。近似最近邻（ANN）用少量准确率换速度和内存。

### 6.1 树结构

KD-tree、Ball-tree 递归划分空间，低维有效；高维向量会受维度灾难影响。

### 6.2 LSH

用随机哈希让相近向量更可能落入同桶，查询只比较相关桶。速度快，但要多个哈希表保证召回。

### 6.3 Product Quantization

把向量分成子空间，各自用码本近似，显著压缩存储；代价是距离近似误差。

### 6.4 HNSW

构建多层近邻图：高层远距离跳转，低层精细搜索。常见参数：

- `M`：每个节点连接数，越大索引更准也更占内存；
- `efConstruction`：建索引搜索宽度；
- `efSearch`：查询搜索宽度，越大召回高、延迟高。

参数必须在真实数据上画 Recall–Latency 曲线。

### 6.5 HNSW 不是一个只调 efSearch 的黑盒

HNSW 查询从稀疏高层图做远跳，再在底层扩大候选集。`M` 与 `efConstruction` 主要影响建库时间、图内存和图质量；`efSearch` 是在线召回—延迟旋钮。索引还要处理：

- 新文档增量插入后图质量是否下降；
- 删除是立即重建还是 tombstone；
- embedding 模型升级时必须全量重算，不能混用向量空间；
- PQ/低精度向量是否让 hard negative 排名改变；
- metadata filter 后有效候选是否不足。

基准测试应固定相同 query、gold neighbors 和硬件，报告 exact brute-force Recall@k、p50/p95 延迟、QPS、索引字节/向量与构建时间。

## 7. 查询增强

### 7.1 对话指代还原

```text
用户：K3 的上下文多长？
用户：那它用什么降低缓存？
```

第二问应改写为“K3 使用什么方法降低长上下文推理的 KV Cache 或状态成本？”但必须保留用户真实意图，不能在改写时偷偷生成答案。

### 7.2 Multi-query

生成多种检索表达，分别召回再合并。提高覆盖率，也会增加延迟和噪声。

### 7.3 问题分解

多跳问题分成子问题，前一步答案用于构造下一查询。每一步都要保留来源，防止早期错误传播。

### 7.4 HyDE

先生成一段“假想相关文档”，用其向量检索真实文档。它能补足短问题语义，但假想内容只能帮助检索，绝不能作为最终证据。

## 8. 重排器：让问题和文档真正交互

| 架构 | 计算方式 | 速度 | 精度直觉 |
|---|---|---:|---:|
| Bi-Encoder | Q/D 各自编码后点积 | 快 | 适合大规模召回 |
| Poly/Late Interaction | 保留多个向量或 token 级迟交互 | 中 | 比单向量细致 |
| Cross-Encoder | `[Q;D]` 一起过模型 | 慢 | 候选少时更准确 |
| LLM Reranker | 用提示比较相关性/证据性 | 更慢 | 灵活但成本和稳定性需控 |

重排分数应判断“是否支持回答”，而不只是主题相关。

<RAGPipelineLab />

## 9. 检索评测公式

若正确证据集合为 $R$：

$$\operatorname{Recall@k}=\frac{|\text{Top-k}\cap R|}{|R|}$$

还有 MRR（第一个正确结果的倒数排名）、nDCG（考虑多级相关性和位置）。对 RAG 最重要的是：生成所需关键证据是否在上下文预算内出现。

### 9.1 手算一组指标

假设 Top-5 相关性标签为 `[0,1,0,1,0]`，全集共有 3 篇 gold evidence：

- `Precision@5 = 2/5 = 0.4`；
- `Recall@5 = 2/3 ≈ 0.667`；
- 第一篇相关证据排第 2，所以 `MRR = 1/2 = 0.5`。

nDCG 还会给高相关文档更高 gain，并用理想排序归一化。实际 gold 标注常不完整：一个未标文档可能同样支持答案，不能未经人工复核就一律当 false positive。对于多跳问题，还应标注“一组证据是否共同完整”，而不仅是单块相关。

### 9.2 Retrieval Recall 与 Answer Recall 不是同一个 k

召回器可能取 100 篇候选供重排，但生成器上下文只放 6 块。应分别报告 Recall@100（粗召回是否找全）、重排 nDCG/MRR，以及 **evidence Recall@context-budget**（真正进入 Prompt 的关键证据）。只报最前面的高 Recall 会掩盖重排或预算阶段丢证据。

## 10. 一次检索故障排查

```text
正确块根本没生成 → 修解析/切块
正确块有但向量不对 → 查 embedding 与输入前缀
正确块进候选但排很后 → 加重排/hard negatives
正确块被过滤 → 查版本、权限、时间条件
Top-k 中重复块太多 → 去重/父子文档策略
```

<ConceptCheck question="为什么 Cross-Encoder 通常放在召回之后？" :options="['它对每个问题-文档对联合编码，精细但无法便宜扫描全部文档','它不能读取文本','它只支持图像']" :answer="0" explanation="先用可预计算的双塔或关键词检索缩小候选，再承担联合编码成本。" />

<ConceptCheck question="把 HNSW 的 efSearch 调大，最直接的典型权衡是什么？" :options='["召回通常提高，但查询延迟和计算也上升", "embedding 维度自动降低", "旧文档权限自动更新"]' :answer="0" explanation="efSearch 扩大在线图搜索候选；必须在真实数据上画 Recall–Latency 曲线。" />

> 本课对应原书第 6.3 节（PDF 第 251–263 页），系统展开知识库、查询增强、稀疏/稠密检索、ANN 索引和重排。

## 11. 课件与论文精读路线

1. [CMU ANLP L10 Slides](https://cmu-l3.github.io/anlp-spring2026/static_files/anlp-s2026-10-rag.pdf)：先按页区分 parametric knowledge、sparse/dense retrieval、RAG 训练与评测；
2. [CS224N L10 第 7–26 页](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture10-rag-agents.pdf#page=7)：对照外部知识、检索和 Agent 工具接口；
3. [HNSW 原论文](https://arxiv.org/pdf/1603.09320.pdf)：画出 `M / efConstruction / efSearch` 对内存、构建、Recall 与延迟的控制关系；
4. 为至少 30 个 query 标注 gold evidence，分别测 BM25、Dense、RRF、Reranker 的 Recall@候选与 Recall@上下文预算；
5. 对每个 miss 标记 `parse/chunk/filter/embed/ANN/rerank/budget`，按最大错误桶决定下一次改动。

<ChapterReadings lesson="23-rag-retrieval" />
