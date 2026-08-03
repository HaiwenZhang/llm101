<script setup lang="ts">
import { computed, ref } from 'vue'

const queryIndex = ref(0)
const topK = ref(3)
const rerank = ref(true)
const examples = [
  {
    query: '公司 2025 年的差旅报销上限是多少？',
    answer: '需要找到 2025 年生效的差旅制度，并排除旧版。',
    docs: [
      ['2025 差旅制度：住宿上限 600 元/晚', .83, .98, true],
      ['2023 差旅制度：住宿上限 450 元/晚', .91, .41, false],
      ['2025 年会议费用管理办法', .72, .35, false],
      ['财务制度版本变更记录', .64, .86, true],
      ['员工手册：请假与考勤', .55, .08, false]
    ]
  },
  {
    query: '为什么长上下文会占更多 KV Cache？',
    answer: '需要同时找到“每个历史 token 保存 K/V”和“显存随长度增长”两条证据。',
    docs: [
      ['每层为历史 token 缓存 Key 与 Value', .86, .96, true],
      ['上下文长度翻倍时缓存近似线性增长', .78, .94, true],
      ['模型参数量的常见计算方法', .82, .31, false],
      ['训练阶段的梯度检查点', .63, .22, false],
      ['MoE 路由负载均衡', .52, .09, false]
    ]
  },
  {
    query: '树袋熊和考拉是不是同一种动物？',
    answer: '需要权威来源明确说明两个名称指向同一物种。',
    docs: [
      ['物种词条：树袋熊又称考拉', .79, .98, true],
      ['澳大利亚常见有袋类动物列表', .88, .47, false],
      ['袋熊科动物的生活习性', .76, .28, false],
      ['物种学名与各地区译名对照', .68, .91, true],
      ['桉树的种类与分布', .58, .15, false]
    ]
  }
]

const ranked = computed(() => examples[queryIndex.value].docs
  .map(([title, recall, rank, evidence]) => ({ title, recall:Number(recall), rank:Number(rank), evidence:Boolean(evidence) }))
  .sort((a,b) => (rerank.value ? b.rank-a.rank : b.recall-a.recall)))
const selected = computed(() => ranked.value.slice(0, topK.value))
const evidenceCount = computed(() => selected.value.filter(doc => doc.evidence).length)
const goldEvidenceCount = computed(() => ranked.value.filter(doc => doc.evidence).length)
const recallAtK = computed(() => evidenceCount.value / goldEvidenceCount.value)
const firstRelevantRank = computed(() => ranked.value.findIndex(doc => doc.evidence) + 1)
const mrr = computed(() => firstRelevantRank.value > 0 ? 1 / firstRelevantRank.value : 0)
const distractorCount = computed(() => selected.value.length - evidenceCount.value)
</script>

<template>
  <section class="lab-shell rag-lab" aria-labelledby="rag-title">
    <h3 id="rag-title">RAG 检索与重排实验</h3>
    <p class="lab-intro">向量召回擅长“先捞一批像的”，重排负责“再找真正能回答的”。换问题、开关重排、改变送给模型的文档数。</p>
    <div class="rag-queries">
      <button v-for="(example,i) in examples" :key="example.query" :class="{active:queryIndex===i}" @click="queryIndex=i">问题 {{ i+1 }}</button>
    </div>
    <div class="rag-question"><span>用户问题</span><strong>{{ examples[queryIndex].query }}</strong></div>
    <div class="rag-controls">
      <label><input v-model="rerank" type="checkbox">启用重排</label>
      <label>送入前 {{ topK }} 篇 <input v-model.number="topK" type="range" min="1" max="5" step="1"></label>
    </div>
    <div class="rag-docs">
      <div v-for="(doc,i) in ranked" :key="doc.title" class="rag-doc" :class="{selected:i<topK,evidence:doc.evidence}">
        <span>{{ i+1 }}</span><p>{{ doc.title }}</p><small>{{ rerank ? '重排' : '召回' }} {{ (rerank ? doc.rank : doc.recall).toFixed(2) }}</small>
      </div>
    </div>
    <div class="rag-result" :class="{warn:evidenceCount===0}">
      <strong>{{ evidenceCount ? `已选中 ${evidenceCount} 篇关键证据` : '没有选中关键证据' }}</strong>
      <span>{{ evidenceCount ? examples[queryIndex].answer : '即使生成模型很强，没有正确证据也可能自信地答错。' }}</span>
    </div>
    <div class="metric-grid rag-metrics">
      <div class="metric-card"><small>Recall@{{ topK }}</small><strong>{{ (recallAtK * 100).toFixed(0) }}%</strong><span>{{ evidenceCount }} / {{ goldEvidenceCount }} 篇 gold evidence</span></div>
      <div class="metric-card"><small>MRR</small><strong>{{ mrr.toFixed(2) }}</strong><span>首篇正确证据排名 {{ firstRelevantRank }}</span></div>
      <div class="metric-card"><small>上下文干扰块</small><strong>{{ distractorCount }}</strong><span>Top-k 中不能回答的块</span></div>
      <div class="metric-card"><small>上下文 token 估计</small><strong>≈ {{ topK * 180 }}</strong><span>教学估计：每块 180 token</span></div>
    </div>
  </section>
</template>

<style scoped>
.rag-queries { display:flex; flex-wrap:wrap; gap:.5rem; margin:1rem 0; }
.rag-queries button { padding:.45rem .7rem; border:1px solid var(--line); border-radius:99px; background:var(--vp-c-bg); color:var(--vp-c-text-2); cursor:pointer; }
.rag-queries button.active { border-color:var(--vp-c-brand-1); background:var(--vp-c-brand-soft); color:var(--vp-c-text-1); font-weight:700; }
.rag-question { display:grid; gap:.2rem; padding:.85rem 1rem; border-left:4px solid var(--coral); background:var(--vp-c-bg-soft); }
.rag-question span { color:var(--ink-muted); font-size:.7rem; }
.rag-controls { display:flex; justify-content:space-between; gap:1rem; margin:1rem 0; font-size:.8rem; font-weight:700; }
.rag-controls label { display:flex; align-items:center; gap:.55rem; }
.rag-docs { display:grid; gap:.45rem; }
.rag-doc { display:grid; grid-template-columns:26px 1fr 74px; align-items:center; gap:.55rem; padding:.55rem .7rem; border:1px solid var(--line); border-radius:9px; opacity:.45; }
.rag-doc.selected { opacity:1; background:var(--vp-c-bg); }
.rag-doc.evidence.selected { border-color:var(--vp-c-brand-1); box-shadow:inset 4px 0 var(--mint); }
.rag-doc p { margin:0; line-height:1.5; font-size:.78rem; }
.rag-doc small { text-align:right; color:var(--ink-muted); font: .65rem var(--vp-font-family-mono); }
.rag-result { display:grid; gap:.2rem; margin-top:1rem; padding:.8rem 1rem; border-radius:10px; background:rgba(185,216,193,.28); }
.rag-result.warn { background:rgba(227,108,72,.12); }
.rag-result span { color:var(--ink-muted); font-size:.77rem; line-height:1.6; }
.rag-metrics { grid-template-columns:repeat(4,1fr); }
@media (max-width:780px) { .rag-metrics { grid-template-columns:1fr 1fr; } }
@media (max-width:700px) { .rag-controls { flex-direction:column; } .rag-doc { grid-template-columns:24px 1fr; } .rag-doc small { grid-column:2; text-align:left; } }
@media (max-width:460px) { .rag-metrics { grid-template-columns:1fr; } }
</style>
