<script setup lang="ts">
import { computed, ref } from 'vue'

type Label = '正面' | '负面'
type Demo = { text: string; label: Label; similarity: number[] }

const queries = [
  { text: '用了两天就坏了', gold: '负面' as Label },
  { text: '包装漂亮，送货也快', gold: '正面' as Label }
]
const demos: Demo[] = [
  { text: '刚买就无法开机', label: '负面', similarity: [0.92, 0.18] },
  { text: '质量差，很快损坏', label: '负面', similarity: [0.88, 0.12] },
  { text: '客服一直不回复', label: '负面', similarity: [0.48, 0.18] },
  { text: '包装精美，没有破损', label: '正面', similarity: [0.16, 0.91] },
  { text: '物流很快，第二天到', label: '正面', similarity: [0.11, 0.87] },
  { text: '功能齐全，很满意', label: '正面', similarity: [0.23, 0.52] }
]

const currentQuery = ref(0)
const exampleCount = ref(4)
const selection = ref<'nearest' | 'coverage'>('nearest')
const order = ref<'similarity' | 'positiveLast' | 'negativeLast'>('similarity')
const recency = ref(40)

const selected = computed(() => {
  const ranked = demos
    .map((demo, index) => ({ ...demo, index, sim: demo.similarity[currentQuery.value] }))
    .sort((a, b) => b.sim - a.sim)
  let result = ranked.slice(0, exampleCount.value)

  if (selection.value === 'coverage' && exampleCount.value >= 2) {
    const bestPositive = ranked.find(item => item.label === '正面')!
    const bestNegative = ranked.find(item => item.label === '负面')!
    const chosen = [bestPositive, bestNegative]
    for (const item of ranked) {
      if (chosen.length >= exampleCount.value) break
      if (!chosen.some(existing => existing.index === item.index)) chosen.push(item)
    }
    result = chosen
  }

  if (order.value === 'positiveLast') {
    result = [...result].sort((a, b) => Number(a.label === '正面') - Number(b.label === '正面'))
  } else if (order.value === 'negativeLast') {
    result = [...result].sort((a, b) => Number(a.label === '负面') - Number(b.label === '负面'))
  } else {
    result = [...result].sort((a, b) => a.sim - b.sim)
  }
  return result
})

const positiveProbability = computed(() => {
  const strength = recency.value / 100
  const raw = selected.value.reduce((score, demo, index) => {
    const position = selected.value.length === 1 ? 1 : index / (selected.value.length - 1)
    const weight = demo.sim * (1 + 5 * strength * position)
    return score + (demo.label === '正面' ? weight : -weight)
  }, 0)
  return 1 / (1 + Math.exp(-1.35 * raw))
})
const prediction = computed<Label>(() => positiveProbability.value >= 0.5 ? '正面' : '负面')
const labelCoverage = computed(() => new Set(selected.value.map(item => item.label)).size)
const estimatedTokens = computed(() => 42 + exampleCount.value * 28 + 16)

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}
</script>

<template>
  <section class="lab-shell icl-lab" aria-labelledby="icl-title">
    <h3 id="icl-title">ICL 敏感性实验：示例没变参数，却能改变答案</h3>
    <p class="lab-intro">这是把“相似度 + 位置偏置”显式化的教学打分器，不是真实语言模型。它用于观察控制变量：选择策略、顺序和 recency sensitivity 怎样共同改变 few-shot 分类。</p>

    <div class="query-row" role="group" aria-label="选择待分类文本">
      <button v-for="(query, index) in queries" :key="query.text" type="button" :class="{ active: currentQuery === index }" @click="currentQuery = index">{{ query.text }}</button>
    </div>
    <div class="controls-grid">
      <label><span>示例数量 K</span><input v-model.number="exampleCount" type="range" min="1" max="6" step="1"><output>{{ exampleCount }}</output></label>
      <label><span>位置敏感度</span><input v-model.number="recency" type="range" min="0" max="100" step="10"><output>{{ recency }}%</output></label>
      <label><span>选择策略</span><select v-model="selection"><option value="nearest">只按相似度</option><option value="coverage">先保证两类覆盖</option></select></label>
      <label><span>示例顺序</span><select v-model="order"><option value="similarity">最相似放最后</option><option value="positiveLast">正面示例放最后</option><option value="negativeLast">负面示例放最后</option></select></label>
    </div>

    <div class="demo-flow" aria-label="最终放入 Prompt 的示例顺序">
      <div v-for="(demo, index) in selected" :key="demo.index" :class="demo.label === '正面' ? 'positive' : 'negative'">
        <small>#{{ index + 1 }} · 相似度 {{ demo.sim.toFixed(2) }}</small>
        <strong>{{ demo.text }}</strong>
        <span>答案：{{ demo.label }}</span>
      </div>
      <div class="query-card">
        <small>待预测</small><strong>{{ queries[currentQuery].text }}</strong><span>?</span>
      </div>
    </div>

    <div class="probability-bar"><span>负面 {{ percent(1 - positiveProbability) }}</span><i><b :style="{ width: percent(positiveProbability) }"></b></i><span>正面 {{ percent(positiveProbability) }}</span></div>
    <div class="metric-grid">
      <div class="metric-card"><small>教学预测</small><strong>{{ prediction }}</strong><span>标准标签：{{ queries[currentQuery].gold }}</span></div>
      <div class="metric-card"><small>标签覆盖</small><strong>{{ labelCoverage }} / 2</strong><span>覆盖不等于比例平衡</span></div>
      <div class="metric-card"><small>估计 Prompt token</small><strong>{{ estimatedTokens }}</strong><span>示意：K 增大就重复付费</span></div>
      <div class="metric-card"><small>当前稳定性</small><strong>{{ prediction === queries[currentQuery].gold ? '答对' : '漂移' }}</strong><span>换顺序后还需复测</span></div>
    </div>

    <p class="teach-note"><strong>建议实验：</strong>对“用了两天就坏了”选择 K=2、“先保证两类覆盖”、位置敏感度 100%，再分别把正面/负面示例放最后。教学预测会翻转，说明不是模型更新了参数，而是条件序列发生变化。真实模型的敏感性不能由此公式预测，必须用固定测试集、多个顺序与置信区间实测。</p>
  </section>
</template>

<style scoped>
.icl-lab { container-type:inline-size; }
.query-row { display:flex; flex-wrap:wrap; gap:.5rem; margin:.9rem 0; }
.query-row button { padding:.5rem .72rem; border:1px solid var(--line); border-radius:9px; color:var(--vp-c-text-1); background:var(--vp-c-bg); cursor:pointer; font-size:.69rem; }
.query-row button.active { border-color:var(--ink); color:var(--vp-c-bg); background:var(--ink); }
.controls-grid { display:grid; grid-template-columns:1fr 1fr; gap:.65rem; }
.controls-grid label { display:grid; grid-template-columns:105px 1fr 48px; align-items:center; gap:.45rem; padding:.65rem .72rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg); }
.controls-grid span { font-size:.64rem; font-weight:700; }
.controls-grid output { text-align:right; font:.63rem var(--vp-font-family-mono); }
.controls-grid select { grid-column:2 / 4; min-width:0; padding:.35rem; border:1px solid var(--line); border-radius:7px; color:var(--vp-c-text-1); background:var(--vp-c-bg); font-size:.62rem; }
.demo-flow { display:flex; gap:.5rem; align-items:stretch; margin:1rem 0; overflow-x:auto; padding-bottom:.3rem; }
.demo-flow>div { flex:1 0 125px; display:grid; align-content:start; gap:.25rem; min-height:105px; padding:.62rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg); }
.demo-flow>div.positive { background:color-mix(in srgb,var(--mint) 11%,var(--vp-c-bg)); }
.demo-flow>div.negative { background:color-mix(in srgb,var(--coral) 8%,var(--vp-c-bg)); }
.demo-flow small,.demo-flow span { color:var(--ink-muted); font-size:.56rem; }
.demo-flow strong { font-size:.66rem; line-height:1.45; }
.demo-flow .query-card { border-style:dashed; border-color:var(--coral); }
.query-card span { margin-top:auto; color:var(--coral); font-size:1rem; font-weight:800; }
.probability-bar { display:grid; grid-template-columns:110px 1fr 110px; align-items:center; gap:.55rem; margin:1rem 0; font:.62rem var(--vp-font-family-mono); }
.probability-bar span:last-child { text-align:right; }
.probability-bar i { position:relative; height:18px; overflow:hidden; border-radius:999px; background:color-mix(in srgb,var(--coral) 24%,var(--vp-c-bg)); }
.probability-bar b { display:block; height:100%; border-radius:inherit; background:var(--mint); transition:width .2s ease; }
.metric-grid { grid-template-columns:repeat(4,1fr); }
@container (max-width:680px) { .controls-grid,.metric-grid { grid-template-columns:1fr 1fr; } .demo-flow { flex-wrap:wrap; overflow:visible; } .demo-flow>div { flex-basis:calc(50% - .5rem); } }
@container (max-width:450px) { .controls-grid,.metric-grid { grid-template-columns:1fr; } .controls-grid label { grid-template-columns:1fr 48px; } .controls-grid label input,.controls-grid label select { grid-column:1 / -1; grid-row:2; } .demo-flow>div { flex-basis:100%; } .probability-bar { grid-template-columns:1fr 1fr; } .probability-bar i { grid-column:1 / -1; grid-row:2; } }
</style>
