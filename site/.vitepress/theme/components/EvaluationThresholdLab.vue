<script setup lang="ts">
import { computed, ref } from 'vue'

const threshold = ref(0.5)
const falseNegativeCost = ref(5)
const presets = [
  { label: '少漏报', threshold: .35, cost: 10, note: '召回优先' },
  { label: '折中', threshold: .5, cost: 5, note: '默认观察点' },
  { label: '少误报', threshold: .7, cost: 1, note: '精度优先' }
]
const samples = [
  { score: .95, label: 1 }, { score: .88, label: 0 }, { score: .82, label: 1 },
  { score: .72, label: 1 }, { score: .69, label: 0 }, { score: .61, label: 1 },
  { score: .55, label: 0 }, { score: .49, label: 1 }, { score: .44, label: 0 },
  { score: .40, label: 0 }, { score: .35, label: 1 }, { score: .32, label: 0 },
  { score: .28, label: 0 }, { score: .23, label: 0 }, { score: .18, label: 0 },
  { score: .14, label: 0 }, { score: .10, label: 0 }, { score: .06, label: 0 },
  { score: .04, label: 0 }, { score: .01, label: 0 }
]

function countsAt(value: number) {
  return samples.reduce((counts, sample) => {
    const prediction = sample.score >= value ? 1 : 0
    if (prediction === 1 && sample.label === 1) counts.tp++
    else if (prediction === 1) counts.fp++
    else if (sample.label === 1) counts.fn++
    else counts.tn++
    return counts
  }, { tp: 0, fp: 0, fn: 0, tn: 0 })
}

const counts = computed(() => countsAt(threshold.value))
const precision = computed(() => counts.value.tp / Math.max(1, counts.value.tp + counts.value.fp))
const recall = computed(() => counts.value.tp / Math.max(1, counts.value.tp + counts.value.fn))
const f1 = computed(() => 2 * precision.value * recall.value / Math.max(1e-9, precision.value + recall.value))
const accuracy = computed(() => (counts.value.tp + counts.value.tn) / samples.length)
const weightedCost = computed(() => counts.value.fp + falseNegativeCost.value * counts.value.fn)
const bestThreshold = computed(() => {
  const candidates = [0, ...samples.map(sample => sample.score), 1]
  return candidates
    .map(value => {
      const c = countsAt(value)
      return { value, cost: c.fp + falseNegativeCost.value * c.fn }
    })
    .sort((a, b) => a.cost - b.cost || b.value - a.value)[0]
})

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

function applyPreset(preset: typeof presets[number]) {
  threshold.value = preset.threshold
  falseNegativeCost.value = preset.cost
}
</script>

<template>
  <section class="lab-shell threshold-lab" aria-labelledby="threshold-title">
    <h3 id="threshold-title">阈值实验：同一模型分数，可以得到不同 Precision / Recall</h3>
    <p class="lab-intro">20 个样本中有 6 个真实正例、14 个真实负例。模型分数固定不变；你只改变“多少分算正例”以及漏报相对误报有多贵。</p>

    <div class="preset-row" aria-label="阈值实验预设">
      <button v-for="preset in presets" :key="preset.label" type="button" :class="{ active: threshold === preset.threshold && falseNegativeCost === preset.cost }" @click="applyPreset(preset)"><strong>{{ preset.label }}</strong><span>{{ preset.note }} · 阈值 {{ preset.threshold.toFixed(2) }} · 漏报 {{ preset.cost }}×</span></button>
    </div>

    <div class="control-row"><label for="eval-threshold">判正阈值</label><input id="eval-threshold" v-model.number="threshold" type="range" min="0" max="1" step="0.01"><output>{{ threshold.toFixed(2) }}</output></div>
    <div class="control-row"><label for="eval-fn-cost">一次漏报成本（误报=1）</label><input id="eval-fn-cost" v-model.number="falseNegativeCost" type="range" min="1" max="10" step="1"><output>{{ falseNegativeCost }}×</output></div>

    <div class="score-strip" aria-label="样本分数、真实标签和当前阈值">
      <div class="threshold-line" :style="{ left: `${threshold * 100}%` }"><span>阈值</span></div>
      <i v-for="(sample, index) in samples" :key="index" :class="sample.label ? 'positive' : 'negative'" :style="{ left: `${sample.score * 100}%`, top: `${12 + (index % 4) * 18}px` }" :title="`分数 ${sample.score}，真实${sample.label ? '正' : '负'}`"></i>
      <div class="axis"><span>0.0</span><span>低于阈值 → 预测负</span><span>高于阈值 → 预测正</span><span>1.0</span></div>
    </div>

    <div class="confusion-grid" aria-label="当前混淆矩阵">
      <div class="corner">当前阈值 {{ threshold.toFixed(2) }}</div><div class="head">实际正</div><div class="head">实际负</div>
      <div class="head">预测正</div><div class="good"><small>TP</small><strong>{{ counts.tp }}</strong></div><div class="bad"><small>FP</small><strong>{{ counts.fp }}</strong></div>
      <div class="head">预测负</div><div class="bad"><small>FN</small><strong>{{ counts.fn }}</strong></div><div class="good"><small>TN</small><strong>{{ counts.tn }}</strong></div>
    </div>

    <div class="metric-grid">
      <div class="metric-card"><small>Precision</small><strong>{{ percent(precision) }}</strong><span>TP / (TP+FP)</span></div>
      <div class="metric-card"><small>Recall</small><strong>{{ percent(recall) }}</strong><span>TP / (TP+FN)</span></div>
      <div class="metric-card"><small>F1</small><strong>{{ percent(f1) }}</strong><span>Precision 与 Recall 调和均值</span></div>
      <div class="metric-card"><small>Accuracy</small><strong>{{ percent(accuracy) }}</strong><span>(TP+TN) / 20</span></div>
      <div class="metric-card"><small>加权错误成本</small><strong>{{ weightedCost }}</strong><span>FP + {{ falseNegativeCost }}×FN</span></div>
    </div>

    <p class="decision-note"><strong>这 20 个样本上的最低成本阈值：</strong>{{ bestThreshold.value.toFixed(2) }}（成本 {{ bestThreshold.cost }}）。这是对教学样本的拟合，不可直接当部署阈值；真实阈值必须在独立 validation set 选择，再在 test set 一次性报告。</p>
    <p class="teach-note"><strong>试一试：</strong>提高阈值通常减少 FP、也增加 FN；把漏报成本从 1× 拉到 10×，成本最优阈值会下降。模型分数完全没变，但业务决策和指标改变了，所以报告 Precision/Recall 时必须同时报告阈值与类别基率。</p>
  </section>
</template>

<style scoped>
.threshold-lab { container-type:inline-size; }
.preset-row { display:grid; grid-template-columns:repeat(3,1fr); gap:.55rem; margin:.9rem 0; }
.preset-row button { padding:.6rem .7rem; border:1px solid var(--line); border-radius:10px; color:var(--vp-c-text-1); background:var(--vp-c-bg); text-align:left; cursor:pointer; }
.preset-row button:hover { border-color:var(--ink); }
.preset-row button.active { border-color:var(--ink); color:var(--vp-c-bg); background:var(--ink); }
.preset-row strong,.preset-row span { display:block; }
.preset-row strong { font-size:.7rem; }
.preset-row span { margin-top:.13rem; color:var(--ink-muted); font-size:.56rem; line-height:1.4; }
.preset-row button.active span { color:color-mix(in srgb,var(--vp-c-bg) 72%,transparent); }
.score-strip { position:relative; height:120px; margin:1rem 0; border:1px solid var(--line); border-radius:11px; background:var(--vp-c-bg); }
.score-strip>i { position:absolute; width:12px; height:12px; margin-left:-6px; border-radius:50%; border:2px solid var(--vp-c-bg); }
.score-strip>i.positive { background:var(--mint); }
.score-strip>i.negative { background:var(--coral); }
.threshold-line { position:absolute; z-index:2; top:5px; bottom:28px; width:2px; background:var(--ink); transition:left .15s ease; }
.threshold-line span { position:absolute; top:0; left:5px; padding:.12rem .25rem; border-radius:5px; color:var(--vp-c-bg); background:var(--ink); font-size:.52rem; white-space:nowrap; }
.axis { position:absolute; inset:auto 8px 7px; display:flex; justify-content:space-between; color:var(--ink-muted); font-size:.53rem; }
.confusion-grid { display:grid; grid-template-columns:110px 1fr 1fr; gap:2px; max-width:560px; margin:1rem auto; overflow:hidden; border-radius:11px; background:var(--line); }
.confusion-grid>div { display:grid; place-items:center; min-height:54px; padding:.4rem; background:var(--vp-c-bg); }
.confusion-grid .head,.confusion-grid .corner { color:var(--ink-muted); background:var(--vp-c-bg-soft); font-size:.61rem; font-weight:700; }
.confusion-grid .good { background:color-mix(in srgb,var(--mint) 12%,var(--vp-c-bg)); }
.confusion-grid .bad { background:color-mix(in srgb,var(--coral) 10%,var(--vp-c-bg)); }
.confusion-grid small,.confusion-grid strong { display:block; }
.confusion-grid small { color:var(--ink-muted); font-size:.55rem; }
.confusion-grid strong { font:800 1rem var(--vp-font-family-mono); }
.metric-grid { grid-template-columns:repeat(5,1fr); }
.decision-note { padding:.7rem .82rem; border-radius:10px; color:var(--ink-muted); background:var(--vp-c-bg); font-size:.65rem; line-height:1.55; }
.decision-note strong { color:var(--vp-c-text-1); }
@container (max-width:700px) { .metric-grid { grid-template-columns:1fr 1fr; } }
@container (max-width:450px) { .preset-row,.metric-grid { grid-template-columns:1fr; } .confusion-grid { grid-template-columns:82px 1fr 1fr; } .axis span:nth-child(2),.axis span:nth-child(3) { display:none; } }
</style>
