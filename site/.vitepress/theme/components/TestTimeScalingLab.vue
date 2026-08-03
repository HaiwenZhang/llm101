<script setup lang="ts">
import { computed, ref } from 'vue'

type Preset = {
  label: string
  note: string
  p: number
  n: number
  rho: number
  selector: number
  candidateTokens: number
  verifierTokens: number
}

const presets: Preset[] = [
  { label: '只答一次', note: '建立 pass@1 基线', p: 35, n: 1, rho: 0, selector: 100, candidateTokens: 900, verifierTokens: 0 },
  { label: '自一致性', note: '多路径，但有相关性', p: 55, n: 16, rho: 20, selector: 90, candidateTokens: 1200, verifierTokens: 0 },
  { label: '弱验证器', note: '覆盖高，未必选得对', p: 20, n: 32, rho: 35, selector: 55, candidateTokens: 1600, verifierTokens: 180 },
  { label: '外部验证器', note: '单元测试式可靠反馈', p: 20, n: 16, rho: 15, selector: 95, candidateTokens: 1600, verifierTokens: 80 }
]

const baseCorrect = ref(35)
const candidates = ref(16)
const correlation = ref(20)
const selectorRecall = ref(85)
const candidateTokens = ref(1200)
const verifierTokens = ref(120)

const p = computed(() => baseCorrect.value / 100)
const rho = computed(() => correlation.value / 100)
const selector = computed(() => selectorRecall.value / 100)
const effectiveN = computed(() => 1 + (candidates.value - 1) * (1 - rho.value))
const independentCoverage = computed(() => 1 - (1 - p.value) ** candidates.value)
const adjustedCoverage = computed(() => 1 - (1 - p.value) ** effectiveN.value)
const selectedAccuracy = computed(() => adjustedCoverage.value * selector.value)
const selectionGap = computed(() => adjustedCoverage.value - selectedAccuracy.value)
const totalTokens = computed(() => candidates.value * (candidateTokens.value + verifierTokens.value))
const costMultiple = computed(() => totalTokens.value / Math.max(candidateTokens.value, 1))

function combination(n: number, k: number) {
  const m = Math.min(k, n - k)
  let result = 1
  for (let i = 1; i <= m; i += 1) result = result * (n - m + i) / i
  return result
}

function binaryMajority(n: number, probability: number) {
  let result = 0
  for (let correct = 0; correct <= n; correct += 1) {
    const mass = combination(n, correct) * probability ** correct * (1 - probability) ** (n - correct)
    if (correct > n / 2) result += mass
    else if (correct === n / 2) result += 0.5 * mass
  }
  return result
}

const majorityAccuracy = computed(() => binaryMajority(candidates.value, p.value))
const rows = computed(() => [1, 4, 8, 16, 32, 64].map((n) => {
  const nEff = 1 + (n - 1) * (1 - rho.value)
  const oracle = 1 - (1 - p.value) ** n
  const diverse = 1 - (1 - p.value) ** nEff
  return { n, oracle, diverse, selected: diverse * selector.value, tokens: n * (candidateTokens.value + verifierTokens.value) }
}))

const diagnosis = computed(() => {
  if (correlation.value >= 60) return { tone: 'warn', title: '先补多样性', text: '候选高度相关，名义上的 N 很大，有效候选数却增长很慢。应改变温度、提示或生成器，而不是继续机械加样本。' }
  if (selectorRecall.value < 65 && candidates.value >= 16) return { tone: 'danger', title: '验证器成为瓶颈', text: '候选集大概率已有正确答案，但选择器经常错过它。继续扩大 N 会给代理漏洞更多机会。' }
  if (baseCorrect.value < 15 && adjustedCoverage.value < 0.5) return { tone: 'warn', title: '生成器覆盖不足', text: '即使多次采样，正确路径仍很少出现。此时应换更强模型、加入工具或改善提示，而不是只训练选择器。' }
  return { tone: 'good', title: '覆盖与选择较匹配', text: '当前设置下，增加候选确实提高覆盖，验证器也能保留大部分收益；仍需用真实任务曲线确认边际收益。' }
})

function applyPreset(preset: Preset) {
  baseCorrect.value = preset.p
  candidates.value = preset.n
  correlation.value = preset.rho
  selectorRecall.value = preset.selector
  candidateTokens.value = preset.candidateTokens
  verifierTokens.value = preset.verifierTokens
}

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

function tokens(value: number) {
  return value >= 1_000_000 ? `${(value / 1_000_000).toFixed(2)}M` : value >= 1000 ? `${(value / 1000).toFixed(1)}K` : value.toLocaleString()
}
</script>

<template>
  <section class="lab-shell tts-lab" aria-labelledby="tts-title">
    <h3 id="tts-title">交互实验：候选多了，为什么仍可能选错？</h3>
    <p class="lab-intro">先让单条候选有概率 <strong>p</strong> 正确，再生成 <strong>N</strong> 条。实验把“候选集中至少有一个正确答案”和“验证器最终选中正确答案”分开计算。</p>

    <div class="preset-grid" aria-label="实验预设">
      <button v-for="preset in presets" :key="preset.label" type="button" @click="applyPreset(preset)">
        <strong>{{ preset.label }}</strong><span>{{ preset.note }}</span>
      </button>
    </div>

    <div class="experiment-grid">
      <div class="controls-panel">
        <label><span>单条候选正确率 p</span><input v-model.number="baseCorrect" type="range" min="5" max="90" step="5"><output>{{ baseCorrect }}%</output></label>
        <label><span>候选数 N</span><input v-model.number="candidates" type="range" min="1" max="64" step="1"><output>{{ candidates }}</output></label>
        <label><span>候选相关性 ρ</span><input v-model.number="correlation" type="range" min="0" max="90" step="5"><output>{{ correlation }}%</output><small>0% 表示教学上的独立假设；越高表示候选越像。</small></label>
        <label><span>选择器召回率 s</span><input v-model.number="selectorRecall" type="range" min="40" max="100" step="5"><output>{{ selectorRecall }}%</output><small>定义为：候选集中已有正确解时，验证器选中它的概率。</small></label>
        <label><span>每条候选 token</span><input v-model.number="candidateTokens" type="range" min="200" max="8000" step="200"><output>{{ candidateTokens.toLocaleString() }}</output></label>
        <label><span>每条验证 token</span><input v-model.number="verifierTokens" type="range" min="0" max="1200" step="40"><output>{{ verifierTokens.toLocaleString() }}</output></label>
      </div>

      <div class="results-panel">
        <div class="metric primary"><small>独立候选的 oracle coverage</small><strong>{{ percent(independentCoverage) }}</strong><span>1 - (1-p)<sup>N</sup></span></div>
        <div class="metric"><small>考虑相关性的教学估计</small><strong>{{ percent(adjustedCoverage) }}</strong><span>有效候选数 N<sub>eff</sub> ≈ {{ effectiveN.toFixed(1) }}</span></div>
        <div class="metric selected"><small>Best-of-N 最终正确率估计</small><strong>{{ percent(selectedAccuracy) }}</strong><span>coverage × selector recall</span></div>

        <div class="bar-stack" aria-label="覆盖到选中的损失">
          <div><span>候选覆盖</span><i><b :style="{ width: percent(adjustedCoverage) }"></b></i><strong>{{ percent(adjustedCoverage) }}</strong></div>
          <div><span>最终选中</span><i><b class="selected-bar" :style="{ width: percent(selectedAccuracy) }"></b></i><strong>{{ percent(selectedAccuracy) }}</strong></div>
        </div>

        <p class="equation">{{ percent(adjustedCoverage) }} × {{ selectorRecall }}% = {{ percent(selectedAccuracy) }}，中间损失 {{ percent(selectionGap) }}</p>
        <div class="cost-strip"><div><small>总 token 账单</small><strong>{{ tokens(totalTokens) }}</strong></div><div><small>约为单答成本</small><strong>{{ costMultiple.toFixed(1) }}×</strong></div></div>
      </div>
    </div>

    <div class="majority-card" :class="{ bad: p < 0.5 && candidates > 1 }">
      <div><small>二元独立假设下的多数投票</small><strong>{{ percent(majorityAccuracy) }}</strong></div>
      <p><strong>{{ p > 0.5 ? 'p > 50%：多数投票会放大单条优势。' : p < 0.5 ? 'p < 50%：多数投票会把系统性错误放大。' : 'p = 50%：增加票数没有信息增益。' }}</strong> 这是把每条输出只分成“正确/错误”的教学模型；真实 Self-Consistency 中错误答案可能分散成多个类别，所以还要看答案归一化与类别分布。</p>
    </div>

    <div class="diagnosis" :class="diagnosis.tone"><strong>{{ diagnosis.title }}</strong><span>{{ diagnosis.text }}</span></div>

    <details class="growth-table"><summary>展开查看 N 从 1 到 64 的收益与账单</summary>
      <div class="table-scroll"><table><thead><tr><th>N</th><th>独立 oracle</th><th>相关性估计</th><th>选中正确</th><th>总 token</th></tr></thead><tbody><tr v-for="row in rows" :key="row.n"><td>{{ row.n }}</td><td>{{ percent(row.oracle) }}</td><td>{{ percent(row.diverse) }}</td><td>{{ percent(row.selected) }}</td><td>{{ tokens(row.tokens) }}</td></tr></tbody></table></div>
    </details>

    <p class="teach-note"><strong>模型边界：</strong><code>N_eff = 1 + (N-1)(1-ρ)</code> 只是帮助理解候选相关性的近似，不是统计定理；“选择器召回率”也被当作固定条件概率。真实系统必须从标注集测 pass@N、候选相似度、排序准确率、校准误差与真实延迟。</p>
  </section>
</template>

<style scoped>
.tts-lab{container-type:inline-size}.preset-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin:1rem 0 1.25rem}.preset-grid button{padding:.72rem;border:1px solid var(--line);border-radius:11px;color:var(--vp-c-text-1);background:var(--vp-c-bg);text-align:left;cursor:pointer}.preset-grid button:hover{border-color:var(--coral);transform:translateY(-1px)}.preset-grid strong,.preset-grid span{display:block}.preset-grid strong{font-size:.8rem}.preset-grid span{margin-top:.18rem;color:var(--ink-muted);font-size:.64rem}.experiment-grid{display:grid;grid-template-columns:minmax(270px,.8fr) minmax(340px,1.2fr);gap:1rem}.controls-panel,.results-panel{padding:1rem;border:1px solid var(--line);border-radius:14px;background:var(--vp-c-bg)}.controls-panel{display:grid;gap:.78rem;align-content:start}.controls-panel label{display:grid;grid-template-columns:145px minmax(80px,1fr) 62px;align-items:center;gap:.45rem}.controls-panel span{font-size:.72rem;font-weight:700}.controls-panel output{text-align:right;color:var(--coral);font:700 .72rem var(--vp-font-family-mono)}.controls-panel small{grid-column:2/4;color:var(--ink-muted);font-size:.61rem;line-height:1.45}.results-panel{display:grid;grid-template-columns:repeat(3,1fr);gap:.65rem}.metric{padding:.78rem;border-radius:12px;background:var(--vp-c-bg-soft)}.metric small,.metric strong,.metric span{display:block}.metric small{min-height:2.4em;color:var(--ink-muted);font-size:.61rem;line-height:1.35}.metric strong{margin:.3rem 0;color:#5368d9;font:800 1.45rem var(--vp-font-family-mono)}.metric span{color:var(--ink-muted);font-size:.59rem}.metric.selected strong{color:var(--coral)}.bar-stack{grid-column:1/-1;display:grid;gap:.55rem;padding:.75rem;border:1px solid var(--line);border-radius:11px}.bar-stack>div{display:grid;grid-template-columns:78px 1fr 55px;align-items:center;gap:.5rem;font-size:.65rem}.bar-stack i{height:10px;overflow:hidden;border-radius:999px;background:var(--vp-c-bg-soft)}.bar-stack b{display:block;height:100%;border-radius:inherit;background:#5368d9}.bar-stack .selected-bar{background:var(--coral)}.bar-stack strong{text-align:right;font-family:var(--vp-font-family-mono)}.equation{grid-column:1/-1;margin:0;padding:.65rem;border-radius:10px;background:color-mix(in srgb,var(--coral) 8%,var(--vp-c-bg));font:600 .68rem var(--vp-font-family-mono)}.cost-strip{grid-column:1/-1;display:grid;grid-template-columns:1fr 1fr;gap:.6rem}.cost-strip>div{padding:.65rem .8rem;border-radius:10px;background:var(--vp-c-bg-soft)}.cost-strip small,.cost-strip strong{display:block}.cost-strip small{color:var(--ink-muted);font-size:.6rem}.cost-strip strong{margin-top:.18rem;font:750 .98rem var(--vp-font-family-mono)}.majority-card{display:grid;grid-template-columns:190px 1fr;gap:1rem;align-items:center;margin:1rem 0;padding:.85rem 1rem;border:1px solid color-mix(in srgb,#2a9d81 40%,var(--line));border-radius:13px;background:color-mix(in srgb,#2a9d81 8%,var(--vp-c-bg))}.majority-card.bad{border-color:color-mix(in srgb,#d85f55 45%,var(--line));background:color-mix(in srgb,#d85f55 8%,var(--vp-c-bg))}.majority-card small,.majority-card strong{display:block}.majority-card small{color:var(--ink-muted);font-size:.62rem}.majority-card>div>strong{margin-top:.22rem;font:800 1.35rem var(--vp-font-family-mono)}.majority-card p{margin:0;font-size:.68rem;line-height:1.65}.diagnosis{display:flex;gap:.7rem;align-items:flex-start;padding:.75rem .9rem;border-radius:11px;font-size:.68rem;line-height:1.6}.diagnosis strong{flex:0 0 auto}.diagnosis.good{background:color-mix(in srgb,#2a9d81 10%,var(--vp-c-bg))}.diagnosis.warn{background:color-mix(in srgb,#dfa037 13%,var(--vp-c-bg))}.diagnosis.danger{background:color-mix(in srgb,#d85f55 11%,var(--vp-c-bg))}.growth-table{margin-top:1rem}.growth-table summary{cursor:pointer;font-size:.72rem;font-weight:700}.table-scroll{overflow-x:auto}.growth-table table{width:100%;margin:.65rem 0 0;font-size:.66rem}.growth-table th,.growth-table td{padding:.45rem .55rem;text-align:right;white-space:nowrap}.growth-table th:first-child,.growth-table td:first-child{text-align:left}.teach-note{margin-bottom:0}
@container (max-width:760px){.preset-grid{grid-template-columns:1fr 1fr}.experiment-grid{grid-template-columns:1fr}.majority-card{grid-template-columns:1fr}.controls-panel label{grid-template-columns:130px 1fr 55px}}
@container (max-width:460px){.preset-grid{grid-template-columns:1fr}.results-panel{grid-template-columns:1fr}.metric,.bar-stack,.equation,.cost-strip{grid-column:1}.controls-panel label{grid-template-columns:1fr 55px}.controls-panel input{grid-column:1}.controls-panel output{grid-column:2;grid-row:2}.controls-panel small{grid-column:1/3}.cost-strip{grid-template-columns:1fr}}
</style>
