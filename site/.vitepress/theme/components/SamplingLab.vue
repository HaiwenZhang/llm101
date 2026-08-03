<script setup lang="ts">
import { computed, ref } from 'vue'

const temperature = ref(1)
const topK = ref(6)
const topP = ref(.9)
const candidates = [
  { token: '学习', logit: 3.1 }, { token: '工作', logit: 2.6 },
  { token: '旅行', logit: 2.1 }, { token: '生活', logit: 1.8 },
  { token: '模型', logit: 1.35 }, { token: '宇宙', logit: .75 },
  { token: '香蕉', logit: .15 }, { token: '……', logit: -.3 }
]

const rows = computed(() => {
  const scaled = candidates.map(item => ({ ...item, score: item.logit / temperature.value }))
  const max = Math.max(...scaled.map(item => item.score))
  const exp = scaled.map(item => Math.exp(item.score - max))
  const total = exp.reduce((a, b) => a + b, 0)
  const withProb = scaled.map((item, i) => ({ ...item, raw: exp[i] / total }))
  const byK = withProb.slice(0, topK.value)
  let cumulative = 0
  let cutoff = byK.length
  for (let i = 0; i < byK.length; i++) {
    cumulative += byK[i].raw
    if (cumulative >= topP.value) { cutoff = i + 1; break }
  }
  const kept = byK.slice(0, Math.max(1, cutoff))
  const keptTotal = kept.reduce((sum, item) => sum + item.raw, 0)
  return withProb.map((item, i) => ({
    ...item,
    kept: i < kept.length,
    probability: i < kept.length ? item.raw / keptTotal : 0
  }))
})

const keptCount = computed(() => rows.value.filter(row => row.kept).length)
</script>

<template>
  <section class="lab-shell sampling-lab" aria-labelledby="sampling-title">
    <div class="lab-title-row">
      <div><h3 id="sampling-title">下一词采样实验</h3><p class="lab-intro">句子是“周末我准备去……”。先调温度，再用 Top-k 和 Top-p 删掉候选，最后重新归一化。</p></div>
      <span class="lab-badge">保留 {{ keptCount }} 个候选</span>
    </div>
    <div class="sampling-controls">
      <label>温度 <input v-model.number="temperature" type="range" min="0.3" max="2" step="0.1"><output>{{ temperature.toFixed(1) }}</output></label>
      <label>Top-k <input v-model.number="topK" type="range" min="1" max="8" step="1"><output>{{ topK }}</output></label>
      <label>Top-p <input v-model.number="topP" type="range" min="0.4" max="1" step="0.05"><output>{{ topP.toFixed(2) }}</output></label>
    </div>
    <div class="sampling-rows">
      <div v-for="row in rows" :key="row.token" class="sampling-row" :class="{ dropped: !row.kept }">
        <strong>{{ row.token }}</strong>
        <div class="sampling-track"><i :style="{ width: `${Math.max(1, row.probability * 100)}%` }"></i></div>
        <output>{{ row.kept ? `${(row.probability * 100).toFixed(1)}%` : '已剔除' }}</output>
      </div>
    </div>
    <p class="teach-note"><strong>观察方法：</strong>先只动温度，看概率形状；再固定温度，只动 Top-k 或 Top-p，看“候选集合”怎样改变。三个旋钮不是同一件事。</p>
  </section>
</template>

<style scoped>
.lab-title-row { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; }
.lab-badge { flex:0 0 auto; padding:.35rem .6rem; border-radius:99px; background:var(--mint); color:#123c38; font-size:.75rem; font-weight:700; }
.sampling-controls { display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; margin:1.2rem 0; }
.sampling-controls label { display:grid; grid-template-columns:auto 1fr 42px; align-items:center; gap:.6rem; padding:.65rem .75rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg); font-size:.78rem; font-weight:700; }
.sampling-controls output { text-align:right; color:var(--coral-dark); font: .7rem var(--vp-font-family-mono); }
.sampling-rows { display:grid; gap:.48rem; }
.sampling-row { display:grid; grid-template-columns:56px 1fr 64px; align-items:center; gap:.7rem; transition:opacity .2s ease; }
.sampling-row strong { font-size:.82rem; }
.sampling-track { height:14px; overflow:hidden; border-radius:99px; background:var(--line); }
.sampling-track i { display:block; height:100%; border-radius:inherit; background:linear-gradient(90deg,var(--mint),var(--coral)); transition:width .25s ease; }
.sampling-row output { text-align:right; color:var(--ink-muted); font:.68rem var(--vp-font-family-mono); }
.sampling-row.dropped { opacity:.34; }
.sampling-row.dropped .sampling-track i { width:0!important; }
@media (max-width:700px) { .sampling-controls { grid-template-columns:1fr; } .lab-title-row { flex-direction:column; } }
</style>
