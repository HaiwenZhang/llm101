<script setup lang="ts">
import { computed, ref } from 'vue'

const expertCount = 4
const tokens = [
  { text: 'def', scores: [2.4, 0.2, 1.5, 0.4] },
  { text: 'return', scores: [2.0, 0.3, 1.4, 0.5] },
  { text: '巴黎', scores: [0.3, 2.2, 0.6, 1.5] },
  { text: '法国', scores: [0.4, 2.0, 0.7, 1.7] },
  { text: '∫', scores: [0.8, 0.5, 2.3, 1.2] },
  { text: '矩阵', scores: [0.7, 0.8, 2.0, 1.5] },
  { text: '你好', scores: [0.9, 1.8, 0.6, 1.6] },
  { text: '<xml>', scores: [1.8, 0.4, 1.3, 0.7] }
]

const current = ref(0)
const topK = ref(2)
const expertOneBias = ref(0)
const capacityFactor = ref(1)

function softmax(values: number[]) {
  const peak = Math.max(...values)
  const exponentials = values.map(value => Math.exp(value - peak))
  const total = exponentials.reduce((sum, value) => sum + value, 0)
  return exponentials.map(value => value / total)
}

const routerProbabilities = computed(() => tokens.map(token => {
  const logits = [...token.scores]
  logits[0] += expertOneBias.value
  return softmax(logits)
}))

const choices = computed(() => routerProbabilities.value.map(probabilities =>
  probabilities
    .map((probability, expert) => ({ probability, expert }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, topK.value)
))

const normalizedCurrentChoices = computed(() => {
  const chosen = choices.value[current.value]
  const total = chosen.reduce((sum, item) => sum + item.probability, 0)
  return chosen.map(item => ({ ...item, gate: item.probability / total }))
})

const load = computed(() => Array.from({ length: expertCount }, (_, expert) =>
  choices.value.reduce((count, tokenChoices) =>
    count + Number(tokenChoices.some(choice => choice.expert === expert)), 0)
))

const capacity = computed(() => Math.ceil(
  capacityFactor.value * tokens.length * topK.value / expertCount
))
const overflow = computed(() => load.value.reduce(
  (sum, count) => sum + Math.max(0, count - capacity.value), 0
))
const averageLoad = computed(() => tokens.length * topK.value / expertCount)
const maxMeanRatio = computed(() => Math.max(...load.value) / averageLoad.value)

const probabilityMass = computed(() => Array.from({ length: expertCount }, (_, expert) =>
  routerProbabilities.value.reduce((sum, probabilities) => sum + probabilities[expert], 0) / tokens.length
))
const auxiliaryLoss = computed(() => expertCount * load.value.reduce((sum, count, expert) => {
  const hardFraction = count / (tokens.length * topK.value)
  return sum + hardFraction * probabilityMass.value[expert]
}, 0))
const routerZLoss = computed(() => tokens.reduce((sum, token) => {
  const logits = [...token.scores]
  logits[0] += expertOneBias.value
  const peak = Math.max(...logits)
  const logZ = peak + Math.log(logits.reduce((subtotal, value) => subtotal + Math.exp(value - peak), 0))
  return sum + logZ ** 2
}, 0) / tokens.length)

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}
</script>

<template>
  <section class="lab-shell moe-lab" aria-labelledby="moe-title">
    <h3 id="moe-title">MoE 批量路由实验：高分不等于有容量</h3>
    <p class="lab-intro">8 个 token 同时竞争 4 位专家。先观察正常 Top-2，再把 E1 的 Router 偏置调高；最后提高容量因子，区分“路由偏斜”和“为偏斜多留显存”这两种做法。</p>

    <div class="control-row"><label for="moe-topk">每 token 激活数 k</label><input id="moe-topk" v-model.number="topK" type="range" min="1" max="2" step="1"><output>{{ topK }} / 4</output></div>
    <div class="control-row"><label for="moe-bias">E1 Router 偏置</label><input id="moe-bias" v-model.number="expertOneBias" type="range" min="0" max="3" step="0.1"><output>+{{ expertOneBias.toFixed(1) }}</output></div>
    <div class="control-row"><label for="moe-capacity">容量因子 CF</label><input id="moe-capacity" v-model.number="capacityFactor" type="range" min="0.5" max="2" step="0.1"><output>{{ capacityFactor.toFixed(1) }}</output></div>

    <div class="token-pills" role="group" aria-label="选择要检查的 token">
      <button v-for="(token, index) in tokens" :key="token.text" type="button" :class="{ active: current === index }" @click="current = index">{{ token.text }}</button>
    </div>

    <div class="route-detail" aria-live="polite">
      <strong>{{ tokens[current].text }}</strong>
      <span v-for="choice in normalizedCurrentChoices" :key="choice.expert">E{{ choice.expert + 1 }}：原概率 {{ percent(choice.probability) }} → Top-k 内权重 {{ percent(choice.gate) }}</span>
    </div>

    <div class="expert-grid" aria-label="每位专家收到的 token 数和容量">
      <div v-for="(count, expert) in load" :key="expert" class="expert" :class="{ overloaded: count > capacity }">
        <div class="expert-head"><strong>E{{ expert + 1 }}</strong><span>{{ count }} / {{ capacity }}</span></div>
        <div class="load-track"><i :style="{ width: `${Math.min(100, count / Math.max(1, capacity) * 100)}%` }"></i><b v-if="count > capacity">+{{ count - capacity }}</b></div>
        <small>软概率质量 {{ percent(probabilityMass[expert]) }}</small>
      </div>
    </div>

    <div class="metric-grid">
      <div class="metric-card"><small>每专家容量 C</small><strong>{{ capacity }}</strong><span>⌈CF × Nk/E⌉</span></div>
      <div class="metric-card"><small>溢出 assignment</small><strong>{{ overflow }}</strong><span>需丢弃、重路由或 dropless</span></div>
      <div class="metric-card"><small>最大/平均负载</small><strong>{{ maxMeanRatio.toFixed(2) }}×</strong><span>1.00× 才完全均匀</span></div>
      <div class="metric-card"><small>归一化辅助损失</small><strong>{{ auxiliaryLoss.toFixed(3) }}</strong><span>均匀时约为 1</span></div>
      <div class="metric-card"><small>Router z-loss</small><strong>{{ routerZLoss.toFixed(2) }}</strong><span>监控 logits 整体变大</span></div>
    </div>

    <p class="teach-note"><strong>试一试：</strong>把 E1 偏置拉到 3.0，E1 会同时获得更高的软概率质量和更多硬 assignment，辅助损失随之上升。把 CF 调高只能减少溢出，却不会修好专家塌缩；它还会增加预留缓冲和显存。真实实现中的 token 顺序、重路由策略和是否 dropless 会改变最终结果。</p>
  </section>
</template>

<style scoped>
.moe-lab { container-type:inline-size; }
.token-pills { display:flex; flex-wrap:wrap; gap:.45rem; margin:1rem 0; }
.token-pills button { padding:.43rem .68rem; border:1px solid var(--line); border-radius:999px; color:var(--vp-c-text-1); background:var(--vp-c-bg); cursor:pointer; font-size:.7rem; }
.token-pills button.active { border-color:var(--ink); color:var(--vp-c-bg); background:var(--ink); }
.route-detail { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; padding:.72rem .8rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg); }
.route-detail strong { min-width:62px; color:var(--coral); font-size:.82rem; }
.route-detail span { padding:.32rem .48rem; border-radius:7px; background:var(--vp-c-bg-soft); font:600 .63rem var(--vp-font-family-mono); }
.expert-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.65rem; margin:1rem 0; }
.expert { padding:.72rem; border:1px solid var(--line); border-radius:11px; background:var(--vp-c-bg); }
.expert.overloaded { border-color:color-mix(in srgb,var(--coral) 65%,var(--line)); background:color-mix(in srgb,var(--coral) 7%,var(--vp-c-bg)); }
.expert-head { display:flex; align-items:center; justify-content:space-between; gap:.4rem; }
.expert-head strong { font:750 .78rem var(--vp-font-family-mono); }
.expert-head span { color:var(--ink-muted); font:.65rem var(--vp-font-family-mono); }
.load-track { position:relative; height:18px; margin:.55rem 0 .38rem; overflow:hidden; border-radius:999px; background:var(--vp-c-bg-soft); }
.load-track i { display:block; height:100%; border-radius:inherit; background:var(--mint); transition:width .2s ease; }
.overloaded .load-track i { background:var(--coral); }
.load-track b { position:absolute; inset:0; display:grid; place-items:center; color:var(--vp-c-text-1); font:.62rem var(--vp-font-family-mono); }
.expert small { color:var(--ink-muted); font-size:.58rem; }
.metric-grid { grid-template-columns:repeat(5,1fr); }
@container (max-width:720px) { .expert-grid { grid-template-columns:1fr 1fr; } .metric-grid { grid-template-columns:1fr 1fr; } }
@container (max-width:430px) { .expert-grid,.metric-grid { grid-template-columns:1fr; } .route-detail { align-items:stretch; } .route-detail strong { width:100%; } }
</style>
