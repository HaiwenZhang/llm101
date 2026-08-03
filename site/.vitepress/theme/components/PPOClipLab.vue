<script setup lang="ts">
import { computed, ref } from 'vue'

type Preset = { label: string; note: string; advantage: number; ratio: number }
const presets: Preset[] = [
  { label: '好动作·正常上涨', note: '正优势，仍在区间内', advantage: 2, ratio: 1.10 },
  { label: '好动作·上涨过多', note: '触发上裁剪', advantage: 2, ratio: 1.50 },
  { label: '坏动作·正常下降', note: '负优势，仍在区间内', advantage: -2, ratio: 0.90 },
  { label: '坏动作·下降过多', note: '触发下裁剪', advantage: -2, ratio: 0.50 }
]

const advantage = ref(2)
const ratio = ref(1.5)
const epsilon = ref(0.2)
const oldProbability = ref(20)

const lower = computed(() => 1 - epsilon.value)
const upper = computed(() => 1 + epsilon.value)
const clippedRatio = computed(() => Math.min(upper.value, Math.max(lower.value, ratio.value)))
const raw = computed(() => ratio.value * advantage.value)
const clipped = computed(() => clippedRatio.value * advantage.value)
const objective = computed(() => Math.min(raw.value, clipped.value))
const isClipped = computed(() => Math.abs(raw.value - objective.value) > 1e-9)
const newProbability = computed(() => Math.min(1, oldProbability.value / 100 * ratio.value))

const xMin = 0.4
const xMax = 1.8
const yMax = computed(() => Math.max(1, Math.abs(advantage.value) * 1.9))
const chartPoints = computed(() => Array.from({ length: 71 }, (_, i) => {
  const r = xMin + (xMax - xMin) * i / 70
  const c = Math.min(upper.value, Math.max(lower.value, r))
  const y = Math.min(r * advantage.value, c * advantage.value)
  const xPx = 36 + (r - xMin) / (xMax - xMin) * 508
  const yPx = 118 - y / yMax.value * 82
  return `${xPx.toFixed(1)},${yPx.toFixed(1)}`
}).join(' '))
const markerX = computed(() => 36 + (ratio.value - xMin) / (xMax - xMin) * 508)
const markerY = computed(() => 118 - objective.value / yMax.value * 82)

const message = computed(() => {
  if (advantage.value === 0) return '优势为 0：无论概率比是多少，这个样本都不给策略方向。'
  if (isClipped.value && advantage.value > 0) return '好动作已经被提高得太多；继续增大概率不再提高代理目标。'
  if (isClipped.value && advantage.value < 0) return '坏动作已经被压低得太多；继续降低概率不再提高代理目标。'
  if (advantage.value > 0 && ratio.value < 1) return '这是坏方向：好动作的概率反而下降，PPO 不会用裁剪替你掩盖它。'
  if (advantage.value < 0 && ratio.value > 1) return '这是坏方向：坏动作的概率反而上升，代理目标会继续惩罚。'
  return '更新方向与优势一致，而且尚未越过裁剪边界。'
})

function applyPreset(preset: Preset) {
  advantage.value = preset.advantage
  ratio.value = preset.ratio
}

function signed(value: number) { return `${value >= 0 ? '+' : ''}${value.toFixed(2)}` }
</script>

<template>
  <section class="lab-shell ppo-lab" aria-labelledby="ppo-lab-title">
    <h3 id="ppo-lab-title">交互实验：PPO 的 min + clip 到底截住哪一边？</h3>
    <p class="lab-intro">选择优势符号并移动概率比。曲线显示 PPO 真正最大化的代理目标，而不是只把 ratio 粗暴夹在区间里。</p>
    <div class="presets">
      <button v-for="preset in presets" :key="preset.label" type="button" @click="applyPreset(preset)"><strong>{{ preset.label }}</strong><span>{{ preset.note }}</span></button>
    </div>
    <div class="experiment">
      <div class="controls">
        <label><span>优势 Â</span><input v-model.number="advantage" type="range" min="-3" max="3" step="0.25"><output>{{ signed(advantage) }}</output></label>
        <label><span>概率比 r</span><input v-model.number="ratio" type="range" min="0.4" max="1.8" step="0.01"><output>{{ ratio.toFixed(2) }}</output></label>
        <label><span>裁剪 ε</span><input v-model.number="epsilon" type="range" min="0.05" max="0.40" step="0.01"><output>{{ epsilon.toFixed(2) }}</output></label>
        <label><span>旧策略概率</span><input v-model.number="oldProbability" type="range" min="1" max="70" step="1"><output>{{ oldProbability }}%</output></label>
        <div class="probability"><span>π<sub>old</sub>={{ oldProbability }}%</span><b>× {{ ratio.toFixed(2) }}</b><strong>π<sub>θ</sub>≈{{ (newProbability * 100).toFixed(1) }}%</strong></div>
      </div>
      <div class="chart-wrap">
        <svg viewBox="0 0 580 150" role="img" aria-label="PPO 裁剪代理目标随概率比变化的曲线">
          <line x1="36" y1="118" x2="544" y2="118" class="axis"/><line x1="36" y1="20" x2="36" y2="132" class="axis"/>
          <rect :x="36 + (lower - xMin) / (xMax - xMin) * 508" y="20" :width="epsilon * 2 / (xMax - xMin) * 508" height="98" class="safe-zone"/>
          <line :x1="36 + (lower - xMin) / (xMax - xMin) * 508" y1="20" :x2="36 + (lower - xMin) / (xMax - xMin) * 508" y2="124" class="bound"/>
          <line :x1="36 + (upper - xMin) / (xMax - xMin) * 508" y1="20" :x2="36 + (upper - xMin) / (xMax - xMin) * 508" y2="124" class="bound"/>
          <polyline :points="chartPoints" class="curve"/>
          <circle :cx="markerX" :cy="markerY" r="6" :class="['dot',{clipped:isClipped}]"/>
          <text x="36" y="144">0.4</text><text x="218" y="144">1−ε</text><text x="290" y="144">r=1</text><text x="365" y="144">1+ε</text><text x="525" y="144">1.8</text>
        </svg>
        <div class="numbers"><div><small>未裁剪 rÂ</small><strong>{{ signed(raw) }}</strong></div><div><small>clip(r)Â</small><strong>{{ signed(clipped) }}</strong></div><div :class="{active:isClipped}"><small>PPO 取 min</small><strong>{{ signed(objective) }}</strong></div></div>
        <p class="message" :class="{warn:isClipped}">{{ message }}</p>
      </div>
    </div>
    <p class="boundary"><strong>边界：</strong>图中只展示单样本、单 token 的 surrogate objective。真实训练还会跨 token/batch 归约、加入 value/entropy/KL 项；clip 也不等价于全局 KL 信赖域。</p>
  </section>
</template>

<style scoped>
.ppo-lab{container-type:inline-size}.presets{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin:1rem 0}.presets button{padding:.7rem;border:1px solid var(--line);border-radius:11px;background:var(--vp-c-bg);color:var(--vp-c-text-1);text-align:left;cursor:pointer}.presets button:hover{border-color:var(--coral);transform:translateY(-1px)}.presets strong,.presets span{display:block}.presets strong{font-size:.72rem}.presets span{margin-top:.18rem;color:var(--ink-muted);font-size:.61rem}.experiment{display:grid;grid-template-columns:minmax(250px,.72fr) minmax(350px,1.28fr);gap:1rem}.controls,.chart-wrap{padding:1rem;border:1px solid var(--line);border-radius:14px;background:var(--vp-c-bg)}.controls{display:grid;gap:.8rem;align-content:start}.controls label{display:grid;grid-template-columns:110px 1fr 60px;gap:.45rem;align-items:center}.controls span{font-size:.7rem;font-weight:700}.controls output{color:var(--coral);text-align:right;font:700 .7rem var(--vp-font-family-mono)}.probability{display:flex;flex-wrap:wrap;gap:.4rem;justify-content:center;padding:.65rem;border-radius:10px;background:var(--vp-c-bg-soft);font:650 .65rem var(--vp-font-family-mono)}.probability b{color:var(--ink-muted)}.chart-wrap svg{display:block;width:100%;height:auto}.axis{stroke:var(--line);stroke-width:1.5}.safe-zone{fill:color-mix(in srgb,#2a9d81 10%,transparent)}.bound{stroke:color-mix(in srgb,#2a9d81 55%,var(--line));stroke-width:1;stroke-dasharray:4 4}.curve{fill:none;stroke:#5368d9;stroke-width:4;stroke-linecap:round;stroke-linejoin:round}.dot{fill:#2a9d81;stroke:var(--vp-c-bg);stroke-width:3}.dot.clipped{fill:#d85f55}.chart-wrap text{fill:var(--ink-muted);font:10px var(--vp-font-family-mono)}.numbers{display:grid;grid-template-columns:repeat(3,1fr);gap:.55rem}.numbers>div{padding:.65rem;border-radius:10px;background:var(--vp-c-bg-soft)}.numbers>div.active{background:color-mix(in srgb,#d85f55 10%,var(--vp-c-bg))}.numbers small,.numbers strong{display:block}.numbers small{color:var(--ink-muted);font-size:.59rem}.numbers strong{margin-top:.18rem;font:750 .9rem var(--vp-font-family-mono)}.message{margin:.65rem 0 0;padding:.65rem .75rem;border-radius:10px;background:color-mix(in srgb,#2a9d81 9%,var(--vp-c-bg));font-size:.66rem;line-height:1.55}.message.warn{background:color-mix(in srgb,#dfa037 12%,var(--vp-c-bg))}.boundary{margin-bottom:0;color:var(--ink-muted);font-size:.64rem;line-height:1.6}
@container (max-width:720px){.presets{grid-template-columns:1fr 1fr}.experiment{grid-template-columns:1fr}}
@container (max-width:430px){.presets{grid-template-columns:1fr}.controls label{grid-template-columns:1fr 58px}.controls input{grid-column:1}.controls output{grid-column:2;grid-row:2}.numbers{grid-template-columns:1fr}}
</style>
