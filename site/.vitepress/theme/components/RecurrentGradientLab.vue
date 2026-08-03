<script setup lang="ts">
import { computed, ref } from 'vue'

type Preset = {
  label: string
  note: string
  jacobian: number
  steps: number
  gate: number
}

const presets: Preset[] = [
  { label: '梯度消失', note: '每步只保留 0.75', jacobian: 0.75, steps: 30, gate: 1 },
  { label: '接近稳定', note: '每步约保留 0.99', jacobian: 0.99, steps: 30, gate: 1 },
  { label: '梯度爆炸', note: '每步放大到 1.18', jacobian: 1.18, steps: 24, gate: 1 },
  { label: '门控保留', note: '少写入，多走加法捷径', jacobian: 0.7, steps: 40, gate: 0.08 }
]

const jacobian = ref(0.75)
const steps = ref(30)
const updateGate = ref(1)
const clipThreshold = ref(5)

const effectiveStep = computed(() => Math.abs(1 - updateGate.value + updateGate.value * jacobian.value))
const rawGradient = computed(() => effectiveStep.value ** steps.value)
const clippedGradient = computed(() => Math.min(rawGradient.value, clipThreshold.value))
const logMagnitude = computed(() => Math.log10(Math.max(rawGradient.value, 1e-30)))

const verdict = computed(() => {
  if (rawGradient.value < 1e-3) {
    return {
      tone: 'vanish',
      title: '远处信号几乎消失',
      copy: '靠近序列开头的参数几乎收不到这次误差的信息；裁剪无法把已经消失的梯度找回来。'
    }
  }
  if (rawGradient.value > clipThreshold.value) {
    return {
      tone: 'explode',
      title: '梯度正在爆炸',
      copy: `原始梯度超过阈值 ${clipThreshold.value.toFixed(1)}；范数裁剪会缩短向量，但保留它的方向。`
    }
  }
  return {
    tone: 'stable',
    title: '这段距离仍可传递学习信号',
    copy: '梯度没有跨越很多数量级。不过真实 RNN 是矩阵连乘，还会受激活函数、不同方向的特征值和数据影响。'
  }
})

const pathPoints = computed(() => {
  const count = 11
  return Array.from({ length: count }, (_, index) => {
    const distance = Math.round((steps.value * index) / (count - 1))
    const magnitude = effectiveStep.value ** distance
    const level = Math.max(-6, Math.min(6, Math.log10(Math.max(magnitude, 1e-12))))
    return {
      distance,
      magnitude,
      opacity: Math.max(0.14, Math.min(1, 0.58 + level * 0.12)),
      scale: Math.max(0.58, Math.min(1.45, 1 + level * 0.08))
    }
  })
})

function applyPreset(preset: Preset) {
  jacobian.value = preset.jacobian
  steps.value = preset.steps
  updateGate.value = preset.gate
}

function formatMagnitude(value: number) {
  if (value === 0) return '0'
  if (value >= 0.001 && value < 1000) return value.toFixed(value < 0.1 ? 5 : 3)
  return value.toExponential(2)
}
</script>

<template>
  <section class="lab-shell recurrent-lab" aria-labelledby="recurrent-gradient-title">
    <h3 id="recurrent-gradient-title">BPTT 实验：一条梯度能走多远？</h3>
    <p class="lab-intro">把高维 Jacobian 暂时简化成一个标量 <code>|λ|</code>。若误差信号要跨过 <code>k</code> 个时间步，量级近似为 <code>|λ|ᵏ</code>。先点四个预设，再调旋钮观察“消失、稳定、爆炸”和门控捷径。</p>

    <div class="preset-grid" aria-label="梯度传播预设">
      <button v-for="preset in presets" :key="preset.label" type="button" @click="applyPreset(preset)">
        <strong>{{ preset.label }}</strong><span>{{ preset.note }}</span>
      </button>
    </div>

    <div class="experiment-grid">
      <div class="controls-card">
        <label>
          <span>普通循环路径每步倍率 |λ|</span>
          <input v-model.number="jacobian" type="range" min="0.55" max="1.3" step="0.01">
          <output>{{ jacobian.toFixed(2) }}</output>
          <small>小于 1 倾向消失，大于 1 倾向爆炸；等于 1 只是理想化边界。</small>
        </label>
        <label>
          <span>要跨越的时间步 k</span>
          <input v-model.number="steps" type="range" min="1" max="80" step="1">
          <output>{{ steps }} 步</output>
          <small>同样的小偏差，连乘次数增加后会变成数量级差异。</small>
        </label>
        <label>
          <span>更新门 z：这一刻写入多少新状态</span>
          <input v-model.number="updateGate" type="range" min="0" max="1" step="0.01">
          <output>{{ updateGate.toFixed(2) }}</output>
          <small>教学式：hₜ=(1-z)hₜ₋₁+z·h̃ₜ。z 越小，越多信号沿“保留旧状态”的加法路径通过。</small>
        </label>
        <label>
          <span>梯度裁剪阈值 c</span>
          <input v-model.number="clipThreshold" type="range" min="0.5" max="20" step="0.5">
          <output>{{ clipThreshold.toFixed(1) }}</output>
          <small>这里只展示一维量级；真实训练通常按整个梯度向量的范数缩放。</small>
        </label>
      </div>

      <div class="result-card">
        <div class="formula-card">
          <small>门控后的每步有效倍率</small>
          <code>|1-z+zλ| = {{ effectiveStep.toFixed(4) }}</code>
          <span>跨 {{ steps }} 步：{{ effectiveStep.toFixed(4) }}<sup>{{ steps }}</sup></span>
        </div>

        <div class="gradient-path" aria-label="梯度沿时间反向传播的量级变化">
          <div v-for="point in pathPoints" :key="point.distance" class="path-node">
            <i :style="{ opacity: point.opacity, transform: `scale(${point.scale})` }"></i>
            <span>{{ point.distance }}</span>
          </div>
        </div>
        <div class="path-caption"><span>当前误差</span><b>沿时间反向传播</b><span>更早状态</span></div>

        <div class="metric-grid">
          <div><small>原始梯度倍率</small><strong>{{ formatMagnitude(rawGradient) }}</strong><span>log₁₀={{ logMagnitude.toFixed(2) }}</span></div>
          <div><small>裁剪后倍率</small><strong>{{ formatMagnitude(clippedGradient) }}</strong><span>最多限制到 {{ clipThreshold.toFixed(1) }}</span></div>
        </div>

        <div class="verdict" :class="verdict.tone">
          <strong>{{ verdict.title }}</strong><span>{{ verdict.copy }}</span>
        </div>
      </div>
    </div>

    <div class="two-fixes">
      <div><strong>Gradient clipping</strong><span>只在梯度过大时把它缩短；能治爆炸，不能恢复已经接近 0 的信号。</span></div>
      <div><strong>LSTM / GRU 的门控加法路径</strong><span>在合适的门值下让导数更接近 1，给长距离信号一条更平缓的路，但不保证永不遗忘。</span></div>
    </div>
    <p class="teach-note"><strong>模型边界：</strong>真实 BPTT 连乘的是随时间变化的 Jacobian 矩阵，不是同一个标量。这个实验只负责建立“少量缩小或放大经过很多步会发生什么”的数量级直觉；不能据此预测某个真实网络的精确梯度。</p>
  </section>
</template>

<style scoped>
.recurrent-lab { container-type:inline-size; }
.preset-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.55rem; margin:1rem 0; }
.preset-grid button { padding:.7rem; border:1px solid var(--line); border-radius:10px; color:var(--vp-c-text-1); background:var(--vp-c-bg); text-align:left; cursor:pointer; }
.preset-grid button:hover { border-color:var(--vp-c-brand-1); transform:translateY(-1px); }
.preset-grid strong,.preset-grid span { display:block; }
.preset-grid strong { font-size:.76rem; }
.preset-grid span { margin-top:.15rem; color:var(--ink-muted); font-size:.62rem; }
.experiment-grid { display:grid; grid-template-columns:minmax(280px,.85fr) minmax(340px,1.15fr); gap:.8rem; }
.controls-card,.result-card { padding:.9rem; border:1px solid var(--line); border-radius:13px; background:var(--vp-c-bg); }
.controls-card { display:grid; gap:.8rem; }
.controls-card label { display:grid; grid-template-columns:1fr auto; gap:.25rem .65rem; }
.controls-card label>span { font-size:.71rem; font-weight:750; }
.controls-card input { grid-column:1; width:100%; accent-color:#7c5ce5; }
.controls-card output { color:#7c5ce5; font:700 .72rem var(--vp-font-family-mono); }
.controls-card small { grid-column:1/-1; color:var(--ink-muted); font-size:.6rem; line-height:1.5; }
.formula-card { display:grid; gap:.25rem; padding:.7rem; border-radius:10px; background:color-mix(in srgb,#7c5ce5 8%,var(--vp-c-bg-soft)); text-align:center; }
.formula-card small,.formula-card span { color:var(--ink-muted); font-size:.62rem; }
.formula-card code { color:var(--vp-c-text-1); background:transparent; font-size:.9rem; font-weight:750; }
.gradient-path { display:grid; grid-template-columns:repeat(11,1fr); align-items:center; gap:.15rem; margin:1.2rem .2rem .25rem; }
.path-node { position:relative; display:grid; justify-items:center; gap:.25rem; }
.path-node:not(:last-child)::after { position:absolute; top:8px; left:58%; width:92%; height:2px; content:""; background:linear-gradient(90deg,#7c5ce5,#ccbdf8); }
.path-node i { position:relative; z-index:1; width:16px; height:16px; border:3px solid #fff; border-radius:50%; background:#7c5ce5; box-shadow:0 0 0 1px #7c5ce5; transform-origin:center; }
.path-node span { color:var(--ink-muted); font:500 .52rem var(--vp-font-family-mono); }
.path-caption { display:flex; justify-content:space-between; color:var(--ink-muted); font-size:.57rem; }
.path-caption b { color:#7c5ce5; }
.metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:.55rem; margin:1rem 0 .65rem; }
.metric-grid>div { padding:.7rem; border-radius:10px; background:var(--vp-c-bg-soft); }
.metric-grid small,.metric-grid strong,.metric-grid span { display:block; }
.metric-grid small,.metric-grid span { color:var(--ink-muted); font-size:.58rem; }
.metric-grid strong { margin:.18rem 0; font:750 1.08rem var(--vp-font-family-mono); }
.verdict { display:grid; gap:.25rem; padding:.75rem; border-left:4px solid #2da68d; border-radius:0 9px 9px 0; background:rgba(45,166,141,.09); }
.verdict.vanish { border-left-color:#4c6fff; background:rgba(76,111,255,.09); }
.verdict.explode { border-left-color:#e36c48; background:rgba(227,108,72,.1); }
.verdict strong { font-size:.75rem; }
.verdict span { color:var(--ink-muted); font-size:.65rem; line-height:1.6; }
.two-fixes { display:grid; grid-template-columns:1fr 1fr; gap:.65rem; margin:1rem 0; }
.two-fixes>div { padding:.75rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg-soft); }
.two-fixes strong,.two-fixes span { display:block; }
.two-fixes strong { font-size:.72rem; }
.two-fixes span { margin-top:.22rem; color:var(--ink-muted); font-size:.63rem; line-height:1.55; }
@media (max-width:800px) { .preset-grid,.two-fixes { grid-template-columns:repeat(2,1fr); } .experiment-grid { grid-template-columns:1fr; } }
@media (max-width:520px) { .preset-grid,.two-fixes,.metric-grid { grid-template-columns:1fr; } .gradient-path { gap:0; } .path-node i { width:12px; height:12px; } }
</style>
