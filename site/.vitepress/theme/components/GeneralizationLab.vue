<script setup lang="ts">
import { computed, ref } from 'vue'

const capacity = ref(6)
const currentStep = ref(45)
const noise = ref(20)

const width = 620
const height = 250
const pad = { left: 46, right: 18, top: 18, bottom: 34 }
const plotWidth = width - pad.left - pad.right
const plotHeight = height - pad.top - pad.bottom

function losses(step: number) {
  const normalizedNoise = noise.value / 100
  const train = 1.14 * Math.exp(-step * (0.025 + capacity.value * 0.0028))
    + 0.045 + normalizedNoise * 0.08 / capacity.value
  const turn = Math.max(0, step - (74 - capacity.value * 5 - normalizedNoise * 20))
  const validation = 0.98 * Math.exp(-step * (0.021 + capacity.value * 0.0015))
    + 0.15 + normalizedNoise * 0.25
    + turn ** 2 * (0.00009 + capacity.value * 0.000035 + normalizedNoise * 0.00008)
  return { train, validation }
}

const points = computed(() => Array.from({ length: 101 }, (_, step) => ({ step, ...losses(step) })))
const maxLoss = computed(() => Math.max(1.25, ...points.value.map(point => Math.max(point.train, point.validation))))

function x(step: number) {
  return pad.left + step / 100 * plotWidth
}

function y(loss: number) {
  return pad.top + (1 - Math.min(loss, maxLoss.value) / maxLoss.value) * plotHeight
}

function path(key: 'train' | 'validation') {
  return points.value.map((point, index) => `${index ? 'L' : 'M'}${x(point.step).toFixed(1)},${y(point[key]).toFixed(1)}`).join(' ')
}

const best = computed(() => points.value.reduce((winner, point) =>
  point.validation < winner.validation ? point : winner, points.value[0]))
const current = computed(() => losses(currentStep.value))
const diagnosis = computed(() => {
  if (currentStep.value < best.value.step - 8) return { label: '还没学够', copy: '训练和验证损失都仍有明显下降空间。' }
  if (currentStep.value > best.value.step + 8) return { label: '开始过拟合', copy: '训练损失继续下降，验证损失已经反弹。' }
  return { label: '接近最佳验证点', copy: '应保存 checkpoint，并在独立 Test 集上只验收一次。' }
})
</script>

<template>
  <section class="lab-shell generalization-lab" aria-labelledby="generalization-title">
    <h3 id="generalization-title">泛化实验：训练更久不一定更好</h3>
    <p class="lab-intro">这是教学模拟，不是真实模型日志。训练集参与更新，验证集只负责选择训练步数和超参数；拖动三个旋钮，观察两条曲线为什么会分叉。</p>

    <div class="control-row"><label for="general-capacity">模型容量</label><input id="general-capacity" v-model.number="capacity" type="range" min="1" max="10" step="1"><output>{{ capacity }} / 10</output></div>
    <div class="control-row"><label for="general-noise">数据噪声</label><input id="general-noise" v-model.number="noise" type="range" min="0" max="50" step="5"><output>{{ noise }}%</output></div>
    <div class="control-row"><label for="general-step">当前训练步</label><input id="general-step" v-model.number="currentStep" type="range" min="0" max="100" step="1"><output>{{ currentStep }}</output></div>

    <div class="curve-wrap">
      <svg :viewBox="`0 0 ${width} ${height}`" role="img" aria-label="训练损失持续下降，验证损失先下降后可能上升的曲线">
        <line :x1="pad.left" :x2="pad.left" :y1="pad.top" :y2="height-pad.bottom" class="axis"/>
        <line :x1="pad.left" :x2="width-pad.right" :y1="height-pad.bottom" :y2="height-pad.bottom" class="axis"/>
        <line :x1="x(best.step)" :x2="x(best.step)" :y1="pad.top" :y2="height-pad.bottom" class="best-line"/>
        <path :d="path('train')" class="train-line"/>
        <path :d="path('validation')" class="validation-line"/>
        <circle :cx="x(currentStep)" :cy="y(current.train)" r="5" class="train-dot"/>
        <circle :cx="x(currentStep)" :cy="y(current.validation)" r="5" class="validation-dot"/>
        <text :x="x(best.step)+6" y="32" class="annotation">最佳验证步 {{ best.step }}</text>
        <text :x="pad.left" :y="height-9" class="axis-label">0</text><text :x="width-pad.right-23" :y="height-9" class="axis-label">训练步</text>
        <text x="8" y="22" class="axis-label">loss</text>
      </svg>
      <div class="legend"><span><i class="train"></i>Train loss</span><span><i class="validation"></i>Validation loss</span><span><i class="best"></i>选 checkpoint</span></div>
    </div>

    <div class="metric-grid">
      <div class="metric-card"><small>当前 Train loss</small><strong>{{ current.train.toFixed(3) }}</strong><span>模型直接看过这批数据</span></div>
      <div class="metric-card"><small>当前 Validation loss</small><strong>{{ current.validation.toFixed(3) }}</strong><span>不参与梯度更新</span></div>
      <div class="metric-card"><small>最佳验证步</small><strong>{{ best.step }}</strong><span>按验证集保存 checkpoint</span></div>
    </div>
    <p class="general-verdict"><strong>{{ diagnosis.label }}：</strong>{{ diagnosis.copy }}</p>
    <p class="teach-note"><strong>试一试：</strong>先把当前训练步拖过最佳点；再提高模型容量或数据噪声。容量越大，模型越能记住训练样本；噪声越高，训练集细节越不值得照抄。Test 集不能拿来反复选步，否则它也会被“间接训练”。</p>
  </section>
</template>

<style scoped>
.generalization-lab { container-type:inline-size; }
.curve-wrap { margin:1rem 0; padding:.75rem; border:1px solid var(--line); border-radius:12px; background:var(--vp-c-bg); }
.curve-wrap svg { display:block; width:100%; height:auto; }
.axis { stroke:var(--line); stroke-width:2; }
.train-line,.validation-line { fill:none; stroke-width:4; stroke-linecap:round; }
.train-line { stroke:var(--vp-c-brand-1); }
.validation-line { stroke:var(--coral); }
.best-line { stroke:var(--gold); stroke-width:2; stroke-dasharray:6 5; }
.train-dot { fill:var(--vp-c-brand-1); stroke:var(--vp-c-bg); stroke-width:3; }
.validation-dot { fill:var(--coral); stroke:var(--vp-c-bg); stroke-width:3; }
.annotation,.axis-label { fill:var(--ink-muted); font:12px var(--vp-font-family-mono); }
.legend { display:flex; flex-wrap:wrap; justify-content:center; gap:.9rem; color:var(--ink-muted); font-size:.66rem; }
.legend span { display:flex; align-items:center; gap:.3rem; }
.legend i { width:18px; height:3px; border-radius:99px; }
.legend .train { background:var(--vp-c-brand-1); }.legend .validation { background:var(--coral); }.legend .best { height:0; border-top:2px dashed var(--gold); }
.general-verdict { margin:.9rem 0 0; padding:.75rem .9rem; border-left:4px solid var(--coral); border-radius:0 9px 9px 0; background:color-mix(in srgb, var(--coral) 8%, var(--vp-c-bg)); color:var(--ink-muted); font-size:.78rem; }
.general-verdict strong { color:var(--vp-c-text-1); }
</style>
