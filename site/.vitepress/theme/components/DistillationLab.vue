<script setup lang="ts">
import { computed, ref } from 'vue'

const temperature = ref(2)
const teacherWeight = ref(0.7)

const tokens = ['巴黎', '里昂', '法国', '东京']
const teacherLogits = [3.4, 1.8, 0.9, -0.7]
const studentLogits = [2.1, 1.7, 0.7, 0]
const goldIndex = 0

function softmax(logits: number[], value: number) {
  const scaled = logits.map(logit => logit / value)
  const peak = Math.max(...scaled)
  const exponentials = scaled.map(logit => Math.exp(logit - peak))
  const total = exponentials.reduce((sum, item) => sum + item, 0)
  return exponentials.map(item => item / total)
}

const teacherProb = computed(() => softmax(teacherLogits, temperature.value))
const studentProbAtTemperature = computed(() => softmax(studentLogits, temperature.value))
const studentProb = computed(() => softmax(studentLogits, 1))

const hardLoss = computed(() => -Math.log(studentProb.value[goldIndex]))
const softLoss = computed(() => teacherProb.value.reduce(
  (sum, target, index) => sum - target * Math.log(studentProbAtTemperature.value[index]),
  0
))
const totalLoss = computed(() =>
  (1 - teacherWeight.value) * hardLoss.value
  + teacherWeight.value * temperature.value ** 2 * softLoss.value
)
const teacherEntropy = computed(() => teacherProb.value.reduce(
  (sum, probability) => sum - probability * Math.log(probability),
  0
))
const softKl = computed(() => teacherProb.value.reduce((sum, target, index) =>
  sum + target * Math.log(target / studentProbAtTemperature.value[index]), 0
))

const gradients = computed(() => tokens.map((_, index) => {
  const hardTarget = index === goldIndex ? 1 : 0
  const hardGradient = (1 - teacherWeight.value) * (studentProb.value[index] - hardTarget)
  const kdGradient = teacherWeight.value * temperature.value
    * (studentProbAtTemperature.value[index] - teacherProb.value[index])
  return hardGradient + kdGradient
}))

const strongestGradient = computed(() => {
  const index = gradients.value.reduce((best, value, current) =>
    Math.abs(value) > Math.abs(gradients.value[best]) ? current : best, 0)
  return {
    token: tokens[index],
    direction: gradients.value[index] < 0 ? '提高概率' : '降低概率',
    value: Math.abs(gradients.value[index])
  }
})

function applyPreset(kind: 'hard' | 'classic' | 'soft') {
  if (kind === 'hard') {
    temperature.value = 1
    teacherWeight.value = 0
  } else if (kind === 'classic') {
    temperature.value = 2
    teacherWeight.value = 0.7
  } else {
    temperature.value = 4
    teacherWeight.value = 1
  }
}

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}
</script>

<template>
  <section class="lab-shell distillation-lab" aria-labelledby="distillation-title">
    <h3 id="distillation-title">蒸馏实验：温度怎样把“次优答案”显露出来</h3>
    <p class="lab-intro">教师和学生面对同一个位置。金标准是“巴黎”，但教师还认为“里昂”比“东京”合理得多。拖动温度与教师权重，观察软分布和真正回传到学生 logits 的梯度。</p>

    <div class="preset-row" aria-label="蒸馏预设">
      <button type="button" @click="applyPreset('hard')">只学硬标签</button>
      <button type="button" @click="applyPreset('classic')">经典混合</button>
      <button type="button" @click="applyPreset('soft')">只学软标签</button>
    </div>

    <div class="control-row">
      <label for="distill-temperature">温度 T</label>
      <input id="distill-temperature" v-model.number="temperature" type="range" min="0.5" max="5" step="0.1">
      <output>{{ temperature.toFixed(1) }}</output>
    </div>
    <div class="control-row">
      <label for="distill-alpha">教师损失权重 α</label>
      <input id="distill-alpha" v-model.number="teacherWeight" type="range" min="0" max="1" step="0.05">
      <output>{{ teacherWeight.toFixed(2) }}</output>
    </div>

    <div class="distribution-table" role="table" aria-label="教师与学生 token 概率及梯度">
      <div class="table-head" role="row">
        <span role="columnheader">候选 token</span>
        <span role="columnheader">教师 q<sub>T</sub></span>
        <span role="columnheader">学生 p<sub>T</sub></span>
        <span role="columnheader">总梯度对学生的要求</span>
      </div>
      <div v-for="(token, index) in tokens" :key="token" class="token-row" role="row">
        <strong role="cell">{{ token }}<small v-if="index === goldIndex">硬标签</small></strong>
        <div role="cell" class="bar-cell"><small>教师</small><i :style="{ width: percent(teacherProb[index]) }"></i><span>{{ percent(teacherProb[index]) }}</span></div>
        <div role="cell" class="bar-cell student"><small>学生</small><i :style="{ width: percent(studentProbAtTemperature[index]) }"></i><span>{{ percent(studentProbAtTemperature[index]) }}</span></div>
        <div role="cell" class="gradient-cell" :class="gradients[index] < 0 ? 'raise' : 'lower'">
          <b>{{ gradients[index] < 0 ? '↑ 提高' : '↓ 降低' }}</b>
          <code>{{ Math.abs(gradients[index]).toFixed(3) }}</code>
        </div>
      </div>
    </div>

    <div class="metric-grid">
      <div class="metric-card"><small>教师分布熵</small><strong>{{ teacherEntropy.toFixed(3) }}</strong><span>T 越高，通常越平</span></div>
      <div class="metric-card"><small>软分布 KL(q‖p)</small><strong>{{ softKl.toFixed(3) }}</strong><span>教师与学生仍差多少</span></div>
      <div class="metric-card"><small>加权教学损失</small><strong>{{ totalLoss.toFixed(3) }}</strong><span>(1−α)CE + αT²CE</span></div>
    </div>

    <p class="gradient-summary"><strong>当前最强更新：</strong>让学生对“{{ strongestGradient.token }}”{{ strongestGradient.direction }}，梯度幅度约 {{ strongestGradient.value.toFixed(3) }}。</p>
    <p class="teach-note"><strong>读图：</strong>α=0 时教师分布完全不参与，模型只被要求把“巴黎”推高；α>0 后，“里昂”和“法国”的相对概率也成为监督。提高 T 会暴露更多候选关系，同时使原始 softmax 梯度缩小，因此经典配方常乘 T² 补偿尺度。T 不是越高越好：过高会把教师真正确信的差异也冲淡。</p>
  </section>
</template>

<style scoped>
.distillation-lab { container-type:inline-size; }
.preset-row { display:flex; flex-wrap:wrap; gap:.5rem; margin:.8rem 0 1rem; }
.preset-row button { padding:.5rem .72rem; border:1px solid var(--line); border-radius:9px; color:var(--vp-c-text-1); background:var(--vp-c-bg); cursor:pointer; font-size:.72rem; font-weight:700; }
.preset-row button:hover { border-color:var(--coral); }
.distribution-table { margin:1.2rem 0; overflow:hidden; border:1px solid var(--line); border-radius:12px; background:var(--vp-c-bg); }
.table-head,.token-row { display:grid; grid-template-columns:110px 1fr 1fr minmax(135px,.8fr); align-items:center; gap:.7rem; padding:.65rem .8rem; }
.table-head { color:var(--ink-muted); background:var(--vp-c-bg-soft); font-size:.65rem; font-weight:700; }
.token-row + .token-row { border-top:1px solid var(--line); }
.token-row > strong { display:flex; align-items:center; gap:.35rem; font-size:.76rem; }
.token-row > strong small { padding:.12rem .3rem; border-radius:99px; color:var(--vp-c-bg); background:var(--coral); font-size:.55rem; }
.bar-cell { position:relative; height:25px; overflow:hidden; border-radius:7px; background:var(--vp-c-bg-soft); }
.bar-cell small { display:none; }
.bar-cell i { display:block; height:100%; min-width:2px; background:color-mix(in srgb, var(--coral) 72%, var(--vp-c-bg)); transition:width .2s ease; }
.bar-cell.student i { background:color-mix(in srgb, var(--vp-c-brand-1) 70%, var(--vp-c-bg)); }
.bar-cell span { position:absolute; inset:0; display:grid; place-items:center; color:var(--vp-c-text-1); font:700 .64rem var(--vp-font-family-mono); }
.gradient-cell { display:flex; align-items:center; justify-content:space-between; gap:.45rem; padding:.35rem .5rem; border-radius:7px; font-size:.65rem; }
.gradient-cell.raise { background:color-mix(in srgb, var(--mint) 24%, var(--vp-c-bg)); }
.gradient-cell.lower { background:color-mix(in srgb, var(--gold) 17%, var(--vp-c-bg)); }
.gradient-cell code { color:var(--vp-c-text-1); background:transparent; font-size:.62rem; }
.gradient-summary { margin:.8rem 0 0; padding:.7rem .85rem; border-radius:9px; background:var(--vp-c-bg); color:var(--ink-muted); font-size:.76rem; }
.gradient-summary strong { color:var(--vp-c-text-1); }
@container (max-width:620px) {
  .table-head { display:none; }
  .token-row { grid-template-columns:1fr 1fr; gap:.55rem; padding:.75rem; }
  .token-row > strong,.gradient-cell { grid-column:1 / -1; }
  .bar-cell { height:42px; }
  .bar-cell small { position:absolute; z-index:2; top:3px; left:6px; display:block; color:var(--ink-muted); font-size:.55rem; }
  .bar-cell span { align-items:end; padding-bottom:3px; }
}
</style>
