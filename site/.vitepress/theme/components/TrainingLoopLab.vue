<script setup lang="ts">
import { computed, ref } from 'vue'

const weight = ref(1)
const steps = ref(0)
const x = 2
const target = 10
const prediction = computed(() => weight.value * x)
const error = computed(() => prediction.value - target)
const loss = computed(() => error.value ** 2)

function train() {
  const gradient = 2 * error.value * x
  weight.value -= 0.05 * gradient
  steps.value += 1
}
function reset() { weight.value = 1; steps.value = 0 }
</script>

<template>
  <section class="lab-shell" aria-labelledby="train-title">
    <h3 id="train-title">一个只有 1 个参数的“模型”</h3>
    <p class="lab-intro">任务：输入 2 时，希望输出 10。模型只有公式 y = w × x，训练要找到合适的 w。</p>
    <div class="tiny-model-flow">
      <div><small>输入 x</small><strong>{{ x }}</strong></div><span>×</span>
      <div class="weight-box"><small>参数 w</small><strong>{{ weight.toFixed(3) }}</strong></div><span>=</span>
      <div><small>预测</small><strong>{{ prediction.toFixed(3) }}</strong></div><span>→</span>
      <div><small>目标</small><strong>{{ target }}</strong></div>
    </div>
    <div class="metric-grid">
      <div class="metric-card"><small>误差：预测 − 目标</small><strong>{{ error.toFixed(3) }}</strong></div>
      <div class="metric-card"><small>损失：误差²</small><strong>{{ loss.toFixed(3) }}</strong></div>
      <div class="metric-card"><small>已经训练</small><strong>{{ steps }} 步</strong></div>
    </div>
    <div class="lab-actions"><button @click="train">训练一步</button><button class="ghost" @click="reset">重新开始</button></div>
    <p class="teach-note">每点一次，梯度告诉 w 应该向哪个方向移动。真实模型一次更新数十亿个参数，但“预测 → 损失 → 梯度 → 更新”的闭环相同。</p>
  </section>
</template>

<style scoped>
.tiny-model-flow { display: flex; align-items: center; justify-content: center; gap: .7rem; flex-wrap: wrap; padding: 1.2rem 0; }
.tiny-model-flow div { display: grid; min-width: 88px; padding: .8rem; border: 1px solid var(--line); border-radius: 9px; background: var(--vp-c-bg); text-align: center; }
.tiny-model-flow small { color: var(--ink-muted); }.tiny-model-flow strong { font: 700 1.2rem var(--vp-font-family-mono); }
.tiny-model-flow .weight-box { border-color: var(--coral); background: rgba(227,108,72,.08); }
.lab-actions { display: flex; justify-content: center; gap: .6rem; margin: 1rem 0; }
.lab-actions button { padding: .65rem 1rem; border: 1px solid var(--ink); border-radius: 8px; color: #fff7ea; background: var(--ink); cursor: pointer; font-weight: 700; }
.lab-actions button.ghost { color: var(--ink); background: transparent; }
</style>
