<script setup lang="ts">
import { computed, ref } from 'vue'

const temperature = ref(1)
const queryAt = ref(5)
const tokens = ['小明', '把', '红色', '书', '放在', '桌上', '。']
const baseScores = [0.2, -0.1, 1.4, 2.25, 0.45, 0.75, -0.6]

const weights = computed(() => {
  const visible = baseScores.map((score, i) => i <= queryAt.value ? score / temperature.value : -Infinity)
  const max = Math.max(...visible)
  const exp = visible.map(x => Number.isFinite(x) ? Math.exp(x - max) : 0)
  const sum = exp.reduce((a, b) => a + b, 0)
  return exp.map(x => x / sum)
})
const topIndex = computed(() => weights.value.indexOf(Math.max(...weights.value)))
</script>

<template>
  <section class="lab-shell attention-lab" aria-labelledby="attention-title">
    <h3 id="attention-title">Attention 温度实验</h3>
    <p class="lab-intro">假设正在计算“桌上”这个位置应该读取谁。低温让注意力更集中，高温让分配更平均。</p>
    <div class="control-row"><label for="temperature">Softmax 温度</label><input id="temperature" v-model.number="temperature" type="range" min="0.2" max="3" step="0.1"><output>{{ temperature.toFixed(1) }}</output></div>
    <div class="control-row"><label for="query">当前 Query 位置</label><input id="query" v-model.number="queryAt" type="range" min="0" :max="tokens.length - 1" step="1"><output>{{ tokens[queryAt] }}</output></div>
    <div class="attention-rows">
      <div v-for="(token, i) in tokens" :key="token + i" class="attention-row" :class="{ masked: i > queryAt, top: i === topIndex }">
        <span class="attention-token">{{ token }}</span>
        <div class="attention-track"><i :style="{ width: `${weights[i] * 100}%` }"></i></div>
        <output>{{ i > queryAt ? 'MASK' : `${(weights[i] * 100).toFixed(1)}%` }}</output>
      </div>
    </div>
    <p class="teach-note"><strong>两件事别混：</strong>causal mask 决定“未来位置能不能看”，softmax 决定“在允许看的位置里怎样分配注意力”。</p>
  </section>
</template>

<style scoped>
.attention-rows { display: grid; gap: .55rem; margin-top: 1.2rem; }
.attention-row { display: grid; grid-template-columns: 70px 1fr 60px; align-items: center; gap: .8rem; }
.attention-token { font-weight: 700; }
.attention-track { height: 15px; overflow: hidden; border-radius: 99px; background: var(--line); }
.attention-track i { display: block; height: 100%; border-radius: inherit; background: var(--mint); transition: width .25s ease; }
.attention-row.top .attention-track i { background: var(--coral); }
.attention-row.masked { opacity: .42; }
.attention-row output { text-align: right; color: var(--ink-muted); font: .72rem var(--vp-font-family-mono); }
</style>
