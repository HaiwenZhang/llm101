<script setup lang="ts">
import { computed, ref } from 'vue'

const props = withDefaults(defineProps<{ focus?: 'returns' | 'policy' | 'gae' }>(), { focus: 'returns' })
const gamma = ref(0.9)
const lambda = ref(0.8)
const terminalReward = ref(5)
const stepCost = ref(0)
const criticScale = ref(0.7)

const rewards = computed(() => [stepCost.value, stepCost.value, stepCost.value, terminalReward.value])
const returns = computed(() => {
  const out = Array(rewards.value.length).fill(0)
  let future = 0
  for (let t = rewards.value.length - 1; t >= 0; t -= 1) {
    future = rewards.value[t] + gamma.value * future
    out[t] = future
  }
  return out
})
const values = computed(() => [...returns.value.map(v => v * criticScale.value), 0])
const deltas = computed(() => rewards.value.map((r, t) => r + gamma.value * values.value[t + 1] - values.value[t]))
const gae = computed(() => {
  const out = Array(rewards.value.length).fill(0)
  let running = 0
  for (let t = rewards.value.length - 1; t >= 0; t -= 1) {
    running = deltas.value[t] + gamma.value * lambda.value * running
    out[t] = running
  }
  return out
})
const targets = computed(() => gae.value.map((a, t) => a + values.value[t]))
const rows = computed(() => rewards.value.map((reward, t) => ({
  t, reward, ret: returns.value[t], value: values.value[t], delta: deltas.value[t], advantage: gae.value[t], target: targets.value[t]
})))

const title = computed(() => props.focus === 'returns'
  ? '交互实验：即时奖励怎样变成未来回报？'
  : props.focus === 'policy'
    ? '交互实验：同一条轨迹给每个动作多大更新权重？'
    : '交互实验：Critic、TD 误差与 GAE 怎样接起来？')
const intro = computed(() => props.focus === 'returns'
  ? '调节折扣、步骤成本和终点奖励，观察越早状态看到的回报如何变化。'
  : props.focus === 'policy'
    ? 'Return 或 GAE 都会乘在动作 log-prob 梯度上；改变奖励位置与折扣，信用会沿时间重新分配。'
    : 'Critic 按真实 Return 的一定比例估值。调节偏差与 λ，观察一步 TD 怎样累积成 GAE。')

function preset(kind: 'sparse' | 'cost' | 'failure') {
  if (kind === 'sparse') { terminalReward.value = 5; stepCost.value = 0; gamma.value = 0.9; criticScale.value = 0.7 }
  if (kind === 'cost') { terminalReward.value = 5; stepCost.value = -0.5; gamma.value = 0.95; criticScale.value = 0.8 }
  if (kind === 'failure') { terminalReward.value = 0; stepCost.value = -0.2; gamma.value = 0.9; criticScale.value = 1.2 }
}
function signed(v:number){ return `${v>=0?'+':''}${v.toFixed(2)}` }
</script>

<template>
  <section class="lab-shell return-lab" :aria-labelledby="`return-lab-${focus}`">
    <h3 :id="`return-lab-${focus}`">{{ title }}</h3>
    <p class="lab-intro">{{ intro }}</p>
    <div class="presets"><button type="button" @click="preset('sparse')"><strong>稀疏终奖</strong><span>前三步 0，末尾成功</span></button><button type="button" @click="preset('cost')"><strong>工具有成本</strong><span>每一步先扣费用</span></button><button type="button" @click="preset('failure')"><strong>失败轨迹</strong><span>没有终奖，仍有成本</span></button></div>
    <div class="controls">
      <label><span>折扣 γ</span><input v-model.number="gamma" type="range" min="0" max="1" step="0.01"><output>{{ gamma.toFixed(2) }}</output></label>
      <label><span>GAE λ</span><input v-model.number="lambda" type="range" min="0" max="1" step="0.01"><output>{{ lambda.toFixed(2) }}</output></label>
      <label><span>终点奖励</span><input v-model.number="terminalReward" type="range" min="0" max="10" step="0.5"><output>{{ terminalReward.toFixed(1) }}</output></label>
      <label><span>每步成本</span><input v-model.number="stepCost" type="range" min="-1" max="0" step="0.1"><output>{{ stepCost.toFixed(1) }}</output></label>
      <label><span>Critic 比例</span><input v-model.number="criticScale" type="range" min="0" max="1.5" step="0.05"><output>{{ criticScale.toFixed(2) }}×</output></label>
    </div>
    <div class="trajectory" aria-label="四步轨迹">
      <div v-for="row in rows" :key="row.t" class="step"><small>t={{ row.t }}</small><strong>r {{ signed(row.reward) }}</strong><i></i><span>G {{ signed(row.ret) }}</span></div>
    </div>
    <div class="table-scroll"><table><thead><tr><th>t</th><th>即时 r</th><th>Return G</th><th>Critic V</th><th>TD δ</th><th>GAE Â</th><th>V target</th></tr></thead><tbody><tr v-for="row in rows" :key="row.t"><td>{{ row.t }}</td><td>{{ signed(row.reward) }}</td><td>{{ signed(row.ret) }}</td><td>{{ signed(row.value) }}</td><td>{{ signed(row.delta) }}</td><td>{{ signed(row.advantage) }}</td><td>{{ signed(row.target) }}</td></tr></tbody></table></div>
    <p class="reading"><strong>读表顺序：</strong>先用右到左递推得到 Return；再用相邻两格 V 得到 TD 误差；最后从右向左按 γλ 累积成 GAE。Actor 使用 Â，Critic 回归 V target = V + Â。</p>
    <p class="boundary"><strong>实验边界：</strong>这里人为让 Critic 等于真实 Return 的固定比例，只为清楚显示估值偏差。真实 Critic 从数据学习，误差会随状态、策略版本和奖励定义变化。</p>
  </section>
</template>

<style scoped>
.return-lab{container-type:inline-size}.presets{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin:1rem 0}.presets button{padding:.7rem;border:1px solid var(--line);border-radius:11px;background:var(--vp-c-bg);color:var(--vp-c-text-1);text-align:left;cursor:pointer}.presets button:hover{border-color:var(--coral)}.presets strong,.presets span{display:block}.presets strong{font-size:.73rem}.presets span{margin-top:.16rem;color:var(--ink-muted);font-size:.61rem}.controls{display:grid;grid-template-columns:repeat(5,1fr);gap:.6rem}.controls label{display:grid;grid-template-columns:1fr auto;gap:.3rem;padding:.65rem;border:1px solid var(--line);border-radius:10px;background:var(--vp-c-bg)}.controls span{font-size:.64rem;font-weight:700}.controls output{color:var(--coral);font:700 .64rem var(--vp-font-family-mono)}.controls input{grid-column:1/3;width:100%}.trajectory{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin:1rem 0}.step{position:relative;padding:.7rem;border-radius:11px;background:var(--vp-c-bg-soft);text-align:center}.step small,.step strong,.step span{display:block}.step small{color:var(--ink-muted);font-size:.58rem}.step strong{font-size:.73rem}.step span{color:#5368d9;font:700 .65rem var(--vp-font-family-mono)}.step i{display:block;height:4px;margin:.35rem 0;border-radius:4px;background:color-mix(in srgb,#5368d9 35%,var(--line))}.table-scroll{overflow-x:auto}.return-lab table{width:100%;margin:0;font-size:.65rem}.return-lab th,.return-lab td{padding:.45rem .52rem;text-align:right;white-space:nowrap}.return-lab th:first-child,.return-lab td:first-child{text-align:left}.reading{padding:.65rem .75rem;border-radius:10px;background:color-mix(in srgb,#2a9d81 9%,var(--vp-c-bg));font-size:.66rem;line-height:1.55}.boundary{margin-bottom:0;color:var(--ink-muted);font-size:.63rem;line-height:1.6}
@container (max-width:760px){.controls{grid-template-columns:1fr 1fr}.trajectory{grid-template-columns:1fr 1fr}}
@container (max-width:430px){.presets,.controls,.trajectory{grid-template-columns:1fr}}
</style>
