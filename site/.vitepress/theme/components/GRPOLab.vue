<script setup lang="ts">
import { computed, ref, watch } from 'vue'

type Preset = { label: string; note: string; g: number; k: number; p: number }

const presets: Preset[] = [
  { label: '全错组', note: '没有相对信号', g: 8, k: 0, p: 10 },
  { label: '一次成功', note: '稀有成功被放大', g: 8, k: 1, p: 10 },
  { label: '难度适中', note: '正负样本都充足', g: 8, k: 3, p: 40 },
  { label: '全对组', note: '同样没有信号', g: 8, k: 8, p: 90 }
]

const groupSize = ref(8)
const successes = ref(1)
const baseSuccess = ref(10)
const responseTokens = ref(2048)
const epsilonLow = ref(20)
const epsilonHigh = ref(28)
const ratio = ref(1.15)
const selectedKind = ref<'success' | 'failure'>('success')

watch(groupSize, (next) => {
  if (successes.value > next) successes.value = next
})

const mean = computed(() => successes.value / groupSize.value)
const std = computed(() => Math.sqrt(mean.value * (1 - mean.value)))
const informative = computed(() => successes.value > 0 && successes.value < groupSize.value)
const successAdvantage = computed(() => informative.value ? (1 - mean.value) / std.value : 0)
const failureAdvantage = computed(() => informative.value ? -mean.value / std.value : 0)
const selectedAdvantage = computed(() => selectedKind.value === 'success' ? successAdvantage.value : failureAdvantage.value)

const p = computed(() => baseSuccess.value / 100)
const informativeProbability = computed(() => 1 - p.value ** groupSize.value - (1 - p.value) ** groupSize.value)
const expectedGroups = computed(() => informativeProbability.value > 0 ? 1 / informativeProbability.value : Number.POSITIVE_INFINITY)
const rawRolloutTokens = computed(() => groupSize.value * responseTokens.value)
const effectiveRolloutTokens = computed(() => rawRolloutTokens.value * expectedGroups.value)

const low = computed(() => 1 - epsilonLow.value / 100)
const high = computed(() => 1 + epsilonHigh.value / 100)
const clippedRatio = computed(() => Math.min(high.value, Math.max(low.value, ratio.value)))
const rawObjective = computed(() => ratio.value * selectedAdvantage.value)
const clippedObjective = computed(() => clippedRatio.value * selectedAdvantage.value)
const ppoObjective = computed(() => Math.min(rawObjective.value, clippedObjective.value))
const isClipped = computed(() => Math.abs(rawObjective.value - ppoObjective.value) > 1e-9)

const samples = computed(() => Array.from({ length: groupSize.value }, (_, index) => {
  const reward = index < successes.value ? 1 : 0
  const advantage = reward ? successAdvantage.value : failureAdvantage.value
  return { index: index + 1, reward, advantage }
}))

const diagnosis = computed(() => {
  if (!informative.value) return '这一组所有奖励相同：减去组均值后全部为 0，GRPO 无法知道该提高哪条轨迹。'
  if (successes.value === 1) return '唯一成功轨迹获得很大的正优势；这很有用，但也会放大验证器误判，必须先保证奖励可靠。'
  if (successes.value === groupSize.value - 1) return '唯一失败轨迹承担很大的负优势；题目已偏易，继续为它生成大量样本不划算。'
  return '这一组同时有正、负样本，可以形成稳定的相对比较信号。'
})

function applyPreset(preset: Preset) {
  groupSize.value = preset.g
  successes.value = preset.k
  baseSuccess.value = preset.p
}

function signed(value: number) {
  if (!Number.isFinite(value)) return '—'
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}`
}

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

function tokens(value: number) {
  if (!Number.isFinite(value)) return '∞'
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`
  return value.toFixed(0)
}
</script>

<template>
  <section class="lab-shell grpo-lab" aria-labelledby="grpo-lab-title">
    <h3 id="grpo-lab-title">交互实验：一组回答怎样变成 GRPO 梯度？</h3>
    <p class="lab-intro">先指定本组有几条回答通过验证器，再观察组内优势；随后调节真实成功率，估算为了拿到一个“有正有负”的有效组需要付出多少 rollout。</p>

    <div class="presets" aria-label="实验预设">
      <button v-for="preset in presets" :key="preset.label" type="button" @click="applyPreset(preset)">
        <strong>{{ preset.label }}</strong><span>{{ preset.note }}</span>
      </button>
    </div>

    <div class="lab-grid">
      <div class="controls">
        <label><span>组大小 G</span><input v-model.number="groupSize" type="range" min="2" max="32" step="1"><output>{{ groupSize }}</output></label>
        <label><span>本组成功数 k</span><input v-model.number="successes" type="range" min="0" :max="groupSize" step="1"><output>{{ successes }}</output></label>
        <label><span>单条真实成功率 p</span><input v-model.number="baseSuccess" type="range" min="1" max="99" step="1"><output>{{ baseSuccess }}%</output></label>
        <label><span>每条回答 token</span><input v-model.number="responseTokens" type="range" min="256" max="8192" step="256"><output>{{ responseTokens.toLocaleString() }}</output></label>
      </div>

      <div class="group-view">
        <div class="group-head"><span>组内均值 μ = {{ mean.toFixed(3) }}</span><span>标准差 σ = {{ std.toFixed(3) }}</span></div>
        <div class="sample-grid" :style="{ '--cols': Math.min(groupSize, 8) }">
          <div v-for="sample in samples" :key="sample.index" class="sample" :class="sample.reward ? 'pass' : 'fail'" :aria-label="`回答 ${sample.index}，奖励 ${sample.reward}，优势 ${signed(sample.advantage)}`">
            <small>#{{ sample.index }}</small><strong>R={{ sample.reward }}</strong><span>A={{ signed(sample.advantage) }}</span>
          </div>
        </div>
        <p class="diagnosis" :class="{ empty: !informative }">{{ diagnosis }}</p>
      </div>
    </div>

    <div class="budget-strip">
      <div><small>随机一组有信息的概率</small><strong>{{ percent(informativeProbability) }}</strong><span>1 − p<sup>G</sup> − (1−p)<sup>G</sup></span></div>
      <div><small>平均需采样的组数</small><strong>{{ expectedGroups.toFixed(2) }} 组</strong><span>动态采样的额外代价</span></div>
      <div><small>每个有效 prompt 的 rollout</small><strong>{{ tokens(effectiveRolloutTokens) }}</strong><span>原始一组 {{ tokens(rawRolloutTokens) }} token</span></div>
    </div>

    <div class="clip-demo">
      <div class="clip-controls">
        <label><span>观察哪类回答</span><select v-model="selectedKind"><option value="success">成功回答</option><option value="failure">失败回答</option></select></label>
        <label><span>新/旧策略概率比 r</span><input v-model.number="ratio" type="range" min="0.5" max="1.6" step="0.01"><output>{{ ratio.toFixed(2) }}</output></label>
        <label><span>下裁剪 ε<sub>low</sub></span><input v-model.number="epsilonLow" type="range" min="5" max="40" step="1"><output>{{ epsilonLow }}%</output></label>
        <label><span>上裁剪 ε<sub>high</sub></span><input v-model.number="epsilonHigh" type="range" min="5" max="50" step="1"><output>{{ epsilonHigh }}%</output></label>
      </div>
      <div class="clip-result">
        <div class="ratio-axis" aria-label="策略概率比及裁剪区间">
          <i class="safe" :style="{ left: `${(low - 0.5) / 1.1 * 100}%`, width: `${(high - low) / 1.1 * 100}%` }"></i>
          <i class="marker" :class="{ clipped: isClipped }" :style="{ left: `${(ratio - 0.5) / 1.1 * 100}%` }"></i>
        </div>
        <div class="axis-labels"><span>0.50</span><span>允许区间 [{{ low.toFixed(2) }}, {{ high.toFixed(2) }}]</span><span>1.60</span></div>
        <p><strong>A = {{ signed(selectedAdvantage) }}</strong>，未裁剪目标 {{ signed(rawObjective) }}，PPO 采用 {{ signed(ppoObjective) }}。<b>{{ isClipped ? '这一方向已被截住。' : '这一更新仍在信任区间内。' }}</b></p>
      </div>
    </div>

    <p class="boundary"><strong>实验边界：</strong>二元奖励和样本独立只是教学假设。真实回答有关联，奖励可能连续且有噪声；动态采样还会改变训练题目的分布，因此必须同时记录被过滤题型、采样版本和实际 token 成本。</p>
  </section>
</template>

<style scoped>
.grpo-lab{container-type:inline-size}.presets{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin:1rem 0}.presets button{padding:.72rem;border:1px solid var(--line);border-radius:11px;background:var(--vp-c-bg);color:var(--vp-c-text-1);text-align:left;cursor:pointer}.presets button:hover{border-color:var(--coral);transform:translateY(-1px)}.presets strong,.presets span{display:block}.presets strong{font-size:.78rem}.presets span{margin-top:.18rem;color:var(--ink-muted);font-size:.64rem}.lab-grid{display:grid;grid-template-columns:minmax(250px,.72fr) minmax(360px,1.28fr);gap:1rem}.controls,.group-view,.clip-demo{border:1px solid var(--line);border-radius:14px;background:var(--vp-c-bg)}.controls{display:grid;gap:.8rem;align-content:start;padding:1rem}.controls label,.clip-controls label{display:grid;grid-template-columns:135px 1fr 65px;gap:.45rem;align-items:center}.controls span,.clip-controls span{font-size:.7rem;font-weight:700}.controls output,.clip-controls output{color:var(--coral);text-align:right;font:700 .7rem var(--vp-font-family-mono)}.group-view{padding:1rem}.group-head{display:flex;justify-content:space-between;gap:.8rem;margin-bottom:.7rem;color:var(--ink-muted);font:650 .66rem var(--vp-font-family-mono)}.sample-grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(54px,1fr));gap:.42rem}.sample{min-width:0;padding:.55rem .3rem;border-radius:9px;text-align:center;background:color-mix(in srgb,#d85f55 10%,var(--vp-c-bg-soft));border:1px solid color-mix(in srgb,#d85f55 28%,var(--line))}.sample.pass{background:color-mix(in srgb,#2a9d81 10%,var(--vp-c-bg-soft));border-color:color-mix(in srgb,#2a9d81 30%,var(--line))}.sample small,.sample strong,.sample span{display:block}.sample small{color:var(--ink-muted);font-size:.56rem}.sample strong{margin:.15rem 0;font-size:.7rem}.sample span{font:650 .61rem var(--vp-font-family-mono)}.diagnosis{margin:.75rem 0 0;padding:.65rem .75rem;border-radius:10px;background:color-mix(in srgb,#2a9d81 9%,var(--vp-c-bg));font-size:.68rem;line-height:1.55}.diagnosis.empty{background:color-mix(in srgb,#dfa037 12%,var(--vp-c-bg))}.budget-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:.65rem;margin:1rem 0}.budget-strip>div{padding:.8rem;border:1px solid var(--line);border-radius:12px;background:var(--vp-c-bg)}.budget-strip small,.budget-strip strong,.budget-strip span{display:block}.budget-strip small{color:var(--ink-muted);font-size:.61rem}.budget-strip strong{margin:.25rem 0;color:var(--coral);font:800 1.2rem var(--vp-font-family-mono)}.budget-strip span{color:var(--ink-muted);font-size:.59rem}.clip-demo{display:grid;grid-template-columns:minmax(260px,.8fr) minmax(320px,1.2fr);gap:1rem;padding:1rem}.clip-controls{display:grid;gap:.75rem}.clip-controls select{min-width:0;padding:.3rem;border:1px solid var(--line);border-radius:7px;background:var(--vp-c-bg);color:var(--vp-c-text-1)}.clip-result{align-self:center}.ratio-axis{position:relative;height:18px;margin:.5rem .35rem;border-radius:999px;background:var(--vp-c-bg-soft)}.ratio-axis .safe{position:absolute;top:0;height:100%;border-radius:inherit;background:color-mix(in srgb,#2a9d81 28%,transparent)}.ratio-axis .marker{position:absolute;top:-5px;width:3px;height:28px;border-radius:2px;background:#2a9d81;transform:translateX(-1px)}.ratio-axis .marker.clipped{background:#d85f55}.axis-labels{display:flex;justify-content:space-between;color:var(--ink-muted);font-size:.58rem}.clip-result p{margin:.75rem 0 0;padding:.7rem;border-radius:10px;background:var(--vp-c-bg-soft);font-size:.68rem;line-height:1.6}.boundary{margin-bottom:0;color:var(--ink-muted);font-size:.65rem;line-height:1.6}
@container (max-width:760px){.presets{grid-template-columns:1fr 1fr}.lab-grid,.clip-demo{grid-template-columns:1fr}.budget-strip{grid-template-columns:1fr}.sample-grid{grid-template-columns:repeat(4,1fr)!important}}
@container (max-width:460px){.presets{grid-template-columns:1fr}.controls label,.clip-controls label{grid-template-columns:1fr 58px}.controls input,.clip-controls input,.clip-controls select{grid-column:1}.controls output,.clip-controls output{grid-column:2;grid-row:2}.sample-grid{grid-template-columns:repeat(2,1fr)!important}.group-head{display:grid}.clip-demo{padding:.8rem}}
</style>
