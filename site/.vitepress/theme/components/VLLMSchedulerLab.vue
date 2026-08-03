<script setup lang="ts">
import { computed, ref, watch } from 'vue'

const tokenBudget = ref(16)
const decodeRequests = ref(4)
const prefillTokens = ref(24)
const chunkedPrefill = ref(true)

const decodeScheduled = computed(() => Math.min(decodeRequests.value, tokenBudget.value))
const budgetAfterDecode = computed(() => tokenBudget.value - decodeScheduled.value)
const prefillScheduled = computed(() => {
  if (chunkedPrefill.value) return Math.min(prefillTokens.value, budgetAfterDecode.value)
  return prefillTokens.value <= budgetAfterDecode.value ? prefillTokens.value : 0
})
const unusedBudget = computed(() => tokenBudget.value - decodeScheduled.value - prefillScheduled.value)
const waitingDecode = computed(() => Math.max(0, decodeRequests.value - decodeScheduled.value))
const tokenCells = computed(() => Array.from({ length: tokenBudget.value }, (_, index) => {
  if (index < decodeScheduled.value) return 'decode'
  if (index < decodeScheduled.value + prefillScheduled.value) return 'prefill'
  return 'idle'
}))

const sequenceTokens = ref(37)
const blockSize = ref(16)
const reservedLength = ref(128)
const sharedPrefix = ref(24)
const physicalIds = [7, 2, 11, 4, 15, 1, 9, 5, 13, 3, 12, 6, 14, 0, 10, 8]

watch(sequenceTokens, value => {
  if (sharedPrefix.value > value) sharedPrefix.value = value
})

const pagedBlocks = computed(() => Math.ceil(sequenceTokens.value / blockSize.value))
const pagedCapacity = computed(() => pagedBlocks.value * blockSize.value)
const pagedWaste = computed(() => pagedCapacity.value - sequenceTokens.value)
const contiguousWaste = computed(() => Math.max(0, reservedLength.value - sequenceTokens.value))
const reusableBlocks = computed(() => Math.min(pagedBlocks.value, Math.floor(sharedPrefix.value / blockSize.value)))
const blockRows = computed(() => Array.from({ length: pagedBlocks.value }, (_, index) => {
  const used = Math.min(blockSize.value, Math.max(0, sequenceTokens.value - index * blockSize.value))
  return {
    logical: index,
    physical: physicalIds[index],
    used,
    percent: used / blockSize.value * 100,
    shared: index < reusableBlocks.value,
  }
}))
</script>

<template>
  <section class="lab-shell vllm-lab" aria-labelledby="vllm-lab-title">
    <h3 id="vllm-lab-title">vLLM 实验一：一个调度步怎样分配 token budget</h3>
    <p class="lab-intro">教学模型采用 V1 的核心思想：Decode 请求每个只需要推进一个 token；剩余预算再交给 Prefill。打开 Chunked Prefill 后，长 Prompt 可以只做一部分，不必等待整段都能塞进本轮。</p>

    <div class="controls-grid">
      <label><span>本轮 token budget</span><input v-model.number="tokenBudget" type="range" min="4" max="32" step="1"><output>{{ tokenBudget }}</output></label>
      <label><span>等待推进的 Decode 请求</span><input v-model.number="decodeRequests" type="range" min="0" max="20" step="1"><output>{{ decodeRequests }}</output></label>
      <label><span>下一个 Prompt 剩余 token</span><input v-model.number="prefillTokens" type="range" min="4" max="48" step="4"><output>{{ prefillTokens }}</output></label>
    </div>
    <label class="switch-row"><input v-model="chunkedPrefill" type="checkbox"><span>允许 Chunked Prefill</span></label>

    <div class="token-budget" role="img" :aria-label="`共 ${tokenBudget} 个 token 槽，其中 Decode ${decodeScheduled}，Prefill ${prefillScheduled}，空闲 ${unusedBudget}`">
      <span v-for="(type,index) in tokenCells" :key="index" :class="type"><small>{{ index + 1 }}</small></span>
    </div>
    <div class="legend-row"><span><i class="decode"></i>Decode</span><span><i class="prefill"></i>Prefill</span><span><i class="idle"></i>未使用</span></div>

    <div class="metric-grid scheduler-metrics">
      <div class="metric-card"><small>本轮 Decode</small><strong>{{ decodeScheduled }}</strong><span>{{ waitingDecode ? `${waitingDecode} 个仍等待` : '所有活跃请求都前进一步' }}</span></div>
      <div class="metric-card"><small>本轮 Prefill</small><strong>{{ prefillScheduled }}</strong><span>{{ prefillScheduled < prefillTokens ? `还剩 ${prefillTokens - prefillScheduled}` : 'Prompt 本轮完成' }}</span></div>
      <div class="metric-card"><small>空闲预算</small><strong>{{ unusedBudget }}</strong><span>{{ unusedBudget ? '本轮没有可装入的工作' : '预算已用满' }}</span></div>
    </div>

    <p v-if="!chunkedPrefill && prefillScheduled === 0" class="teach-note"><strong>队头阻塞：</strong>长 Prompt 需要 {{ prefillTokens }} 个槽，但 Decode 后只剩 {{ budgetAfterDecode }} 个；禁止切块时，本轮一个 Prefill token 都进不来。</p>
    <p v-else class="teach-note"><strong>观察：</strong>调度器决定“哪些 token 本轮执行”，ModelRunner 再把它们压成 GPU 批次。这里展示的是机制，不是复刻 vLLM 所有优先级、投机解码和多模态约束。</p>

    <div class="lab-divider"></div>
    <h3>vLLM 实验二：逻辑块不连续，序列仍然连续</h3>
    <p class="lab-intro">请求只看到连续 token；Block Table 把逻辑块映射到任意空闲物理块。最后一页可能没装满，但不必按最大上下文提前为每个请求留出整段显存。</p>

    <div class="controls-grid">
      <label><span>当前序列 token</span><input v-model.number="sequenceTokens" type="range" min="8" :max="reservedLength" step="1"><output>{{ sequenceTokens }}</output></label>
      <label><span>KV block size</span><select v-model.number="blockSize"><option :value="8">8</option><option :value="16">16</option><option :value="32">32</option></select><output>tokens</output></label>
      <label><span>第二请求相同前缀</span><input v-model.number="sharedPrefix" type="range" min="0" :max="sequenceTokens" step="1"><output>{{ sharedPrefix }}</output></label>
    </div>

    <div class="block-table" role="img" :aria-label="`${pagedBlocks} 个逻辑块映射到不连续物理块，可复用 ${reusableBlocks} 个完整前缀块`">
      <div v-for="block in blockRows" :key="block.logical" :class="{ shared:block.shared }">
        <header><strong>L{{ block.logical }} → P{{ block.physical }}</strong><span>{{ block.used }}/{{ blockSize }}</span></header>
        <i><b :style="{ width:`${block.percent}%` }"></b></i>
        <small>{{ block.shared ? '第二请求可复用' : '本请求独占/新分配' }}</small>
      </div>
    </div>

    <div class="metric-grid paging-metrics">
      <div class="metric-card"><small>分页实际容量</small><strong>{{ pagedCapacity }}</strong><span>只浪费末页 {{ pagedWaste }} token 槽</span></div>
      <div class="metric-card"><small>若预留最大长度</small><strong>{{ reservedLength }}</strong><span>当前空置 {{ contiguousWaste }} token 槽</span></div>
      <div class="metric-card"><small>可复用完整前缀块</small><strong>{{ reusableBlocks }}</strong><span>未满 block 不进入本实验的缓存命中</span></div>
    </div>
  </section>
</template>

<style scoped>
.vllm-lab { container-type:inline-size; }
.controls-grid { display:grid; grid-template-columns:1fr 1fr; gap:.55rem; }
.controls-grid label { display:grid; grid-template-columns:1fr 1.1fr 58px; gap:.45rem; align-items:center; padding:.55rem .65rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg); }
.controls-grid span,.controls-grid output { font-size:.6rem; }
.controls-grid span { font-weight:750; }
.controls-grid output { color:var(--coral); text-align:right; font-family:var(--vp-font-family-mono); }
.controls-grid select { width:100%; }
.switch-row { display:flex; gap:.48rem; align-items:center; margin:.8rem 0; color:var(--ink-muted); font-size:.63rem; }
.token-budget { display:grid; grid-template-columns:repeat(auto-fit,minmax(18px,1fr)); gap:3px; margin:.9rem 0 .5rem; }
.token-budget span { display:grid; place-items:center; height:34px; border-radius:5px; background:var(--vp-c-bg); border:1px solid var(--line); }
.token-budget span.decode { background:color-mix(in srgb,var(--mint) 52%,var(--vp-c-bg)); border-color:var(--mint); }
.token-budget span.prefill { background:color-mix(in srgb,var(--gold) 30%,var(--vp-c-bg)); border-color:var(--gold); }
.token-budget small { color:var(--ink-muted); font:500 .47rem var(--vp-font-family-mono); }
.legend-row { display:flex; flex-wrap:wrap; gap:.8rem; color:var(--ink-muted); font-size:.57rem; }
.legend-row span { display:flex; gap:.3rem; align-items:center; }
.legend-row i { width:10px; height:10px; border:1px solid var(--line); border-radius:3px; background:var(--vp-c-bg); }
.legend-row i.decode { background:var(--mint); }.legend-row i.prefill { background:var(--gold); }
.scheduler-metrics,.paging-metrics { grid-template-columns:repeat(3,1fr); }
.lab-divider { height:1px; margin:1.4rem 0; background:var(--line); }
.block-table { display:grid; grid-template-columns:repeat(3,1fr); gap:.55rem; margin:.9rem 0; }
.block-table>div { min-width:0; padding:.62rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg); }
.block-table>div.shared { border-color:var(--mint); background:color-mix(in srgb,var(--mint) 14%,var(--vp-c-bg)); }
.block-table header { display:flex; justify-content:space-between; gap:.35rem; font-size:.57rem; }
.block-table header strong { font-family:var(--vp-font-family-mono); }
.block-table header span,.block-table small { color:var(--ink-muted); }
.block-table i { display:block; height:9px; margin:.5rem 0; overflow:hidden; border-radius:99px; background:var(--vp-c-bg-soft); }
.block-table b { display:block; height:100%; border-radius:inherit; background:var(--coral); }
.block-table>div.shared b { background:var(--mint); }
.block-table small { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.5rem; }
@container (max-width:650px) { .controls-grid,.block-table { grid-template-columns:1fr; } .scheduler-metrics,.paging-metrics { grid-template-columns:1fr; } }
@container (max-width:430px) { .controls-grid label { grid-template-columns:1fr 52px; } .controls-grid input,.controls-grid select { grid-column:1; grid-row:2; } .controls-grid output { grid-column:2; grid-row:2; } .token-budget { grid-template-columns:repeat(8,1fr); } }
</style>
