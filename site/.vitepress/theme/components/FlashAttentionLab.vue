<script setup lang="ts">
import { computed, ref } from 'vue'

const sequence = ref(4096)
const batch = ref(1)
const heads = ref(32)
const headDim = ref(128)
const bytes = ref(2)
const causal = ref(true)
const processedBlocks = ref(1)

const sequenceOptions = [1024, 4096, 16384, 32768]
const scoreEntries = computed(() => batch.value * heads.value * sequence.value ** 2 * (causal.value ? .5 : 1))
const oneMatrixBytes = computed(() => scoreEntries.value * bytes.value)
const splitKernelIntermediates = computed(() => oneMatrixBytes.value * 2)
const flashStatsBytes = computed(() => batch.value * heads.value * sequence.value * 2 * 4)
const commonQkvoBytes = computed(() => batch.value * heads.value * sequence.value * headDim.value * bytes.value * 4)
const attentionFlops = computed(() => 4 * batch.value * heads.value * sequence.value ** 2 * headDim.value * (causal.value ? .5 : 1))
const workspaceRatio = computed(() => splitKernelIntermediates.value / flashStatsBytes.value)
const flashBarWidth = computed(() => Math.max(1.5, flashStatsBytes.value / splitKernelIntermediates.value * 100))

const blocks = [
  { label: '块 1', scores: [1, 2], values: [10, 20] },
  { label: '块 2', scores: [4, -1], values: [30, 40] }
]

const online = computed(() => {
  let maximum = -Infinity
  let denominator = 0
  let numerator = 0
  let oldScale = 0
  const seenScores:number[] = []
  const seenValues:number[] = []

  for (const block of blocks.slice(0, processedBlocks.value)) {
    const blockMaximum = Math.max(...block.scores)
    const newMaximum = Math.max(maximum, blockMaximum)
    oldScale = maximum === -Infinity ? 0 : Math.exp(maximum - newMaximum)
    denominator *= oldScale
    numerator *= oldScale
    block.scores.forEach((score, index) => {
      const weight = Math.exp(score - newMaximum)
      denominator += weight
      numerator += weight * block.values[index]
      seenScores.push(score)
      seenValues.push(block.values[index])
    })
    maximum = newMaximum
  }

  const weights = seenScores.map(score => Math.exp(score - maximum) / denominator)
  return {
    maximum,
    denominator,
    numerator,
    output: numerator / denominator,
    oldScale,
    weights,
    seenValues
  }
})

function formatBytes(value:number) {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let current = value
  let index = 0
  while (current >= 1024 && index < units.length - 1) {
    current /= 1024
    index++
  }
  return `${current.toFixed(current >= 100 ? 0 : current >= 10 ? 1 : 2)} ${units[index]}`
}

function formatEntries(value:number) {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`
  return value.toLocaleString()
}

function formatFlops(value:number) {
  if (value >= 1e15) return `${(value / 1e15).toFixed(2)} PFLOPs`
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} TFLOPs`
  return `${(value / 1e9).toFixed(2)} GFLOPs`
}
</script>

<template>
  <section class="lab-shell flash-lab" aria-labelledby="flash-lab-title">
    <h3 id="flash-lab-title">FlashAttention 实验：省掉的是哪一块内存</h3>
    <p class="lab-intro">教学账本假设普通 Attention 用分开的 kernel，把分数 S 和概率 P 都写入 HBM；FlashAttention 不 materialize 这两个完整矩阵，只保存每行的最大值 m、分母 ℓ 和最终输出。不同框架的实际 workspace 会不同，但 N² 与 N 的差距不变。</p>

    <div class="preset-row" aria-label="序列长度预设">
      <button v-for="option in sequenceOptions" :key="option" type="button" :class="{ active: sequence === option }" @click="sequence = option">{{ (option / 1024).toFixed(0) }}K tokens</button>
    </div>

    <div class="control-grid">
      <label><span>Batch</span><input v-model.number="batch" type="range" min="1" max="8" step="1"><output>{{ batch }}</output></label>
      <label><span>Attention heads</span><input v-model.number="heads" type="range" min="1" max="64" step="1"><output>{{ heads }}</output></label>
      <label><span>Head dimension</span><select v-model.number="headDim"><option :value="64">64</option><option :value="128">128</option><option :value="256">256</option></select><output>d</output></label>
      <label><span>元素精度</span><select v-model.number="bytes"><option :value="2">FP16/BF16</option><option :value="4">FP32</option></select><output>{{ bytes }} B</output></label>
    </div>
    <label class="causal-toggle"><input v-model="causal" type="checkbox"><span>只计算 causal 下三角（教学近似按一半元素计）</span></label>

    <div class="metric-grid">
      <div class="metric-card"><small>实际计算的 score</small><strong>{{ formatEntries(scoreEntries) }}</strong><span>密集 Attention 仍是 N²</span></div>
      <div class="metric-card"><small>若 materialize S + P</small><strong>{{ formatBytes(splitKernelIntermediates) }}</strong><span>两个 N×N 中间矩阵</span></div>
      <div class="metric-card"><small>Flash 行统计 m + ℓ</small><strong>{{ formatBytes(flashStatsBytes) }}</strong><span>FP32，随 N 线性增长</span></div>
    </div>

    <div class="workspace-bars" aria-label="普通 Attention 与 FlashAttention 的 HBM 中间量对比">
      <div><span>普通：S + P</span><i><b style="width:100%"></b></i><output>{{ formatBytes(splitKernelIntermediates) }}</output></div>
      <div><span>Flash：m + ℓ</span><i><b class="flash" :style="{ width: `${flashBarWidth}%` }"></b></i><output>{{ formatBytes(flashStatsBytes) }}</output></div>
    </div>

    <p class="ledger-note">这组配置中，S+P 与行统计相差约 <strong>{{ workspaceRatio.toLocaleString(undefined,{ maximumFractionDigits:0 }) }}×</strong>。但 Q/K/V/O 仍需约 <strong>{{ formatBytes(commonQkvoBytes) }}</strong>，两次矩阵乘的算术量仍约 <strong>{{ formatFlops(attentionFlops) }}</strong>；FlashAttention 没把密集 Attention 变成线性时间。Flash 条设置了最小可见宽度，比例请以右侧数值为准。</p>

    <div class="lab-divider"></div>
    <h3>Online Softmax：第二块出现更大分数时怎样修正旧结果</h3>
    <p class="lab-intro">把一行 score 分成 [1,2] 与 [4,−1]，对应 Value 为 [10,20] 与 [30,40]。先只处理块 1，再合并块 2；观察旧累计量乘上 exp(2−4)，最终仍得到四个 score 的精确 softmax。</p>

    <div class="block-buttons" aria-label="Online softmax 已处理的数据块">
      <button type="button" :class="{ active: processedBlocks === 1 }" @click="processedBlocks = 1">只处理块 1</button>
      <button type="button" :class="{ active: processedBlocks === 2 }" @click="processedBlocks = 2">合并块 2</button>
    </div>

    <div class="softmax-blocks">
      <div v-for="(block,index) in blocks" :key="block.label" :class="{ pending: index >= processedBlocks }">
        <strong>{{ block.label }}</strong>
        <span v-for="(score,j) in block.scores" :key="score">s={{ score }}<small>V={{ block.values[j] }}</small></span>
      </div>
    </div>

    <div class="metric-grid online-metrics">
      <div class="metric-card"><small>运行最大值 m</small><strong>{{ online.maximum.toFixed(0) }}</strong><span>{{ processedBlocks === 2 ? 'max(2,4)' : '块 1 的最大值' }}</span></div>
      <div class="metric-card"><small>运行分母 ℓ</small><strong>{{ online.denominator.toFixed(4) }}</strong><span>Σ exp(s−m)</span></div>
      <div class="metric-card"><small>当前加权输出 O</small><strong>{{ online.output.toFixed(3) }}</strong><span>未保存完整概率矩阵</span></div>
    </div>

    <div class="weight-row" aria-label="当前已处理分数的 softmax 权重">
      <span v-for="(weight,index) in online.weights" :key="index"><i :style="{ height: `${Math.max(4,weight * 100)}%` }"></i><strong>{{ (weight * 100).toFixed(1) }}%</strong><small>V={{ online.seenValues[index] }}</small></span>
    </div>

    <p class="decision-note" v-if="processedBlocks === 1"><strong>现在还不是最终答案：</strong>块 1 内部归一化后输出约 {{ online.output.toFixed(3) }}。模型尚未看到 score=4，因此不能把这两个权重直接写死。</p>
    <p class="decision-note" v-else><strong>合并时的关键：</strong>全局最大值从 2 变 4，旧分母与旧分子都先乘 exp(2−4)={{ online.oldScale.toFixed(4) }}，再加入块 2。最终权重与一次性对 [1,2,4,−1] 做 softmax 相同；“精确”来自代数重缩放，不是逐步逼近。</p>
  </section>
</template>

<style scoped>
.flash-lab { container-type:inline-size; }
.preset-row,.block-buttons { display:flex; flex-wrap:wrap; gap:.45rem; margin:.8rem 0; }
.preset-row button,.block-buttons button { padding:.4rem .65rem; border:1px solid var(--line); border-radius:8px; color:var(--vp-c-text-1); background:var(--vp-c-bg); cursor:pointer; font-size:.61rem; }
.preset-row button:hover,.block-buttons button:hover { border-color:var(--ink); }
.preset-row button.active,.block-buttons button.active { border-color:var(--ink); color:var(--vp-c-bg); background:var(--ink); }
.control-grid { display:grid; grid-template-columns:1fr 1fr; gap:.55rem; }
.control-grid label { display:grid; grid-template-columns:115px 1fr 46px; gap:.45rem; align-items:center; padding:.52rem .62rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg); }
.control-grid span,.control-grid output { font-size:.6rem; }
.control-grid span { font-weight:750; }
.control-grid output { color:var(--coral); text-align:right; font-family:var(--vp-font-family-mono); }
.control-grid select { width:100%; }
.causal-toggle { display:flex; gap:.48rem; align-items:center; margin:.75rem 0; color:var(--ink-muted); font-size:.62rem; }
.metric-grid { grid-template-columns:repeat(3,1fr); }
.workspace-bars { display:grid; gap:.62rem; margin:1rem 0; }
.workspace-bars>div { display:grid; grid-template-columns:110px 1fr 78px; gap:.55rem; align-items:center; font-size:.59rem; }
.workspace-bars i { display:block; height:12px; overflow:hidden; border-radius:999px; background:var(--vp-c-bg-soft); }
.workspace-bars b { display:block; height:100%; border-radius:inherit; background:var(--coral); }
.workspace-bars b.flash { background:var(--mint); }
.workspace-bars output { text-align:right; font-family:var(--vp-font-family-mono); }
.ledger-note,.decision-note { padding:.72rem .82rem; border-radius:10px; color:var(--ink-muted); background:var(--vp-c-bg); font-size:.65rem; line-height:1.58; }
.ledger-note strong,.decision-note strong { color:var(--vp-c-text-1); }
.lab-divider { height:1px; margin:1.3rem 0; background:var(--line); }
.softmax-blocks { display:grid; grid-template-columns:1fr 1fr; gap:.7rem; margin:.8rem 0; }
.softmax-blocks>div { display:grid; grid-template-columns:70px 1fr 1fr; gap:.45rem; align-items:center; padding:.7rem; border:1px solid var(--line); border-radius:11px; background:var(--vp-c-bg); }
.softmax-blocks>div.pending { opacity:.38; }
.softmax-blocks>div>strong { font-size:.65rem; }
.softmax-blocks span { display:grid; place-items:center; min-height:48px; border-radius:8px; background:var(--vp-c-bg-soft); font:750 .65rem var(--vp-font-family-mono); }
.softmax-blocks small { color:var(--ink-muted); font-size:.51rem; }
.weight-row { display:grid; grid-template-columns:repeat(4,1fr); gap:.5rem; height:120px; margin:1rem 0; align-items:end; }
.weight-row span { display:grid; grid-template-rows:1fr auto auto; gap:.15rem; height:100%; place-items:center; font-size:.56rem; }
.weight-row i { width:26px; align-self:end; border-radius:7px 7px 2px 2px; background:var(--coral); transition:height .18s ease; }
.weight-row strong { font-family:var(--vp-font-family-mono); }
.weight-row small { color:var(--ink-muted); }
@container (max-width:620px) { .control-grid,.softmax-blocks { grid-template-columns:1fr; } .metric-grid { grid-template-columns:1fr; } }
@container (max-width:430px) { .control-grid label { grid-template-columns:1fr 44px; } .control-grid input,.control-grid select { grid-column:1; } .control-grid output { grid-column:2; grid-row:2; } .workspace-bars>div { grid-template-columns:82px 1fr 64px; } }
</style>
