<script setup lang="ts">
import { computed, ref, watch } from 'vue'

type Preset = {
  label: string
  note: string
  layers: number
  qHeads: number
  kvHeads: number
  headDim: number
  tokens: number
}

const presets: Preset[] = [
  { label: 'MHA', note: '每个 Query head 独享 K/V', layers: 32, qHeads: 32, kvHeads: 32, headDim: 128, tokens: 8192 },
  { label: 'GQA', note: '4 个 Query heads 共用一组 K/V', layers: 32, qHeads: 32, kvHeads: 8, headDim: 128, tokens: 8192 },
  { label: 'MQA', note: '所有 Query heads 共用一组 K/V', layers: 32, qHeads: 32, kvHeads: 1, headDim: 128, tokens: 8192 },
  { label: '长上下文 GQA', note: '把上下文拉到 128K', layers: 32, qHeads: 32, kvHeads: 8, headDim: 128, tokens: 131072 }
]

const layers = ref(32)
const queryHeads = ref(32)
const kvHeads = ref(8)
const headDim = ref(128)
const tokens = ref(8192)
const bytesPerElement = ref(2)

const possibleQueryHeads = [8, 16, 24, 32, 40, 48, 64, 80, 96, 128]
const possibleKvHeads = computed(() => [1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 80, 96, 128]
  .filter((value) => value <= queryHeads.value && queryHeads.value % value === 0))

watch(queryHeads, () => {
  if (!possibleKvHeads.value.includes(kvHeads.value)) kvHeads.value = queryHeads.value
})

const modelDim = computed(() => queryHeads.value * headDim.value)
const kvWidth = computed(() => kvHeads.value * headDim.value)
const groupSize = computed(() => queryHeads.value / kvHeads.value)
const cacheBytes = computed(() => 2 * layers.value * tokens.value * kvHeads.value * headDim.value * bytesPerElement.value)
const mhaCacheBytes = computed(() => 2 * layers.value * tokens.value * queryHeads.value * headDim.value * bytesPerElement.value)
const savedFraction = computed(() => 1 - cacheBytes.value / mhaCacheBytes.value)
const bytesPerNewToken = computed(() => 2 * layers.value * kvHeads.value * headDim.value * bytesPerElement.value)

const qParams = computed(() => modelDim.value * modelDim.value)
const kvParams = computed(() => 2 * modelDim.value * kvWidth.value)
const oParams = computed(() => modelDim.value * modelDim.value)
const attnParamsPerLayer = computed(() => qParams.value + kvParams.value + oParams.value)
const mhaParamsPerLayer = computed(() => 4 * modelDim.value * modelDim.value)

const visibleHeads = computed(() => Array.from({ length: Math.min(queryHeads.value, 32) }, (_, index) => ({
  index,
  group: Math.floor(index / groupSize.value),
  color: `hsl(${222 + (Math.floor(index / groupSize.value) * 37) % 118} 68% ${54 + (Math.floor(index / groupSize.value) % 2) * 7}%)`
})))

function applyPreset(preset: Preset) {
  layers.value = preset.layers
  queryHeads.value = preset.qHeads
  kvHeads.value = preset.kvHeads
  headDim.value = preset.headDim
  tokens.value = preset.tokens
}

function formatBytes(value: number) {
  const gib = value / 1024 ** 3
  if (gib >= 1024) return `${(gib / 1024).toFixed(2)} TiB`
  if (gib >= 1) return `${gib.toFixed(gib < 10 ? 2 : 1)} GiB`
  const mib = value / 1024 ** 2
  return `${mib.toFixed(mib < 10 ? 2 : 1)} MiB`
}

function formatParams(value: number) {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`
  return value.toLocaleString()
}
</script>

<template>
  <section class="lab-shell decoder-lab" aria-labelledby="decoder-lab-title">
    <h3 id="decoder-lab-title">现代 Decoder 实验：GQA 到底省了哪一笔？</h3>
    <p class="lab-intro">Query head 数决定模型提出多少组问题；KV head 数决定历史中要保存多少组 Key/Value。切换 MHA、GQA、MQA，观察参数和 KV Cache 分开变化。</p>

    <div class="preset-grid" aria-label="Attention 头配置预设">
      <button v-for="preset in presets" :key="preset.label" type="button" @click="applyPreset(preset)">
        <strong>{{ preset.label }}</strong><span>{{ preset.note }}</span>
      </button>
    </div>

    <div class="lab-grid">
      <div class="controls-card">
        <label><span>Transformer 层数 L</span><input v-model.number="layers" type="range" min="1" max="96" step="1"><output>{{ layers }}</output></label>
        <label><span>Query heads h</span><select v-model.number="queryHeads"><option v-for="value in possibleQueryHeads" :key="value" :value="value">{{ value }}</option></select><output>d={{ modelDim }}</output></label>
        <label><span>KV heads h<sub>kv</sub></span><select v-model.number="kvHeads"><option v-for="value in possibleKvHeads" :key="value" :value="value">{{ value }}</option></select><output>{{ groupSize }} Q / KV</output></label>
        <label><span>每头维度 d<sub>h</sub></span><select v-model.number="headDim"><option :value="64">64</option><option :value="128">128</option><option :value="256">256</option></select><output>{{ headDim }}</output></label>
        <label><span>缓存上下文 T</span><input v-model.number="tokens" type="range" min="1024" max="131072" step="1024"><output>{{ (tokens / 1024).toFixed(0) }}K</output></label>
        <label><span>KV 精度</span><select v-model.number="bytesPerElement"><option :value="2">BF16 / FP16</option><option :value="1">INT8</option><option :value="4">FP32</option></select><output>{{ bytesPerElement }} byte</output></label>
      </div>

      <div class="visual-card">
        <div class="head-map">
          <div class="map-label"><strong>{{ queryHeads }} 个 Query heads</strong><span v-if="queryHeads > 32">图中只画前 32 个</span></div>
          <div class="query-row">
            <i v-for="head in visibleHeads" :key="head.index" :style="{ background: head.color }" :title="`Query head ${head.index + 1} → KV group ${head.group + 1}`"></i>
          </div>
          <div class="connector-copy">每 {{ groupSize }} 个 Query heads 共用 1 组 K/V</div>
          <div class="kv-row">
            <i v-for="index in Math.min(kvHeads, 32)" :key="index" :style="{ background: `hsl(${222 + ((index - 1) * 37) % 118} 68% ${54 + ((index - 1) % 2) * 7}%)` }"></i>
          </div>
          <div class="map-label"><strong>{{ kvHeads }} 组 K/V</strong><span>每层、每个历史 token 都要保存</span></div>
        </div>

        <div class="formula-card">
          <small>KV Cache</small>
          <code>2 × L × T × h<sub>kv</sub> × d<sub>h</sub> × bytes</code>
          <strong>{{ formatBytes(cacheBytes) }}</strong>
        </div>
      </div>
    </div>

    <div class="metric-grid">
      <div><small>当前 KV Cache</small><strong>{{ formatBytes(cacheBytes) }}</strong><span>只算单 batch 的 K/V，不含权重与激活</span></div>
      <div><small>同配置 MHA</small><strong>{{ formatBytes(mhaCacheBytes) }}</strong><span>令 h<sub>kv</sub>=h={{ queryHeads }}</span></div>
      <div><small>缓存节省</small><strong>{{ (savedFraction * 100).toFixed(1) }}%</strong><span>每新增 token 增加 {{ formatBytes(bytesPerNewToken) }}</span></div>
      <div><small>每层 Attention 参数</small><strong>{{ formatParams(attnParamsPerLayer) }}</strong><span>MHA 基线 {{ formatParams(mhaParamsPerLayer) }}</span></div>
    </div>

    <div class="boundary-grid">
      <div><strong>省下来的</strong><span>K/V 投影宽度、每 token 的 KV Cache、Decode 时要读取的 K/V 字节。</span></div>
      <div><strong>没有自动省下来的</strong><span>Query 数、Q/O 投影、每个 Query 对历史位置的打分，以及任何质量损失风险。</span></div>
    </div>
    <p class="teach-note"><strong>边界：</strong>公式假设普通缓存布局、单 batch、所有层同样的头配置。分页缓存、量化 scale、张量对齐、滑窗、MLA 或混合层都会改变真实占用；实验只负责把 GQA 的核心账算清楚。</p>
  </section>
</template>

<style scoped>
.decoder-lab { container-type:inline-size; }
.preset-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.55rem; margin:1rem 0; }
.preset-grid button { padding:.72rem; border:1px solid var(--line); border-radius:10px; color:var(--vp-c-text-1); background:var(--vp-c-bg); text-align:left; cursor:pointer; }
.preset-grid button:hover { border-color:var(--vp-c-brand-1); transform:translateY(-1px); }
.preset-grid strong,.preset-grid span { display:block; }
.preset-grid strong { font-size:.76rem; }.preset-grid span { margin-top:.16rem; color:var(--ink-muted); font-size:.61rem; line-height:1.45; }
.lab-grid { display:grid; grid-template-columns:minmax(280px,.8fr) minmax(340px,1.2fr); gap:.8rem; }
.controls-card,.visual-card { padding:.9rem; border:1px solid var(--line); border-radius:13px; background:var(--vp-c-bg); }
.controls-card { display:grid; gap:.72rem; align-content:start; }
.controls-card label { display:grid; grid-template-columns:130px 1fr auto; align-items:center; gap:.45rem; }
.controls-card span { font-size:.69rem; font-weight:700; }.controls-card output { min-width:58px; color:#596fe5; font:650 .65rem var(--vp-font-family-mono); text-align:right; }
.controls-card select { min-width:0; padding:.42rem; border:1px solid var(--line); border-radius:8px; color:var(--vp-c-text-1); background:var(--vp-c-bg-soft); font-size:.69rem; }.controls-card input { width:100%; accent-color:#596fe5; }
.visual-card { display:grid; gap:.9rem; }.head-map { padding:.8rem; border-radius:11px; background:var(--vp-c-bg-soft); }.map-label { display:flex; justify-content:space-between; gap:1rem; align-items:center; font-size:.68rem; }.map-label span,.connector-copy { color:var(--ink-muted); font-size:.57rem; }
.query-row,.kv-row { display:grid; grid-template-columns:repeat(16,1fr); gap:.22rem; margin:.55rem 0; }.query-row i,.kv-row i { height:18px; border-radius:4px; box-shadow:inset 0 0 0 1px rgba(255,255,255,.45); }.kv-row { grid-template-columns:repeat(16,1fr); }.connector-copy { padding:.35rem; border-top:1px dashed var(--line); border-bottom:1px dashed var(--line); text-align:center; }
.formula-card { display:grid; grid-template-columns:1fr auto; gap:.25rem .7rem; align-items:end; padding:.8rem; border-radius:11px; background:color-mix(in srgb,#596fe5 8%,var(--vp-c-bg-soft)); }.formula-card small { color:var(--ink-muted); font-size:.6rem; }.formula-card code { grid-column:1; color:var(--vp-c-text-1); background:transparent; font-size:.66rem; }.formula-card strong { grid-column:2; grid-row:1/3; color:#596fe5; font:750 1.45rem var(--vp-font-family-mono); }
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.55rem; margin:1rem 0; }.metric-grid>div { padding:.72rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg); }.metric-grid small,.metric-grid strong,.metric-grid span { display:block; }.metric-grid small,.metric-grid span { color:var(--ink-muted); font-size:.57rem; line-height:1.45; }.metric-grid strong { margin:.2rem 0; font:750 1rem var(--vp-font-family-mono); }
.boundary-grid { display:grid; grid-template-columns:1fr 1fr; gap:.65rem; }.boundary-grid>div { padding:.75rem; border-left:4px solid #2da68d; border-radius:0 10px 10px 0; background:rgba(45,166,141,.08); }.boundary-grid>div+div { border-left-color:#e36c48; background:rgba(227,108,72,.08); }.boundary-grid strong,.boundary-grid span { display:block; }.boundary-grid strong { font-size:.7rem; }.boundary-grid span { margin-top:.2rem; color:var(--ink-muted); font-size:.62rem; line-height:1.55; }
@media (max-width:820px) { .preset-grid,.metric-grid { grid-template-columns:repeat(2,1fr); }.lab-grid { grid-template-columns:1fr; } }
@media (max-width:520px) { .preset-grid,.metric-grid,.boundary-grid { grid-template-columns:1fr; }.controls-card label { grid-template-columns:1fr auto; }.controls-card label input,.controls-card label select { grid-column:1; }.query-row,.kv-row { grid-template-columns:repeat(8,1fr); } }
</style>
