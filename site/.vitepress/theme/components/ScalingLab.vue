<script setup lang="ts">
import { computed, ref, watch } from 'vue'

type FfnType = 'standard' | 'swiglu'

type Preset = {
  label: string
  note: string
  vocab: number
  layers: number
  dim: number
  heads: number
  kvHeads: number
  ffn: number
  ffnType: FfnType
  tied: boolean
  tokens: number
}

const presets: Preset[] = [
  { label: '教学小模型', note: '先看懂每一笔从哪里来', vocab: 8192, layers: 6, dim: 512, heads: 8, kvHeads: 8, ffn: 2048, ffnType: 'standard', tied: true, tokens: 1 },
  { label: 'GPT-2 124M 式', note: '经典 MHA + 两层 FFN', vocab: 50257, layers: 12, dim: 768, heads: 12, kvHeads: 12, ffn: 3072, ffnType: 'standard', tied: true, tokens: 10 },
  { label: 'LLaMA 7B 式', note: 'SwiGLU + MHA，约 6.7B', vocab: 32000, layers: 32, dim: 4096, heads: 32, kvHeads: 32, ffn: 11008, ffnType: 'swiglu', tied: false, tokens: 140 },
  { label: 'GQA 8B 式', note: '8 个 KV heads，观察 Attention 变小', vocab: 128256, layers: 32, dim: 4096, heads: 32, kvHeads: 8, ffn: 14336, ffnType: 'swiglu', tied: false, tokens: 160 }
]

const vocabSize = ref(32000)
const layers = ref(32)
const modelDim = ref(4096)
const heads = ref(32)
const kvHeads = ref(32)
const ffnDim = ref(11008)
const ffnType = ref<FfnType>('swiglu')
const tiedEmbeddings = ref(false)
const tokens = ref(140)
const utilization = ref(40)

const dimensionOptions = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 8192, 12288, 16384]
const possibleHeads = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128]
const headOptions = computed(() => possibleHeads.filter((value) => modelDim.value % value === 0 && modelDim.value / value >= 64))
const kvHeadOptions = computed(() => possibleHeads.filter((value) => value <= heads.value && heads.value % value === 0))

watch(modelDim, (next) => {
  if (!headOptions.value.includes(heads.value)) heads.value = headOptions.value.at(-1) ?? 1
  if (ffnDim.value < next || ffnDim.value > next * 8) ffnDim.value = next * 4
})

watch(heads, () => {
  if (!kvHeadOptions.value.includes(kvHeads.value)) kvHeads.value = heads.value
})

const headDim = computed(() => modelDim.value / heads.value)
const kvDim = computed(() => headDim.value * kvHeads.value)

const tokenEmbeddingParams = computed(() => vocabSize.value * modelDim.value)
const outputHeadParams = computed(() => tiedEmbeddings.value ? 0 : vocabSize.value * modelDim.value)
const embeddingParams = computed(() => tokenEmbeddingParams.value + outputHeadParams.value)

const queryParams = computed(() => modelDim.value * modelDim.value)
const keyParams = computed(() => modelDim.value * kvDim.value)
const valueParams = computed(() => modelDim.value * kvDim.value)
const outputProjectionParams = computed(() => modelDim.value * modelDim.value)
const attentionPerBlock = computed(() => queryParams.value + keyParams.value + valueParams.value + outputProjectionParams.value)
const ffnMatrices = computed(() => ffnType.value === 'swiglu' ? 3 : 2)
const ffnPerBlock = computed(() => ffnMatrices.value * modelDim.value * ffnDim.value)
const normPerBlock = computed(() => 2 * modelDim.value)
const blockParams = computed(() => attentionPerBlock.value + ffnPerBlock.value + normPerBlock.value)

const attentionParams = computed(() => attentionPerBlock.value * layers.value)
const ffnParams = computed(() => ffnPerBlock.value * layers.value)
const normParams = computed(() => normPerBlock.value * layers.value + modelDim.value)
const transformerBodyParams = computed(() => attentionParams.value + ffnParams.value + normParams.value)
const totalParams = computed(() => embeddingParams.value + transformerBodyParams.value)

const parts = computed(() => [
  { label: tiedEmbeddings.value ? '词嵌入（输出层共享）' : '词嵌入 + 输出层', value: embeddingParams.value, color: '#4c6fff' },
  { label: `${layers.value} 层 Attention`, value: attentionParams.value, color: '#8b5cf6' },
  { label: `${layers.value} 层 FFN`, value: ffnParams.value, color: '#ed8c3b' },
  { label: 'RMSNorm', value: normParams.value, color: '#28a68a' }
].map((part) => ({ ...part, share: part.value / totalParams.value * 100 })))

const tokensPerParameter = computed(() => tokens.value / (totalParams.value / 1e9))
const computeFlops = computed(() => 6 * transformerBodyParams.value * tokens.value * 1e9)
const gpuDays = computed(() => computeFlops.value / (312e12 * utilization.value / 100 * 86400))
const targetTokens = computed(() => totalParams.value / 1e9 * 20)
const weightBytes = computed(() => totalParams.value * 2)
const gradientBytes = computed(() => totalParams.value * 2)
const adamBytes = computed(() => totalParams.value * 8)
const trainingStateBytes = computed(() => weightBytes.value + gradientBytes.value + adamBytes.value)

const regime = computed(() => {
  if (tokensPerParameter.value < 10) return { label: '参数偏多、数据偏少', tone: 'warn', copy: '每个参数看到的 token 较少，模型可能还没有被充分训练。' }
  if (tokensPerParameter.value <= 30) return { label: '接近教学平衡区', tone: 'good', copy: '这里用约 20 token/参数建立 Chinchilla 风格的预算直觉。' }
  return { label: '数据更充足、计算更重', tone: 'rich', copy: '更多 token 增加覆盖面，也会线性增加训练计算。' }
})

function applyPreset(preset: Preset) {
  vocabSize.value = preset.vocab
  modelDim.value = preset.dim
  layers.value = preset.layers
  heads.value = preset.heads
  kvHeads.value = preset.kvHeads
  ffnDim.value = preset.ffn
  ffnType.value = preset.ffnType
  tiedEmbeddings.value = preset.tied
  tokens.value = preset.tokens
}

function formatParams(value: number) {
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)}T`
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return value.toLocaleString()
}

function formatBytes(value: number) {
  const gib = value / 1024 ** 3
  if (gib >= 1024) return `${(gib / 1024).toFixed(2)} TiB`
  return `${gib.toFixed(gib < 10 ? 2 : 1)} GiB`
}

function formatCompute(value: number) {
  if (value >= 1e21) return `${(value / 1e21).toFixed(2)} ZFLOPs`
  if (value >= 1e18) return `${(value / 1e18).toFixed(2)} EFLOPs`
  return `${(value / 1e15).toFixed(1)} PFLOPs`
}

function formatGpuTime(days: number) {
  if (days >= 1) return `${days.toFixed(0)} GPU·天`
  if (days * 24 >= 1) return `${(days * 24).toFixed(1)} GPU·小时`
  return `${(days * 24 * 60).toFixed(0)} GPU·分钟`
}

function formatTokenBillions(value: number) {
  if (value >= 1000) return `${(value / 1000).toFixed(2)}T`
  if (value >= 1) return `${value.toFixed(value < 10 ? 2 : 0)}B`
  return `${(value * 1000).toFixed(0)}M`
}
</script>

<template>
  <section class="lab-shell scaling-lab" aria-labelledby="scaling-title">
    <h3 id="scaling-title">Scaling 实验：先把参数量算出来</h3>
    <p class="lab-intro">选择一套架构后，实验会逐笔计算 Embedding、Attention、FFN 与 Norm；得到的参数量会自动进入显存与 <strong>6ND</strong> 预算。所有公式按现代 bias-free Decoder-only Transformer 做教学近似。</p>

    <div class="preset-grid" aria-label="模型预设">
      <button v-for="preset in presets" :key="preset.label" type="button" @click="applyPreset(preset)">
        <strong>{{ preset.label }}</strong><span>{{ preset.note }}</span>
      </button>
    </div>

    <div class="step-title"><span>1</span><div><strong>搭模型</strong><small>先决定矩阵的形状</small></div></div>
    <div class="builder-grid">
      <div class="builder-controls">
        <label class="compact-control"><span>词表大小 V</span><input v-model.number="vocabSize" type="number" min="4096" max="262144" step="256"><small>每个 token 需要一行向量</small></label>
        <label class="compact-control"><span>Transformer 层数 L</span><input v-model.number="layers" type="range" min="1" max="128" step="1"><output>{{ layers }} 层</output></label>
        <label class="compact-control"><span>隐藏维度 d</span><select v-model.number="modelDim"><option v-for="value in dimensionOptions" :key="value" :value="value">{{ value.toLocaleString() }}</option></select><small>残差流中每个 token 的宽度</small></label>
        <label class="compact-control"><span>Query heads h</span><select v-model.number="heads"><option v-for="value in headOptions" :key="value" :value="value">{{ value }}</option></select><small>每个 head 维度：{{ headDim }}</small></label>
        <label class="compact-control"><span>KV heads h<sub>kv</sub></span><select v-model.number="kvHeads"><option v-for="value in kvHeadOptions" :key="value" :value="value">{{ value }}</option></select><small>{{ kvHeads === heads ? 'MHA：每个 Query head 有自己的 K/V' : `GQA：${heads / kvHeads} 个 Query heads 共享一组 K/V` }}</small></label>
        <label class="compact-control"><span>FFN 中间维度 d<sub>ff</sub></span><input :key="`ffn-${modelDim}`" v-model.number="ffnDim" type="range" :min="modelDim" :max="modelDim * 8" step="256"><output>{{ ffnDim.toLocaleString() }}（{{ (ffnDim / modelDim).toFixed(2) }}d）</output></label>

        <fieldset class="choice-control">
          <legend>FFN 类型</legend>
          <button type="button" :class="{ active: ffnType === 'standard' }" @click="ffnType = 'standard'">GELU / ReLU<span>2 个矩阵</span></button>
          <button type="button" :class="{ active: ffnType === 'swiglu' }" @click="ffnType = 'swiglu'">SwiGLU<span>3 个矩阵</span></button>
        </fieldset>

        <label class="toggle-control"><input v-model="tiedEmbeddings" type="checkbox"><span><strong>输入、输出共享词表矩阵</strong><small>开启后不再单独计算 LM Head 的 V × d</small></span></label>
      </div>

      <div class="parameter-receipt">
        <div class="receipt-head"><span>参数小票</span><strong>{{ formatParams(totalParams) }}</strong></div>
        <div class="mix-bar" aria-label="参数量构成">
          <span v-for="part in parts" :key="part.label" :style="{ width: `${part.share}%`, background: part.color }" :title="`${part.label} ${part.share.toFixed(1)}%`"></span>
        </div>
        <div class="receipt-list">
          <div v-for="part in parts" :key="part.label"><i :style="{ background: part.color }"></i><span>{{ part.label }}</span><strong>{{ formatParams(part.value) }}</strong><small>{{ part.share.toFixed(1) }}%</small></div>
        </div>

        <div class="block-receipt">
          <p><strong>只看 1 个 Transformer Block</strong><span>最后再乘 {{ layers }} 层</span></p>
          <div><span>Attention</span><code>{{ formatParams(attentionPerBlock) }}</code><small>Q {{ formatParams(queryParams) }} + K {{ formatParams(keyParams) }} + V {{ formatParams(valueParams) }} + O {{ formatParams(outputProjectionParams) }}</small></div>
          <div><span>{{ ffnType === 'swiglu' ? 'SwiGLU FFN' : '普通 FFN' }}</span><code>{{ formatParams(ffnPerBlock) }}</code><small>{{ ffnMatrices }} × d × d<sub>ff</sub></small></div>
          <div><span>2 个 RMSNorm</span><code>{{ formatParams(normPerBlock) }}</code><small>2 × d；和矩阵相比通常很小</small></div>
        </div>
      </div>
    </div>

    <div class="formula-strip">
      <div><small>Embedding</small><code>{{ tiedEmbeddings ? 'Vd' : '2Vd' }}</code></div>
      <b>+</b>
      <div><small>重复 L 次</small><code>L × (Attention + FFN + 2d)</code></div>
      <b>+</b>
      <div><small>最终 Norm</small><code>d</code></div>
      <b>=</b>
      <div class="formula-total"><small>总参数</small><code>{{ totalParams.toLocaleString() }}</code></div>
    </div>

    <p class="insight-card"><strong>最容易数错的地方：</strong>把 {{ heads }} 个 Attention heads 再乘一遍。这里每个 head 只有 {{ headDim }} 维，{{ heads }} × {{ headDim }} = {{ modelDim }}，合起来仍是一个 d 维投影。只有当 KV heads 少于 Query heads 时，K/V 两个矩阵才会变窄；当前是 {{ kvHeads }} / {{ heads }}。</p>

    <div class="step-title"><span>2</span><div><strong>把 N 放进资源账本</strong><small>参数、数据和算力现在联动了</small></div></div>
    <div class="memory-ledger">
      <div><small>BF16 权重</small><strong>{{ formatBytes(weightBytes) }}</strong><span>2 bytes × 总参数</span></div>
      <div><small>BF16 梯度</small><strong>{{ formatBytes(gradientBytes) }}</strong><span>2 bytes × 总参数</span></div>
      <div><small>Adam 一、二阶矩</small><strong>{{ formatBytes(adamBytes) }}</strong><span>8 bytes × 总参数</span></div>
      <div class="ledger-total"><small>训练状态小计</small><strong>{{ formatBytes(trainingStateBytes) }}</strong><span>约 12 bytes/参数；还没算激活</span></div>
    </div>

    <div class="control-row"><label for="scale-tokens">训练 token D</label><input id="scale-tokens" v-model.number="tokens" type="range" min="1" max="3000" step="1"><output>{{ tokens.toLocaleString() }}B</output></div>
    <div class="control-row"><label for="scale-util">GPU 有效利用率</label><input id="scale-util" v-model.number="utilization" type="range" min="20" max="65" step="5"><output>{{ utilization }}%</output></div>

    <div class="metric-grid">
      <div class="metric-card"><small>每个参数看到</small><strong>{{ tokensPerParameter.toFixed(1) }} token</strong><span>教学参考值约 20</span></div>
      <div class="metric-card"><small>近似训练计算</small><strong>{{ formatCompute(computeFlops) }}</strong><span>6 × 非 Embedding 参数 N × D</span></div>
      <div class="metric-card"><small>A100 80G 等效时间</small><strong>{{ formatGpuTime(gpuDays) }}</strong><span>312 TFLOPS × {{ utilization }}%</span></div>
    </div>

    <div class="scaling-verdict" :class="regime.tone">
      <strong>{{ regime.label }}</strong>
      <span>{{ regime.copy }} 当前总参数量的 20× 教学参考数据约为 {{ formatTokenBillions(targetTokens) }} token。</span>
    </div>
    <p class="teach-note"><strong>边界：</strong>参数量公式忽略 bias，并按两个 RMSNorm/Block 计数；12 bytes/参数不含激活、临时张量与通信缓冲。6ND 对短上下文 dense Transformer 是很有用的量级估算，但 MoE、长序列 Attention、重计算和硬件差异都会让真实成本偏离。</p>
  </section>
</template>

<style scoped>
.scaling-lab { container-type: inline-size; }
.preset-grid { display:grid; grid-template-columns:repeat(4, 1fr); gap:.6rem; margin:1rem 0 1.5rem; }
.preset-grid button { padding:.75rem; border:1px solid var(--line); border-radius:11px; color:var(--vp-c-text-1); background:var(--vp-c-bg); text-align:left; cursor:pointer; }
.preset-grid button:hover { border-color:var(--vp-c-brand-1); transform:translateY(-1px); }
.preset-grid strong, .preset-grid span { display:block; }
.preset-grid strong { font-size:.82rem; }
.preset-grid span { margin-top:.22rem; color:var(--ink-muted); font-size:.66rem; line-height:1.45; }
.step-title { display:flex; align-items:center; gap:.7rem; margin:1.4rem 0 .85rem; }
.step-title > span { display:grid; width:2rem; height:2rem; place-items:center; border-radius:50%; color:#fff; background:var(--coral); font:700 .9rem var(--vp-font-family-mono); }
.step-title strong, .step-title small { display:block; }
.step-title strong { font-size:.95rem; }
.step-title small { color:var(--ink-muted); font-size:.7rem; }
.builder-grid { display:grid; grid-template-columns:minmax(280px, .85fr) minmax(330px, 1.15fr); gap:1rem; }
.builder-controls, .parameter-receipt { padding:1rem; border:1px solid var(--line); border-radius:14px; background:var(--vp-c-bg); }
.builder-controls { display:grid; gap:.75rem; align-content:start; }
.compact-control { display:grid; grid-template-columns:125px minmax(0, 1fr); align-items:center; gap:.35rem .7rem; }
.compact-control > span { font-size:.76rem; font-weight:700; }
.compact-control input[type='number'], .compact-control select { min-width:0; width:100%; padding:.45rem .55rem; border:1px solid var(--line); border-radius:8px; color:var(--vp-c-text-1); background:var(--vp-c-bg-soft); font:500 .76rem var(--vp-font-family-mono); }
.compact-control small, .compact-control output { grid-column:2; color:var(--ink-muted); font-size:.65rem; }
.compact-control output { font-family:var(--vp-font-family-mono); }
.choice-control { display:grid; grid-template-columns:1fr 1fr; gap:.4rem; margin:.15rem 0 0; padding:.6rem; border:1px solid var(--line); border-radius:10px; }
.choice-control legend { padding:0 .3rem; font-size:.72rem; font-weight:700; }
.choice-control button { padding:.5rem; border:1px solid var(--line); border-radius:8px; color:var(--vp-c-text-1); background:var(--vp-c-bg-soft); font-size:.72rem; cursor:pointer; }
.choice-control button.active { border-color:var(--coral); background:color-mix(in srgb, var(--coral) 12%, var(--vp-c-bg)); }
.choice-control span { display:block; margin-top:.1rem; color:var(--ink-muted); font-size:.6rem; }
.toggle-control { display:flex; align-items:flex-start; gap:.55rem; padding:.65rem; border-radius:10px; background:var(--vp-c-bg-soft); }
.toggle-control input { margin-top:.2rem; accent-color:var(--coral); }
.toggle-control strong, .toggle-control small { display:block; }
.toggle-control strong { font-size:.73rem; }
.toggle-control small { margin-top:.12rem; color:var(--ink-muted); font-size:.62rem; line-height:1.5; }
.receipt-head { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; }
.receipt-head span { color:var(--ink-muted); font-size:.75rem; font-weight:700; }
.receipt-head strong { color:var(--coral); font:800 clamp(1.65rem, 5cqi, 2.5rem)/1 var(--vp-font-family-mono); }
.mix-bar { display:flex; height:16px; overflow:hidden; margin:.8rem 0; border-radius:99px; background:var(--vp-c-bg-soft); }
.mix-bar span { min-width:2px; transition:width .25s ease; }
.receipt-list { display:grid; gap:.35rem; }
.receipt-list > div { display:grid; grid-template-columns:9px 1fr auto 42px; align-items:center; gap:.5rem; font-size:.7rem; }
.receipt-list i { width:9px; height:9px; border-radius:3px; }
.receipt-list strong { font-family:var(--vp-font-family-mono); }
.receipt-list small { color:var(--ink-muted); text-align:right; }
.block-receipt { margin-top:1rem; padding:.8rem; border-radius:11px; background:var(--vp-c-bg-soft); }
.block-receipt p { display:flex; justify-content:space-between; gap:1rem; margin:0 0 .55rem; font-size:.72rem; }
.block-receipt p span { color:var(--ink-muted); }
.block-receipt > div { display:grid; grid-template-columns:1fr auto; gap:.1rem .6rem; padding:.42rem 0; border-top:1px dashed var(--line); }
.block-receipt > div > span, .block-receipt code { font-size:.7rem; }
.block-receipt code { color:var(--vp-c-text-1); background:transparent; font-weight:700; }
.block-receipt small { grid-column:1 / -1; color:var(--ink-muted); font-size:.61rem; }
.formula-strip { display:flex; align-items:stretch; gap:.45rem; margin:1rem 0; padding:.75rem; overflow-x:auto; border:1px solid var(--line); border-radius:12px; background:var(--vp-c-bg); }
.formula-strip > div { display:grid; min-width:max-content; place-items:center; padding:.45rem .65rem; border-radius:8px; background:var(--vp-c-bg-soft); }
.formula-strip small { color:var(--ink-muted); font-size:.59rem; }
.formula-strip code { color:var(--vp-c-text-1); background:transparent; font-size:.68rem; }
.formula-strip b { align-self:center; color:var(--ink-muted); }
.formula-strip .formula-total { background:color-mix(in srgb, var(--coral) 12%, var(--vp-c-bg)); }
.insight-card { margin:0; padding:.8rem .9rem; border-left:4px solid #8b5cf6; border-radius:0 10px 10px 0; background:color-mix(in srgb, #8b5cf6 9%, var(--vp-c-bg)); color:var(--ink-muted); font-size:.74rem; line-height:1.7; }
.insight-card strong { color:var(--vp-c-text-1); }
.memory-ledger { display:grid; grid-template-columns:repeat(4, 1fr); gap:.55rem; margin:.7rem 0 1.2rem; }
.memory-ledger > div { padding:.75rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg); }
.memory-ledger small, .memory-ledger strong, .memory-ledger span { display:block; }
.memory-ledger small { color:var(--ink-muted); font-size:.63rem; }
.memory-ledger strong { margin:.18rem 0; font:700 1.05rem var(--vp-font-family-mono); }
.memory-ledger span { color:var(--ink-muted); font-size:.58rem; line-height:1.45; }
.memory-ledger .ledger-total { border-color:color-mix(in srgb, var(--coral) 55%, var(--line)); background:color-mix(in srgb, var(--coral) 8%, var(--vp-c-bg)); }
.scaling-verdict { display:grid; gap:.3rem; margin:1rem 0; padding:.9rem 1rem; border-radius:10px; border-left:4px solid var(--coral); background:rgba(227,108,72,.08); }
.scaling-verdict.good { border-left-color:var(--mint); background:rgba(185,216,193,.22); }
.scaling-verdict.rich { border-left-color:var(--gold); background:rgba(231,189,90,.13); }
.scaling-verdict span { color:var(--ink-muted); font-size:.78rem; line-height:1.65; }
@media (max-width: 850px) {
  .preset-grid { grid-template-columns:repeat(2, 1fr); }
  .builder-grid { grid-template-columns:1fr; }
  .memory-ledger { grid-template-columns:repeat(2, 1fr); }
}
@media (max-width: 520px) {
  .preset-grid, .memory-ledger { grid-template-columns:1fr; }
  .compact-control { grid-template-columns:110px minmax(0, 1fr); }
  .receipt-list > div { grid-template-columns:9px 1fr auto; }
  .receipt-list small { display:none; }
}
</style>
