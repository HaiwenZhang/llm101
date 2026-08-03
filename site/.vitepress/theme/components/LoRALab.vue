<script setup lang="ts">
import { computed, ref } from 'vue'

const hidden = ref(4096)
const ffn = ref(11008)
const layers = ref(32)
const rank = ref(16)
const target = ref<'qv' | 'attn' | 'all'>('qv')
const baseBillions = ref(7)
const baseBits = ref(4)
const trainingBytes = ref(12)

const loraPerLayer = computed(() => {
  const square = rank.value * (hidden.value + hidden.value)
  if (target.value === 'qv') return 2 * square
  if (target.value === 'attn') return 4 * square
  const ffnMatrices = 3 * rank.value * (hidden.value + ffn.value)
  return 4 * square + ffnMatrices
})
const fullTargetPerLayer = computed(() => {
  const square = hidden.value ** 2
  if (target.value === 'qv') return 2 * square
  if (target.value === 'attn') return 4 * square
  return 4 * square + 3 * hidden.value * ffn.value
})
const loraParameters = computed(() => loraPerLayer.value * layers.value)
const fullTargetParameters = computed(() => fullTargetPerLayer.value * layers.value)
const trainableRatio = computed(() => loraParameters.value / (baseBillions.value * 1e9) * 100)
const targetRatio = computed(() => loraParameters.value / fullTargetParameters.value * 100)
const baseWeightGib = computed(() => baseBillions.value * 1e9 * baseBits.value / 8 / 2 ** 30)
const adapterCheckpointMib = computed(() => loraParameters.value * 2 / 2 ** 20)
const loraTrainingGib = computed(() => loraParameters.value * trainingBytes.value / 2 ** 30)
const fullTargetTrainingGib = computed(() => fullTargetParameters.value * trainingBytes.value / 2 ** 30)

function millions(value: number) {
  return `${(value / 1e6).toFixed(value >= 1e9 ? 0 : 2)}M`
}
</script>

<template>
  <section class="lab-shell lora-lab" aria-labelledby="lora-lab-title">
    <h3 id="lora-lab-title">LoRA 账本实验：rank 只是一个乘数，target modules 才决定乘多少次</h3>
    <p class="lab-intro">按常见 dense Attention + SwiGLU 教学结构估算。Q/K/V/O 都视为 d×d，FFN 有两个 d→dff 和一个 dff→d；真实 GQA、fused projection 和层间异构必须按 checkpoint shape 重算。</p>

    <div class="controls-grid">
      <label><span>隐藏维 d</span><select v-model.number="hidden"><option :value="2048">2,048</option><option :value="4096">4,096</option><option :value="8192">8,192</option></select></label>
      <label><span>FFN 维 dff</span><select v-model.number="ffn"><option :value="5504">5,504</option><option :value="11008">11,008</option><option :value="28672">28,672</option></select></label>
      <label><span>挂载层数 L</span><input v-model.number="layers" type="range" min="4" max="96" step="4"><output>{{ layers }}</output></label>
      <label><span>低秩 r</span><input v-model.number="rank" type="range" min="1" max="128" step="1"><output>{{ rank }}</output></label>
      <label><span>目标矩阵</span><select v-model="target"><option value="qv">仅 Q / V</option><option value="attn">Q / K / V / O</option><option value="all">Attention + 3 个 FFN</option></select></label>
      <label><span>底座参数</span><input v-model.number="baseBillions" type="range" min="1" max="70" step="1"><output>{{ baseBillions }}B</output></label>
      <label><span>底座存储精度</span><select v-model.number="baseBits"><option :value="4">4 bit</option><option :value="8">8 bit</option><option :value="16">16 bit</option></select></label>
      <label><span>每训练参数状态</span><select v-model.number="trainingBytes"><option :value="12">12 B：权重2+梯度2+Adam8</option><option :value="16">16 B：另含 FP32 master</option></select></label>
    </div>

    <div class="equation-strip">
      <code>每层 LoRA = Σ r(d<sub>in</sub>+d<sub>out</sub>) = {{ loraPerLayer.toLocaleString() }}</code>
      <span>× {{ layers }} 层</span>
      <strong>{{ millions(loraParameters) }}</strong>
    </div>

    <div class="metric-grid">
      <div class="metric-card"><small>LoRA 可训练参数</small><strong>{{ millions(loraParameters) }}</strong><span>占 {{ baseBillions }}B 底座 {{ trainableRatio.toFixed(3) }}%</span></div>
      <div class="metric-card"><small>相对目标矩阵</small><strong>{{ targetRatio.toFixed(2) }}%</strong><span>full target 为 {{ millions(fullTargetParameters) }}</span></div>
      <div class="metric-card"><small>Adapter checkpoint</small><strong>{{ adapterCheckpointMib.toFixed(1) }} MiB</strong><span>仅按 BF16/FP16 参数 2 B</span></div>
      <div class="metric-card"><small>LoRA 训练状态</small><strong>{{ loraTrainingGib.toFixed(2) }} GiB</strong><span>不含激活和底座</span></div>
      <div class="metric-card"><small>若全训这些目标矩阵</small><strong>{{ fullTargetTrainingGib.toFixed(1) }} GiB</strong><span>同一 bytes/参数口径</span></div>
      <div class="metric-card"><small>冻结底座权重</small><strong>{{ baseWeightGib.toFixed(1) }} GiB</strong><span>仅理论 packed {{ baseBits }}-bit</span></div>
    </div>

    <p class="teach-note"><strong>边界：</strong>底座冻结只省它的梯度与优化器状态，不省前向权重，也不自动省激活。真实显存还包括量化 scale/zero-point、反量化 workspace、embedding/head、临时张量、CUDA context 和碎片；4-bit 权重也不会让 Attention 激活变成 4-bit。</p>
  </section>
</template>

<style scoped>
.lora-lab { container-type:inline-size; }
.controls-grid { display:grid; grid-template-columns:1fr 1fr; gap:.62rem; margin:.9rem 0; }
.controls-grid label { display:grid; grid-template-columns:118px 1fr 56px; align-items:center; gap:.45rem; padding:.62rem .72rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg); }
.controls-grid span { font-size:.63rem; font-weight:700; }
.controls-grid output { text-align:right; font:.62rem var(--vp-font-family-mono); }
.controls-grid select { grid-column:2 / 4; min-width:0; padding:.35rem; border:1px solid var(--line); border-radius:7px; color:var(--vp-c-text-1); background:var(--vp-c-bg); font-size:.61rem; }
.equation-strip { display:flex; flex-wrap:wrap; align-items:center; gap:.65rem; margin:1rem 0; padding:.75rem .85rem; border:1px solid var(--line); border-radius:11px; background:var(--vp-c-bg); }
.equation-strip code { flex:1 1 280px; color:var(--vp-c-text-1); background:transparent; font-size:.65rem; }
.equation-strip span { color:var(--ink-muted); font-size:.65rem; }
.equation-strip strong { color:var(--coral); font:800 .92rem var(--vp-font-family-mono); }
.metric-grid { grid-template-columns:repeat(3,1fr); }
@container (max-width:680px) { .controls-grid,.metric-grid { grid-template-columns:1fr 1fr; } }
@container (max-width:460px) { .controls-grid,.metric-grid { grid-template-columns:1fr; } .controls-grid label { grid-template-columns:1fr 58px; } .controls-grid label input,.controls-grid label select { grid-column:1 / -1; grid-row:2; } }
</style>
