<script setup lang="ts">
import { computed, ref } from 'vue'

const parameters = ref(7)
const gpuMemory = ref(24)
const formatIndex = ref(1)
const formats = [
  { name: 'FP16', bits: 16, note: '常见训练/推理基线' },
  { name: 'INT8', bits: 8, note: '权重约缩小到一半' },
  { name: 'INT4', bits: 4, note: '权重约缩小到四分之一' }
]

const format = computed(() => formats[formatIndex.value])
const weightGiB = computed(() => parameters.value * 1e9 * format.value.bits / 8 / 1024 ** 3)
const fp16GiB = computed(() => parameters.value * 1e9 * 2 / 1024 ** 3)
const compression = computed(() => fp16GiB.value / weightGiB.value)
const minGpus = computed(() => Math.max(1, Math.ceil(weightGiB.value / gpuMemory.value)))
const headroom = computed(() => gpuMemory.value * minGpus.value - weightGiB.value)
const formatGiB = (value: number) => value < 1 ? `${(value * 1024).toFixed(0)} MiB` : `${value.toFixed(1)} GiB`
</script>

<template>
  <section class="lab-shell quant-lab" aria-labelledby="quant-title">
    <h3 id="quant-title">量化与权重显存实验</h3>
    <p class="lab-intro">先只计算模型权重：同一个参数量改用 FP16、INT8 或 INT4，需要多少显存？这不是整台服务的总显存。</p>

    <div class="format-pills" role="group" aria-label="选择权重精度">
      <button v-for="(item, index) in formats" :key="item.name" :class="{ active: formatIndex === index }" @click="formatIndex = index">
        <strong>{{ item.name }}</strong><small>{{ item.bits }} bit</small>
      </button>
    </div>

    <div class="control-row"><label for="quant-params">模型参数量</label><input id="quant-params" v-model.number="parameters" type="range" min="0.5" max="80" step="0.5"><output>{{ parameters.toFixed(1) }}B</output></div>
    <div class="control-row"><label for="quant-gpu">单卡显存</label><input id="quant-gpu" v-model.number="gpuMemory" type="range" min="16" max="192" step="8"><output>{{ gpuMemory }} GiB</output></div>

    <div class="metric-grid">
      <div class="metric-card"><small>{{ format.name }} 权重</small><strong>{{ formatGiB(weightGiB) }}</strong><span>{{ format.note }}</span></div>
      <div class="metric-card"><small>相对 FP16 压缩</small><strong>{{ compression.toFixed(1) }}×</strong><span>只比较权重字节数</span></div>
      <div class="metric-card"><small>至少需要</small><strong>{{ minGpus }} 张卡</strong><span>剩余约 {{ formatGiB(headroom) }}</span></div>
    </div>

    <p class="teach-note"><strong>别把“权重装得下”当成“服务跑得动”：</strong>真实推理还要为 KV Cache、激活、临时 workspace、量化比例尺和显存碎片留空间；低 bit 也需要匹配的硬件与 kernel 才可能转化为速度。</p>
  </section>
</template>

<style scoped>
.format-pills { display:grid; grid-template-columns:repeat(3,1fr); gap:.6rem; margin:1rem 0 1.25rem; }
.format-pills button { display:grid; gap:.18rem; padding:.8rem; border:1px solid var(--line); border-radius:10px; color:var(--ink); background:var(--vp-c-bg); cursor:pointer; }
.format-pills button.active { border-color:var(--coral); box-shadow:inset 0 -3px var(--coral); background:rgba(227,108,72,.06); }
.format-pills small { color:var(--ink-muted); font:.66rem var(--vp-font-family-mono); }
@media (max-width:620px) { .format-pills { grid-template-columns:1fr; } }
</style>
