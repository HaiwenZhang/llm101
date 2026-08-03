<script setup lang="ts">
import { computed, ref } from 'vue'

const time = ref(0.35)
const size = 9
const clean = Array.from({ length: size * size }, (_, index) => {
  const x = index % size
  const y = Math.floor(index / size)
  const head = (x - 4) ** 2 + (y - 4) ** 2 < 15
  const eyes = (y === 3 && (x === 3 || x === 5))
  const smile = (y === 5 && x >= 3 && x <= 5) || (y === 4 && (x === 2 || x === 6))
  return head ? (eyes || smile ? -1 : .85) : -.75
})
const noise = Array.from({ length: size * size }, (_, index) => {
  const value = Math.sin((index + 3) * 12.9898) * 43758.5453
  return (value - Math.floor(value)) * 2 - 1
})

const alphaBar = computed(() => Math.cos(time.value * Math.PI / 2) ** 2)
const signalScale = computed(() => Math.sqrt(alphaBar.value))
const noiseScale = computed(() => Math.sqrt(1 - alphaBar.value))
const noisy = computed(() => clean.map((value, index) => signalScale.value * value + noiseScale.value * noise[index]))
const snr = computed(() => alphaBar.value / Math.max(1e-6, 1 - alphaBar.value))

function cellStyle(value: number) {
  const amount = Math.max(0, Math.min(100, (value + 1) * 50))
  return { background: `color-mix(in srgb, var(--coral) ${amount.toFixed(1)}%, var(--vp-c-bg))` }
}
</script>

<template>
  <section class="lab-shell diffusion-noise-lab" aria-labelledby="noise-lab-title">
    <h3 id="noise-lab-title">前向加噪实验：模型在随机时刻究竟看见什么</h3>
    <p class="lab-intro">这里用余弦日程演示累计信号比例 ᾱ。左边是固定的干净小图和固定噪声，右边严格按 xₜ = √ᾱ·x₀ + √(1−ᾱ)·ε 混合。</p>

    <div class="preset-row" aria-label="噪声时刻预设">
      <button v-for="preset in [0, .35, .7, 1]" :key="preset" type="button" :class="{ active: time === preset }" @click="time = preset">t={{ preset.toFixed(2) }}</button>
    </div>
    <div class="control-row"><label for="diffusion-time">时间 t</label><input id="diffusion-time" v-model.number="time" type="range" min="0" max="1" step="0.01"><output>{{ time.toFixed(2) }}</output></div>

    <div class="image-flow">
      <div><strong>干净数据 x₀</strong><div class="pixel-grid" role="img" aria-label="干净的九乘九示意图"><i v-for="(value,index) in clean" :key="index" :style="cellStyle(value)"></i></div></div>
      <div class="operator"><span>× {{ signalScale.toFixed(3) }}</span><b>＋</b><span>噪声 × {{ noiseScale.toFixed(3) }}</span></div>
      <div><strong>当前 xₜ</strong><div class="pixel-grid" role="img" :aria-label="`时间 ${time.toFixed(2)} 的带噪示意图`"><i v-for="(value,index) in noisy" :key="index" :style="cellStyle(value)"></i></div></div>
    </div>

    <div class="metric-grid">
      <div class="metric-card"><small>信号系数</small><strong>{{ signalScale.toFixed(3) }}</strong><span>乘在 x₀ 上</span></div>
      <div class="metric-card"><small>噪声系数</small><strong>{{ noiseScale.toFixed(3) }}</strong><span>乘在 ε 上</span></div>
      <div class="metric-card"><small>SNR</small><strong>{{ snr > 999 ? '∞' : snr.toFixed(2) }}</strong><span>ᾱ / (1−ᾱ)</span></div>
    </div>

    <p class="decision-note"><strong>训练样本：</strong>随机抽一张 x₀、一个时刻 t 和一份已知噪声 ε，构造右图后让网络预测这份 ε。因为答案已知，一步就能计算监督损失；这不是让网络“凭空猜原图”。</p>
    <p class="teach-note"><strong>试一试：</strong>在 t=0.35 时轮廓仍明显；到 t=0.70，单个像素已经很难判断，但网络会利用整幅图和文字条件；t=1.00 时信号系数为 0，输入只剩噪声。</p>
  </section>
</template>

<style scoped>
.diffusion-noise-lab { container-type:inline-size; }
.preset-row { display:flex; flex-wrap:wrap; gap:.45rem; margin:.8rem 0; }
.preset-row button { padding:.38rem .62rem; border:1px solid var(--line); border-radius:8px; color:var(--vp-c-text-1); background:var(--vp-c-bg); cursor:pointer; font-size:.61rem; }
.preset-row button:hover { border-color:var(--ink); }
.preset-row button.active { border-color:var(--ink); color:var(--vp-c-bg); background:var(--ink); }
.image-flow { display:grid; grid-template-columns:1fr 140px 1fr; gap:1rem; align-items:center; max-width:680px; margin:1rem auto; }
.image-flow>div>strong { display:block; margin-bottom:.45rem; text-align:center; font-size:.67rem; }
.pixel-grid { display:grid; grid-template-columns:repeat(9,1fr); gap:2px; aspect-ratio:1; padding:5px; border:1px solid var(--line); border-radius:12px; background:var(--vp-c-bg); }
.pixel-grid i { display:block; border-radius:2px; }
.operator { display:grid; gap:.35rem; place-items:center; color:var(--ink-muted); text-align:center; font-size:.62rem; }
.operator b { color:var(--coral); font-size:1.2rem; }
.metric-grid { grid-template-columns:repeat(3,1fr); }
.decision-note { padding:.72rem .82rem; border-radius:10px; color:var(--ink-muted); background:var(--vp-c-bg); font-size:.65rem; line-height:1.58; }
.decision-note strong { color:var(--vp-c-text-1); }
@container (max-width:560px) { .image-flow { grid-template-columns:1fr 64px 1fr; gap:.5rem; } .operator span { font-size:.5rem; } .metric-grid { grid-template-columns:1fr; } }
</style>
