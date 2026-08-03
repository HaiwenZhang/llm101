<script setup lang="ts">
import { computed, ref } from 'vue'

type WordVector = { word: string; x: number; y: number; note: string }

const vectors: WordVector[] = [
  { word: '猫', x: 0.86, y: 0.48, note: '动物、宠物相关方向' },
  { word: '狗', x: 0.80, y: 0.55, note: '与“猫”共享大量上下文' },
  { word: '汽车', x: -0.72, y: 0.38, note: '交通工具相关方向' },
  { word: '快乐', x: 0.12, y: 0.92, note: '情绪相关方向' }
]

const anchorIndex = ref(0)
const candidateIndex = ref(1)
const anchor = computed(() => vectors[anchorIndex.value])
const candidate = computed(() => vectors[candidateIndex.value])

const dot = computed(() => anchor.value.x * candidate.value.x + anchor.value.y * candidate.value.y)
const anchorNorm = computed(() => Math.hypot(anchor.value.x, anchor.value.y))
const candidateNorm = computed(() => Math.hypot(candidate.value.x, candidate.value.y))
const cosine = computed(() => dot.value / (anchorNorm.value * candidateNorm.value))

const originX = 260
const originY = 135
const scale = 105
const point = (vector: WordVector) => ({
  x: originX + vector.x * scale,
  y: originY - vector.y * scale
})
const anchorPoint = computed(() => point(anchor.value))
const candidatePoint = computed(() => point(candidate.value))
</script>

<template>
  <section class="lab-shell vector-lab" aria-labelledby="vector-lab-title">
    <h3 id="vector-lab-title">向量方向实验：换一个词，相似度怎样变</h3>
    <p class="lab-intro">这里把真实的数千维词向量压成二维教学示意。先比较“猫—狗”，再把候选词换成“汽车”或“快乐”。</p>

    <div class="vector-controls">
      <label>
        基准词
        <select v-model.number="anchorIndex">
          <option v-for="(item, index) in vectors" :key="`anchor-${item.word}`" :value="index">{{ item.word }}</option>
        </select>
      </label>
      <label>
        比较词
        <select v-model.number="candidateIndex">
          <option v-for="(item, index) in vectors" :key="`candidate-${item.word}`" :value="index">{{ item.word }}</option>
        </select>
      </label>
    </div>

    <svg class="vector-plane" viewBox="0 0 520 270" role="img" :aria-label="`${anchor.word}与${candidate.word}的二维向量方向及余弦相似度 ${cosine.toFixed(3)}`">
      <line x1="40" :y1="originY" x2="480" :y2="originY" class="axis" />
      <line :x1="originX" y1="22" :x2="originX" y2="242" class="axis" />
      <line :x1="originX" :y1="originY" :x2="anchorPoint.x" :y2="anchorPoint.y" class="vector anchor" />
      <line :x1="originX" :y1="originY" :x2="candidatePoint.x" :y2="candidatePoint.y" class="vector candidate" />
      <circle :cx="anchorPoint.x" :cy="anchorPoint.y" r="7" class="point anchor" />
      <circle :cx="candidatePoint.x" :cy="candidatePoint.y" r="7" class="point candidate" />
      <text :x="anchorPoint.x + 10" :y="anchorPoint.y - 8" class="label">{{ anchor.word }}</text>
      <text :x="candidatePoint.x + 10" :y="candidatePoint.y + 20" class="label">{{ candidate.word }}</text>
      <text x="470" :y="originY - 9" class="axis-label">维度 1</text>
      <text :x="originX + 9" y="30" class="axis-label">维度 2</text>
    </svg>

    <div class="metric-grid">
      <div class="metric-card"><small>点积</small><strong>{{ dot.toFixed(3) }}</strong><span>同时受方向与长度影响</span></div>
      <div class="metric-card"><small>长度乘积</small><strong>{{ (anchorNorm * candidateNorm).toFixed(3) }}</strong><span>余弦的归一化分母</span></div>
      <div class="metric-card"><small>余弦相似度</small><strong>{{ cosine.toFixed(3) }}</strong><span>越接近 1，方向越相似</span></div>
    </div>

    <p class="teach-note">{{ anchor.word }}：{{ anchor.note }}；{{ candidate.word }}：{{ candidate.note }}。图中的二维坐标只是教学投影，不能把某一维直接命名为“动物维”或“情绪维”。</p>
  </section>
</template>

<style scoped>
.vector-controls { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; margin: 1rem 0; }
.vector-controls label { display: grid; gap: .35rem; color: var(--ink-muted); font-size: .78rem; font-weight: 700; }
.vector-controls select { width: 100%; padding: .62rem .7rem; border: 1px solid var(--line); border-radius: 8px; color: var(--ink); background: var(--vp-c-bg); }
.vector-plane { display: block; width: 100%; min-height: 250px; border: 1px dashed var(--line); border-radius: 12px; background: var(--vp-c-bg); }
.axis { stroke: var(--line); stroke-width: 1.5; }
.vector { stroke-width: 5; stroke-linecap: round; opacity: .86; }
.vector.anchor { stroke: var(--vp-c-brand-1); }
.vector.candidate { stroke: var(--coral); }
.point.anchor { fill: var(--vp-c-brand-1); }
.point.candidate { fill: var(--coral); }
.label { fill: var(--vp-c-text-1); font-size: 15px; font-weight: 700; }
.axis-label { fill: var(--ink-muted); font-size: 11px; }
@media (max-width: 620px) { .vector-controls { grid-template-columns: 1fr; }.vector-plane { min-height: 210px; } }
</style>
