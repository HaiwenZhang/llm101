<script setup lang="ts">
import { computed, ref } from 'vue'

const stage = ref(0)
const selected = ref(2)
const tokens = [
  { text: '我', id: 105, vector: [0.2, -0.4, 0.8, 0.1], meaning: '说话者 / 主语线索' },
  { text: '喜欢', id: 2841, vector: [0.7, 0.3, -0.2, 0.6], meaning: '偏好 / 动作线索' },
  { text: '机器', id: 9132, vector: [-0.1, 0.9, 0.4, 0.3], meaning: '技术对象的一部分' },
  { text: '学习', id: 662, vector: [0.5, 0.8, -0.1, 0.7], meaning: '行为；与“机器”组合成领域词' }
]
const labels = ['原始文字', '切成 token', '换成编号', '查表变向量']
const current = computed(() => tokens[selected.value])
</script>

<template>
  <section class="lab-shell token-lab" aria-labelledby="token-lab-title">
    <h3 id="token-lab-title">亲手走一遍：文字怎样进入模型</h3>
    <div class="stage-switch" role="tablist" aria-label="处理阶段">
      <button v-for="(label, i) in labels" :key="label" :class="{ active: stage === i }" @click="stage = i"><span>{{ i + 1 }}</span>{{ label }}</button>
    </div>
    <div class="token-stage">
      <p v-if="stage === 0" class="raw-sentence">我喜欢机器学习</p>
      <div v-else class="token-pieces">
        <button v-for="(token, i) in tokens" :key="token.text" :class="{ selected: selected === i }" @click="selected = i">
          <span>{{ token.text }}</span>
          <small v-if="stage >= 2">ID {{ token.id }}</small>
        </button>
      </div>
      <div v-if="stage === 3" class="vector-readout">
        <span>“{{ current.text }}” 的教学版 4 维向量</span>
        <div><i v-for="(value, i) in current.vector" :key="i" :class="{ negative: value < 0 }" :style="{ height: `${Math.abs(value) * 70 + 12}px` }"><b>{{ value }}</b></i></div>
        <small>{{ current.meaning }}。真实模型通常有数千维，单独一维一般没有固定中文含义。</small>
      </div>
    </div>
    <p class="teach-note">编号只是查表地址，不是“语义分数”。向量才是后续矩阵运算真正处理的对象。</p>
  </section>
</template>

<style scoped>
.stage-switch { display: grid; grid-template-columns: repeat(4, 1fr); gap: .45rem; margin: 1rem 0; }
.stage-switch button { display: flex; align-items: center; gap: .45rem; padding: .65rem; border: 1px solid var(--line); border-radius: 8px; color: var(--ink-muted); background: var(--vp-c-bg); cursor: pointer; font-size: .75rem; }
.stage-switch button span { display: grid; width: 22px; height: 22px; place-items: center; border-radius: 50%; background: var(--line); font: .65rem var(--vp-font-family-mono); }
.stage-switch button.active { border-color: var(--ink); color: var(--ink); }
.stage-switch button.active span { color: white; background: var(--coral); }
.token-stage { min-height: 215px; padding: 1.25rem; border: 1px dashed var(--line); border-radius: 12px; background: var(--vp-c-bg); }
.raw-sentence { margin: 3rem 0; text-align: center; font: 800 2rem 'Noto Serif SC', serif; }
.token-pieces { display: flex; flex-wrap: wrap; justify-content: center; gap: .5rem; margin: 1rem 0; }
.token-pieces button { display: grid; min-width: 75px; gap: .2rem; padding: .8rem; border: 2px solid transparent; border-radius: 8px; color: var(--ink); background: var(--paper-warm); cursor: pointer; }
.token-pieces button.selected { border-color: var(--coral); }
.token-pieces small { color: var(--ink-muted); font: .6rem var(--vp-font-family-mono); }
.vector-readout { display: grid; gap: .7rem; margin-top: 1.2rem; text-align: center; }
.vector-readout > span { font-size: .8rem; color: var(--ink-muted); }
.vector-readout > div { display: flex; justify-content: center; align-items: end; gap: .7rem; height: 92px; border-bottom: 1px solid var(--line); }
.vector-readout i { position: relative; width: 34px; background: var(--mint); border-radius: 5px 5px 0 0; }
.vector-readout i.negative { background: var(--coral); }
.vector-readout b { position: absolute; top: -1.2rem; left: 50%; transform: translateX(-50%); font: .6rem var(--vp-font-family-mono); }
.vector-readout small { color: var(--ink-muted); line-height: 1.6; }
@media (max-width: 620px) { .stage-switch { grid-template-columns: 1fr 1fr; }.stage-switch button { font-size: .68rem; } }
</style>
