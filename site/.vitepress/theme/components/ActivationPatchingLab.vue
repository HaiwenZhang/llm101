<script setup lang="ts">
import { computed, ref } from 'vue'

const positions = ['France', '’s', 'capital', 'is', '最后位置']
const recovery = [
  [5, 2, 1, 0, 0],
  [18, 4, 3, 2, 1],
  [55, 8, 5, 4, 3],
  [82, 13, 8, 6, 5],
  [48, 10, 12, 8, 7],
  [20, 6, 18, 10, 15]
]
const selectedLayer = ref(3)
const selectedPosition = ref(0)
const cleanProbability = .82
const corruptedProbability = .12

const selectedRecovery = computed(() => recovery[selectedLayer.value][selectedPosition.value])
const patchedProbability = computed(() => corruptedProbability + selectedRecovery.value / 100 * (cleanProbability - corruptedProbability))

function choose(layer: number, position: number) {
  selectedLayer.value = layer
  selectedPosition.value = position
}

function cellStyle(value: number) {
  return { background: `color-mix(in srgb, var(--coral) ${Math.max(7, value)}%, var(--vp-c-bg))` }
}
</script>

<template>
  <section class="lab-shell patching-lab" aria-labelledby="patching-title">
    <h3 id="patching-title">Activation Patching 实验：把 clean 激活贴到 corrupted 运行</h3>
    <p class="lab-intro">Clean 输入“France’s capital is”让 Paris 概率为 82%；Corrupted 输入把 France 改成 Italy，Paris 概率降到 12%。点击任意层与位置，把 clean 残差激活替换进去，看目标概率恢复多少。</p>

    <div class="run-strip">
      <div><small>Clean</small><strong>France → Paris</strong><span>p(Paris)=82%</span></div>
      <div><small>Corrupted</small><strong>Italy → Rome</strong><span>p(Paris)=12%</span></div>
      <div class="patched"><small>Patched</small><strong>L{{ selectedLayer }} · {{ positions[selectedPosition] }}</strong><span>p(Paris)={{ (patchedProbability * 100).toFixed(1) }}%</span></div>
    </div>

    <div class="heatmap-wrap">
      <div class="heatmap" role="grid" aria-label="不同层和 token 位置的 activation patch 恢复率">
        <span class="corner"></span><strong v-for="position in positions" :key="position">{{ position }}</strong>
        <template v-for="(row, layer) in recovery" :key="layer">
          <strong>L{{ layer }}</strong>
          <button v-for="(value, position) in row" :key="`${layer}-${position}`" type="button" :class="{ active: selectedLayer === layer && selectedPosition === position }" :style="cellStyle(value)" :aria-label="`第 ${layer} 层 ${positions[position]} 位置，恢复率 ${value}%`" @click="choose(layer, position)">{{ value }}%</button>
        </template>
      </div>
    </div>

    <div class="metric-grid">
      <div class="metric-card"><small>Clean 概率</small><strong>82.0%</strong><span>实验上界</span></div>
      <div class="metric-card"><small>Patched 概率</small><strong>{{ (patchedProbability * 100).toFixed(1) }}%</strong><span>只替换一个站点</span></div>
      <div class="metric-card"><small>恢复率</small><strong>{{ selectedRecovery }}%</strong><span>(patched−bad)/(clean−bad)</span></div>
    </div>

    <p class="decision-note"><strong>当前结论：</strong>在这组教学数据中，L{{ selectedLayer }} 的“{{ positions[selectedPosition] }}”位置 patch 可恢复 {{ selectedRecovery }}% 的 clean/corrupted 差距。{{ selectedRecovery >= 50 ? '较高恢复率支持“这里携带了影响答案的因果信息”，但还不能证明它独自构成完整 circuit。' : '恢复有限，当前站点不足以单独解释行为；还要检查相邻层、其他位置与组合路径。' }}</p>
    <p class="teach-note"><strong>试一试：</strong>比较 L3/France 与最后位置，再点相邻层。真实研究会对许多 clean/corrupted 对重复整张扫描，并加入随机位置 patch、均值替换和无关任务作为对照。</p>
  </section>
</template>

<style scoped>
.patching-lab { container-type:inline-size; }
.run-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:.6rem; margin:1rem 0; }
.run-strip>div { padding:.7rem; border:1px solid var(--line); border-radius:11px; background:var(--vp-c-bg); }
.run-strip .patched { background:color-mix(in srgb,var(--mint) 10%,var(--vp-c-bg)); }
.run-strip small,.run-strip strong,.run-strip span { display:block; }
.run-strip small,.run-strip span { color:var(--ink-muted); font-size:.56rem; }
.run-strip strong { margin:.2rem 0; font-size:.68rem; }
.heatmap-wrap { overflow-x:auto; padding-bottom:.2rem; }
.heatmap { display:grid; grid-template-columns:64px repeat(5,minmax(76px,1fr)); gap:4px; min-width:500px; }
.heatmap>strong { display:grid; place-items:center; min-height:34px; color:var(--ink-muted); font-size:.56rem; }
.heatmap button { min-height:46px; border:1px solid transparent; border-radius:8px; color:var(--vp-c-text-1); cursor:pointer; font:750 .61rem var(--vp-font-family-mono); }
.heatmap button:hover { border-color:var(--ink); }
.heatmap button.active { border-color:var(--ink); outline:2px solid color-mix(in srgb,var(--ink) 24%,transparent); outline-offset:1px; }
.metric-grid { grid-template-columns:repeat(3,1fr); }
.decision-note { padding:.72rem .82rem; border-radius:10px; color:var(--ink-muted); background:var(--vp-c-bg); font-size:.65rem; line-height:1.58; }
.decision-note strong { color:var(--vp-c-text-1); }
@container (max-width:500px) { .run-strip,.metric-grid { grid-template-columns:1fr; } }
</style>
