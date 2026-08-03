<script setup lang="ts">
import { computed, ref } from 'vue'

const beta = ref(0.15)
const responses = [
  { name: 'A · 稳健回答', preview: '清楚回答，并保留必要边界。', reward: 0.76, kl: 0.18 },
  { name: 'B · 更讨喜', preview: '更有感染力，但偏离参考策略较多。', reward: 0.92, kl: 1.35 },
  { name: 'C · 奖励投机', preview: '迎合评分器，出现夸张和不可靠断言。', reward: 0.99, kl: 3.10 },
  { name: 'D · 接近基线', preview: '变化很小，收益也比较有限。', reward: 0.68, kl: 0.03 }
]

const scored = computed(() => responses.map(response => ({
  ...response,
  objective: response.reward - beta.value * response.kl
})).sort((a, b) => b.objective - a.objective))
const winner = computed(() => scored.value[0].name)
</script>

<template>
  <section class="lab-shell preference-lab" aria-labelledby="preference-title">
    <h3 id="preference-title">RLHF：奖励与 KL 约束实验</h3>
    <p class="lab-intro">教学目标写成：奖励 − β × KL。β 太小可能追着奖励模型投机；β 太大又会让策略几乎不敢改变。</p>

    <div class="control-row"><label for="kl-beta">KL 系数 β</label><input id="kl-beta" v-model.number="beta" type="range" min="0" max="0.8" step="0.02"><output>{{ beta.toFixed(2) }}</output></div>

    <div class="preference-equation"><span>目标</span><strong>J = Reward − {{ beta.toFixed(2) }} × KL</strong><small>当前最优：{{ winner }}</small></div>
    <div class="response-list">
      <article v-for="(response, index) in scored" :key="response.name" :class="{ winner: index === 0 }">
        <div><strong>{{ response.name }}</strong><span v-if="index === 0">当前选择</span></div>
        <p>{{ response.preview }}</p>
        <dl><div><dt>奖励</dt><dd>{{ response.reward.toFixed(2) }}</dd></div><div><dt>KL</dt><dd>{{ response.kl.toFixed(2) }}</dd></div><div><dt>目标</dt><dd>{{ response.objective.toFixed(3) }}</dd></div></dl>
      </article>
    </div>
    <p class="teach-note"><strong>试一试：</strong>先把 β 调到 0，看谁只靠奖励胜出；再慢慢增大。真实 RLHF 的 KL 通常按 token 计算，奖励模型也会犯错，所以“目标更高”不自动等于“人类更满意”。</p>
  </section>
</template>

<style scoped>
.preference-equation { display:grid; grid-template-columns:auto 1fr auto; align-items:center; gap:1rem; margin:1rem 0; padding:.85rem 1rem; border-radius:10px; color:#fff7ea; background:var(--ink); }
.preference-equation span,.preference-equation small { color:#b9ccc7; font-size:.7rem; }
.preference-equation strong { font:700 .92rem var(--vp-font-family-mono); }
.response-list { display:grid; grid-template-columns:1fr 1fr; gap:.65rem; }
.response-list article { padding:.85rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg); }
.response-list article.winner { border-color:var(--coral); box-shadow:inset 4px 0 var(--coral); }
.response-list article > div { display:flex; justify-content:space-between; gap:.6rem; }
.response-list article > div span { padding:.18rem .4rem; border-radius:99px; color:#123c38; background:var(--mint); font-size:.62rem; font-weight:700; }
.response-list p { min-height:2.6em; color:var(--ink-muted); font-size:.75rem; line-height:1.55; }
.response-list dl { display:grid; grid-template-columns:repeat(3,1fr); gap:.4rem; margin:0; }
.response-list dl div { display:grid; gap:.12rem; }
.response-list dt { color:var(--ink-muted); font-size:.62rem; }
.response-list dd { margin:0; font:.72rem var(--vp-font-family-mono); }
@media (max-width:700px) { .response-list { grid-template-columns:1fr; }.preference-equation { grid-template-columns:1fr; gap:.3rem; } }
</style>
