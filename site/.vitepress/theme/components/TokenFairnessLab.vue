<script setup lang="ts">
import { computed, ref } from 'vue'

type Mode = 'legacy' | 'balanced'

const alpha = ref(0.5)
const mode = ref<Mode>('legacy')
const trainingTokens = 1_000_000
const contextTokens = 2_048
const languages = [
  { code: 'EN', name: '英语', raw: .8, legacy: 21, balanced: 21 },
  { code: 'SO', name: '索马里语', raw: .15, legacy: 30, balanced: 24 },
  { code: 'TH', name: '泰语', raw: .05, legacy: 36, balanced: 25 }
]

const rows = computed(() => {
  const weights = languages.map(language => language.raw ** alpha.value)
  const denominator = weights.reduce((sum, weight) => sum + weight, 0)
  return languages.map((language, index) => {
    const share = weights[index] / denominator
    const tokenLength = language[mode.value]
    return {
      ...language,
      share,
      tokenLength,
      oversample: share / language.raw,
      messages: trainingTokens * share / tokenLength,
      contextMessages: Math.floor(contextTokens / tokenLength)
    }
  })
})

const englishTokens = computed(() => rows.value[0].tokenLength)

function applyPreset(kind: 'raw' | 'resample' | 'joint') {
  if (kind === 'raw') {
    alpha.value = 1
    mode.value = 'legacy'
  } else if (kind === 'resample') {
    alpha.value = .5
    mode.value = 'legacy'
  } else {
    alpha.value = .5
    mode.value = 'balanced'
  }
}

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}
</script>

<template>
  <section class="lab-shell fairness-lab" aria-labelledby="fairness-title">
    <h3 id="fairness-title">Token 公平预算实验：同样 100 万训练 Token，谁真正学得多</h3>
    <p class="lab-intro">三种语言的原始网页占比分别为 80%、15%、5%。每行的“语义样本”是假设一条等价消息需要对应 token 数后，100 万训练 token 大约能容纳多少条消息。</p>

    <div class="preset-grid" aria-label="多语言训练策略预设">
      <button type="button" :class="{ active: alpha === 1 && mode === 'legacy' }" @click="applyPreset('raw')"><strong>原样采样</strong><span>α=1.0 · 英语偏置 tokenizer</span></button>
      <button type="button" :class="{ active: alpha === .5 && mode === 'legacy' }" @click="applyPreset('resample')"><strong>只做重采样</strong><span>α=0.5 · 切分成本不变</span></button>
      <button type="button" :class="{ active: alpha === .5 && mode === 'balanced' }" @click="applyPreset('joint')"><strong>联合改进</strong><span>α=0.5 · 更均衡 tokenizer</span></button>
    </div>

    <div class="control-row"><label for="language-alpha">采样温度指数 α</label><input id="language-alpha" v-model.number="alpha" type="range" min="0.2" max="1" step="0.05"><output>{{ alpha.toFixed(2) }}</output></div>
    <div class="mode-row" role="group" aria-label="选择 tokenizer">
      <span>Tokenizer</span>
      <button type="button" :class="{ active: mode === 'legacy' }" @click="mode = 'legacy'">英语偏置</button>
      <button type="button" :class="{ active: mode === 'balanced' }" @click="mode = 'balanced'">较均衡</button>
    </div>

    <div class="distribution" aria-label="当前训练 Token 分配">
      <div v-for="row in rows" :key="row.code" class="language-row">
        <div class="row-label"><strong>{{ row.name }}</strong><span>原始 {{ percent(row.raw) }} → 训练 {{ percent(row.share) }}</span></div>
        <div class="bar"><i :style="{ width: `${row.share * 100}%` }"></i></div>
        <output>{{ row.oversample.toFixed(2) }}×</output>
      </div>
    </div>

    <div class="table-wrap">
      <table>
        <thead><tr><th>语言</th><th>等价消息</th><th>相对价格</th><th>100 万 token 中的语义样本</th><th>2048 上下文可放</th></tr></thead>
        <tbody>
          <tr v-for="row in rows" :key="row.code">
            <th>{{ row.name }}</th>
            <td>{{ row.tokenLength }} token</td>
            <td>{{ (row.tokenLength / englishTokens).toFixed(2) }}×</td>
            <td>{{ Math.round(row.messages).toLocaleString() }} 条</td>
            <td>{{ row.contextMessages }} 条</td>
          </tr>
        </tbody>
      </table>
    </div>

    <p class="decision-note"><strong>读表方法：</strong>α 变小只能改变“训练 token 份额”；tokenizer 决定每条等价内容要花多少 token。只做低资源语言过采样，能增加它拿到的预算，却不会自动消除上下文、延迟和按 token 计价的不公平。</p>
    <p class="teach-note"><strong>试一试：</strong>先点“原样采样”，再点“只做重采样”，最后点“联合改进”。观察泰语的训练份额、语义样本数和同一上下文能放下的消息数分别由哪一个旋钮改变。</p>
  </section>
</template>

<style scoped>
.fairness-lab { container-type:inline-size; }
.preset-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.55rem; margin:.9rem 0; }
.preset-grid button,.mode-row button { border:1px solid var(--line); color:var(--vp-c-text-1); background:var(--vp-c-bg); cursor:pointer; }
.preset-grid button { padding:.65rem .7rem; border-radius:10px; text-align:left; }
.preset-grid button:hover,.mode-row button:hover { border-color:var(--ink); }
.preset-grid button.active,.mode-row button.active { border-color:var(--ink); color:var(--vp-c-bg); background:var(--ink); }
.preset-grid strong,.preset-grid span { display:block; }
.preset-grid strong { font-size:.7rem; }
.preset-grid span { margin-top:.14rem; color:var(--ink-muted); font-size:.56rem; }
.preset-grid button.active span { color:color-mix(in srgb,var(--vp-c-bg) 72%,transparent); }
.mode-row { display:flex; align-items:center; gap:.45rem; margin:.7rem 0 1rem; }
.mode-row>span { margin-right:auto; font-size:.65rem; font-weight:750; }
.mode-row button { padding:.36rem .62rem; border-radius:8px; font-size:.61rem; }
.distribution { display:grid; gap:.62rem; margin:1rem 0; }
.language-row { display:grid; grid-template-columns:175px 1fr 48px; gap:.6rem; align-items:center; }
.row-label strong,.row-label span { display:block; }
.row-label strong { font-size:.66rem; }
.row-label span { color:var(--ink-muted); font-size:.55rem; }
.bar { height:12px; overflow:hidden; border-radius:999px; background:var(--vp-c-bg-soft); }
.bar i { display:block; height:100%; border-radius:inherit; background:var(--coral); transition:width .18s ease; }
.language-row output { color:var(--ink-muted); text-align:right; font:700 .6rem var(--vp-font-family-mono); }
.table-wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:.61rem; }
th,td { padding:.55rem .45rem; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th:first-child,td:first-child { text-align:left; }
thead th { color:var(--ink-muted); font-size:.55rem; }
.decision-note { padding:.72rem .82rem; border-radius:10px; color:var(--ink-muted); background:var(--vp-c-bg); font-size:.65rem; line-height:1.58; }
.decision-note strong { color:var(--vp-c-text-1); }
@container (max-width:560px) { .preset-grid { grid-template-columns:1fr; } .language-row { grid-template-columns:118px 1fr 42px; } }
</style>
