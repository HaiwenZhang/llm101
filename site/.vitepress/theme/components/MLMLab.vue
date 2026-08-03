<script setup lang="ts">
import { computed, ref } from 'vue'

type Example = {
  name: string
  tokens: string[]
  targetIndex: number
  randomToken: string
  candidates: string[]
  bidirectional: number[]
  leftOnly: number[]
}

const examples: Example[] = [
  {
    name: '餐厅',
    tokens: ['[CLS]', '这家', '餐厅', '的', '面条', '很好吃', '[SEP]'],
    targetIndex: 2,
    randomToken: '电影院',
    candidates: ['餐厅', '商店', '学校', '公园'],
    bidirectional: [0.74, 0.11, 0.07, 0.08],
    leftOnly: [0.31, 0.27, 0.22, 0.20]
  },
  {
    name: '银行',
    tokens: ['[CLS]', '他', '去', '银行', '办理', '贷款', '[SEP]'],
    targetIndex: 3,
    randomToken: '河岸',
    candidates: ['银行', '公司', '学校', '河岸'],
    bidirectional: [0.82, 0.07, 0.06, 0.05],
    leftOnly: [0.29, 0.27, 0.23, 0.21]
  },
  {
    name: '河岸',
    tokens: ['[CLS]', '她', '沿着', '河岸', '看', '水流', '[SEP]'],
    targetIndex: 3,
    randomToken: '银行',
    candidates: ['河岸', '道路', '走廊', '银行'],
    bidirectional: [0.77, 0.10, 0.08, 0.05],
    leftOnly: [0.33, 0.28, 0.22, 0.17]
  }
]

const current = ref(0)
const visibility = ref<'bidirectional' | 'leftOnly'>('bidirectional')
const corruption = ref<'mask' | 'random' | 'keep'>('mask')

const example = computed(() => examples[current.value])
const inputTokens = computed(() => example.value.tokens.map((token, index) => {
  if (index !== example.value.targetIndex) return token
  if (corruption.value === 'mask') return '[MASK]'
  if (corruption.value === 'random') return example.value.randomToken
  return token
}))
const probabilities = computed(() => example.value[visibility.value])
const targetProbability = computed(() => probabilities.value[0])
const loss = computed(() => -Math.log(targetProbability.value))

function percent(value: number) {
  return `${(value * 100).toFixed(0)}%`
}
</script>

<template>
  <section class="lab-shell mlm-lab" aria-labelledby="mlm-title">
    <h3 id="mlm-title">MLM 实验：右文、替换方式与 loss mask</h3>
    <p class="lab-intro">这是一个用于理解监督规则的概率示意，不是真实 BERT 输出。选择句子、可见范围和 80/10/10 中的一种替换结果，观察输入变了什么、标签又留在哪里。</p>

    <div class="preset-row" aria-label="选择 MLM 句子">
      <button v-for="(item, index) in examples" :key="item.name" type="button" :class="{ active: current === index }" @click="current = index">预测“{{ item.name }}”</button>
    </div>
    <div class="choice-grid">
      <fieldset>
        <legend>该位置怎样呈现给模型</legend>
        <label><input v-model="corruption" type="radio" value="mask"> `[MASK]`（选中位置中的 80%）</label>
        <label><input v-model="corruption" type="radio" value="random"> 随机词（10%）</label>
        <label><input v-model="corruption" type="radio" value="keep"> 保持原词（10%）</label>
      </fieldset>
      <fieldset>
        <legend>教学对照：能看哪些位置</legend>
        <label><input v-model="visibility" type="radio" value="bidirectional"> 双向 Encoder：左右都能看</label>
        <label><input v-model="visibility" type="radio" value="leftOnly"> 只看左侧：遮住右文</label>
      </fieldset>
    </div>

    <div class="token-board" aria-label="MLM 输入 token 和监督标签">
      <div class="board-label">模型输入</div>
      <div class="token-row">
        <span v-for="(token, index) in inputTokens" :key="`input-${index}`" :class="{ selected: index === example.targetIndex }">{{ token }}</span>
      </div>
      <div class="board-label">labels</div>
      <div class="token-row labels">
        <span v-for="(token, index) in example.tokens" :key="`label-${index}`" :class="{ selected: index === example.targetIndex }">{{ index === example.targetIndex ? token : '−100' }}</span>
      </div>
    </div>

    <div class="candidate-list" aria-label="候选词教学概率">
      <div v-for="(candidate, index) in example.candidates" :key="candidate">
        <strong>{{ candidate }}</strong>
        <i><b :style="{ width: percent(probabilities[index]) }"></b></i>
        <output>{{ percent(probabilities[index]) }}</output>
      </div>
    </div>

    <div class="metric-grid">
      <div class="metric-card"><small>原始 target</small><strong>{{ example.tokens[example.targetIndex] }}</strong><span>无论怎样替换都不变</span></div>
      <div class="metric-card"><small>target 概率</small><strong>{{ percent(targetProbability) }}</strong><span>教学分布中的正确项</span></div>
      <div class="metric-card"><small>该位置交叉熵</small><strong>{{ loss.toFixed(3) }}</strong><span>−ln p(target)</span></div>
      <div class="metric-card"><small>直接监督位置</small><strong>1 / {{ example.tokens.length }}</strong><span>其余 label 是 −100</span></div>
    </div>

    <p class="teach-note"><strong>读图：</strong>即使选中“保持原词”的 10% 情况，labels 中仍保留原 token，所以这一格仍计算 loss；`−100` 只表示其他未选中位置不计 MLM loss，不等于它们不参与 Attention。切到“只看左侧”后，教学分布变平，用来说明右文为何能帮助消歧。</p>
  </section>
</template>

<style scoped>
.mlm-lab { container-type:inline-size; }
.preset-row { display:flex; flex-wrap:wrap; gap:.5rem; margin:.9rem 0; }
.preset-row button { padding:.48rem .72rem; border:1px solid var(--line); border-radius:9px; color:var(--vp-c-text-1); background:var(--vp-c-bg); cursor:pointer; font-size:.69rem; }
.preset-row button.active { border-color:var(--ink); color:var(--vp-c-bg); background:var(--ink); }
.choice-grid { display:grid; grid-template-columns:1fr 1fr; gap:.7rem; }
.choice-grid fieldset { display:grid; gap:.42rem; min-width:0; margin:0; padding:.75rem .85rem; border:1px solid var(--line); border-radius:11px; background:var(--vp-c-bg); }
.choice-grid legend { padding:0 .3rem; font-size:.68rem; font-weight:750; }
.choice-grid label { display:flex; align-items:center; gap:.42rem; color:var(--ink-muted); font-size:.64rem; }
.token-board { display:grid; grid-template-columns:72px 1fr; gap:.55rem; align-items:center; margin:1rem 0; padding:.8rem; border:1px solid var(--line); border-radius:12px; background:var(--vp-c-bg); }
.board-label { color:var(--ink-muted); font:650 .61rem var(--vp-font-family-mono); }
.token-row { display:grid; grid-template-columns:repeat(7,minmax(42px,1fr)); gap:.35rem; }
.token-row span { display:grid; place-items:center; min-height:38px; padding:.25rem; border-radius:7px; background:var(--vp-c-bg-soft); font-size:.63rem; text-align:center; }
.token-row span.selected { outline:2px solid var(--coral); background:color-mix(in srgb,var(--coral) 10%,var(--vp-c-bg)); }
.token-row.labels span { color:var(--ink-muted); font-family:var(--vp-font-family-mono); }
.token-row.labels span.selected { color:var(--vp-c-text-1); font-family:var(--vp-font-family-base); }
.candidate-list { display:grid; gap:.45rem; margin:1rem 0; padding:.8rem; border:1px solid var(--line); border-radius:12px; background:var(--vp-c-bg); }
.candidate-list>div { display:grid; grid-template-columns:72px 1fr 42px; align-items:center; gap:.55rem; }
.candidate-list strong { font-size:.67rem; }
.candidate-list i { height:13px; overflow:hidden; border-radius:999px; background:var(--vp-c-bg-soft); }
.candidate-list b { display:block; height:100%; border-radius:inherit; background:var(--mint); transition:width .2s ease; }
.candidate-list output { text-align:right; font:.64rem var(--vp-font-family-mono); }
.metric-grid { grid-template-columns:repeat(4,1fr); }
@container (max-width:650px) { .choice-grid,.metric-grid { grid-template-columns:1fr 1fr; } .token-board { grid-template-columns:1fr; } }
@container (max-width:450px) { .choice-grid,.metric-grid { grid-template-columns:1fr; } .token-row { grid-template-columns:repeat(4,1fr); } }
</style>
