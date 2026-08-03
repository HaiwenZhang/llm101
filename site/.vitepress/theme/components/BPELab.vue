<script setup lang="ts">
import { computed, ref } from 'vue'

type Entry = { word: string; count: number; symbols: string[] }
type Merge = { left: string; right: string; count: number }

const presets = [
  {
    name: '经典英文',
    text: 'low low low low low lower lower newest newest newest newest newest newest widest widest widest'
  },
  {
    name: '中文短语',
    text: '大模型 大模型 大模型 语言模型 语言模型 模型训练 模型推理'
  }
]

const corpus = ref(presets[0].text)
const entries = ref<Entry[]>([])
const history = ref<Merge[]>([])

function reset() {
  const counts = new Map<string, number>()
  for (const word of corpus.value.trim().split(/\s+/).filter(Boolean)) {
    counts.set(word, (counts.get(word) || 0) + 1)
  }
  entries.value = [...counts.entries()].map(([word, count]) => ({
    word,
    count,
    symbols: [...word, '</w>']
  }))
  history.value = []
}

function usePreset(index: number) {
  corpus.value = presets[index].text
  reset()
}

const pairs = computed(() => {
  const counts = new Map<string, { left: string; right: string; count: number }>()
  for (const entry of entries.value) {
    for (let i = 0; i < entry.symbols.length - 1; i++) {
      const left = entry.symbols[i]
      const right = entry.symbols[i + 1]
      const key = JSON.stringify([left, right])
      const current = counts.get(key) || { left, right, count: 0 }
      current.count += entry.count
      counts.set(key, current)
    }
  }
  return [...counts.values()].sort((a, b) => b.count - a.count || `${a.left}${a.right}`.localeCompare(`${b.left}${b.right}`))
})

const bestPair = computed(() => pairs.value[0])
const vocabulary = computed(() => new Set(entries.value.flatMap(entry => entry.symbols)).size)
const totalPieces = computed(() => entries.value.reduce((sum, entry) => sum + entry.symbols.length * entry.count, 0))

function mergeBest() {
  const best = bestPair.value
  if (!best) return
  entries.value = entries.value.map(entry => {
    const next: string[] = []
    for (let i = 0; i < entry.symbols.length; i++) {
      if (entry.symbols[i] === best.left && entry.symbols[i + 1] === best.right) {
        next.push(best.left + best.right)
        i += 1
      } else {
        next.push(entry.symbols[i])
      }
    }
    return { ...entry, symbols: next }
  })
  history.value.push({ ...best })
}

const pretty = (symbol: string) => symbol.replace('</w>', '‹END›')

reset()
</script>

<template>
  <section class="lab-shell bpe-lab" aria-labelledby="bpe-title">
    <div class="lab-title-row">
      <div><h3 id="bpe-title">BPE 逐轮合并实验</h3><p class="lab-intro">相同词出现得越多，其中的相邻符号对计数越高。每轮把最高频的一对合成一个新 token。</p></div>
      <span class="lab-badge">已合并 {{ history.length }} 轮</span>
    </div>

    <div class="preset-row">
      <button v-for="(preset, index) in presets" :key="preset.name" @click="usePreset(index)">{{ preset.name }}</button>
    </div>
    <label class="corpus-editor">训练小语料（空格分词；重复词表示更高词频）<textarea v-model="corpus" rows="3"></textarea></label>
    <div class="bpe-actions"><button @click="reset">按当前语料重新开始</button><button class="merge" :disabled="!bestPair" @click="mergeBest">合并最高频相邻对</button></div>

    <div class="metric-grid">
      <div class="metric-card"><small>当前子词表</small><strong>{{ vocabulary }}</strong><span>不同符号数量</span></div>
      <div class="metric-card"><small>语料总片段</small><strong>{{ totalPieces }}</strong><span>按词频加权</span></div>
      <div class="metric-card"><small>下一对</small><strong>{{ bestPair ? `${pretty(bestPair.left)} + ${pretty(bestPair.right)}` : '完成' }}</strong><span>{{ bestPair ? `出现 ${bestPair.count} 次` : '没有相邻对' }}</span></div>
    </div>

    <div class="bpe-workspace">
      <div>
        <h4>当前词表切分</h4>
        <div class="word-entries">
          <div v-for="entry in entries" :key="entry.word">
            <span class="word-count">× {{ entry.count }}</span>
            <strong>{{ entry.word }}</strong>
            <div><i v-for="(symbol, index) in entry.symbols" :key="`${symbol}-${index}`">{{ pretty(symbol) }}</i></div>
          </div>
        </div>
      </div>
      <div>
        <h4>相邻对排行榜</h4>
        <ol class="pair-list">
          <li v-for="pair in pairs.slice(0, 6)" :key="`${pair.left}-${pair.right}`">
            <span>{{ pretty(pair.left) }}</span><b>+</b><span>{{ pretty(pair.right) }}</span><output>{{ pair.count }}</output>
          </li>
        </ol>
      </div>
    </div>

    <div v-if="history.length" class="merge-history"><strong>已经学到的 merge 规则</strong><span v-for="(merge, index) in history" :key="index">{{ index + 1 }}. {{ pretty(merge.left) }} + {{ pretty(merge.right) }} → {{ pretty(merge.left + merge.right) }}</span></div>
    <p class="teach-note"><strong>教学简化：</strong>为了让中文符号可读，这里从 Unicode 字符开始。GPT-2 一类 byte-level BPE 会先把文本编码成 UTF-8 字节，再学习相同的“统计相邻对并逐轮合并”规则；特殊 token 和正规化也由具体 tokenizer 单独处理。</p>
  </section>
</template>

<style scoped>
.lab-title-row { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; }
.lab-badge { flex:0 0 auto; padding:.35rem .6rem; border-radius:99px; color:#123c38; background:var(--mint); font-size:.72rem; font-weight:700; }
.preset-row { display:flex; gap:.5rem; margin:1rem 0 .65rem; }
.preset-row button,.bpe-actions button { padding:.52rem .75rem; border:1px solid var(--line); border-radius:8px; color:var(--ink); background:var(--vp-c-bg); cursor:pointer; font-weight:700; }
.corpus-editor { display:grid; gap:.35rem; color:var(--ink-muted); font-size:.75rem; font-weight:700; }
.corpus-editor textarea { width:100%; resize:vertical; padding:.7rem; border:1px solid var(--line); border-radius:9px; color:var(--ink); background:var(--vp-c-bg); font: .78rem/1.6 var(--vp-font-family-mono); }
.bpe-actions { display:flex; justify-content:flex-end; gap:.6rem; margin:.7rem 0 1rem; }
.bpe-actions button.merge { border-color:var(--ink); color:#fff7ea; background:var(--ink); }
.bpe-actions button:disabled { opacity:.4; cursor:not-allowed; }
.bpe-workspace { display:grid; grid-template-columns:1.35fr .65fr; gap:1rem; margin:1rem 0; }
.bpe-workspace > div { padding:1rem; border:1px solid var(--line); border-radius:10px; background:var(--vp-c-bg); }
.bpe-workspace h4 { margin:0 0 .7rem; font-size:.85rem; }
.word-entries { display:grid; gap:.6rem; }
.word-entries > div { display:grid; grid-template-columns:40px 80px 1fr; align-items:center; gap:.5rem; }
.word-count { color:var(--ink-muted); font:.65rem var(--vp-font-family-mono); }
.word-entries strong { font-size:.78rem; }
.word-entries > div > div { display:flex; flex-wrap:wrap; gap:.25rem; }
.word-entries i { padding:.22rem .38rem; border-radius:5px; color:#123c38; background:var(--mint); font:normal .65rem var(--vp-font-family-mono); }
.pair-list { display:grid; gap:.42rem; margin:0; padding:0; list-style:none; }
.pair-list li { display:grid; grid-template-columns:1fr auto 1fr 30px; align-items:center; gap:.3rem; padding:.38rem .45rem; border-bottom:1px solid var(--line); font:.67rem var(--vp-font-family-mono); }
.pair-list li:first-child { color:var(--coral-dark); font-weight:700; }
.pair-list output { text-align:right; }
.merge-history { display:flex; flex-wrap:wrap; gap:.4rem; padding:.75rem; border-radius:9px; background:var(--vp-c-bg-soft); }
.merge-history strong { width:100%; font-size:.76rem; }
.merge-history span { padding:.28rem .45rem; border:1px solid var(--line); border-radius:99px; font:.62rem var(--vp-font-family-mono); }
@media (max-width:700px) { .lab-title-row { flex-direction:column; }.bpe-workspace { grid-template-columns:1fr; }.word-entries > div { grid-template-columns:35px 65px 1fr; } }
</style>
