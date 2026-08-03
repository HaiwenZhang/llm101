<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import paperIndex from '../data/papers.json'

type Paper = typeof paperIndex.papers[number]
const query = ref('')
const filter = ref('全部')
const groups = ['全部', '必读', '架构', '训练', 'Agent', '系统']

const architecture = new Set([1,2,3,4,5,7,13,14,15,16,17,32])
const training = new Set([6,8,9,10,12,18,19,20,21,22])
const agent = new Set([9,10,11,12,22,23,24])
const systems = new Set([25,26,27,28,29,30,31,32])

function tags(paper: Paper) {
  const result: string[] = []
  if (paper.order <= 12) result.push('必读')
  if (architecture.has(paper.order)) result.push('架构')
  if (training.has(paper.order)) result.push('训练')
  if (agent.has(paper.order)) result.push('Agent')
  if (systems.has(paper.order)) result.push('系统')
  return result.length ? result : ['拓展']
}
function level(paper: Paper) {
  if (paper.order === 0) return '主线'
  if (paper.order <= 12) return '精读'
  return '选读'
}
const papers = computed(() => paperIndex.papers.filter(paper => {
  const keyword = query.value.trim().toLowerCase()
  const haystack = `${paper.title} ${paper.focus.join(' ')} ${paper.arxiv_id}`.toLowerCase()
  return (!keyword || haystack.includes(keyword)) && (filter.value === '全部' || tags(paper).includes(filter.value))
}))
</script>

<template>
  <section class="paper-library">
    <div class="paper-toolbar">
      <label class="paper-search"><span>⌕</span><input v-model="query" type="search" placeholder="搜索标题、概念或 arXiv ID" aria-label="搜索论文"></label>
      <div class="paper-filters" role="group" aria-label="论文分类">
        <button v-for="group in groups" :key="group" :class="{ active: filter === group }" @click="filter = group">{{ group }}</button>
      </div>
    </div>
    <p class="paper-count">找到 {{ papers.length }} 篇 · 点击标题进入中文拆解学习页</p>
    <div class="paper-grid">
      <article v-for="paper in papers" :key="paper.arxiv_id" class="paper-card">
        <div class="paper-top"><span class="paper-order">{{ String(paper.order).padStart(2, '0') }}</span><span class="paper-level">{{ level(paper) }}</span></div>
        <h3><a :href="withBase(`/papers/${paper.slug}`)">{{ paper.title }}</a></h3>
        <p>{{ paper.focus.join(' · ') }}</p>
        <div class="paper-tags"><span v-for="tag in tags(paper)" :key="tag">{{ tag }}</span></div>
        <footer><a :href="withBase(`/papers/${paper.slug}`)">进入学习页 →</a><a :href="paper.source_url" target="_blank" rel="noopener">原文 ↗</a></footer>
      </article>
    </div>
  </section>
</template>

<style scoped>
.paper-toolbar { display: flex; gap: 1rem; align-items: center; justify-content: space-between; margin: 1.5rem 0 1rem; }
.paper-search { display: flex; align-items: center; gap: .6rem; min-width: min(360px, 100%); padding: .72rem .9rem; border: 1px solid var(--line); border-radius: 10px; background: var(--vp-c-bg); }
.paper-search input { width: 100%; border: 0; outline: 0; color: var(--ink); background: transparent; }
.paper-filters { display: flex; flex-wrap: wrap; gap: .4rem; }
.paper-filters button { padding: .45rem .7rem; border: 1px solid var(--line); border-radius: 99px; color: var(--ink-muted); background: var(--vp-c-bg); cursor: pointer; }
.paper-filters button.active { border-color: var(--ink); color: #fff7ea; background: var(--ink); }
.paper-count { color: var(--ink-muted); font-size: .78rem; }
.paper-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1.2rem; }
.paper-card { display: flex; flex-direction: column; min-height: 250px; padding: 1.2rem; border: 1px solid var(--line); border-radius: 14px; background: var(--vp-c-bg); transition: transform .2s ease, box-shadow .2s ease; }
.paper-card:hover { transform: translateY(-3px); box-shadow: var(--shadow); }
.paper-top { display: flex; justify-content: space-between; align-items: center; }
.paper-order { color: var(--coral); font: 700 .72rem var(--vp-font-family-mono); }
.paper-level { padding: .2rem .45rem; border-radius: 99px; color: var(--ink); background: var(--mint); font-size: .66rem; }
.paper-card h3 { margin: .8rem 0 .6rem; font-size: 1rem; line-height: 1.45; }
.paper-card h3 a { color: var(--ink); text-decoration: none; }
.paper-card p { margin: 0 0 1rem; color: var(--ink-muted); font-size: .75rem; line-height: 1.6; }
.paper-tags { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: auto; }
.paper-tags span { padding: .25rem .45rem; border: 1px solid var(--line); border-radius: 5px; color: var(--ink-muted); font-size: .65rem; }
.paper-card footer { display: flex; justify-content: space-between; margin-top: .9rem; padding-top: .75rem; border-top: 1px solid var(--line); color: var(--ink-muted); font: .65rem var(--vp-font-family-mono); }
@media (max-width: 800px) { .paper-toolbar { align-items: stretch; flex-direction: column; }.paper-grid { grid-template-columns: 1fr; } }
</style>
