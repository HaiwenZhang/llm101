<script setup lang="ts">
import { computed, nextTick, onMounted, watch } from 'vue'
import { Content, useData, useRoute, withBase } from 'vitepress'
import DefaultTheme from 'vitepress/theme'

const { frontmatter } = useData()
const route = useRoute()
const isLanding = computed(() => frontmatter.value.layout === 'landing')

async function renderMermaid() {
  if (typeof window === 'undefined') return
  await nextTick()
  const nodes = document.querySelectorAll<HTMLElement>('.mermaid:not([data-processed])')
  if (!nodes.length) return
  const mermaid = (await import('mermaid')).default
  mermaid.initialize({
    startOnLoad: false,
    theme: document.documentElement.classList.contains('dark') ? 'dark' : 'neutral',
    securityLevel: 'loose',
    fontFamily: 'Inter, PingFang SC, system-ui, sans-serif',
    themeVariables: {
      primaryColor: '#d9eadb',
      primaryTextColor: '#123c38',
      primaryBorderColor: '#54877d',
      lineColor: '#e06b46',
      secondaryColor: '#f5dfaa',
      tertiaryColor: '#f7f2e9'
    }
  })
  for (const node of nodes) {
    try {
      await mermaid.run({ nodes: [node] })
    } catch (error) {
      node.dataset.processed = 'true'
      node.classList.add('mermaid-error')
      node.textContent = '图表暂时未能显示，请刷新页面。'
      console.error('Mermaid render failed', error)
    }
  }
}

onMounted(renderMermaid)
watch(() => route.path, renderMermaid)
</script>

<template>
  <div v-if="isLanding" class="landing-shell">
    <header class="landing-nav">
      <a class="landing-brand" :href="withBase('/')" aria-label="大模型系统课首页">
        <span class="brand-mark">LLM</span>
        <span>大模型系统课</span>
      </a>
      <nav aria-label="主导航">
        <a :href="withBase('/beginner/')">系统教程</a>
        <a :href="withBase('/curriculum/sources')">课程来源</a>
        <a :href="withBase('/guide/ch00')">K3 案例</a>
        <a :href="withBase('/labs/')">交互实验</a>
        <a :href="withBase('/papers/')">论文库</a>
      </nav>
      <a class="nav-cta" :href="withBase('/beginner/')">开始学习</a>
    </header>
    <main><Content /></main>
    <footer class="landing-footer">
      <strong>大模型系统课</strong>
      <span>从计算直觉、模型原理与训练系统，学到应用落地与真实模型案例。</span>
    </footer>
  </div>
  <DefaultTheme.Layout v-else />
</template>
