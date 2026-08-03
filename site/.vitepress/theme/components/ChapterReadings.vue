<script setup lang="ts">
import { withBase } from 'vitepress'
import readings from '../data/chapter-readings.json'

type Lecture = {
  course: string
  session: string
  title: string
  label: string
  url: string
  pages: number
}

type Paper = {
  level: '必读' | '选读'
  role: string
  title: string
  url: string
  guide_url?: string
  pages: number
  source: string
  question: string
}

type ChapterReading = {
  lesson_title: string
  reading_question: string
  lectures: Lecture[]
  papers: Paper[]
}

const props = defineProps<{ lesson: string }>()
const chapter = (readings as Record<string, ChapterReading>)[props.lesson]
</script>

<template>
  <section v-if="chapter" class="chapter-readings" aria-labelledby="chapter-readings-title">
    <h2 id="chapter-readings-title">本章参考论文与推荐阅读</h2>
    <p class="reading-intro"><strong>不用一次读完。</strong>先完成“必读”的摘要、方法总图和结论，再带着一个问题回到正文：{{ chapter.reading_question }}</p>

    <div v-if="chapter.lectures.length" class="lecture-sources">
      <h3>对应课程讲义</h3>
      <div class="lecture-link-grid">
        <a v-for="lecture in chapter.lectures" :key="lecture.url" :href="withBase(lecture.url)" target="_blank">
          <span>{{ lecture.course }} · {{ lecture.session }}</span>
          <strong>{{ lecture.title }}</strong>
          <small>{{ lecture.label }}<template v-if="lecture.pages"> · {{ lecture.pages }} 页</template></small>
        </a>
      </div>
    </div>

    <div class="paper-reading-list">
      <article v-for="paper in chapter.papers" :key="paper.url" class="paper-reading-card">
        <div class="paper-reading-meta">
          <span :class="['reading-level', { optional: paper.level === '选读' }]">{{ paper.level }}</span>
          <span>{{ paper.role }}</span>
          <span>{{ paper.source }}<template v-if="paper.pages"> · {{ paper.pages }} 页</template></span>
        </div>
        <h3><a :href="withBase(paper.url)" target="_blank">{{ paper.title }}</a></h3>
        <p><strong>读时只回答：</strong>{{ paper.question }}</p>
        <a v-if="paper.guide_url" class="paper-guide-link" :href="withBase(paper.guide_url)">先看中文精读页 →</a>
      </article>
    </div>

    <p class="reading-method"><strong>闭卷标准：</strong>合上 PDF 后，用自己的话写出“问题 → 核心改动 → 关键证据 → 代价/边界”四行；写不出哪一行，就只回看对应页面，不必从头重读。</p>
  </section>
</template>

<style scoped>
.chapter-readings { margin-top: 2.8rem; padding-top: .2rem; }
.reading-intro { margin: .4rem 0 1.2rem; padding: .9rem 1rem; border-left: 4px solid var(--gold); border-radius: 0 10px 10px 0; color: var(--ink-muted); background: color-mix(in srgb, var(--gold) 10%, var(--vp-c-bg)); line-height: 1.75; }
.reading-intro strong { color: var(--vp-c-text-1); }
.lecture-sources h3 { margin: 1.35rem 0 .65rem; font-size: .95rem; }
.lecture-link-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .65rem; }
.lecture-link-grid a { display: grid; gap: .18rem; padding: .8rem .9rem; border: 1px solid var(--line); border-radius: 11px; color: var(--vp-c-text-1); background: var(--vp-c-bg-soft); text-decoration: none; }
.lecture-link-grid a:hover { border-color: var(--vp-c-brand-1); transform: translateY(-1px); }
.lecture-link-grid span, .lecture-link-grid small { color: var(--ink-muted); font-size: .68rem; }
.lecture-link-grid strong { font-size: .82rem; line-height: 1.45; }
.paper-reading-list { display: grid; gap: .75rem; margin-top: 1rem; }
.paper-reading-card { padding: 1rem 1.05rem; border: 1px solid var(--line); border-radius: 12px; background: var(--vp-c-bg); }
.paper-reading-meta { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem .65rem; color: var(--ink-muted); font-size: .68rem; }
.reading-level { padding: .12rem .42rem; border-radius: 999px; color: var(--vp-c-bg); background: var(--vp-c-brand-1); font-weight: 700; }
.reading-level.optional { color: var(--vp-c-text-2); background: var(--vp-c-bg-soft); }
.paper-reading-card h3 { margin: .55rem 0 .35rem; font-size: .94rem; line-height: 1.45; }
.paper-reading-card h3 a { color: var(--vp-c-text-1); }
.paper-reading-card p { margin: 0; color: var(--ink-muted); font-size: .78rem; line-height: 1.7; }
.paper-reading-card p strong { color: var(--vp-c-text-1); }
.paper-guide-link { display: inline-block; margin-top: .45rem; font-size: .72rem; font-weight: 700; }
.reading-method { margin-top: .9rem; padding: .85rem 1rem; border-radius: 10px; color: var(--ink-muted); background: var(--vp-c-bg-soft); font-size: .78rem; line-height: 1.7; }
.reading-method strong { color: var(--vp-c-text-1); }
@media (max-width: 720px) {
  .lecture-link-grid { grid-template-columns: 1fr; }
}
</style>
