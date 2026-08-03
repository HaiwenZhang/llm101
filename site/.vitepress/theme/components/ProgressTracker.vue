<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

const lessons = [
  '主教程 00 · 模型、参数与训练', '主教程 01 · Token 与分词',
  '主教程 02 · 向量与 Word Embeddings', '主教程 03 · 多语言与 Token 公平性',
  '主教程 04 · 损失、梯度与训练', '主教程 05 · 语言模型演化',
  '主教程 06 · Attention', '主教程 07 · 完整 Transformer',
  '主教程 08 · BERT 与 Encoder-only', '主教程 09 · T5、BART 与 Encoder–Decoder',
  '主教程 10 · GPT、LLaMA 与 SSM', '主教程 11 · 架构全景',
  '主教程 12 · 生成与 KV Cache', '主教程 13 · MoE',
  '主教程 14 · 数据与 Scaling Laws', '主教程 15 · 训练工程与 GPU',
  '主教程 16 · 分布式训练', '主教程 17 · 后训练总览',
  '主教程 18 · Prompt 基础', '主教程 19 · Prompt 进阶',
  '主教程 20 · PEFT', '主教程 21 · LoRA', '主教程 22 · 模型编辑',
  '主教程 23 · SFT、RLHF、DPO 与推理 RL',
  '主教程 24 · 推理、验证器与测试时计算',
  '主教程 25 · LLM 作为 RL 问题', '主教程 26 · MDP 与价值函数',
  '主教程 27 · 策略梯度', '主教程 28 · Actor-Critic 与 GAE',
  '主教程 29 · PPO', '主教程 30 · RLHF 与 DPO',
  '主教程 31 · GRPO 与可验证奖励', '主教程 32 · Agent RL',
  '主教程 33 · RL 系统、评测与安全', '主教程 34 · 知识蒸馏',
  '主教程 35 · 解码与采样', '主教程 36 · 量化',
  '主教程 37 · 高效 Attention 与长上下文', '主教程 38 · 在线服务',
  '主教程 39 · RAG 架构', '主教程 40 · RAG 检索',
  '主教程 41 · RAG 生成与实践', '主教程 42 · Agent 与 Deep Research',
  '主教程 43 · 多模态与具身智能', '主教程 44 · 扩散模型与 Flow Matching',
  '主教程 45 · 大模型应用', '主教程 46 · 大模型评测',
  '主教程 47 · 高级评测与实验设计', '主教程 48 · 模型可解释性',
  '主教程 49 · 安全与攻击防护', '主教程 50 · 部署与成本',
  '主教程 51 · 研究方法', '主教程 52 · Kimi K3 全景拼装',
  '主教程 53 · K3 完整毕业项目',
  'K3-00 · Kimi K3 全景', 'K3-01 · 自回归语言模型', 'K3-02 · KV Cache',
  'K3-03 · MLA', 'K3-04 · MoE', 'K3-05 · KDA', 'K3-06 · AttnRes',
  'K3-07 · Stable LatentMoE', 'K3-08 · 原生视觉',
  'K3-09 · Scaling 与长上下文', 'K3-10 · SFT 与 RL',
  'K3-11 · 多教师蒸馏', 'K3-12 · Agent', 'K3-13 · 训练与服务系统',
  'K3-14 · 评测与论文判断', 'K3-15 · K3 三遍阅读法', 'K3-16 · 核心论文精读'
]
const done = ref<string[]>([])
const mounted = ref(false)
const percent = computed(() => Math.round(done.value.length / lessons.length * 100))

onMounted(() => {
  try {
    const saved = JSON.parse(localStorage.getItem('llm-system-course-progress-v2') || '[]')
    done.value = Array.isArray(saved) ? saved.filter(item => lessons.includes(item)) : []
  } catch { done.value = [] }
  mounted.value = true
})
watch(done, value => {
  if (mounted.value) localStorage.setItem('llm-system-course-progress-v2', JSON.stringify(value))
}, { deep: true })
</script>

<template>
  <section class="lab-shell progress-board" aria-labelledby="progress-title">
    <div class="progress-head">
      <div><h3 id="progress-title">我的学习进度</h3><p class="lab-intro">勾选已经真正讲得明白的章节，进度只保存在你的浏览器里。</p></div>
      <span class="progress-count">{{ done.length }} / {{ lessons.length }} · {{ percent }}%</span>
    </div>
    <div class="progress-rail" role="progressbar" :aria-valuenow="percent" aria-valuemin="0" aria-valuemax="100">
      <div class="progress-fill" :style="{ width: `${percent}%` }"></div>
    </div>
    <div class="progress-list">
      <label v-for="lesson in lessons" :key="lesson" class="progress-item" :class="{ done: done.includes(lesson) }">
        <input v-model="done" type="checkbox" :value="lesson">
        <span>{{ lesson }}</span>
      </label>
    </div>
  </section>
</template>
