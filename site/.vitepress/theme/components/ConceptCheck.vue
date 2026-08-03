<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  question: string
  options: string[]
  answer: number
  explanation: string
}>()
const selected = ref<number | null>(null)
</script>

<template>
  <section class="concept-check">
    <div class="concept-question">想一想：{{ question }}</div>
    <div class="concept-options">
      <button v-for="(option, i) in props.options" :key="option" class="concept-option" :class="{ correct: selected !== null && i === answer, wrong: selected === i && i !== answer }" @click="selected = i">
        {{ String.fromCharCode(65 + i) }}. {{ option }}
      </button>
    </div>
    <p v-if="selected !== null" class="concept-feedback">{{ selected === answer ? '答对了。' : '再想一步。' }} {{ explanation }}</p>
  </section>
</template>
