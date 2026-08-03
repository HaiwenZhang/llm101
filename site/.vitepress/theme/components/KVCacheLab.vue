<script setup lang="ts">
import { computed, ref, watch } from 'vue'

const sequence = ref(32768)
const layers = ref(64)
const heads = ref(64)
const kvHeads = ref(8)
const headDim = ref(128)
const batch = ref(1)
const bytes = ref(2)

watch(heads, value => {
  if (kvHeads.value > value) kvHeads.value = value
})

const gib = (n: number) => n / 1024 ** 3
const mha = computed(() => gib(batch.value * sequence.value * layers.value * heads.value * headDim.value * 2 * bytes.value))
const gqa = computed(() => gib(batch.value * sequence.value * layers.value * kvHeads.value * headDim.value * 2 * bytes.value))
// 教学近似：每个 token 每层保留一个 576 维 latent 和 64 维解耦位置分量。
const mla = computed(() => gib(batch.value * sequence.value * layers.value * (576 + 64) * bytes.value))
const format = (n: number) => n < 1 ? `${(n * 1024).toFixed(0)} MiB` : `${n.toFixed(n > 10 ? 1 : 2)} GiB`
</script>

<template>
  <section class="lab-shell" aria-labelledby="kv-title">
    <h3 id="kv-title">KV Cache 显存计算器</h3>
    <p class="lab-intro">拖动参数，比较 MHA、GQA 和 MLA 在生成时需要保留多少历史状态。</p>
    <div class="control-row"><label for="seq">上下文长度 T</label><input id="seq" v-model.number="sequence" type="range" min="2048" max="1048576" step="2048"><output>{{ sequence.toLocaleString() }}</output></div>
    <div class="control-row"><label for="layers">层数 L</label><input id="layers" v-model.number="layers" type="range" min="8" max="128" step="8"><output>{{ layers }}</output></div>
    <div class="control-row"><label for="heads">Query 头数 H</label><input id="heads" v-model.number="heads" type="range" min="8" max="128" step="8"><output>{{ heads }}</output></div>
    <div class="control-row"><label for="kvheads">KV 头数 Hkv</label><input id="kvheads" v-model.number="kvHeads" type="range" min="1" :max="heads" step="1"><output>{{ kvHeads }}</output></div>
    <div class="control-row"><label for="headdim">每头维度 d<sub>h</sub></label><input id="headdim" v-model.number="headDim" type="range" min="32" max="256" step="32"><output>{{ headDim }}</output></div>
    <div class="control-row"><label for="batch">并发请求 B</label><input id="batch" v-model.number="batch" type="range" min="1" max="32" step="1"><output>{{ batch }}</output></div>
    <div class="control-row"><label for="kvbytes">每元素字节</label><input id="kvbytes" v-model.number="bytes" type="range" min="1" max="4" step="1"><output>{{ bytes }} B</output></div>
    <div class="metric-grid">
      <div class="metric-card"><small>标准 MHA</small><strong>{{ format(mha) }}</strong><span>保存所有 head 的 K 和 V</span></div>
      <div class="metric-card"><small>GQA</small><strong>{{ format(gqa) }}</strong><span>多个 query 头共享 KV</span></div>
      <div class="metric-card"><small>MLA 教学近似</small><strong>{{ format(mla) }}</strong><span>缓存低维 latent 状态</span></div>
    </div>
    <p class="teach-note"><strong>观察：</strong>上下文和并发翻倍，缓存也线性翻倍。MLA 主要压缩的是推理时的 KV 状态，不等于把整个模型或全部计算都压缩了。</p>
  </section>
</template>
