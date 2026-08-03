<script setup lang="ts">
import { computed, ref } from 'vue'
type Preset={label:string;note:string;g:number;length:number;verify:number;concurrency:number;informative:number}
const presets:Preset[]=[
 {label:'数学验证',note:'答案检查快，生成主导',g:8,length:2048,verify:8,concurrency:256,informative:70},
 {label:'代码沙箱',note:'测试较慢，环境可能堵塞',g:8,length:3072,verify:800,concurrency:64,informative:55},
 {label:'工具 Agent',note:'轨迹长，环境尾延迟高',g:4,length:8192,verify:2500,concurrency:32,informative:45}
]
const prompts=ref(256),groupSize=ref(8),length=ref(2048),rolloutTps=ref(50000),trainTps=ref(180000)
const verifierMs=ref(8),verifierConcurrency=ref(256),informative=ref(70),policyLag=ref(1),maxLag=ref(3)
const samples=computed(()=>prompts.value*groupSize.value)
const tokensTotal=computed(()=>samples.value*length.value)
const rolloutSeconds=computed(()=>tokensTotal.value/rolloutTps.value)
const verifierSeconds=computed(()=>samples.value*verifierMs.value/1000/verifierConcurrency.value)
const trainSeconds=computed(()=>tokensTotal.value*informative.value/100/trainTps.value)
const cycleSeconds=computed(()=>Math.max(rolloutSeconds.value,verifierSeconds.value,trainSeconds.value))
const usefulTokens=computed(()=>tokensTotal.value*informative.value/100)
const wastedTokens=computed(()=>tokensTotal.value-usefulTokens.value)
const effectivePrompts=computed(()=>prompts.value*informative.value/100)
const stages=computed(()=>[
 {name:'Rollout',seconds:rolloutSeconds.value,color:'rollout'},
 {name:'Verifier',seconds:verifierSeconds.value,color:'verify'},
 {name:'Trainer',seconds:trainSeconds.value,color:'train'}
])
const bottleneck=computed(()=>stages.value.reduce((a,b)=>a.seconds>b.seconds?a:b).name)
const lagState=computed(()=>policyLag.value>maxLag.value?'danger':policyLag.value===maxLag.value?'warn':'good')
function apply(p:Preset){groupSize.value=p.g;length.value=p.length;verifierMs.value=p.verify;verifierConcurrency.value=p.concurrency;informative.value=p.informative}
function duration(v:number){return v<60?`${v.toFixed(1)} 秒`:`${(v/60).toFixed(1)} 分钟`}
function num(v:number){return v>=1e6?`${(v/1e6).toFixed(2)}M`:v>=1e3?`${(v/1e3).toFixed(1)}K`:v.toFixed(0)}
</script>

<template>
  <section class="lab-shell rollout-lab" aria-labelledby="rollout-lab-title">
    <h3 id="rollout-lab-title">交互实验：一轮 RL 到底卡在生成、验证还是训练？</h3>
    <p class="lab-intro">调节组大小、轨迹长度、验证耗时和有效组率。实验用流水线稳态近似比较三阶段吞吐，并把被过滤 rollout 的 token 单独记账。</p>
    <div class="presets"><button v-for="p in presets" :key="p.label" type="button" @click="apply(p)"><strong>{{p.label}}</strong><span>{{p.note}}</span></button></div>
    <div class="controls">
      <label><span>Prompt 数 B</span><input v-model.number="prompts" type="range" min="32" max="1024" step="32"><output>{{prompts}}</output></label>
      <label><span>每题样本 G</span><input v-model.number="groupSize" type="range" min="1" max="32" step="1"><output>{{groupSize}}</output></label>
      <label><span>平均长度</span><input v-model.number="length" type="range" min="256" max="16384" step="256"><output>{{num(length)}}</output></label>
      <label><span>Rollout tok/s</span><input v-model.number="rolloutTps" type="range" min="10000" max="300000" step="10000"><output>{{num(rolloutTps)}}</output></label>
      <label><span>Train tok/s</span><input v-model.number="trainTps" type="range" min="20000" max="500000" step="20000"><output>{{num(trainTps)}}</output></label>
      <label><span>单样本验证 ms</span><input v-model.number="verifierMs" type="range" min="0" max="5000" step="50"><output>{{verifierMs}}</output></label>
      <label><span>验证并发</span><input v-model.number="verifierConcurrency" type="range" min="1" max="512" step="1"><output>{{verifierConcurrency}}</output></label>
      <label><span>有效组率</span><input v-model.number="informative" type="range" min="5" max="100" step="5"><output>{{informative}}%</output></label>
    </div>
    <div class="stage-view">
      <div v-for="stage in stages" :key="stage.name" class="stage"><div><strong>{{stage.name}}</strong><span>{{duration(stage.seconds)}}</span></div><i><b :class="stage.color" :style="{width:`${Math.max(3,stage.seconds/cycleSeconds*100)}%`}"></b></i></div>
      <p>稳态瓶颈：<strong>{{bottleneck}}</strong>；一轮下界约 <strong>{{duration(cycleSeconds)}}</strong>。这是阶段可重叠时的理想近似，不含 P99 尾延迟和通信。</p>
    </div>
    <div class="metrics"><div><small>总 rollout</small><strong>{{num(tokensTotal)}} token</strong><span>{{samples.toLocaleString()}} 条轨迹</span></div><div><small>进入有效更新</small><strong>{{num(usefulTokens)}} token</strong><span>约 {{effectivePrompts.toFixed(0)}} 个 prompt</span></div><div><small>过滤/无效成本</small><strong>{{num(wastedTokens)}} token</strong><span>仍已消耗生成算力</span></div></div>
    <div class="lag"><label><span>样本策略落后版本数</span><input v-model.number="policyLag" type="range" min="0" max="10" step="1"><output>{{policyLag}}</output></label><label><span>允许最大版本差</span><input v-model.number="maxLag" type="range" min="0" max="10" step="1"><output>{{maxLag}}</output></label><p :class="lagState">{{lagState==='danger'?'样本超过版本门限：应丢弃、隔离或只做离线分析。':lagState==='warn'?'已到版本门限：ratio 与 clip fraction 需要重点检查。':'版本差在门限内，但仍要用实际 ratio/KL 判断陈旧程度。'}}</p></div>
    <p class="boundary"><strong>模型边界：</strong>各阶段被假设为稳定吞吐且能完全重叠；真实生成长度分布、沙箱超时、网络传输、checkpoint 发布和动态采样都会放大尾延迟。</p>
  </section>
</template>

<style scoped>
.rollout-lab{container-type:inline-size}.presets{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin:1rem 0}.presets button{padding:.7rem;border:1px solid var(--line);border-radius:11px;background:var(--vp-c-bg);color:var(--vp-c-text-1);text-align:left;cursor:pointer}.presets button:hover{border-color:var(--coral)}.presets strong,.presets span{display:block}.presets strong{font-size:.73rem}.presets span{margin-top:.16rem;color:var(--ink-muted);font-size:.61rem}.controls{display:grid;grid-template-columns:1fr 1fr;gap:.55rem}.controls label,.lag label{display:grid;grid-template-columns:130px 1fr 62px;gap:.45rem;align-items:center;padding:.55rem .65rem;border:1px solid var(--line);border-radius:10px;background:var(--vp-c-bg)}.controls span,.lag span{font-size:.65rem;font-weight:700}.controls output,.lag output{color:var(--coral);text-align:right;font:700 .65rem var(--vp-font-family-mono)}.stage-view{display:grid;gap:.65rem;margin:1rem 0;padding:.9rem;border:1px solid var(--line);border-radius:13px;background:var(--vp-c-bg)}.stage>div{display:flex;justify-content:space-between;font-size:.67rem}.stage i{display:block;height:10px;margin-top:.25rem;border-radius:999px;background:var(--vp-c-bg-soft)}.stage b{display:block;height:100%;border-radius:inherit}.stage b.rollout{background:#5368d9}.stage b.verify{background:#cf8b2d}.stage b.train{background:#28987d}.stage-view p{margin:.2rem 0 0;font-size:.66rem}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem}.metrics>div{padding:.7rem;border-radius:11px;background:var(--vp-c-bg-soft)}.metrics small,.metrics strong,.metrics span{display:block}.metrics small,.metrics span{color:var(--ink-muted);font-size:.59rem}.metrics strong{margin:.2rem 0;font:750 .9rem var(--vp-font-family-mono)}.lag{display:grid;grid-template-columns:1fr 1fr;gap:.55rem;margin-top:1rem}.lag p{grid-column:1/3;margin:0;padding:.65rem .75rem;border-radius:10px;font-size:.65rem}.lag p.good{background:color-mix(in srgb,#2a9d81 9%,var(--vp-c-bg))}.lag p.warn{background:color-mix(in srgb,#dfa037 12%,var(--vp-c-bg))}.lag p.danger{background:color-mix(in srgb,#d85f55 11%,var(--vp-c-bg))}.boundary{margin-bottom:0;color:var(--ink-muted);font-size:.63rem;line-height:1.6}
@container (max-width:700px){.controls,.metrics,.lag{grid-template-columns:1fr}.lag p{grid-column:1}}
@container (max-width:430px){.presets{grid-template-columns:1fr}.controls label,.lag label{grid-template-columns:1fr 58px}.controls input,.lag input{grid-column:1}.controls output,.lag output{grid-column:2;grid-row:2}}
</style>
