<script setup lang="ts">
import { computed, ref } from 'vue'
type Preset={label:string;note:string;h:number;p:number;q:number;primitive:number}
const presets:Preset[]=[
 {label:'原始点击',note:'20 步网页操作',h:20,p:95,q:20,primitive:20},
 {label:'封装工具',note:'4 个可验证 API 动作',h:4,p:97,q:55,primitive:20},
 {label:'长程桌面',note:'错误状态难恢复',h:35,p:93,q:10,primitive:35}
]
const horizon=ref(20),stepAccuracy=ref(95),recovery=ref(20),primitiveSteps=ref(20),riskyRate=ref(10),approvalRecall=ref(90)
const p=computed(()=>stepAccuracy.value/100),q=computed(()=>recovery.value/100)
const neverMistake=computed(()=>p.value**horizon.value)
const trackSeries=computed(()=>{let on=1;const out=[on];for(let t=0;t<horizon.value;t++){on=on*p.value+(1-on)*q.value;out.push(on)}return out})
const finalOnTrack=computed(()=>trackSeries.value.at(-1)??0)
const checkpoints=computed(()=>[0,Math.round(horizon.value/4),Math.round(horizon.value/2),Math.round(horizon.value*3/4),horizon.value].filter((v,i,a)=>a.indexOf(v)===i).map(t=>({t,p:trackSeries.value[t]})))
const compression=computed(()=>primitiveSteps.value/horizon.value)
const expectedRisky=computed(()=>horizon.value*riskyRate.value/100)
const blockedRisky=computed(()=>expectedRisky.value*approvalRecall.value/100)
const escapedRisky=computed(()=>expectedRisky.value-blockedRisky.value)
function apply(x:Preset){horizon.value=x.h;stepAccuracy.value=x.p;recovery.value=x.q;primitiveSteps.value=x.primitive}
function pct(v:number){return `${(v*100).toFixed(1)}%`}
</script>

<template>
  <section class="lab-shell agent-shift-lab" aria-labelledby="agent-shift-title">
    <h3 id="agent-shift-title">交互实验：单步 95% 正确，为什么长程 Agent 仍会失败？</h3>
    <p class="lab-intro">每一步在正确轨道上时，以 p 保持正确；进入分布外状态后，每一步以 q 恢复。比较“从不犯错”和“允许恢复”的最终成功概率。</p>
    <div class="presets"><button v-for="x in presets" :key="x.label" type="button" @click="apply(x)"><strong>{{x.label}}</strong><span>{{x.note}}</span></button></div>
    <div class="controls">
      <label><span>决策步数 H</span><input v-model.number="horizon" type="range" min="1" max="50" step="1"><output>{{horizon}}</output></label>
      <label><span>轨道内单步正确 p</span><input v-model.number="stepAccuracy" type="range" min="70" max="100" step="0.5"><output>{{stepAccuracy.toFixed(1)}}%</output></label>
      <label><span>分布外恢复率 q</span><input v-model.number="recovery" type="range" min="0" max="100" step="5"><output>{{recovery}}%</output></label>
      <label><span>原始动作步数</span><input v-model.number="primitiveSteps" type="range" min="1" max="100" step="1"><output>{{primitiveSteps}}</output></label>
    </div>
    <div class="metrics"><div><small>整条轨迹从不犯错</small><strong>{{pct(neverMistake)}}</strong><span>p<sup>H</sup></span></div><div><small>含恢复的最终在轨率</small><strong>{{pct(finalOnTrack)}}</strong><span>P' = Pp + (1−P)q</span></div><div><small>动作抽象压缩</small><strong>{{compression.toFixed(1)}}×</strong><span>{{primitiveSteps}} 原始步 → {{horizon}} 决策步</span></div></div>
    <div class="track"><div v-for="point in checkpoints" :key="point.t"><span>t={{point.t}}</span><i><b :style="{width:pct(point.p)}"></b></i><strong>{{pct(point.p)}}</strong></div></div>
    <div class="safety">
      <label><span>危险动作比例</span><input v-model.number="riskyRate" type="range" min="0" max="50" step="1"><output>{{riskyRate}}%</output></label>
      <label><span>审批拦截率</span><input v-model.number="approvalRecall" type="range" min="0" max="100" step="5"><output>{{approvalRecall}}%</output></label>
      <p>每条轨迹期望出现 <strong>{{expectedRisky.toFixed(2)}}</strong> 次危险动作；审批拦截约 <strong>{{blockedRisky.toFixed(2)}}</strong> 次，仍可能漏过 <strong>{{escapedRisky.toFixed(2)}}</strong> 次。低概率 × 长时域并不自动安全。</p>
    </div>
    <p class="boundary"><strong>模型边界：</strong>每步概率被假设为固定且只分“在轨/离轨”两态；真实错误相关、状态难度不同，恢复动作也可能制造新风险。这里用于建立乘法直觉，不用于预测真实部署成功率。</p>
  </section>
</template>

<style scoped>
.agent-shift-lab{container-type:inline-size}.presets{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin:1rem 0}.presets button{padding:.7rem;border:1px solid var(--line);border-radius:11px;background:var(--vp-c-bg);color:var(--vp-c-text-1);text-align:left;cursor:pointer}.presets button:hover{border-color:var(--coral)}.presets strong,.presets span{display:block}.presets strong{font-size:.73rem}.presets span{margin-top:.16rem;color:var(--ink-muted);font-size:.61rem}.controls{display:grid;grid-template-columns:1fr 1fr;gap:.55rem}.controls label,.safety label{display:grid;grid-template-columns:145px 1fr 70px;gap:.45rem;align-items:center;padding:.55rem .65rem;border:1px solid var(--line);border-radius:10px;background:var(--vp-c-bg)}.controls span,.safety span{font-size:.65rem;font-weight:700}.controls output,.safety output{color:var(--coral);text-align:right;font:700 .65rem var(--vp-font-family-mono)}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin:1rem 0}.metrics>div{padding:.7rem;border-radius:11px;background:var(--vp-c-bg-soft)}.metrics small,.metrics strong,.metrics span{display:block}.metrics small,.metrics span{color:var(--ink-muted);font-size:.59rem}.metrics strong{margin:.2rem 0;color:var(--coral);font:800 1.15rem var(--vp-font-family-mono)}.track{display:grid;gap:.5rem;padding:.8rem;border:1px solid var(--line);border-radius:12px;background:var(--vp-c-bg)}.track>div{display:grid;grid-template-columns:45px 1fr 55px;gap:.5rem;align-items:center;font-size:.63rem}.track i{height:9px;border-radius:999px;background:var(--vp-c-bg-soft)}.track b{display:block;height:100%;border-radius:inherit;background:#5368d9}.track strong{text-align:right;font-family:var(--vp-font-family-mono)}.safety{display:grid;grid-template-columns:1fr 1fr;gap:.55rem;margin-top:1rem}.safety p{grid-column:1/3;margin:0;padding:.65rem .75rem;border-radius:10px;background:color-mix(in srgb,#dfa037 12%,var(--vp-c-bg));font-size:.65rem;line-height:1.55}.boundary{margin-bottom:0;color:var(--ink-muted);font-size:.63rem;line-height:1.6}
@container (max-width:680px){.controls,.metrics,.safety{grid-template-columns:1fr}.safety p{grid-column:1}}
@container (max-width:430px){.presets{grid-template-columns:1fr}.controls label,.safety label{grid-template-columns:1fr 62px}.controls input,.safety input{grid-column:1}.controls output,.safety output{grid-column:2;grid-row:2}}
</style>
