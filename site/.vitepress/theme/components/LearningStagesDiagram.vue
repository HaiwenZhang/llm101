<template>
  <section class="learning-stages" aria-labelledby="learning-stages-title">
    <div class="diagram-head">
      <p>一张图看完能力形成过程</p>
      <h3 id="learning-stages-title">同一个模型，四次不同的“上课”</h3>
    </div>

    <div class="stage-track">
      <article class="stage-card pretrain">
        <span class="stage-no">01</span>
        <div class="stage-icon" aria-hidden="true">书海</div>
        <h4>Pre-training</h4>
        <strong>学世界中的规律</strong>
        <p>输入海量文本、图像和代码，练习预测下一个 token。</p>
        <small>直接信号：正确 token</small>
      </article>
      <i class="track-arrow" aria-hidden="true">→</i>
      <article class="stage-card sft">
        <span class="stage-no">02</span>
        <div class="stage-icon" aria-hidden="true">示范</div>
        <h4>SFT</h4>
        <strong>学会怎样回答</strong>
        <p>模仿高质量回答与工具轨迹，把有用行为调到高概率区域。</p>
        <small>直接信号：老师答案</small>
      </article>
      <i class="track-arrow" aria-hidden="true">→</i>
      <article class="stage-card rl">
        <span class="stage-no">03</span>
        <div class="stage-icon" aria-hidden="true">试错</div>
        <h4>Preference / RL</h4>
        <strong>学会为结果负责</strong>
        <p>自己采样轨迹，由人类偏好、环境或 verifier 给反馈。</p>
        <small>直接信号：比较或奖励</small>
      </article>
      <i class="track-arrow" aria-hidden="true">→</i>
      <article class="stage-card deploy">
        <span class="stage-no">04</span>
        <div class="stage-icon" aria-hidden="true">上岗</div>
        <h4>Distill &amp; Serve</h4>
        <strong>整合并部署能力</strong>
        <p>把多位教师的能力合入学生，并适配量化与在线服务。</p>
        <small>直接约束：质量、速度、成本</small>
      </article>
    </div>

    <div class="agent-loop">
      <b>模型成为 Agent 后：</b>
      <span>观察</span><i>→</i><span>决定</span><i>→</i><span>调用工具</span><i>→</i><span>环境变化</span><i>↺</i>
      <small>这是一套运行闭环，不是第五次训练。</small>
    </div>
  </section>
</template>

<style scoped>
.learning-stages {
  margin: 1.7rem 0 2.3rem;
  padding: 1.25rem;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: var(--vp-c-bg-soft);
  box-shadow: var(--shadow);
}

.diagram-head p { margin: 0; color: var(--coral-dark); font: 700 .68rem var(--vp-font-family-mono); letter-spacing: .1em; }
.diagram-head h3 { margin: .2rem 0 1.1rem; font-size: 1.2rem; }
.stage-track { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr; align-items: stretch; gap: .45rem; }
.stage-card { position: relative; min-width: 0; padding: .9rem .75rem .8rem; border: 1px solid var(--line); border-top: 4px solid var(--mint); background: var(--vp-c-bg); }
.stage-card.sft { border-top-color: var(--gold); }
.stage-card.rl { border-top-color: var(--coral); }
.stage-card.deploy { border-top-color: var(--vp-c-brand-1); }
.stage-no { position: absolute; top: .55rem; right: .6rem; color: var(--ink-muted); font: .62rem var(--vp-font-family-mono); }
.stage-icon { display: grid; width: 2.65rem; height: 2.65rem; place-items: center; margin-bottom: .7rem; border-radius: 50%; color: var(--ink); background: color-mix(in srgb, var(--mint) 55%, var(--vp-c-bg)); font-size: .68rem; font-weight: 800; }
.stage-card.sft .stage-icon { background: color-mix(in srgb, var(--gold) 45%, var(--vp-c-bg)); }
.stage-card.rl .stage-icon { background: color-mix(in srgb, var(--coral) 30%, var(--vp-c-bg)); }
.stage-card.deploy .stage-icon { color: #fff; background: var(--vp-c-brand-1); }
.stage-card h4 { margin: 0; font: 700 .72rem var(--vp-font-family-mono); }
.stage-card strong { display: block; margin: .25rem 0 .35rem; font-size: .82rem; }
.stage-card p { margin: 0; color: var(--ink-muted); font-size: .72rem; line-height: 1.6; }
.stage-card small { display: block; margin-top: .65rem; padding-top: .55rem; border-top: 1px dashed var(--line); color: var(--ink-muted); font-size: .62rem; }
.track-arrow { align-self: center; color: var(--coral); font-size: 1.2rem; font-style: normal; }
.agent-loop { display: flex; flex-wrap: wrap; align-items: center; gap: .45rem; margin-top: 1rem; padding: .75rem .85rem; border-left: 4px solid var(--gold); background: var(--vp-c-bg); font-size: .72rem; }
.agent-loop b { margin-right: .25rem; }
.agent-loop span { padding: .2rem .4rem; border: 1px solid var(--line); background: var(--vp-c-bg-soft); }
.agent-loop i { color: var(--coral); font-style: normal; }
.agent-loop small { width: 100%; color: var(--ink-muted); }

@media (max-width: 760px) {
  .stage-track { grid-template-columns: 1fr; }
  .track-arrow { justify-self: center; transform: rotate(90deg); }
}
</style>
