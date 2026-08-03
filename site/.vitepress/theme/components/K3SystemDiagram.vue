<template>
  <section class="k3-map" aria-labelledby="k3-map-title">
    <div class="map-title">
      <p>不要背模块清单</p>
      <h3 id="k3-map-title">先问“它在解决哪一种规模问题？”</h3>
    </div>

    <div class="map-canvas">
      <article class="axis-card sequence">
        <span>序列轴</span>
        <h4>怎样读百万 token？</h4>
        <p><b>KDA</b> 固定状态递推</p>
        <p><b>MLA</b> 压缩后的全局检索</p>
      </article>
      <article class="axis-card depth">
        <span>深度轴</span>
        <h4>怎样找回早期表示？</h4>
        <p><b>AttnRes</b> 在历史层之间做选择</p>
      </article>
      <div class="core-card">
        <small>共同核心</small>
        <strong>Kimi K3</strong>
        <p>token 表示在序列、深度、宽度和模态之间流动</p>
      </div>
      <article class="axis-card width">
        <span>宽度轴</span>
        <h4>怎样扩到 2.8T？</h4>
        <p><b>LatentMoE</b> 896 选 16</p>
        <p><b>Shared experts</b> 保留通用路径</p>
      </article>
      <article class="axis-card vision">
        <span>视觉轴</span>
        <h4>图片怎样变成同类输入？</h4>
        <p><b>MoonViT-V2</b> → 视觉 token</p>
      </article>
    </div>

    <div class="foundation-row">
      <div><b>能力线</b><span>预训练 → SFT → 多领域 RL → 多教师蒸馏</span></div>
      <div><b>系统线</b><span>并行 kernel、通信、缓存、Agent 环境与调度</span></div>
    </div>
  </section>
</template>

<style scoped>
.k3-map {
  margin: 1.7rem 0 2.3rem;
  padding: 1.25rem;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: var(--vp-c-bg-soft);
  box-shadow: var(--shadow);
}

.map-title { text-align: center; }
.map-title p { margin: 0; color: var(--coral-dark); font: 700 .68rem var(--vp-font-family-mono); letter-spacing: .1em; }
.map-title h3 { margin: .2rem 0 1.15rem; font-size: 1.2rem; }
.map-canvas { display: grid; grid-template-columns: 1fr .9fr 1fr; grid-template-areas: 'sequence core width' 'depth core vision'; gap: .65rem; }
.axis-card { min-width: 0; padding: .8rem; border: 1px solid var(--line); background: var(--vp-c-bg); }
.axis-card.sequence { grid-area: sequence; border-left: 4px solid var(--coral); }
.axis-card.depth { grid-area: depth; border-left: 4px solid var(--gold); }
.axis-card.width { grid-area: width; border-right: 4px solid var(--vp-c-brand-1); }
.axis-card.vision { grid-area: vision; border-right: 4px solid var(--mint); }
.axis-card span { color: var(--ink-muted); font: 700 .62rem var(--vp-font-family-mono); }
.axis-card h4 { margin: .2rem 0 .45rem; font: 700 .84rem var(--vp-font-family-base); }
.axis-card p { margin: .15rem 0; color: var(--ink-muted); font-size: .7rem; line-height: 1.55; }
.core-card { grid-area: core; display: grid; align-content: center; justify-items: center; padding: 1rem; border: 2px solid var(--vp-c-brand-1); background: var(--ink); color: #f8f3e9; text-align: center; }
.core-card small { color: var(--mint); font-size: .62rem; }
.core-card strong { display: block; margin: .2rem 0; font: 800 1.35rem 'Noto Serif SC', serif; }
.core-card p { margin: 0; color: #dce9e4; font-size: .68rem; line-height: 1.55; }
.foundation-row { display: grid; grid-template-columns: 1fr 1fr; gap: .65rem; margin-top: .65rem; }
.foundation-row div { padding: .7rem .8rem; border-top: 3px solid var(--coral); background: var(--vp-c-bg); }
.foundation-row div:last-child { border-top-color: var(--gold); }
.foundation-row b, .foundation-row span { display: block; }
.foundation-row b { margin-bottom: .2rem; font-size: .76rem; }
.foundation-row span { color: var(--ink-muted); font-size: .67rem; line-height: 1.5; }

@media (max-width: 700px) {
  .map-canvas { grid-template-columns: 1fr; grid-template-areas: 'core' 'sequence' 'depth' 'width' 'vision'; }
  .core-card { min-height: 9rem; }
  .axis-card.width, .axis-card.vision { border-right: 1px solid var(--line); border-left: 4px solid var(--vp-c-brand-1); }
  .axis-card.vision { border-left-color: var(--mint); }
  .foundation-row { grid-template-columns: 1fr; }
}
</style>
