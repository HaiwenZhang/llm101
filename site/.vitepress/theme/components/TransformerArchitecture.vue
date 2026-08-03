<script setup lang="ts">
import { ref } from 'vue'

type View = 'overview' | 'encoder' | 'decoder' | 'modern'

const active = ref<View>('overview')
const views: { id: View; label: string }[] = [
  { id: 'overview', label: '整机全景' },
  { id: 'encoder', label: '只看编码器' },
  { id: 'decoder', label: '只看解码器' },
  { id: 'modern', label: '对比现代 LLM' }
]
</script>

<template>
  <section class="transformer-explorer" aria-labelledby="transformer-explorer-title">
    <div class="explorer-head">
      <div>
        <p class="eyebrow">可交互结构图</p>
        <h3 id="transformer-explorer-title">把原始 Transformer 拆成四张图</h3>
      </div>
      <p>先看全景，再一次只追一条数据流。</p>
    </div>

    <div class="view-tabs" role="tablist" aria-label="Transformer 结构视图">
      <button
        v-for="view in views"
        :id="`tab-${view.id}`"
        :key="view.id"
        type="button"
        role="tab"
        :aria-selected="active === view.id"
        :aria-controls="`panel-${view.id}`"
        :class="{ active: active === view.id }"
        @click="active = view.id"
      >
        {{ view.label }}
      </button>
    </div>

    <div
      v-if="active === 'overview'"
      id="panel-overview"
      class="view-panel"
      role="tabpanel"
      aria-labelledby="tab-overview"
    >
      <div class="overview-grid">
        <div class="architecture-column encoder-column">
          <div class="column-title"><span>输入侧</span><strong>编码器 Encoder</strong></div>
          <div class="sentence-card"><small>原句</small>机器 学习 很 有趣</div>
          <span class="flow-arrow">↓</span>
          <div class="stage neutral">词向量 + 位置编码</div>
          <span class="flow-arrow">↓</span>
          <div class="stack-shell">
            <b>Encoder Block × N</b>
            <div class="stage attention">双向 Self-Attention</div>
            <div class="stage compute">逐位置 FFN</div>
            <small>每个子层外都有 Residual + Norm</small>
          </div>
          <span class="flow-arrow">↓</span>
          <div class="memory-card"><small>整句的编码记忆</small><strong>提供 K、V</strong></div>
        </div>

        <div class="cross-bridge" aria-label="编码器记忆流向解码器交叉注意力">
          <span>编码记忆</span><b>→</b><small>Cross-Attention</small>
        </div>

        <div class="architecture-column decoder-column">
          <div class="column-title"><span>输出侧</span><strong>解码器 Decoder</strong></div>
          <div class="sentence-card"><small>已经生成</small>&lt;开始&gt; Machine learning</div>
          <span class="flow-arrow">↓</span>
          <div class="stage neutral">词向量 + 位置编码</div>
          <span class="flow-arrow">↓</span>
          <div class="stack-shell">
            <b>Decoder Block × N</b>
            <div class="stage attention">带遮罩的 Self-Attention</div>
            <div class="stage cross">Cross-Attention</div>
            <div class="stage compute">逐位置 FFN</div>
            <small>每个子层外都有 Residual + Norm</small>
          </div>
          <span class="flow-arrow">↓</span>
          <div class="stage output">Linear + Softmax → 下一个词</div>
        </div>
      </div>
      <p class="panel-takeaway"><b>一句话：</b>编码器先把原句读完整；解码器一边看自己已经写出的内容，一边查询编码器留下的整句记忆。</p>
    </div>

    <div
      v-else-if="active === 'encoder'"
      id="panel-encoder"
      class="view-panel"
      role="tabpanel"
      aria-labelledby="tab-encoder"
    >
      <ol class="step-flow">
        <li><span>1</span><div><b>输入表示</b><p>每个 token 的词向量加上位置编码，得到带有“内容 + 顺序”的向量。</p></div></li>
        <li><span>2</span><div><b>双向 Self-Attention</b><p>每个位置都可以读取整句中的其他位置；它没有“不能看未来”的生成限制。</p></div></li>
        <li><span>3</span><div><b>Add &amp; Norm</b><p>把注意力结果加回原输入，再做 LayerNorm。原始论文采用这种 PostNorm 顺序。</p></div></li>
        <li><span>4</span><div><b>逐位置 FFN</b><p>每个位置独立通过同一套小网络，不在这里交换 token 间的信息。</p></div></li>
        <li><span>5</span><div><b>Add &amp; Norm，再重复 N 层</b><p>最终得到一组上下文化向量，供每一个 Decoder block 查询。</p></div></li>
      </ol>
      <p class="panel-takeaway"><b>检查点：</b>编码器不是把一句话压成一个向量，而是输出“一串已经互相交流过的向量”。</p>
    </div>

    <div
      v-else-if="active === 'decoder'"
      id="panel-decoder"
      class="view-panel"
      role="tabpanel"
      aria-labelledby="tab-decoder"
    >
      <div class="qkv-origin">
        <div><small>Decoder 当前状态</small><b>提供 Query</b><span>“我现在要找什么？”</span></div>
        <i>查询</i>
        <div><small>Encoder 整句记忆</small><b>提供 Key、Value</b><span>“原句里哪里有答案？”</span></div>
      </div>
      <ol class="step-flow compact">
        <li><span>1</span><div><b>Masked Self-Attention</b><p>当前位置只能看见已经生成的前缀，不能偷看正确答案后面的词。</p></div></li>
        <li><span>2</span><div><b>Cross-Attention</b><p>用 Decoder 的 Query 去匹配 Encoder 的 Key，再按权重读取 Encoder 的 Value。</p></div></li>
        <li><span>3</span><div><b>FFN 与输出头</b><p>继续加工后，由 Linear 得到全词表 logits，Softmax 变成下一个 token 的概率。</p></div></li>
      </ol>
      <p class="panel-takeaway"><b>最容易混淆的点：</b>Decoder 自注意力的 Q/K/V 都来自 Decoder；交叉注意力只有 Q 来自 Decoder，K/V 来自 Encoder。</p>
    </div>

    <div
      v-else
      id="panel-modern"
      class="view-panel"
      role="tabpanel"
      aria-labelledby="tab-modern"
    >
      <div class="comparison-grid">
        <article>
          <small>2017 原始 Transformer</small>
          <h4>Encoder–Decoder</h4>
          <p>输入原句与输出前缀走两条路径。Decoder 通过 Cross-Attention 查询 Encoder。</p>
          <code>Encoder × N → Decoder × N</code>
        </article>
        <div class="comparison-arrow">→</div>
        <article class="modern-card">
          <small>GPT 类现代大语言模型</small>
          <h4>Decoder-only</h4>
          <p>提示词和回答排在同一串 token 中，统一通过因果自注意力处理。</p>
          <code>Causal Block × N → logits</code>
        </article>
      </div>
      <div class="difference-list">
        <p><b>删掉：</b>独立 Encoder 与 Cross-Attention。</p>
        <p><b>保留：</b>多头注意力、FFN、Residual、Norm、逐 token 输出。</p>
        <p><b>继续演化：</b>位置方法、Norm 放置、Attention 和 FFN 的内部实现。</p>
      </div>
      <p class="panel-takeaway"><b>所以：</b>这张经典架构图是理解 Transformer 的祖谱，但不能原封不动地当作 Kimi K3 的结构图。</p>
    </div>
  </section>
</template>

<style scoped>
.transformer-explorer {
  margin: 1.8rem 0 2.4rem;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: var(--vp-c-bg-soft);
  box-shadow: var(--shadow);
}

.explorer-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1.5rem;
  padding: 1.25rem 1.35rem 1rem;
}

.explorer-head h3 { margin: .1rem 0 0; font-size: 1.25rem; }
.explorer-head > p { max-width: 18rem; margin: .2rem 0 0; color: var(--ink-muted); font-size: .82rem; }
.eyebrow { margin: 0; color: var(--coral-dark); font: 700 .68rem var(--vp-font-family-mono); letter-spacing: .12em; }

.view-tabs {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  padding: 0 1.35rem;
  border-bottom: 1px solid var(--line);
}

.view-tabs button {
  padding: .75rem .5rem;
  border: 0;
  border-bottom: 3px solid transparent;
  color: var(--ink-muted);
  background: transparent;
  cursor: pointer;
  font: 700 .78rem var(--vp-font-family-base);
}

.view-tabs button:hover { color: var(--vp-c-text-1); }
.view-tabs button.active { border-bottom-color: var(--coral); color: var(--vp-c-text-1); }
.view-tabs button:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: -2px; }

.view-panel { padding: 1.4rem; background: var(--vp-c-bg); }
.overview-grid { display: grid; grid-template-columns: 1fr 96px 1fr; align-items: center; gap: 1rem; }
.architecture-column { min-width: 0; text-align: center; }
.column-title { display: flex; align-items: baseline; justify-content: space-between; gap: .7rem; margin-bottom: .7rem; text-align: left; }
.column-title span { color: var(--ink-muted); font-size: .72rem; }
.column-title strong { font: 700 .98rem var(--vp-font-family-base); }
.sentence-card, .memory-card, .stage, .stack-shell { border: 1px solid var(--line); background: var(--vp-c-bg-soft); }
.sentence-card { padding: .75rem; font-weight: 700; }
.sentence-card small, .memory-card small { display: block; margin-bottom: .15rem; color: var(--ink-muted); font-size: .66rem; font-weight: 500; }
.flow-arrow { display: block; height: 1.5rem; color: var(--ink-muted); line-height: 1.5rem; }
.stage { padding: .6rem .7rem; font-size: .78rem; font-weight: 700; }
.stack-shell { display: grid; gap: .45rem; padding: .65rem; border-width: 2px; border-color: var(--vp-c-brand-1); }
.stack-shell > b { color: var(--vp-c-brand-1); font: 700 .72rem var(--vp-font-family-mono); }
.stack-shell > small { color: var(--ink-muted); font-size: .64rem; }
.stage.attention { border-color: color-mix(in srgb, var(--coral) 60%, var(--line)); background: color-mix(in srgb, var(--coral) 10%, var(--vp-c-bg)); }
.stage.cross { border-color: color-mix(in srgb, var(--gold) 75%, var(--line)); background: color-mix(in srgb, var(--gold) 14%, var(--vp-c-bg)); }
.stage.compute { border-color: color-mix(in srgb, var(--mint) 85%, var(--line)); background: color-mix(in srgb, var(--mint) 20%, var(--vp-c-bg)); }
.stage.output { color: #fff; border-color: var(--vp-c-brand-1); background: var(--vp-c-brand-1); }
.memory-card { padding: .7rem; border-color: var(--vp-c-brand-1); }
.memory-card strong { display: block; color: var(--vp-c-brand-1); font-size: .82rem; }
.cross-bridge { display: grid; justify-items: center; gap: .15rem; color: var(--ink-muted); text-align: center; }
.cross-bridge span, .cross-bridge small { font-size: .65rem; }
.cross-bridge b { color: var(--coral); font-size: 1.8rem; line-height: 1; }

.panel-takeaway { margin: 1.25rem 0 0; padding: .8rem .95rem; border-left: 4px solid var(--gold); background: color-mix(in srgb, var(--gold) 11%, var(--vp-c-bg)); font-size: .84rem; line-height: 1.7; }
.step-flow { display: grid; gap: .75rem; margin: 0; padding: 0; list-style: none; }
.step-flow li { display: grid; grid-template-columns: 2rem 1fr; gap: .8rem; align-items: start; }
.step-flow li > span { display: grid; width: 2rem; height: 2rem; place-items: center; border-radius: 50%; color: #fff; background: var(--vp-c-brand-1); font: 700 .75rem var(--vp-font-family-mono); }
.step-flow b { font-size: .9rem; }
.step-flow p { margin: .12rem 0 0; color: var(--ink-muted); font-size: .8rem; line-height: 1.65; }
.step-flow.compact { margin-top: 1.2rem; }
.qkv-origin { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 1rem; }
.qkv-origin > div { padding: .9rem; border: 1px solid var(--line); background: var(--vp-c-bg-soft); }
.qkv-origin small, .qkv-origin span { display: block; color: var(--ink-muted); font-size: .68rem; }
.qkv-origin b { display: block; margin: .15rem 0; font-size: .9rem; }
.qkv-origin i { color: var(--coral); font-size: .72rem; font-style: normal; font-weight: 700; }
.comparison-grid { display: grid; grid-template-columns: 1fr 2.5rem 1fr; align-items: stretch; gap: .8rem; }
.comparison-grid article { padding: 1rem; border: 1px solid var(--line); background: var(--vp-c-bg-soft); }
.comparison-grid article.modern-card { border-color: var(--vp-c-brand-1); }
.comparison-grid small { color: var(--ink-muted); font-size: .68rem; }
.comparison-grid h4 { margin: .25rem 0 .5rem; font: 700 1.05rem var(--vp-font-family-base); }
.comparison-grid p { margin: 0 0 .75rem; color: var(--ink-muted); font-size: .78rem; line-height: 1.65; }
.comparison-grid code { display: block; padding: .5rem; white-space: normal; font-size: .68rem; }
.comparison-arrow { display: grid; place-items: center; color: var(--coral); font-size: 1.6rem; }
.difference-list { display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; margin-top: 1rem; }
.difference-list p { margin: 0; padding: .7rem; border-top: 3px solid var(--mint); background: var(--vp-c-bg-soft); font-size: .75rem; line-height: 1.55; }

@media (max-width: 700px) {
  .explorer-head { display: block; }
  .explorer-head > p { max-width: none; }
  .view-tabs { grid-template-columns: repeat(2, 1fr); padding: 0 .7rem; }
  .overview-grid { grid-template-columns: 1fr; }
  .cross-bridge { grid-template-columns: auto auto auto; justify-content: center; }
  .cross-bridge b { transform: rotate(90deg); }
  .qkv-origin, .comparison-grid, .difference-list { grid-template-columns: 1fr; }
  .qkv-origin i, .comparison-arrow { transform: rotate(90deg); justify-self: center; }
}
</style>
