<script setup lang="ts">
import { withBase } from 'vitepress'

const stages = [
  { no: '01', range: '00–05', title: '建立计算直觉', subtitle: 'Token、向量、梯度与语言模型演化', meta: '真正零基础', href: '/beginner/00-model', tone: 'mint' },
  { no: '02', range: '06–13', title: '拆开模型架构', subtitle: 'Attention、Transformer、主流架构、生成与 MoE', meta: '算法原理', href: '/beginner/04-attention', tone: 'gold' },
  { no: '03', range: '14–16', title: '理解预训练与规模化', subtitle: '数据、Scaling Laws、GPU 与分布式训练', meta: '训练系统', href: '/beginner/25-data-scaling', tone: 'coral' },
  { no: '04', range: '17–34', title: '学习后训练与强化学习', subtitle: 'Prompt、PEFT、SFT、PPO、RLHF、GRPO 与 Agent RL', meta: '能力塑造', href: '/beginner/08-post-training', tone: 'mint' },
  { no: '05', range: '35–38', title: '掌握高效推理与服务', subtitle: '解码、量化、FlashAttention、KV Cache 与在线服务', meta: '系统优化', href: '/beginner/11-decoding', tone: 'gold' },
  { no: '06', range: '39–45', title: '构建大模型应用', subtitle: 'RAG、Agent、Deep Research、多模态与扩散生成', meta: '真实场景', href: '/beginner/22-rag', tone: 'coral' },
  { no: '07', range: '46–51', title: '评测、安全与部署', subtitle: 'Benchmark、LLM Judge、可解释性、红队、监控与研究方法', meta: '可信落地', href: '/beginner/12-evaluation', tone: 'mint' },
  { no: '08', range: '52–53', title: '完成真实模型案例', subtitle: '用 Kimi K3 串联架构、训练、Agent、服务与证据', meta: '毕业案例', href: '/beginner/09-k3-map', tone: 'gold' }
]

const systemLayers = [
  { tag: '表示', name: 'Token → Embedding', copy: '文字与图像怎样变成模型可以计算的向量', x: 'top' },
  { tag: '架构', name: 'Attention → Transformer', copy: '信息怎样沿序列、深度与专家通道流动', x: 'middle' },
  { tag: '训练', name: 'Pretrain → SFT → RL', copy: '参数怎样从语料、示范、偏好与奖励中改变', x: 'bottom' },
  { tag: '落地', name: 'RAG → Agent → Serving', copy: '模型怎样接入知识、工具、评测、安全与部署', x: 'service' }
]

const materials = [
  {
    label: '主线教材',
    metric: '54 课',
    title: '按先修关系重新组织',
    copy: '不是把大学课件依次粘贴，而是从一个概念只增加一步开始，连续学到完整大模型系统。',
    href: '/beginner/'
  },
  {
    label: '理解工具',
    metric: '12 个实验',
    title: '概念不只靠文字解释',
    copy: '教学图、最小公式、手算例子、张量形状和交互实验共同解释“输入、状态、输出”。',
    href: '/labs/'
  },
  {
    label: '证据体系',
    metric: '7 门课 + 33 论文',
    title: '每个结论都能回到来源',
    copy: '373 份课程资料与核心论文用于溯源；Kimi K3 作为最后的真实模型综合案例。',
    href: '/curriculum/sources'
  }
]
</script>

<template>
  <div class="home-page">
    <section class="hero-section">
      <div class="hero-copy">
        <div class="eyebrow"><span></span> 大语言模型 · 系统自学教材</div>
        <h1>从 Token 到系统，<br><em>完整学懂大模型。</em></h1>
        <p class="hero-lead">为零基础学习者重组的一条连续知识链：从向量、Attention 和 Transformer，走到数据与预训练、后训练与强化学习、高效推理、RAG、Agent、多模态、评测、安全和部署，最后用真实前沿模型检验理解。</p>
        <div class="hero-actions">
          <a class="primary-action" :href="withBase('/beginner/00-model')">开始第 00 课 <span>→</span></a>
          <a class="text-action" :href="withBase('/beginner/')">查看完整课程目录</a>
        </div>
        <div class="hero-proof">
          <div><strong>54</strong><span>节连续主教程</span></div>
          <div><strong>17</strong><span>章真实模型案例</span></div>
          <div><strong>7</strong><span>门名校课程融合</span></div>
          <div><strong>33</strong><span>篇核心论文精读</span></div>
        </div>
      </div>

      <div class="concept-stack" aria-label="大语言模型课程完整知识链">
        <div class="stack-caption">课程不是模块清单，而是一条数据流</div>
        <div v-for="(layer, index) in systemLayers" :key="layer.name" class="concept-layer" :class="layer.x">
          <div class="layer-index">0{{ index + 1 }}</div>
          <div class="layer-body">
            <span>{{ layer.tag }}</span>
            <strong>{{ layer.name }}</strong>
            <small>{{ layer.copy }}</small>
          </div>
          <div class="layer-signal" aria-hidden="true"><i></i><i></i><i></i></div>
        </div>
        <div class="stack-input"><span>数据</span><span>算力</span><span>反馈</span><span>真实任务</span></div>
      </div>
    </section>

    <section class="promise-strip" aria-label="教学特点">
      <span>先看图建立直觉</span><i></i><span>再用小数字手算</span><i></i><span>补最小公式与资源账本</span><i></i><span>回到论文核对证据</span>
    </section>

    <section class="material-section">
      <header class="section-heading compact-heading">
        <span class="section-kicker">这套教材现在包含什么</span>
        <h2>一套课程，三层学习材料</h2>
        <p>主教程负责建立体系，图解与实验负责真正理解，课程原稿和论文负责深挖与核对。</p>
      </header>
      <div class="material-grid">
        <a v-for="material in materials" :key="material.label" :href="withBase(material.href)" class="material-card">
          <div><span>{{ material.label }}</span><strong>{{ material.metric }}</strong></div>
          <h3>{{ material.title }}</h3>
          <p>{{ material.copy }}</p>
          <b>进入查看 →</b>
        </a>
      </div>
    </section>

    <section class="path-section">
      <header class="section-heading">
        <span class="section-kicker">完整教学主线</span>
        <h2>八个阶段，从计算直觉走到真实系统</h2>
        <p>54 节课已经按依赖关系连续编号。初学者沿顺序学习；有基础的读者可以根据课号范围直接进入对应阶段。</p>
      </header>
      <div class="stage-list">
        <a v-for="stage in stages" :key="stage.no" :href="withBase(stage.href)" class="stage-card" :class="stage.tone">
          <span class="stage-no">{{ stage.no }}</span>
          <div>
            <h3>{{ stage.title }}</h3>
            <p>{{ stage.subtitle }}</p>
          </div>
          <span class="stage-range">第 {{ stage.range }} 课</span>
          <span class="stage-meta">{{ stage.meta }}</span>
          <span class="stage-arrow">↗</span>
        </a>
      </div>
    </section>

    <section class="method-section">
      <div class="method-copy">
        <span class="section-kicker">每一课怎么学</span>
        <h2>掰开揉碎，<br>一层只加一个新东西</h2>
        <p>一页不靠堆术语显得专业。先建立画面，再追踪输入、状态和输出，最后才进入公式、工程代价与论文证据。</p>
        <a :href="withBase('/start/paper-reading')">看看本站的论文阅读法 →</a>
      </div>
      <ol class="method-steps">
        <li><span>1</span><div><strong>图解与生活类比</strong><small>先知道它在解决什么问题，形成可以复述的画面</small></div></li>
        <li><span>2</span><div><strong>小数字与最小公式</strong><small>手算一次，再引入当下真正需要的符号</small></div></li>
        <li><span>3</span><div><strong>张量与资源账本</strong><small>同时追踪形状、计算、显存、通信和延迟</small></div></li>
        <li><span>4</span><div><strong>实验与原始证据</strong><small>动手改变参数，并区分原文事实和教学推论</small></div></li>
      </ol>
    </section>

    <section class="lab-teaser">
      <div class="lab-mini-chart" aria-hidden="true">
        <span style="height: 22%"></span><span style="height: 35%"></span><span style="height: 62%"></span><span style="height: 88%"></span><span style="height: 54%"></span><span style="height: 76%"></span>
      </div>
      <div>
        <span class="section-kicker">不只看，还要动手</span>
        <h2>亲手算一次，胜过多背十遍</h2>
        <p>逐轮训练 BPE、点击一次梯度更新、调节 Scaling 与 KL 系数，再观察采样、缓存、量化、MoE 路由和 RAG 证据怎样变化。</p>
      </div>
      <a :href="withBase('/labs/')">进入交互实验室 →</a>
    </section>

    <section class="source-section">
      <div>
        <span class="source-big">14,488</span>
        <span class="source-label">页名校课程资料<br>已结构化解析并逐讲映射</span>
      </div>
      <p>七门课程、161 个讲次、373 份资料已经合入同一条教学主线；《大模型基础》提供系统底座，33 篇核心论文负责机制与证据，Kimi K3 负责最后的整机案例。</p>
      <a :href="withBase('/curriculum/sources')">查看课程与论文来源</a>
    </section>

    <section class="final-cta">
      <span>START</span>
      <h2>先把“模型”真正弄明白。</h2>
      <p>第 00 课不要求数学和编程背景。学完以后，你会先分清程序、模型、参数、训练和推理，再继续走向 Token 与向量。</p>
      <a :href="withBase('/beginner/00-model')">开始第 00 课 <b>→</b></a>
    </section>
  </div>
</template>
