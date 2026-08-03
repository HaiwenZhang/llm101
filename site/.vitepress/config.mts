import { defineConfig } from 'vitepress'
import paperIndex from './theme/data/papers.json'

const base = process.env.VITEPRESS_BASE || '/'

const chapters = [
  ['00', 'Kimi K3 到底是什么'],
  ['01', '从 Transformer 走向语言模型'],
  ['02', 'Prefill、Decode 与 KV Cache'],
  ['03', 'MLA：压缩 KV'],
  ['04', 'MoE：稀疏的宽度'],
  ['05', '从线性 Attention 到 KDA'],
  ['06', 'Attention Residuals'],
  ['07', 'Stable LatentMoE'],
  ['08', 'MoonViT-V2 与原生视觉'],
  ['09', 'Scaling、Muon 与长上下文'],
  ['10', 'Pre-training、SFT 与 RL'],
  ['11', '多教师蒸馏与部署'],
  ['12', 'Agent：推理、行动、环境'],
  ['13', '从并行训练到在线服务'],
  ['14', '评测、成本与批判性阅读'],
  ['15', 'K3 原论文三遍阅读法'],
  ['16', '核心论文浓缩精读卡']
].map(([id, text]) => ({ text: `K3-${id} · ${text}`, link: `/guide/ch${id}` }))

const paperItems = paperIndex.papers.map(paper => ({
  text: `${String(paper.order).padStart(2, '0')} · ${paper.title}`,
  link: `/papers/${paper.slug}`
}))

const fullCourseSidebar = [
  { text: '学习总览', items: [
    { text: '大模型系统课 · 完整路线', link: '/beginner/' },
    { text: '名校课程知识覆盖表', link: '/curriculum/sources' },
    { text: '两门课逐页深度审计', link: '/curriculum/two-course-depth-audit' },
    { text: '8 周学习路线', link: '/roadmap' }
  ]},
  { text: '阶段一 · 先建立计算直觉', items: [
    { text: '00 · 模型、参数与训练', link: '/beginner/00-model' },
    { text: '01 · 从文字到 Token', link: '/beginner/01-token' },
    { text: '02 · 向量与 Word Embeddings', link: '/beginner/02-vector' },
    { text: '03 · 多语言与 Token 公平性', link: '/beginner/50-multilingual' },
    { text: '04 · 损失、梯度与训练', link: '/beginner/03-training' },
    { text: '05 · 语言模型的演化', link: '/beginner/10-language-models' }
  ]},
  { text: '阶段二 · 模型架构', collapsed: true, items: [
    { text: '06 · Attention 从零拆解', link: '/beginner/04-attention' },
    { text: '07 · 图解完整 Transformer', link: '/beginner/05-transformer' },
    { text: '08 · BERT 与 Encoder-only', link: '/beginner/14-bert' },
    { text: '09 · T5、BART 与 Encoder–Decoder', link: '/beginner/15-encoder-decoder' },
    { text: '10 · GPT、LLaMA、SSM 与 TTT', link: '/beginner/16-decoder-ssm' },
    { text: '11 · 架构全景', link: '/beginner/13-architectures' },
    { text: '12 · 生成、Prefill 与 KV Cache', link: '/beginner/06-generation' },
    { text: '13 · MoE 从零拆解', link: '/beginner/07-moe' }
  ]},
  { text: '阶段三 · 预训练与规模化', collapsed: true, items: [
    { text: '14 · Scaling：参数、数据与算力怎样配平', link: '/beginner/25-data-scaling' },
    { text: '15 · 自动微分、优化器、框架与 GPU', link: '/beginner/26-training-engineering' },
    { text: '16 · 分布式训练与通信', link: '/beginner/27-distributed-training' }
  ]},
  { text: '阶段四 · 后训练、适配与强化学习', collapsed: true, items: [
    { text: '17 · 从预训练到 Agent 总览', link: '/beginner/08-post-training' },
    { text: '18 · Prompt 与上下文学习', link: '/beginner/17-prompting' },
    { text: '19 · Prompt 进阶', link: '/beginner/18-prompt-advanced' },
    { text: '20 · PEFT 方法全景', link: '/beginner/19-peft' },
    { text: '21 · LoRA 深入与实验', link: '/beginner/20-lora' },
    { text: '22 · 模型编辑', link: '/beginner/21-model-editing' },
    { text: '23 · SFT、RLHF、DPO 与推理 RL', link: '/beginner/28-alignment-rl' },
    { text: '24 · 推理、验证器与测试时计算', link: '/beginner/49-reasoning-test-time' },
    { text: '25 · 把语言模型写成 RL 问题', link: '/beginner/40-rl-language-model' },
    { text: '26 · MDP、回报与价值函数', link: '/beginner/41-rl-mdp-value' },
    { text: '27 · 策略梯度与 REINFORCE', link: '/beginner/42-rl-policy-gradient' },
    { text: '28 · Actor-Critic 与 GAE', link: '/beginner/43-rl-actor-critic' },
    { text: '29 · 重要性采样、TRPO 与 PPO', link: '/beginner/44-rl-ppo' },
    { text: '30 · 奖励模型、RLHF 与 DPO', link: '/beginner/45-rlhf-preference' },
    { text: '31 · GRPO 与可验证奖励', link: '/beginner/46-verifiable-rewards' },
    { text: '32 · 离线 RL、探索与 Agent', link: '/beginner/47-rl-agent' },
    { text: '33 · RL 系统、评测与安全', link: '/beginner/48-rl-systems' },
    { text: '34 · 知识蒸馏与多教师学习', link: '/beginner/29-distillation' }
  ]},
  { text: '阶段五 · 高效推理与服务', collapsed: true, items: [
    { text: '35 · 解码与采样', link: '/beginner/11-decoding' },
    { text: '36 · 量化与低精度计算', link: '/beginner/30-quantization' },
    { text: '37 · FlashAttention 与长上下文', link: '/beginner/31-efficient-attention' },
    { text: '38 · vLLM 与在线服务', link: '/beginner/32-serving-systems' }
  ]},
  { text: '阶段六 · 知识、Agent 与应用', collapsed: true, items: [
    { text: '39 · RAG 基础架构', link: '/beginner/22-rag' },
    { text: '40 · RAG 检索系统', link: '/beginner/23-rag-retrieval' },
    { text: '41 · RAG 生成增强与实践', link: '/beginner/24-rag-generation' },
    { text: '42 · Agent、工具与 Deep Research', link: '/beginner/33-agents' },
    { text: '43 · 多模态、生成与具身智能', link: '/beginner/34-multimodal' },
    { text: '44 · 扩散模型与 Flow Matching', link: '/beginner/51-diffusion-flow' },
    { text: '45 · 大模型应用全景', link: '/beginner/35-applications' }
  ]},
  { text: '阶段七 · 评测、安全与落地', collapsed: true, items: [
    { text: '46 · 大模型评测基础', link: '/beginner/12-evaluation' },
    { text: '47 · 基准、LLM Judge 与实验设计', link: '/beginner/36-evaluation-research' },
    { text: '48 · 模型可解释性', link: '/beginner/52-interpretability' },
    { text: '49 · 安全与攻击防护', link: '/beginner/37-safety' },
    { text: '50 · 部署、监控与成本', link: '/beginner/38-deployment' },
    { text: '51 · 怎样做一次可信研究', link: '/beginner/39-research-method' }
  ]},
  { text: '阶段八 · Kimi K3 案例（独立章号）', collapsed: true, items: [
    { text: '52 · Kimi K3 全景拼装', link: '/beginner/09-k3-map' },
    ...chapters,
    { text: '53 · K3 完整毕业项目', link: '/beginner/53-k3-capstone' }
  ]},
  { text: '补充讲义与速查', collapsed: true, items: [
    { text: '数学热身', link: '/start/math' },
    { text: 'Transformer 第一课', link: '/start/transformer' },
    { text: '怎样读懂论文', link: '/start/paper-reading' },
    { text: '核心术语表', link: '/appendix/glossary' },
    { text: '公式速查', link: '/appendix/formulas' }
  ]}
]

const k3CaseSidebar = [
  { text: 'Kimi K3 · K3-00～K3-16', items: [
    ...chapters,
    { text: '毕业项目 · 从零拆解 K3', link: '/beginner/53-k3-capstone' }
  ] },
  { text: '先修与证据', collapsed: true, items: [
    { text: '返回大模型系统课', link: '/beginner/' },
    { text: '52 · K3 全景拼装', link: '/beginner/09-k3-map' },
    { text: 'K3 原论文与核心论文库', link: '/papers/' }
  ]}
]

const courseSourceSidebar = [
  { text: '课程来源说明', items: [
    { text: '名校课程知识覆盖表', link: '/curriculum/sources' },
    { text: 'CS224N + CMU 逐页审计', link: '/curriculum/two-course-depth-audit' },
    { text: '原始资料库首页', link: '/courses/' },
    { text: '返回大模型系统课', link: '/beginner/' }
  ]},
  { text: '七门来源课程', items: [
    { text: 'Stanford CS336 · 2026', link: '/courses/cs336-2026' },
    { text: 'Stanford CS224N · 2026', link: '/courses/cs224n-2026' },
    { text: '台湾大学 ADL · 2025', link: '/courses/ntu-adl-2025' },
    { text: 'CMU Advanced NLP · 2026', link: '/courses/cmu-anlp-2026' },
    { text: 'LLM Systems · 2025', link: '/courses/llm-systems-2025' },
    { text: 'CMU LLM Applications · 2026', link: '/courses/cmu-llm-applications-2026' },
    { text: 'Berkeley Deep RL · 2026', link: '/courses/berkeley-deeprl-2026' }
  ]}
]

export default defineConfig({
  base,
  lang: 'zh-CN',
  title: '大模型系统课',
  description: '从计算直觉、算法原理与训练系统，学到后训练、推理优化、应用、安全与真实模型案例',
  cleanUrls: true,
  lastUpdated: false,
  head: [
    ['meta', { name: 'theme-color', content: '#123c38' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['link', { rel: 'icon', href: `${base}favicon.png`, type: 'image/png' }]
  ],
  themeConfig: {
    logo: { src: '/logo.png', alt: '大模型系统课' },
    siteTitle: '大模型系统课',
    search: { provider: 'local' },
    nav: [
      { text: '系统课程', link: '/beginner/' },
      { text: 'K3 案例', link: '/guide/ch00' },
      { text: '交互实验', link: '/labs/' },
      { text: '课程来源', link: '/curriculum/sources' },
      { text: '论文库', link: '/papers/' },
      { text: '8 周路线', link: '/roadmap' }
    ],
    sidebar: {
      '/beginner/': fullCourseSidebar,
      '/foundations/': fullCourseSidebar,
      '/curriculum/': courseSourceSidebar,
      '/start/': [
        { text: '零基础起点', items: [
          { text: '先从这里开始', link: '/start/' },
          { text: '数学热身：只学会用的', link: '/start/math' },
          { text: 'Transformer 第一课', link: '/start/transformer' },
          { text: '怎样读懂一篇论文', link: '/start/paper-reading' }
        ]}
      ],
      '/guide/': k3CaseSidebar,
      '/labs/': [
        { text: '动手理解', items: [
          { text: '实验室首页', link: '/labs/' },
          { text: '文字进入模型', link: '/labs/token' },
          { text: 'BPE 逐轮合并', link: '/labs/bpe' },
          { text: '词向量方向实验', link: '/labs/vector' },
          { text: '单参数训练循环', link: '/labs/training' },
          { text: 'Attention 温度实验', link: '/labs/attention' },
          { text: 'MoE 路由模拟器', link: '/labs/moe' },
          { text: '参数量与 Scaling 配平', link: '/labs/scaling' },
          { text: 'RLHF 奖励与 KL', link: '/labs/rlhf' },
          { text: '下一词采样实验', link: '/labs/sampling' },
          { text: 'KV Cache 计算器', link: '/labs/kv-cache' },
          { text: '量化与权重显存', link: '/labs/quantization' },
          { text: 'RAG 检索与重排', link: '/labs/rag' }
        ]}
      ],
      '/papers/': [
        { text: '论文学习库', items: [
          { text: '全部 33 篇', link: '/papers/' },
          { text: '如何建立阅读顺序', link: '/papers/reading-map' }
        ]},
        { text: '核心精读 · 00–12', collapsed: true, items: paperItems.slice(0, 13) },
        { text: '方向选读 · 13–32', collapsed: true, items: paperItems.slice(13) }
      ],
      '/courses/': courseSourceSidebar,
      '/appendix/': [
        { text: '附录', items: [
          { text: '核心术语表', link: '/appendix/glossary' },
          { text: '公式速查', link: '/appendix/formulas' },
          { text: '最小实现伪代码', link: '/appendix/pseudocode' },
          { text: '综合练习与答案', link: '/appendix/exercises' }
        ]}
      ]
    },
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一课', next: '下一课' },
    darkModeSwitchLabel: '切换主题',
    lightModeSwitchTitle: '切换到浅色',
    darkModeSwitchTitle: '切换到深色',
    sidebarMenuLabel: '课程目录',
    returnToTopLabel: '回到顶部',
    langMenuLabel: '语言'
  },
  markdown: {
    lineNumbers: true,
    math: true,
    theme: { light: 'github-light', dark: 'github-dark' },
    config(md) {
      if (base !== '/') {
        const baseSegment = base.slice(1)
        const addBase = (html: string) => html.replace(
          /\bhref="\/(?!\/)([^"]*)"/g,
          (match, path) => path.startsWith(baseSegment)
            ? match
            : `href="${base}${path}"`
        )
        md.core.ruler.push('github-pages-raw-html-base', state => {
          const visit = (tokens: typeof state.tokens) => {
            for (const token of tokens) {
              if (token.type === 'html_block' || token.type === 'html_inline') {
                token.content = addBase(token.content)
              }
              if (token.children) visit(token.children)
            }
          }
          visit(state.tokens)
        })
      }

      const fence = md.renderer.rules.fence!
      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        if (tokens[idx].info.trim() === 'mermaid') {
          // `<pre>` keeps Mermaid's statement newlines during SSR/minification.
          // A normal `<div>` collapses them to spaces and turns valid diagrams
          // into one long, invalid Mermaid statement in the browser.
          return `<pre class="mermaid">${md.utils.escapeHtml(tokens[idx].content)}</pre>`
        }
        return fence(tokens, idx, options, env, self)
      }
    }
  }
})
