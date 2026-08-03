import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')
const sourcePath = resolve(root, 'study/03_kimi_k3_textbook.md')
const guideDir = resolve(root, 'site/guide')
const appendixDir = resolve(root, 'site/appendix')
const paperDir = resolve(root, 'site/papers')
const source = await readFile(sourcePath, 'utf8')
const lines = source.split('\n')

await mkdir(guideDir, { recursive: true })
await mkdir(appendixDir, { recursive: true })
await mkdir(paperDir, { recursive: true })

const leads = {
  0: '先不要记模块名。我们从六个“规模变大后哪里会坏”的问题出发，给整门课搭一张地图。',
  1: '这一章把你已经看到的 Transformer 骨架改造成能逐 token 生成的语言模型。重点是训练与推理的差别。',
  2: '一次生成请求其实有两个形状完全不同的阶段。理解它们，才能看懂后面所有缓存与服务优化。',
  3: 'MLA 的核心目标很具体：减少每层、每个历史 token 需要留下的 KV 状态。别把它误解成压缩一切。',
  4: 'MoE 像拥有很多科室、每次只去少数科室。它增加参数容量，却把单 token 计算控制在可接受范围。',
  5: '把 attention 从“保存全部历史”改写成“维护一个会擦写的记忆状态”，KDA 的故事就从这里开始。',
  6: '标准残差把历史层不断相加；AttnRes 让当前层按内容选择该回看哪一层。Attention 被旋转到了深度轴。',
  7: '896 选 16 不只是把专家数写大。低维路径、有限激活与负载平衡必须一起工作。',
  8: '“能看图”和“原生视觉”不是同义词。本章沿视觉 token 的数据流，看图像怎样进入统一自回归模型。',
  9: 'Scaling law、优化器和上下文课程回答三个不同问题：模型做多大、怎样训稳、怎样逐步学会更长文本。',
  10: 'Pre-training、SFT 与 RL 不是三个版本的同一种训练。它们使用不同数据，优化不同目标，也带来不同能力。',
  11: '多个专长教师怎样合回一个可部署模型？为什么量化与草稿模型要在后训练阶段就参与设计？',
  12: 'Agent 不是让模型多写几步思维链，而是模型、工具、环境、记忆、预算与验证器组成的闭环系统。',
  13: '当模型达到万亿参数、上下文达到百万 token，通信、状态与调度会成为模型能力能否落地的必要条件。',
  14: '论文表格不是事实清单。本章教你检查评测设置、成本口径、污染风险与证据强度。',
  15: '同一篇 K3 报告读三遍，每遍目标不同：先搭骨架，再拆机制，最后审系统与证据。',
  16: '把核心论文压成一条技术演化链。每张卡只回答：问题、机制、证据、与 K3 的关系和适用边界。'
}

const beginnerBridges = {
  0: '如果“自回归、MoE、KDA”仍然像缩写墙，请先学 [零基础第 52 课：K3 全景拼装](/beginner/09-k3-map)。那一页先按五个规模问题讲，不要求公式。',
  1: '如果还分不清 Attention、FFN、Residual 与 Norm，请先学 [零基础第 07 课：Transformer Block](/beginner/05-transformer)。',
  2: '如果 Prefill、Decode、KV Cache 是第一次见，请先学 [零基础第 12 课：生成与 KV Cache](/beginner/06-generation)，那里会用一次聊天请求逐步演示。',
  3: '本章默认已经理解 KV Cache。零基础请先完成 [第 12 课](/beginner/06-generation)，再回来只追踪“每个历史 token 到底保存什么”。',
  4: '本章原始教材写给工程师，确实会跳步。请先学 [零基础第 13 课：MoE 从零拆解](/beginner/07-moe)：它从一个 dense FFN、4 个专家、Top-2 手算开始，再进入这里的资源账本。',
  5: '先修只需要两个直觉：[Attention 是内容检索](/beginner/04-attention)，[KV Cache 保存历史状态](/beginner/06-generation)。学完再看固定状态怎样替代逐 token 档案。',
  6: '如果 residual stream 还不熟，请先回到 [零基础第 07 课](/beginner/05-transformer#_4-residual-不要每层都从头重写)，先把标准残差展开成历史更新之和。',
  7: '先用 [零基础第 13 课](/beginner/07-moe) 掌握 router、Top-k、shared expert 和 All-to-All；本章只增加 latent path 与三种稳定化组件。',
  8: '第一次接触视觉 token 时，先回看 [文字到 token](/beginner/01-token)，再把“文字片段”换成“图像 patch 经过视觉塔后的向量”。',
  9: 'Scaling law 讨论的是“固定预算怎样分配”，不是简单的大模型更强。需要时先复习 [损失、梯度与训练](/beginner/03-training)。',
  10: '先完成 [零基础第 17 课：从预训练到 Agent](/beginner/08-post-training)，确认每个阶段的数据来源和直接目标，再读算法细节。',
  11: '如果 on-policy 与 distillation 第一次出现，先看 [零基础第 17 课](/beginner/08-post-training#_7-蒸馏-把教师能力合进学生)。',
  12: '先把 Agent 当成系统而不是模型昵称：[零基础第 17 课](/beginner/08-post-training#_8-agent-不只是-多想几步)。',
  13: '本章会同时出现多种并行与缓存。先用 [零基础第 12 课](/beginner/06-generation) 分清 prefill/decode/cache，再逐项记计算、内存、通信三本账。',
  14: '这是“怎样判断论文是否证明了结论”的课。可先使用 [一页论文阅读模板](/start/paper-reading)，再带着模板读本章。',
  15: '不需要一次读懂整篇 K3。先完成 [零基础 K3 全景](/beginner/09-k3-map)，再按本章的三遍目标逐层深入。',
  16: '每篇论文现在都有独立的“类比 → 机制 → 证据 → K3 关系 → 原文地图”学习页，可从 [33 篇论文学习库](/papers/) 逐篇进入。'
}

const chapterVisuals = {
  0: ['beginner-k3-panorama.webp', 'K3 模型城市全景图', '把 K3 看成一座共同规划的城市：长序列、网络深度、专家通道、视觉入口和基础设施必须协同。'],
  1: ['transformer-reader-writer.webp', 'Transformer 阅读与写作数据流', '一边追踪 token 如何读取上下文，一边追踪下一个 token 如何产生；先看箭头，再看公式。'],
  2: ['generation-kv-archive.webp', 'Prefill、Decode 与 KV Cache 档案室', 'Prefill 一次建立历史档案；Decode 每步读取旧档案并追加一张新卡。'],
  3: ['guide-mla-compression.webp', 'MLA 把大量 KV 档案压成潜在表示', 'MLA 的目标不是压缩一切，而是减少每层、每个历史 token 需要长期保存的 KV 状态。'],
  4: ['moe-workshop.webp', 'MoE 专家工坊与稀疏路由', '每个 token 只去少数专家工坊；模型可以拥有很大容量，而单次不必启动所有参数。'],
  5: ['guide-kda-memory.webp', 'KDA 可擦写固定记忆板', '新信息不是无限叠加：门决定是否写入，delta 更新先擦掉冲突，再写入差值。'],
  6: ['guide-attnres-depth.webp', 'Attention Residuals 沿深度回看历史层', '当前层不只接收上一层，而是按内容从多层历史表示中选择需要的信息。'],
  7: ['guide-latentmoe.webp', 'LatentMoE 低维路由与专家通道', 'token 先经过低维路径，再访问少数专家与共享专家，最后稳定地展开回模型维度。'],
  8: ['guide-vision-tokens.webp', '图像切块并转成视觉 token', '图像先切成 patch，经视觉编码与压缩后变成 token，与文字 token 一起进入统一模型。'],
  9: ['guide-scaling-context.webp', '模型、数据、算力与上下文课程的平衡', '规模不是只把参数做大；模型、数据、训练预算与上下文阶段需要共同分配。'],
  10: ['beginner-learning-journey.webp', '从预训练到后训练与工具实践', '预训练、SFT、RL 与 Agent 环境使用不同数据和反馈，不能当成同一种训练反复执行。'],
  11: ['guide-multi-teacher.webp', '多位专长教师向一个学生模型蒸馏', '数学、代码、语言等教师把能力合入同一学生，量化与草稿模型也要提前进入部署设计。'],
  12: ['guide-agent-loop.webp', 'Agent 从规划、工具、观察到验证的闭环', 'Agent 能力来自模型与工具、环境、记忆、预算、验证器组成的循环，而不是模型昵称。'],
  13: ['guide-infrastructure.webp', '训练并行与在线服务基础设施', '模型分片、流水线、Prefill/Decode、缓存和调度共同决定超大模型能否训练并稳定服务。'],
  14: ['guide-evaluation.webp', '评测证据、成本与污染检查', '分数必须和基线、成本、延迟、污染风险与不确定性一起读，表格不是结论本身。'],
  15: ['guide-paper-three-pass.webp', '论文三遍阅读法', '第一遍画地图，第二遍拆机制，第三遍审证据；三遍目标不同，不要求第一次吃透所有公式。'],
  16: ['guide-paper-lineage.webp', 'K3 核心论文技术演化河流', '沿问题与代价追踪 Attention、MoE、长上下文、后训练和系统方法怎样逐步汇入 K3。']
}

const chapterSecondaryVisuals = {
  0: ['### 0.5 K3 的生命周期', 'posttraining-objectives-sparse.webp', '预训练、SFT、RL 与 Agent 环境的目标区别', '同一模型在四个阶段接收不同信号：token 监督、示范、结果奖励和工具环境反馈。'],
  1: ['### 1.4 next-token prediction 为什么能学到“知识”', 'training-loop-sparse.webp', '语言模型从前向预测到反向更新的训练循环', 'next-token 是直接目标；损失与反向传播把大量语境中的统计规律逐步写入参数。'],
  2: ['### 2.1 一次请求的两个阶段', 'prefill-decode-sparse.webp', 'Prefill 与 Decode 的不同计算形状', 'Prefill 并行读完整提示并建缓存；Decode 每步读取缓存、生成一个 token、再追加状态。'],
  3: ['### 3.1 从低秩联合压缩开始', 'mla-cache-compare-sparse.webp', '普通 KV Cache 与 MLA 压缩缓存对比', '左边每个头保存完整 K/V；右边主要缓存共享 latent，需要计算时再恢复各头内容。'],
  4: ['### 4.1 从 dense FFN 到专家集合', 'moe-routing-sparse.webp', 'MoE Router 将 token 稀疏分配给专家', '每个 token 只经过少数路由专家和共享专家；负载均衡与通信决定这种稀疏是否真的高效。'],
  5: ['### 5.4 Delta rule：读出旧值，再写误差', 'kda-delta-sparse.webp', 'KDA 固定状态中的读取、擦除与差值写入', '先读旧值，再擦掉与新目标冲突的部分，最后写入差值；状态大小不随序列增长。'],
  6: ['### 6.2 把序列 attention 的思想转到深度', 'attnres-depth-sparse.webp', '当前层按内容选择历史层表示', 'AttnRes 把 Query–Key 选择从 token 轴转到深度轴，因此当前层不必只接受上一层。'],
  7: ['### 7.2 低维 routed path 省了什么', 'latentmoe-sparse.webp', 'LatentMoE 先降维路由、再稳定合并升维', '跨专家传输和专家计算发生在较窄 latent 通道，最后再稳定地回到模型宽度。'],
  8: ['### 8.2 这和“图片变 token”不是一回事', 'vision-patch-pipeline-sparse.webp', '图像从 Patch 变成视觉 token 并进入统一模型', '图片先切块、编码和压缩；视觉 token 与文本 token 最终进入同一自回归模型。'],
  9: ['### 9.5 从 8K 到 1M 的 curriculum', 'scaling-curriculum-sparse.webp', '模型数据算力平衡与上下文渐进课程', 'Scaling 先平衡模型、数据与计算；长上下文再从 8K 逐级扩到 1M，避免早期浪费最昂贵计算。'],
  10: ['### 10.1 Pre-training：学习条件分布', 'posttraining-objectives-sparse.webp', '预训练、SFT、RL 与 Agent 实践目标', '四阶段优化的直接信号不同，不能把后训练理解为继续重复预训练。'],
  11: ['### 11.2 On-policy distillation 的关键差异', 'distillation-loop-sparse.webp', '学生当前轨迹接受多个教师指导并蒸馏', '教师不是只提供离线答案，而是围绕学生当前会走到的轨迹给出专长信号。'],
  12: ['### 12.5 Verifiable Agent tasks', 'prompt-plan-verify-sparse.webp', 'Agent 将目标拆成工具子任务并由验证器闭环检查', '规划、真实工具观察和验证器共同形成闭环；不通过时只重试失败步骤。'],
  13: ['### 13.10 Fleet scheduling', 'infrastructure-flow-sparse.webp', '请求经过调度、Prefill、KV 转移和 Decode 集群', '在线服务把读题与逐 token 生成分池，并依赖模型分片、缓存和全局调度。'],
  14: ['### 14.8 一条证据阶梯', 'evaluation-lenses-sparse.webp', '同一结果接受词面语义事实和成本多把尺检查', '一个 benchmark 分数只看一面；证据强度、成本、延迟与不确定性必须一起报告。'],
  15: ['### 15.1 第一遍：90 分钟，只搭骨架', 'paper-three-pass-sparse.webp', '同一篇论文的地图机制证据三遍阅读法', '三遍不是重复阅读：先画地图，再拆机制，最后审证据与边界。'],
  16: ['### 16.2　语料索引', 'paper-three-pass-sparse.webp', '用地图机制证据三个视角组织论文索引', '先按问题链建立地图，再进入机制和证据，避免把论文库读成互不相干的摘要。']
}

const chapterStarts = []
for (let i = 0; i < lines.length; i++) {
  const match = lines[i].match(/^## 第\s*(\d+)\s*章[　\s]*(.+)$/)
  if (match) chapterStarts.push({ index: i, id: Number(match[1]), title: match[2].trim() })
}
const appendixStart = lines.findIndex(line => /^# 附录 A/.test(line))

function normalizeLinks(text) {
  return text
    // VitePress MathJax uses dollar-delimited display math. The source textbook
    // intentionally keeps LaTeX's bracket form for editor portability.
    .replace(/^\\\[$/gm, '$$$$')
    .replace(/^\\\]$/gm, '$$$$')
    .replace(/\[PDF\]\(\.\.\/papers\/([^)]+\.pdf)\)/g, '[本地 PDF](/papers/$1)')
    .replace(/\[解析全文\]\(\.\.\/output\/papers\/[^)]+\)/g, '[论文学习库](/papers/)')
    .replace(/\[带页码解析全文\]\(\.\.\/output\/papers\/[^)]+\)/g, '[论文学习库](/papers/)')
    .replace(/<!-- END GENERATED PAPER DIGEST -->\n?/, '')
}

for (let pos = 0; pos < chapterStarts.length; pos++) {
  const chapter = chapterStarts[pos]
  const next = chapterStarts[pos + 1]?.index ?? appendixStart
  const bodyLines = lines.slice(chapter.index, next)
  bodyLines[0] = `# 第 ${chapter.id} 章　${chapter.title}`
  let body = normalizeLinks(bodyLines.join('\n').trim())
  const lead = `<div class="lesson-lead">${leads[chapter.id]}</div>`
  const bridge = `\n::: info 零基础入口\n${beginnerBridges[chapter.id]}\n:::\n`
  const [visualFile, visualAlt, visualCaption] = chapterVisuals[chapter.id]
  const visual = `<figure class="teaching-figure"><img src="/illustrations/${visualFile}" alt="${visualAlt}"><figcaption>${visualCaption}</figcaption></figure>\n`
  body = body.replace(/^(#[^\n]+)\n/, `$1\n\n${lead}\n${bridge}\n${visual}`)
  const [secondaryAnchor, secondaryFile, secondaryAlt, secondaryCaption] = chapterSecondaryVisuals[chapter.id]
  const secondaryVisual = `<figure class="teaching-figure concept-figure"><img src="/illustrations/${secondaryFile}" alt="${secondaryAlt}"><figcaption>${secondaryCaption}</figcaption></figure>`
  body = body.replace(secondaryAnchor, `${secondaryAnchor}\n\n${secondaryVisual}`)
  const id = String(chapter.id).padStart(2, '0')
  const frontmatter = `---\ntitle: 第 ${chapter.id} 章 ${chapter.title}\ndescription: Kimi K3 自学教程第 ${chapter.id} 章\n---\n\n`
  await writeFile(resolve(guideDir, `ch${id}.md`), frontmatter + body + '\n', 'utf8')
}

const appendixSpecs = [
  { marker: /^# 附录 A/, next: /^# 附录 B/, file: 'glossary.md', title: '核心术语表' },
  { marker: /^# 附录 B/, next: /^# 附录 C/, file: 'formulas.md', title: '公式速查' },
  { marker: /^# 附录 C/, next: /^# 附录 D/, file: 'pseudocode.md', title: '最小实现伪代码' },
  { marker: /^# 附录 D/, next: /^# 结语/, file: 'exercises.md', title: '综合练习与参考要点' }
]

for (const spec of appendixSpecs) {
  const start = lines.findIndex(line => spec.marker.test(line))
  const end = lines.findIndex((line, i) => i > start && spec.next.test(line))
  const body = lines.slice(start, end === -1 ? undefined : end).join('\n').trim()
  const frontmatter = `---\ntitle: ${spec.title}\ndescription: Kimi K3 自学教程附录\n---\n\n`
  await writeFile(resolve(appendixDir, spec.file), frontmatter + normalizeLinks(body) + '\n', 'utf8')
}

const roadmap = await readFile(resolve(root, 'study/02_learning_plan.md'), 'utf8')
await writeFile(
  resolve(root, 'site/roadmap.md'),
  `---\ntitle: 8 周学习路线\ndescription: 从零基础到读懂 Kimi K3 的每周计划\n---\n\n${roadmap}`,
  'utf8'
)

const paperIndex = JSON.parse(await readFile(resolve(root, 'output/papers/index.json'), 'utf8'))
const coreNotes = JSON.parse(await readFile(resolve(root, 'study/paper_notes.json'), 'utf8')).papers
const extraNotes = JSON.parse(await readFile(resolve(root, 'study/paper_extra_notes.json'), 'utf8')).papers
const notesBySlug = new Map([...coreNotes, ...extraNotes].map(note => [note.slug, note]))

const analogies = {
  kimi_k3: '像设计一座超大城市：道路、楼层、专业部门、学校和物流系统必须一起规划，单独拓宽一条路救不了整座城。',
  deepseek_v2: '像把每本借过的书都留下整本副本改成只留一张高密度索引卡，需要时再按索引恢复要用的信息。',
  gated_deltanet: '像可擦写白板：写新答案前先看同一位置的旧答案，只擦掉差值，而不是在上面无限叠字。',
  kimi_linear: '像“随身笔记 + 定期回图书馆”：大多数时候读固定大小的笔记，隔几层再做一次全库检索。',
  attention_residuals: '像写论文时不只看上一版草稿，而是根据当前问题，从所有历史版本里挑最有用的一版。',
  deepseek_moe: '像综合医院把大科室拆成更细专科，同时保留一个所有病人都先去的全科门诊。',
  deepseek_v3: '像高铁系统：车体、轨道、信号、调度与供电必须共同设计，单看某个零件解释不了整体速度。',
  latent_moe: '像专家会诊前先把厚病历压成标准摘要，减少每位专家要读和跨院传输的材料。',
  chinchilla: '像用固定预算备考：不能把钱全花在买更厚的书，也要留下足够时间真正做题。',
  kimi_k2: '像从“会回答”升级到“会办事”：底座、训练数据、工具环境和执行反馈一起决定能力。',
  kimi_k1_5: '像训练马拉松选手：不能只看最后是否到终点，还要处理超长轨迹、等待和中途续跑。',
  kimi_k2_5: '像让助手同时拥有眼睛、思考预算和多个协作者，并在真实工具环境中练习。',
  deepseek_r1: '像只告诉学生最终答案是否正确，让他通过大量尝试逐渐长出自己的解题策略。',
  fast_weight_programmers: '像一块由当前句子实时编程的便签板，读完一句就可以丢弃，不写进长期模型参数。',
  mamba2: '像同一段音乐既能逐音符播放，也能分小节并行排练；两种视角描述同一结构。',
  switch_transformers: '像分诊台每次只把病人送去一个科室，医院很大，但一次就诊不需要全院出动。',
  roformer_rope: '像给每个位置的箭头转不同角度，两支箭头的夹角自然记录它们相隔多远。',
  yarn: '像拉长一把多刻度尺：粗刻度可以压缩，细刻度要尽量保留，否则近距离分辨率会丢失。',
  muon_scalable: '像不只逐个拧音量旋钮，而是校正整张调音台的方向，让不同通道的更新更均衡。',
  instructgpt: '像先看老师示范，再学会给多个答案排序，最后根据评分规则反复练习。',
  dpo: '像直接从“A 比 B 好”的成对选择学习，不再先训练一个裁判、再让选手和裁判在线对练。',
  deepseek_math: '像数学集训：先补专业教材，再做大量有标准答案的练习，并按同题同组相对评分。',
  mopd: '像让学生自己答题，再把当前每一步分别拿给数学、写作、代码老师现场批改。',
  kimi_vl: '像把高分辨率图片切成视觉词块，再压缩成语言模型能够一起阅读的“视觉句子”。',
  react: '像边查资料边写报告：想一步、做一步、看结果，再决定下一步。',
  megatron_lm: '像多人搬一张巨桌：关键不是人多，而是从合适接缝切开，减少来回交接。',
  zero: '像多人合作时不再每个人都背一整套工具，而是把工具、材料和清单分开携带，用时再汇合。',
  flash_attention_2: '像在小厨房分批取食材并立刻烹饪，避免每一步都把整桌半成品搬回远处仓库。',
  ring_attention: '像围桌传阅资料：每个人保留自己的问题，资料包沿圆桌传一圈后，每个人都看过全部资料。',
  deepspeed_ulysses: '像先按书的章节分工，讨论人物时临时改成按人物分工，讨论完再换回章节分工。',
  pagedattention_vllm: '像操作系统的虚拟内存：读者看到连续书页，仓库里却可以把每页放在任何空闲格子。',
  mooncake: '像把备菜厨房与出餐窗口分开，再建设共享冷库，由总调度根据订单和缓存位置派工。',
  deepseek_v4: '像同时使用快速索引和高度压缩档案处理百万页资料，并重新设计楼层间的信息通道。'
}

function extractAbstract(text) {
  const match = text.match(/(?:^|\n)#{0,5}\s*Abstract\s*\n+([\s\S]*?)(?=\n#{1,5}\s|\n<!-- page:|\n- Table|\n\|)/i)
  if (!match) return ''
  return match[1]
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 1800)
}

function extractSections(text) {
  return [...text.matchAll(/^#{2,4}\s+(.+)$/gm)]
    .map(match => match[1].replace(/[#*_`]/g, '').trim())
    .filter(title => !/^(abstract|references|acknowledg|appendix)/i.test(title))
    .slice(0, 12)
}

const chapterNames = {
  0: 'K3 全景', 2: 'KV Cache', 3: 'MLA', 4: 'MoE', 5: 'KDA', 6: 'AttnRes',
  7: 'Stable LatentMoE', 8: '原生视觉', 9: 'Scaling 与长上下文', 10: 'SFT 与 RL',
  11: '蒸馏与部署', 12: 'Agent', 13: '训练与服务系统', 14: '论文评测', 15: '三遍阅读法'
}

for (const paper of paperIndex.papers) {
  const note = notesBySlug.get(paper.slug)
  if (!note) throw new Error(`Missing tutorial note for ${paper.slug}`)
  const parsed = await readFile(resolve(root, paper.markdown), 'utf8')
  const abstract = extractAbstract(parsed)
  const sections = extractSections(parsed)
  const chapterLinks = paper.textbook_chapters
    .filter(id => chapterNames[id])
    .map(id => `[第 ${id} 章 · ${chapterNames[id]}](/guide/ch${String(id).padStart(2, '0')})`)
    .join(' · ')
  const mechanism = note.mechanism.map((item, i) => `${i + 1}. **${['先看输入', '再看核心变化', '最后看输出', '系统如何执行', '为什么有效'][i] || `步骤 ${i + 1}`}**：${item}`).join('\n')
  const evidence = note.evidence.map(item => `- ${item}`).join('\n')
  const bridge = note.k3_bridge.map(item => `- ${item}`).join('\n')
  const caution = note.caution.map(item => `- ${item}`).join('\n')
  const sectionList = sections.map((title, i) => `${i + 1}. ${title}`).join('\n')
  const figure = paper.slug === 'flash_attention_2'
    ? `\n<figure class="paper-figure"><img src="/paper-figures/flash-attention-2/tiling.png" alt="FlashAttention 分块与在线 softmax 原论文示意图"><figcaption>原论文图：分块计算 attention，并通过重缩放得到精确结果。第一次看只追踪 Q、K、V 块的移动方向。</figcaption></figure>\n`
    : ''
  const content = `---
title: ${paper.title.replace(/:/g, '：')}
description: ${note.one_sentence.replace(/\$/g, '')}
---

# ${paper.title}

<div class="paper-lesson-meta"><span>${paper.order === 0 ? '主线论文' : paper.order <= 12 ? '核心精读' : '方向选读'}</span><span>${paper.pages} 页</span><span>arXiv ${paper.arxiv_id}</span></div>

<div class="lesson-lead">${note.one_sentence}</div>

## 先用一个类比

<div class="analogy-card"><strong>把它想成：</strong>${analogies[paper.slug]}</div>

类比只负责搭第一座桥。下面我们回到论文真正处理的对象、状态和证据。

## 它为什么会出现

**前一代方法遇到的问题：**${note.problem}

**它在整条学习链中的位置：**${note.position}

::: tip 新手阅读目标
第一次读完，只要求你能说清“旧方法哪里痛、它改变了哪一步、代价转移到了哪里”。不用一上来复现全部公式。
:::

## 核心机制：一步一步走

${mechanism}
${figure}
## 论文拿什么证明

${evidence}

这里要区分“论文测到了什么”和“我们为什么认为它有效”。前者是实验事实，后者可能仍是机制解释。

## 它怎样接到 Kimi K3

${bridge}

${chapterLinks ? `继续补背景：${chapterLinks}` : '这篇是方向扩展材料，先在论文链中理解它的位置即可。'}

## 不要从论文中过度推出什么

${caution}

## 原文应该怎么读

**推荐范围：**${note.reading}

<div class="paper-source-row"><a href="/${paper.pdf}" target="_blank">打开本地 PDF</a><a href="${paper.source_url}" target="_blank" rel="noopener">核对官方原文 ↗</a><span>解析语料：${paper.extraction.characters.toLocaleString()} 字符 · ${paper.extraction.headings} 个标题</span></div>

### 原文章节地图

${sectionList || '这份解析文本没有稳定提取出章节标题；请按 PDF 页码阅读。'}

${abstract ? `<details class="paper-abstract"><summary>展开查看论文英文摘要</summary>\n\n${abstract}\n\n</details>` : ''}

## 闭卷检查

<div class="checkpoint-card"><strong>合上论文后完成：</strong>${note.checkpoint}</div>

如果做不出来，不要重读整篇。只回到“核心机制”和原论文对应方法图，再用自己的话重画一次。
`
  await writeFile(resolve(paperDir, `${paper.slug}.md`), normalizeLinks(content), 'utf8')
}

console.log(`Generated ${chapterStarts.length} chapters, ${appendixSpecs.length} appendices, ${paperIndex.papers.length} paper tutorials, and the roadmap.`)
