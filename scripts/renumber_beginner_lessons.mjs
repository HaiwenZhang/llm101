import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

// Stable URLs deliberately keep their historical filename. Only the visible
// course number follows the pedagogical order below, so old bookmarks do not
// break when a lesson is moved.
const lessonOrder = [
  '00-model',
  '01-token',
  '02-vector',
  '50-multilingual',
  '03-training',
  '10-language-models',
  '04-attention',
  '05-transformer',
  '14-bert',
  '15-encoder-decoder',
  '16-decoder-ssm',
  '13-architectures',
  '06-generation',
  '07-moe',
  '25-data-scaling',
  '26-training-engineering',
  '27-distributed-training',
  '08-post-training',
  '17-prompting',
  '18-prompt-advanced',
  '19-peft',
  '20-lora',
  '21-model-editing',
  '28-alignment-rl',
  '49-reasoning-test-time',
  '40-rl-language-model',
  '41-rl-mdp-value',
  '42-rl-policy-gradient',
  '43-rl-actor-critic',
  '44-rl-ppo',
  '45-rlhf-preference',
  '46-verifiable-rewards',
  '47-rl-agent',
  '48-rl-systems',
  '29-distillation',
  '11-decoding',
  '30-quantization',
  '31-efficient-attention',
  '32-serving-systems',
  '22-rag',
  '23-rag-retrieval',
  '24-rag-generation',
  '33-agents',
  '34-multimodal',
  '51-diffusion-flow',
  '35-applications',
  '12-evaluation',
  '36-evaluation-research',
  '52-interpretability',
  '37-safety',
  '38-deployment',
  '39-research-method',
  '09-k3-map',
  '53-k3-capstone'
]

const numberBySlug = new Map(
  lessonOrder.map((slug, index) => [slug, String(index).padStart(2, '0')])
)

function renumberLinkLabel(label, slug) {
  const number = numberBySlug.get(slug)
  if (!number) return label

  let result = label.replace(
    /^(零基础)?第\s*\d{1,2}\s*课/,
    (_, prefix = '') => `${prefix}第 ${number} 课`
  )
  if (result !== label) return result

  result = label.replace(/^(\d{1,2})\s*[–—-]\s*(\d{1,2})(\s*·)/, (_, start, end, suffix) => {
    const width = Number(end) - Number(start)
    const newEnd = String(Number(number) + width).padStart(2, '0')
    return `${number}–${newEnd}${suffix}`
  })
  if (result !== label) return result

  return label.replace(/^\d{1,2}(\s*·)/, `${number}$1`)
}

function renumberMarkdown(text) {
  return text.replace(
    /\[([^\]]+)\]\(\/beginner\/([^/)#]+)(#[^)]+)?\)/g,
    (full, label, slug, hash = '') =>
      `[${renumberLinkLabel(label, slug)}](/beginner/${slug}${hash})`
  )
}

function markdownFiles(root) {
  const files = []
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const file = path.join(root, entry.name)
    if (entry.isDirectory()) files.push(...markdownFiles(file))
    else if (entry.isFile() && entry.name.endsWith('.md')) files.push(file)
  }
  return files
}

for (const [slug, number] of numberBySlug) {
  const file = path.join(repoRoot, 'site', 'beginner', `${slug}.md`)
  let text = fs.readFileSync(file, 'utf8')
  text = text.replace(/^title: 第\s*\d{1,2}\s*课/m, `title: 第 ${number} 课`)
  text = text.replace(/^# 第\s*\d{1,2}\s*课/m, `# 第 ${number} 课`)
  fs.writeFileSync(file, text)
}

for (const root of ['site', 'study']) {
  for (const file of markdownFiles(path.join(repoRoot, root))) {
    const text = renumberMarkdown(fs.readFileSync(file, 'utf8'))
    fs.writeFileSync(file, text)
  }
}

const configFile = path.join(repoRoot, 'site', '.vitepress', 'config.mts')
let config = fs.readFileSync(configFile, 'utf8')
config = config.replace(
  /(\{\s*text:\s*')\d{1,2}(\s*·[^']*',\s*link:\s*'\/beginner\/([^']+)')/g,
  (full, prefix, suffix, slug) => {
    const number = numberBySlug.get(slug)
    return number ? `${prefix}${number}${suffix}` : full
  }
)
fs.writeFileSync(configFile, config)

const validationErrors = []

for (const [slug, expected] of numberBySlug) {
  const file = path.join(repoRoot, 'site', 'beginner', `${slug}.md`)
  const text = fs.readFileSync(file, 'utf8')
  const titleNumber = text.match(/^title: 第\s*(\d{1,2})\s*课/m)?.[1]?.padStart(2, '0')
  const headingNumber = text.match(/^# 第\s*(\d{1,2})\s*课/m)?.[1]?.padStart(2, '0')
  if (titleNumber !== expected) {
    validationErrors.push(`${slug}: frontmatter is ${titleNumber ?? 'missing'}, expected ${expected}`)
  }
  if (headingNumber !== expected) {
    validationErrors.push(`${slug}: H1 is ${headingNumber ?? 'missing'}, expected ${expected}`)
  }
}

function visibleNumber(label) {
  const lesson = label.match(/^(?:零基础)?第\s*(\d{1,2})\s*课/)
  if (lesson) return lesson[1].padStart(2, '0')
  const compact = label.match(/^(\d{1,2})(?:\s*[–—-]\s*\d{1,2})?\s*·/)
  return compact?.[1]?.padStart(2, '0')
}

for (const file of markdownFiles(path.join(repoRoot, 'site'))) {
  const text = fs.readFileSync(file, 'utf8')
  const links = text.matchAll(/\[([^\]]+)\]\(\/beginner\/([^/)#]+)(?:#[^)]+)?\)/g)
  for (const [, label, slug] of links) {
    const actual = visibleNumber(label)
    const expected = numberBySlug.get(slug)
    if (actual && expected && actual !== expected) {
      validationErrors.push(`${path.relative(repoRoot, file)}: "${label}" is ${actual}, expected ${expected}`)
    }
  }
}

for (const [, label, slug] of config.matchAll(/text:\s*'([^']+)'[^\n]*link:\s*'\/beginner\/([^']+)'/g)) {
  const actual = visibleNumber(label)
  const expected = numberBySlug.get(slug)
  if (actual && expected && actual !== expected) {
    validationErrors.push(`sidebar: "${label}" is ${actual}, expected ${expected}`)
  }
}

if (validationErrors.length) {
  throw new Error(`Beginner lesson numbering is inconsistent:\n${validationErrors.join('\n')}`)
}

console.log(`Renumbered and validated ${lessonOrder.length} beginner lessons from 00 to 53.`)
