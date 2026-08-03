import { readFile, readdir, writeFile } from 'node:fs/promises'
import { extname, join, resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')
const site = join(root, 'site')
const courseManifest = JSON.parse(
  await readFile(join(root, 'output/course-materials/combined-manifest.json'), 'utf8')
)
const paperIndex = JSON.parse(
  await readFile(join(root, 'output/papers/index.json'), 'utf8')
)

const replacements = new Map()

for (const course of courseManifest.courses) {
  for (const session of course.sessions) {
    for (const item of [...(session.materials || []), ...(session.readings || [])]) {
      if (item.status !== 'available' || !item.public_url) continue
      replacements.set(item.public_url, item.pdf_url || item.url)
    }
  }
}

for (const paper of paperIndex.papers) {
  replacements.set(`/${paper.pdf}`, paper.source_url)
}

replacements.set('/papers/rome-2202.05262.pdf', 'https://arxiv.org/pdf/2202.05262')
replacements.set('/papers/transformer-patcher-2301.09785.pdf', 'https://arxiv.org/pdf/2301.09785')

const sourceExtensions = new Set(['.md', '.json', '.vue', '.ts', '.mts'])
let changedFiles = 0
let changedLinks = 0

async function migrate(path) {
  const entries = await readdir(path, { withFileTypes: true })
  for (const entry of entries) {
    const target = join(path, entry.name)
    if (entry.isDirectory()) {
      await migrate(target)
      continue
    }
    if (!sourceExtensions.has(extname(entry.name))) continue

    let text = await readFile(target, 'utf8')
    const original = text
    for (const [localUrl, officialUrl] of replacements) {
      if (!text.includes(localUrl)) continue
      const occurrences = text.split(localUrl).length - 1
      text = text.split(localUrl).join(officialUrl)
      changedLinks += occurrences
    }
    text = text
      .replaceAll('本地 PDF', '官方 PDF')
      .replaceAll('本地来源', '论文来源')
      .replaceAll('本地的 [', '[')

    if (text !== original) {
      await writeFile(target, text)
      changedFiles += 1
    }
  }
}

await migrate(site)
console.log(`Migrated ${changedLinks} PDF links across ${changedFiles} site files.`)
