import { readFile, readdir, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'

const base = process.env.VITEPRESS_BASE || '/'
if (base === '/') {
  console.log('Root deployment: no static sub-app base rewrite needed.')
  process.exit(0)
}

const root = resolve(import.meta.dirname, '..')
const lectureRoot = join(root, 'site/.vitepress/dist/lectures')
const indexPath = join(lectureRoot, 'index.html')
const lectureBase = `${base}lectures`.replace(/\/$/, '')

let html = await readFile(indexPath, 'utf8')
html = html
  .replace(/^\s*<link rel="icon"[^>]+>\s*$/m, '')
  .replaceAll('href="/lectures/', `href="${lectureBase}/`)
  .replaceAll('src="/lectures/', `src="${lectureBase}/`)
await writeFile(indexPath, html)

const assets = join(lectureRoot, 'assets')
for (const entry of await readdir(assets)) {
  if (!entry.endsWith('.js')) continue
  const assetPath = join(assets, entry)
  const source = await readFile(assetPath, 'utf8')
  const updated = source.replaceAll('basename:"/lectures"', `basename:"${lectureBase}"`)
  if (updated !== source) await writeFile(assetPath, updated)
}

console.log(`Rebased executable lectures to ${lectureBase}/`)
