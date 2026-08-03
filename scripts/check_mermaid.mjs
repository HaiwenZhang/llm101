#!/usr/bin/env node
/** Validate every Mermaid fence before VitePress starts or builds. */

import { readdir, readFile } from 'node:fs/promises'
import { extname, join, relative } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const root = fileURLToPath(new URL('../site/', import.meta.url))
const mermaidChunks = fileURLToPath(
  new URL('../node_modules/mermaid/dist/chunks/mermaid.core/', import.meta.url)
)

const chunkFiles = await readdir(mermaidChunks)
const flowChunk = chunkFiles.find(name => /^flowDiagram-[^.]+\.mjs$/.test(name))
if (!flowChunk) throw new Error('Could not locate Mermaid flowchart parser')
const { createFlowDiagram } = await import(pathToFileURL(join(mermaidChunks, flowChunk)))

function parseFlowchart(source) {
  const diagram = createFlowDiagram()
  const db = diagram.db
  // Mermaid's Node build has no browser DOMPurify instance. Sanitization still
  // runs in the browser; bypass only that DOM step so the grammar can be checked.
  db.sanitizeText = value => value
  diagram.parser.yy = db
  diagram.parser.parser.yy = db
  diagram.parser.parse(source)
}

function checkStateDiagram(source) {
  const lines = source.split('\n').map(line => line.trim()).filter(Boolean)
  if (lines[0] !== 'stateDiagram-v2' || !lines.slice(1).some(line => line.includes('-->'))) {
    throw new Error('stateDiagram-v2 must contain at least one transition')
  }
}

async function markdownFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const nested = await Promise.all(entries.map(async entry => {
    const target = join(directory, entry.name)
    if (entry.isDirectory()) return markdownFiles(target)
    return extname(entry.name) === '.md' ? [target] : []
  }))
  return nested.flat()
}

const files = await markdownFiles(root)
const failures = []
let diagramCount = 0

for (const file of files) {
  const source = await readFile(file, 'utf8')
  const diagrams = source.matchAll(/```mermaid\s*\n([\s\S]*?)\n```/g)
  let index = 0
  for (const match of diagrams) {
    index += 1
    diagramCount += 1
    try {
      const source = match[1].trim()
      if (/^(flowchart|graph)\s/.test(source)) parseFlowchart(source)
      else if (source.startsWith('stateDiagram-v2')) checkStateDiagram(source)
      else throw new Error(`Unsupported diagram type: ${source.split(/\s/)[0]}`)
    } catch (error) {
      failures.push({
        file: relative(root, file),
        diagram: index,
        message: error instanceof Error ? error.message : String(error)
      })
    }
  }
}

if (failures.length) {
  for (const failure of failures) {
    console.error(`${failure.file} · diagram ${failure.diagram}: ${failure.message}`)
  }
  process.exitCode = 1
} else {
  console.log(`Validated ${diagramCount} Mermaid diagrams in ${files.length} Markdown files.`)
}
