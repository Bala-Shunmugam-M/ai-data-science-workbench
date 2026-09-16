/**
 * Decide whether the production build is stale.
 *
 * Exit 0 = build is current, just start it.
 * Exit 1 = rebuild first.
 *
 * The launcher calls this so a warm start is instant instead of paying ~30s
 * for a `next build` that would produce identical output. Comparing file times
 * is fiddly in batch, so it lives here where it is readable and testable.
 */
import { readdirSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const buildId = path.join(root, '.next', 'BUILD_ID')

let builtAt
try {
  builtAt = statSync(buildId).mtimeMs
} catch {
  console.log('no production build found')
  process.exit(1)
}

/** Directories and files whose changes invalidate the build. */
const WATCHED = ['app', 'components', 'lib', 'hooks', 'public']
const WATCHED_FILES = [
  'package.json',
  'package-lock.json',
  'next.config.ts',
  'tsconfig.json',
  'postcss.config.mjs',
]

const SKIP = new Set(['node_modules', '.next', '.next-verify', '.uploads', '.git'])

function newestUnder(dir) {
  let newest = 0
  let entries
  try {
    entries = readdirSync(dir, { withFileTypes: true })
  } catch {
    return 0
  }
  for (const entry of entries) {
    if (SKIP.has(entry.name)) continue
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      newest = Math.max(newest, newestUnder(full))
    } else {
      try {
        newest = Math.max(newest, statSync(full).mtimeMs)
      } catch {
        /* vanished mid-walk; ignore */
      }
    }
  }
  return newest
}

let newestSource = 0
let culprit = ''
for (const name of WATCHED) {
  const at = newestUnder(path.join(root, name))
  if (at > newestSource) {
    newestSource = at
    culprit = name + '/'
  }
}
for (const name of WATCHED_FILES) {
  try {
    const at = statSync(path.join(root, name)).mtimeMs
    if (at > newestSource) {
      newestSource = at
      culprit = name
    }
  } catch {
    /* optional file */
  }
}

if (newestSource > builtAt) {
  console.log(`source changed since last build (${culprit})`)
  process.exit(1)
}

console.log('production build is current')
process.exit(0)
