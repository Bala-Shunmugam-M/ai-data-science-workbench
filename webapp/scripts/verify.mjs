/**
 * Typecheck, unit-check, then build into a throwaway directory.
 *
 * The build goes to `.next-verify` rather than `.next` so it cannot clobber a
 * dev server running in another terminal. Sharing `.next` between `next dev`
 * and `next build` leaves the browser reporting `Cannot find module './873.js'`,
 * which points at webpack internals and gives no hint that a concurrent build
 * was the cause.
 *
 * Every step runs through `process.execPath` against the package's own JS
 * entry point. No shell: `npx` is a `.cmd` shim on Windows that spawnSync
 * cannot resolve unshelled, and `shell: true` both breaks on the space in
 * "C:\Program Files\nodejs" and trips Node's DEP0190 warning.
 */
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const bin = (...parts) => path.join(root, 'node_modules', ...parts)

const steps = [
  { label: 'typecheck', args: [bin('typescript', 'bin', 'tsc'), '--noEmit'] },
  { label: 'unit', args: [path.join(root, 'lib', 'utils.test.ts')] },
  {
    label: 'build',
    args: [bin('next', 'dist', 'bin', 'next'), 'build'],
    env: { NEXT_DIST_DIR: '.next-verify' },
  },
]

for (const step of steps) {
  process.stdout.write(`\n=== ${step.label} ===\n`)
  const result = spawnSync(process.execPath, step.args, {
    stdio: 'inherit',
    cwd: root,
    env: { ...process.env, ...step.env },
  })
  if (result.error) {
    process.stdout.write(`\n${step.label} could not start: ${result.error.message}\n`)
    process.exit(1)
  }
  if (result.status !== 0) {
    process.stdout.write(`\n${step.label} failed\n`)
    process.exit(result.status ?? 1)
  }
}

process.stdout.write('\nall checks passed\n')
