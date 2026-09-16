/**
 * Server-only: the single place this app spawns Python.
 *
 * Everything goes through `webapi/bridge.py`, which guarantees stdout is pure
 * JSON and sends all logging to stderr. Keeping the spawn in one module means
 * the interpreter choice, the working directory and the stderr handling are
 * decided once instead of drifting between two route handlers.
 *
 * Working directory is always the Python repo root (`PROJECT_ROOT`), not
 * `webapp/`. `config/paths.py` resolves from `__file__` so cwd does not
 * strictly matter, but `bridge.py`'s own relative imports do.
 */
import 'server-only'

import { spawn } from 'node:child_process'
import path from 'node:path'

import { PROJECT_ROOT } from '@/lib/workspace'

/**
 * Interpreter to use. `python` is correct on this machine (the workbench uses
 * the global 3.13 install, no venv); `WORKBENCH_PYTHON` overrides it for anyone
 * whose Python is elsewhere.
 */
export const PYTHON = process.env.WORKBENCH_PYTHON ?? 'python'

const BRIDGE = path.join(PROJECT_ROOT, 'webapi', 'bridge.py')

export class PythonError extends Error {
  constructor(
    message: string,
    readonly exitCode: number | null,
    readonly stderr: string,
  ) {
    super(message)
    this.name = 'PythonError'
  }
}

/**
 * Run a bridge subcommand that returns a single JSON object.
 *
 * Used by `detect`, which is fast enough (a few seconds — pandas import plus a
 * profile pass) to await in a request handler.
 */
export function runBridgeJson<T>(args: string[], timeoutMs = 120_000): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const child = spawn(PYTHON, [BRIDGE, ...args], {
      cwd: PROJECT_ROOT,
      windowsHide: true,
    })

    let stdout = ''
    let stderr = ''
    const timer = setTimeout(() => {
      child.kill()
      reject(new PythonError(`Python timed out after ${timeoutMs}ms`, null, stderr))
    }, timeoutMs)

    child.stdout.on('data', (chunk) => (stdout += chunk))
    child.stderr.on('data', (chunk) => (stderr += chunk))

    child.on('error', (error) => {
      clearTimeout(timer)
      reject(
        new PythonError(
          `Could not start ${PYTHON}. Is Python on PATH? (${error.message})`,
          null,
          stderr,
        ),
      )
    })

    child.on('close', (code) => {
      clearTimeout(timer)
      if (code !== 0) {
        reject(new PythonError(lastMeaningfulLine(stderr) || `Python exited ${code}`, code, stderr))
        return
      }
      try {
        // `bridge.py` strips its own BOM, but a stray one here would be a
        // confusing parse error rather than an obvious encoding problem.
        resolve(JSON.parse(stdout.replace(/^﻿/, '')) as T)
      } catch {
        reject(new PythonError('Python returned output that was not JSON', code, stderr))
      }
    })
  })
}

/**
 * Run a bridge subcommand that streams one JSON object per line.
 *
 * `onLine` fires for every parseable line as it arrives; unparseable lines are
 * ignored rather than fatal, because a stray print from a dependency should not
 * kill a five-minute training run.
 */
export function runBridgeStream(
  args: string[],
  onLine: (value: unknown) => void,
  timeoutMs = 900_000,
): Promise<{ code: number | null; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, [BRIDGE, ...args], {
      cwd: PROJECT_ROOT,
      windowsHide: true,
    })

    let stderr = ''
    let buffer = ''
    const timer = setTimeout(() => {
      child.kill()
      reject(new PythonError(`Python timed out after ${timeoutMs}ms`, null, stderr))
    }, timeoutMs)

    const drain = (flush: boolean) => {
      const parts = buffer.split(/\r?\n/)
      buffer = flush ? '' : (parts.pop() ?? '')
      for (const part of parts) {
        const trimmed = part.trim()
        if (!trimmed) continue
        try {
          onLine(JSON.parse(trimmed))
        } catch {
          /* not a JSON line — ignore */
        }
      }
    }

    child.stdout.on('data', (chunk) => {
      buffer += chunk
      drain(false)
    })
    child.stderr.on('data', (chunk) => (stderr += chunk))

    child.on('error', (error) => {
      clearTimeout(timer)
      reject(new PythonError(`Could not start ${PYTHON}: ${error.message}`, null, stderr))
    })

    child.on('close', (code) => {
      clearTimeout(timer)
      drain(true)
      resolve({ code, stderr })
    })
  })
}

/** The most useful line of a Python traceback: the last non-empty one. */
export function lastMeaningfulLine(stderr: string): string {
  const lines = stderr
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
  return lines.length ? lines[lines.length - 1] : ''
}
