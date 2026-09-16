/**
 * Server-only: generic readers the nine stage pages share.
 *
 * Every stage does the same three things - read a CSV into a table, read a JSON
 * or text artifact, list the PNGs in a directory - so those live here once
 * rather than nine times. Each reader returns `null` for a missing file instead
 * of throwing, because "this stage has not run" is a normal state that the page
 * renders as an empty state, not an error.
 */
import 'server-only'

import { readFile, readdir, stat } from 'node:fs/promises'
import path from 'node:path'

import type { StageDefinition } from '@/lib/stages'
import { isSafeSlug, parseCsv, workspaceRoot } from '@/lib/workspace'

export interface ArtifactTable {
  columns: string[]
  rows: string[][]
  totalRows: number
  /** Relative path, shown so a reader can find the file on disk. */
  source: string
}

export interface ArtifactFigure {
  name: string
  /** Human label derived from the file name. */
  label: string
  url: string
}

export function resolveInWorkspace(slug: string, relative: string): string | null {
  if (!isSafeSlug(slug)) return null
  const base = workspaceRoot(slug)
  const target = path.resolve(base, relative)
  if (target !== base && !target.startsWith(base + path.sep)) return null
  return target
}

export async function pathExists(slug: string, relative: string): Promise<boolean> {
  const target = resolveInWorkspace(slug, relative)
  if (!target) return false
  try {
    await stat(target)
    return true
  } catch {
    return false
  }
}

/**
 * True when a probe holds something. A file counts; a directory counts only if a
 * file exists somewhere beneath it.
 *
 * Directory probes (`results/eda`) made an empty directory read as a finished
 * stage. Stages call `ensure_dirs()` before they do any work, so a stage that
 * fails on its first step leaves the directory behind and the app reported
 * "10 of 10 stages produced artifacts" for a project whose EDA had crashed.
 */
async function probeHasContent(slug: string, relative: string): Promise<boolean> {
  const target = resolveInWorkspace(slug, relative)
  if (!target) return false
  try {
    if (!(await stat(target)).isDirectory()) return true
    const entries = await readdir(target, { withFileTypes: true, recursive: true })
    return entries.some((entry) => entry.isFile())
  } catch {
    return false
  }
}

/** True when any of a stage's probe artifacts exist and are non-empty. */
export async function stageHasRun(slug: string, stage: StageDefinition): Promise<boolean> {
  for (const probe of stage.probes) {
    if (await probeHasContent(slug, probe)) return true
  }
  return false
}

export async function readTable(
  slug: string,
  relative: string,
  maxRows = 200,
): Promise<ArtifactTable | null> {
  const target = resolveInWorkspace(slug, relative)
  if (!target) return null
  let text: string
  try {
    text = await readFile(target, 'utf8')
  } catch {
    return null
  }
  const { columns, rows } = parseCsv(text, maxRows)
  if (!columns.length) return null
  const totalRows = Math.max(0, text.split(/\r?\n/).filter((line) => line.length > 0).length - 1)
  return { columns, rows, totalRows, source: relative }
}

/**
 * Size in bytes, or null when absent.
 *
 * Exists so a page can say "2.2 MB" and offer a download without reading the
 * whole file into the render — the project report alone is megabytes.
 */
export async function artifactSize(slug: string, relative: string): Promise<number | null> {
  const target = resolveInWorkspace(slug, relative)
  if (!target) return null
  try {
    return (await stat(target)).size
  } catch {
    return null
  }
}

export async function readTextArtifact(slug: string, relative: string): Promise<string | null> {
  const target = resolveInWorkspace(slug, relative)
  if (!target) return null
  try {
    return await readFile(target, 'utf8')
  } catch {
    return null
  }
}

export async function readJsonArtifact<T>(slug: string, relative: string): Promise<T | null> {
  const text = await readTextArtifact(slug, relative)
  if (text === null) return null
  try {
    return JSON.parse(text) as T
  } catch {
    return null
  }
}

/** One object per line; malformed lines are skipped rather than fatal. */
export async function readJsonLines<T>(slug: string, relative: string): Promise<T[]> {
  const text = await readTextArtifact(slug, relative)
  if (!text) return []
  const records: T[] = []
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    try {
      records.push(JSON.parse(trimmed) as T)
    } catch {
      /* skip */
    }
  }
  return records
}

function humanise(name: string): string {
  return name
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase())
    .trim()
}

/** PNGs inside one workspace-relative directory, sorted by name. */
export async function readFigures(slug: string, relative: string): Promise<ArtifactFigure[]> {
  const target = resolveInWorkspace(slug, relative)
  if (!target) return []
  let files: string[]
  try {
    files = await readdir(target)
  } catch {
    return []
  }
  return files
    .filter((file) => file.toLowerCase().endsWith('.png'))
    .sort()
    .map((file) => ({
      name: path.basename(file, '.png'),
      label: humanise(path.basename(file, '.png')),
      url: `/api/artifact/${slug}/${[...relative.split(/[\\/]/), file].map(encodeURIComponent).join('/')}`,
    }))
}

/** Figures from several candidate directories, first hit wins per directory. */
export async function readFiguresFrom(
  slug: string,
  directories: string[],
): Promise<ArtifactFigure[]> {
  const all: ArtifactFigure[] = []
  for (const directory of directories) {
    all.push(...(await readFigures(slug, directory)))
  }
  return all
}

/** Every CSV in a directory, as tables. Used by the EDA and validation pages. */
export async function readTablesIn(
  slug: string,
  relative: string,
  maxRows = 50,
): Promise<ArtifactTable[]> {
  const target = resolveInWorkspace(slug, relative)
  if (!target) return []
  let files: string[]
  try {
    files = await readdir(target)
  } catch {
    return []
  }

  const tables: ArtifactTable[] = []
  for (const file of files.filter((name) => name.toLowerCase().endsWith('.csv')).sort()) {
    const table = await readTable(slug, path.posix.join(relative, file), maxRows)
    if (table) tables.push(table)
  }
  return tables
}

export function tableTitle(source: string): string {
  return humanise(path.basename(source, path.extname(source)))
}
