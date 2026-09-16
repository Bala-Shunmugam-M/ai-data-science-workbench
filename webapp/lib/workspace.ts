/**
 * Server-only: where the Python workbench keeps each project's files.
 *
 * This mirrors `config/paths.py` exactly, and the mirroring is the whole point —
 * get it wrong and the app reads a different project's artifacts than the one
 * the pipeline wrote. The rule from `config/paths.py:43-52`:
 *
 *   california-housing  ->  PROJECT_ROOT          (the showcase keeps the flat
 *                                                  legacy layout, because its
 *                                                  model registry stores
 *                                                  absolute paths)
 *   anything else       ->  PROJECT_ROOT/workspaces/<slug>
 *
 * Two kinds of project exist. **Built-ins** are declared in
 * `config/datasets.yaml` and were set up by the CLI. **Uploaded** projects are
 * registered by `webapi/bridge.py analyze`, which writes a `dataset.json` into
 * their workspace. Both are listed; only the second kind can be created here.
 */
import 'server-only'

import { readFile, readdir, stat } from 'node:fs/promises'
import path from 'node:path'
import { parse as parseYaml } from 'yaml'

import type { Metric, ProjectSummary, Task } from '@/lib/types'

/** Slugs that resolve to the flat repo-root layout (`config/paths.py:43`). */
const HOUSING_ALIASES = new Set(['', 'california-housing', 'california_housing'])

/** The Python repo root. Next runs with cwd = `webapp/`, so the parent. */
export const PROJECT_ROOT = path.resolve(process.cwd(), '..')

export const WORKSPACES_DIR = path.join(PROJECT_ROOT, 'workspaces')

export function workspaceRoot(slug: string): string {
  return HOUSING_ALIASES.has(slug)
    ? PROJECT_ROOT
    : path.join(WORKSPACES_DIR, slug)
}

/**
 * Reject anything that could escape the workspaces directory.
 *
 * Slugs arrive from the URL, so `../../etc/passwd` is a real request shape.
 * `slugify` in `webapi/bridge.py` only ever emits `[a-z0-9-]`, so anything else
 * is either a typo or an attack and both deserve the same answer.
 */
export function isSafeSlug(slug: string): boolean {
  return /^[a-z0-9][a-z0-9_-]*$/i.test(slug) && !slug.includes('..')
}

async function exists(target: string): Promise<boolean> {
  try {
    await stat(target)
    return true
  } catch {
    return false
  }
}

export async function readJson<T>(file: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(file, 'utf8')) as T
  } catch {
    return null
  }
}

// ---------------------------------------------------------------------------
// The dataset registry
// ---------------------------------------------------------------------------

export interface DatasetDescriptor {
  slug: string
  displayName: string
  task: Task
  target: string
  positiveClass?: string | null
  file?: string
  description?: string
  origin: 'builtin' | 'uploaded'
}

interface YamlDataset {
  display_name?: string
  task?: string
  target?: string
  positive_class?: string
  file?: string
  description?: string
}

/** Built-in projects, from `config/datasets.yaml` — the same file the CLI reads. */
export async function builtinProjects(): Promise<DatasetDescriptor[]> {
  const file = path.join(PROJECT_ROOT, 'config', 'datasets.yaml')
  let parsed: { datasets?: Record<string, YamlDataset> } | null = null
  try {
    parsed = parseYaml(await readFile(file, 'utf8'))
  } catch {
    return []
  }

  return Object.entries(parsed?.datasets ?? {}).map(([key, value]) => ({
    // The YAML key is snake_case; the workspace directory is kebab-case.
    slug: key.replace(/_/g, '-'),
    displayName: value.display_name ?? key,
    task: (value.task === 'classification' ? 'classification' : 'regression') as Task,
    target: value.target ?? '',
    positiveClass: value.positive_class ?? null,
    file: value.file,
    description: value.description?.trim(),
    origin: 'builtin' as const,
  }))
}

/** Uploaded projects — anything with a `workspaces/<slug>/dataset.json`. */
export async function uploadedProjects(): Promise<DatasetDescriptor[]> {
  let entries: string[]
  try {
    entries = await readdir(WORKSPACES_DIR)
  } catch {
    return []
  }

  const found: DatasetDescriptor[] = []
  for (const slug of entries) {
    if (!isSafeSlug(slug)) continue
    const descriptor = await readJson<{
      display_name?: string
      task?: string
      target?: string
      positive_class?: string | null
      file?: string
      description?: string
    }>(path.join(WORKSPACES_DIR, slug, 'dataset.json'))
    if (!descriptor) continue

    found.push({
      slug,
      displayName: descriptor.display_name ?? slug,
      task: (descriptor.task === 'classification' ? 'classification' : 'regression') as Task,
      target: descriptor.target ?? '',
      positiveClass: descriptor.positive_class ?? null,
      file: descriptor.file,
      description: descriptor.description,
      origin: 'uploaded',
    })
  }
  return found
}

export async function findProject(slug: string): Promise<DatasetDescriptor | null> {
  if (!isSafeSlug(slug)) return null
  const all = [...(await uploadedProjects()), ...(await builtinProjects())]
  return all.find((project) => project.slug === slug) ?? null
}

// ---------------------------------------------------------------------------
// Artifacts
// ---------------------------------------------------------------------------

export interface ModelSelection {
  task?: string
  selection_metric?: string
  champion_name?: string
  champion_version?: string
  generated_at?: string
  test_metrics?: Record<string, number>
  validation_metrics?: Record<string, number>
  selection_rationale?: string
}

export function artifactsDir(slug: string): string {
  return path.join(workspaceRoot(slug), 'artifacts')
}

export async function readSelection(slug: string): Promise<ModelSelection | null> {
  return readJson<ModelSelection>(
    path.join(artifactsDir(slug), 'final_model_selection.json'),
  )
}

/**
 * Figures the pipeline produced, newest layout first.
 *
 * Auto-EDA figures only exist for uploaded projects (`auto_report.run()` writes
 * them); the built-ins put their evaluation charts under `evaluation/figures`.
 * Both are listed so a project shows whatever it actually has.
 */
export async function listFigures(slug: string): Promise<{ name: string; rel: string }[]> {
  const roots = [
    path.join(artifactsDir(slug), 'auto_eda'),
    path.join(artifactsDir(slug), 'evaluation', 'figures'),
    path.join(artifactsDir(slug), 'explainability'),
  ]

  const figures: { name: string; rel: string }[] = []
  for (const root of roots) {
    if (!(await exists(root))) continue
    let files: string[]
    try {
      files = await readdir(root)
    } catch {
      continue
    }
    for (const file of files.filter((f) => f.toLowerCase().endsWith('.png'))) {
      figures.push({ name: path.basename(file, '.png'), rel: path.join(path.basename(root), file) })
    }
  }
  return figures
}

/** Resolve a figure request to an absolute path, or null if it escapes. */
export function resolveFigure(slug: string, relative: string): string | null {
  if (!isSafeSlug(slug)) return null
  const base = artifactsDir(slug)
  const target = path.resolve(base, relative)
  // The containment check is the point of this function.
  if (!target.startsWith(base + path.sep)) return null
  if (!target.toLowerCase().endsWith('.png')) return null
  return target
}

export async function readReport(slug: string): Promise<string | null> {
  const candidates = [
    path.join(artifactsDir(slug), 'reports', 'auto_report.md'),
    path.join(artifactsDir(slug), 'reports', 'project_report.md'),
  ]
  for (const candidate of candidates) {
    try {
      return await readFile(candidate, 'utf8')
    } catch {
      /* try the next one */
    }
  }
  return null
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

/**
 * Minimal CSV reader for previews.
 *
 * Deliberately not a full parser: it handles quoted fields containing commas
 * and doubled quotes, which is everything the workbench's own outputs and
 * ordinary exports produce. Embedded newlines inside quotes are not supported —
 * a preview is allowed to be approximate, and a real parser is a dependency
 * this app does not otherwise need.
 */
export function parseCsv(text: string, maxRows: number): { columns: string[]; rows: string[][] } {
  const lines = text.split(/\r?\n/).filter((line) => line.length > 0)
  if (lines.length === 0) return { columns: [], rows: [] }

  const splitLine = (line: string): string[] => {
    const cells: string[] = []
    let cell = ''
    let quoted = false
    for (let i = 0; i < line.length; i += 1) {
      const char = line[i]
      if (quoted) {
        if (char === '"') {
          if (line[i + 1] === '"') {
            cell += '"'
            i += 1
          } else {
            quoted = false
          }
        } else {
          cell += char
        }
      } else if (char === '"') {
        quoted = true
      } else if (char === ',') {
        cells.push(cell)
        cell = ''
      } else {
        cell += char
      }
    }
    cells.push(cell)
    return cells
  }

  return {
    columns: splitLine(lines[0]),
    rows: lines.slice(1, maxRows + 1).map(splitLine),
  }
}

export async function readCsvPreview(
  file: string,
  maxRows: number,
): Promise<{ columns: string[]; rows: string[][]; totalRows: number } | null> {
  let text: string
  try {
    text = await readFile(file, 'utf8')
  } catch {
    return null
  }
  const totalRows = Math.max(0, text.split(/\r?\n/).filter((l) => l.length > 0).length - 1)
  const { columns, rows } = parseCsv(text, maxRows)
  return { columns, rows, totalRows }
}

/** The project's raw input CSV, wherever the descriptor says it lives. */
export async function rawDataFile(descriptor: DatasetDescriptor): Promise<string | null> {
  const raw = path.join(workspaceRoot(descriptor.slug), 'data', 'raw')
  if (descriptor.file) {
    const named = path.join(raw, descriptor.file)
    if (await exists(named)) return named
  }
  try {
    const files = (await readdir(raw)).filter((f) => f.toLowerCase().endsWith('.csv'))
    return files.length ? path.join(raw, files[0]) : null
  } catch {
    return null
  }
}

// ---------------------------------------------------------------------------
// Metric presentation
// ---------------------------------------------------------------------------

/** Metric keys whose *lower* value is better — used to pick a tone. */
const LOWER_IS_BETTER = new Set(['rmse', 'mae', 'mape', 'logloss', 'brier'])

export function metricLabel(key: string): string {
  const labels: Record<string, string> = {
    rmse: 'Test RMSE',
    mae: 'Test MAE',
    r2: 'Test R²',
    mape: 'Test MAPE',
    roc_auc: 'Test ROC-AUC',
    accuracy: 'Test accuracy',
    precision: 'Test precision',
    recall: 'Test recall',
    f1: 'Test F1',
  }
  return labels[key] ?? key.replace(/_/g, ' ')
}

/**
 * Format one metric for display, with a tone.
 *
 * Tone is only applied to bounded, comparable metrics (R², ROC-AUC, accuracy…).
 * RMSE on house prices has no universal "good" value, so it stays neutral
 * rather than being coloured by an invented threshold.
 */
export function toMetric(key: string, value: number): Metric {
  const bounded = ['r2', 'roc_auc', 'accuracy', 'precision', 'recall', 'f1'].includes(key)
  let tone: Metric['tone'] = 'neutral'
  if (bounded) {
    tone = value >= 0.8 ? 'good' : value >= 0.6 ? 'warn' : 'bad'
  }

  const formatted = LOWER_IS_BETTER.has(key)
    ? value >= 1000
      ? Math.round(value).toLocaleString('en-US')
      : value.toFixed(2)
    : value.toFixed(4)

  return {
    label: metricLabel(key),
    value: key === 'mape' ? `${value.toFixed(1)}%` : formatted,
    tone,
  }
}

/** The single metric that best summarises a project, for the history list. */
export function headlineMetricFor(selection: ModelSelection | null): Metric | null {
  const metrics = selection?.test_metrics
  if (!metrics) return null
  for (const key of ['r2', 'roc_auc', 'accuracy', 'rmse']) {
    if (typeof metrics[key] === 'number') return toMetric(key, metrics[key])
  }
  const [first] = Object.entries(metrics)
  return first ? toMetric(first[0], first[1]) : null
}

export async function projectSummaries(): Promise<ProjectSummary[]> {
  const descriptors = [...(await uploadedProjects()), ...(await builtinProjects())]
  const seen = new Set<string>()

  const summaries: ProjectSummary[] = []
  for (const descriptor of descriptors) {
    if (seen.has(descriptor.slug)) continue
    seen.add(descriptor.slug)

    const selection = await readSelection(descriptor.slug)
    const raw = await rawDataFile(descriptor)
    let rows = 0
    let createdAt = selection?.generated_at ?? ''
    if (raw) {
      try {
        const info = await stat(raw)
        if (!createdAt) createdAt = info.mtime.toISOString()
        // Counting rows means reading the file; acceptable for a local tool and
        // it is the only honest source for "rows analysed".
        const text = await readFile(raw, 'utf8')
        rows = Math.max(0, text.split(/\r?\n/).filter((l) => l.length > 0).length - 1)
      } catch {
        /* leave rows at 0 */
      }
    }

    summaries.push({
      slug: descriptor.slug,
      displayName: descriptor.displayName,
      createdAt,
      task: descriptor.task,
      target: descriptor.target,
      rows,
      status: selection ? 'ready' : 'running',
      headlineMetric: headlineMetricFor(selection),
    })
  }

  return summaries.sort((a, b) => (b.createdAt ?? '').localeCompare(a.createdAt ?? ''))
}
