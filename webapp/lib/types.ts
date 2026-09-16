// Shared vocabulary for the workbench web app.
// Source of truth: docs/WEBAPP_PLAN.md §3. Copied verbatim — do not
// rename fields or "improve" this file; three other work packages
// (P4, P5, P6) import these types directly.

export type Task = 'regression' | 'classification'

export interface ColumnProfile {
  name: string; dtype: string; missing: number; missingPct: number
  unique: number; example: string
}

export interface Detection {
  target: string; task: Task; positiveClass: string | null
  categoricalColumns: string[]; dropColumns: string[]
  selectionMetric: string; nClasses: number | null; warnings: string[]
}

export interface PreviewTable { columns: string[]; rows: (string | number | null)[][] }

export interface DetectResponse {
  fileId: string; fileName: string; sizeBytes: number
  rows: number; columns: number
  detection: Detection; profile: ColumnProfile[]; preview: PreviewTable
}

export type StageId = 'upload' | 'scan' | 'extract' | 'analytics' | 'insights'

export interface StageEvent {
  stage: StageId; status: 'started' | 'done' | 'failed'
  message?: string; slug?: string; elapsedMs?: number
}

export interface Metric { label: string; value: string; tone?: 'neutral' | 'good' | 'warn' | 'bad'; hint?: string }

export interface Insight { title: string; body: string; tone: 'good' | 'warn' | 'bad' | 'neutral' }

export interface ResultsPayload {
  slug: string; displayName: string; createdAt: string
  task: Task; target: string; rows: number; columns: number
  headline: Metric[]              // top summary bar
  champion: { name: string; version: string; metrics: Metric[]; rationale: string }
  comparison: { rank: number; model: string; version: string; metrics: Record<string, number> }[]
  drivers: { rank: number; feature: string; coefficient: number; direction: 'positive' | 'negative'; meaning: string }[]
  insights: Insight[]
  figures: { name: string; caption: string; url: string }[]
  reportMarkdown: string | null
  extracted: PreviewTable
}

export interface ProjectSummary {
  slug: string; displayName: string; createdAt: string
  task: Task; target: string; rows: number
  status: 'ready' | 'running' | 'failed'
  headlineMetric: Metric | null
}
