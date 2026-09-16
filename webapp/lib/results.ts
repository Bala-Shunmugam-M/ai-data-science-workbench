/**
 * Assemble everything the results screen renders, from artifacts on disk.
 *
 * Nothing is computed here: every number already exists because the pipeline
 * wrote it. This module's whole job is to find the files, tolerate the ones a
 * given project does not have, and shape the rest into one payload.
 *
 * It lives in `lib/` rather than inside the route handler so the page (a server
 * component) and `GET /api/results/[slug]` produce byte-identical results. Two
 * copies of this mapping would drift within a week.
 */
import 'server-only'

import { readFile } from 'node:fs/promises'
import path from 'node:path'

import type { Insight, Metric, ResultsPayload } from '@/lib/types'
import {
  artifactsDir,
  findProject,
  headlineMetricFor,
  listFigures,
  metricLabel,
  parseCsv,
  rawDataFile,
  readCsvPreview,
  readJson,
  readReport,
  readSelection,
  toMetric,
} from '@/lib/workspace'

export const PREVIEW_ROWS = 50

interface CoefficientArtifact {
  intercept?: number
  coefficients?: {
    rank: number
    feature: string
    coefficient: number
    direction: string
    abs_importance: number
    meaning: string
  }[]
}

/** Captions for the figures the pipeline is known to produce. */
const FIGURE_CAPTIONS: Record<string, string> = {
  predicted_vs_actual: 'Predicted against actual values on the held-out test split.',
  residuals_vs_predicted: 'Residuals against predictions. A flat band means the errors are unbiased.',
  residual_histogram: 'Distribution of prediction errors.',
  top_drivers: 'The features that move the prediction furthest, by standardized coefficient.',
  confusion_matrix: 'Correct and incorrect classifications on the test split.',
  roc_curve: 'True-positive rate against false-positive rate across every threshold.',
}

function captionFor(name: string): string {
  return FIGURE_CAPTIONS[name] ?? name.replace(/[_-]/g, ' ')
}

export async function buildResults(slug: string): Promise<ResultsPayload | null> {
  const descriptor = await findProject(slug)
  if (!descriptor) return null

  const selection = await readSelection(slug)
  const base = artifactsDir(slug)

  const testMetrics = selection?.test_metrics ?? {}
  const championMetrics: Metric[] = Object.entries(testMetrics).map(([key, value]) =>
    toMetric(key, value),
  )
  const headlineMetric = headlineMetricFor(selection)

  const raw = await rawDataFile(descriptor)
  const preview = raw ? await readCsvPreview(raw, PREVIEW_ROWS) : null

  const headline: Metric[] = [
    {
      label: 'Champion model',
      value: selection
        ? `${selection.champion_name ?? '—'} ${selection.champion_version ?? ''}`.trim()
        : 'Not analysed',
      tone: selection ? 'good' : 'warn',
    },
    ...(headlineMetric ? [headlineMetric] : []),
    {
      label: 'Rows analysed',
      value: preview ? preview.totalRows.toLocaleString('en-US') : '—',
      tone: 'neutral',
    },
    {
      label: 'Columns',
      value: preview ? String(preview.columns.length) : '—',
      tone: 'neutral',
    },
  ]

  // --- model comparison ---------------------------------------------------
  const comparison: ResultsPayload['comparison'] = []
  try {
    const text = await readFile(path.join(base, 'evaluation', 'model_comparison.csv'), 'utf8')
    const { columns, rows } = parseCsv(text, 50)
    for (const row of rows) {
      const record = Object.fromEntries(columns.map((column, index) => [column, row[index]]))
      const metrics: Record<string, number> = {}
      for (const [key, value] of Object.entries(record)) {
        if (['rank', 'model', 'version'].includes(key)) continue
        const parsed = Number(value)
        if (Number.isFinite(parsed)) metrics[key] = parsed
      }
      comparison.push({
        rank: Number(record.rank) || comparison.length + 1,
        model: String(record.model ?? ''),
        version: String(record.version ?? ''),
        metrics,
      })
    }
  } catch {
    /* a project without a comparison artifact simply shows none */
  }

  // --- drivers ------------------------------------------------------------
  const coefficients = await readJson<CoefficientArtifact>(
    path.join(base, 'explainability', 'coefficient_interpretation.json'),
  )
  const drivers: ResultsPayload['drivers'] = (coefficients?.coefficients ?? [])
    .slice(0, 12)
    .map((entry) => ({
      rank: entry.rank,
      feature: entry.feature,
      coefficient: entry.coefficient,
      // The artifact phrases direction as prose ("increases value"); the
      // contract wants the sign, so derive it rather than string-match.
      direction: entry.coefficient >= 0 ? 'positive' : 'negative',
      meaning: entry.meaning,
    }))

  // --- insights -----------------------------------------------------------
  const insights: Insight[] = []
  if (selection) {
    const metricKey = Object.keys(testMetrics).find((key) =>
      ['r2', 'roc_auc', 'accuracy'].includes(key),
    )
    if (metricKey) {
      const value = testMetrics[metricKey]
      insights.push({
        title:
          value >= 0.8
            ? `Strong fit: ${metricLabel(metricKey)} of ${value.toFixed(3)}`
            : value >= 0.6
              ? `Moderate fit: ${metricLabel(metricKey)} of ${value.toFixed(3)}`
              : `Weak fit: ${metricLabel(metricKey)} of ${value.toFixed(3)}`,
        body:
          value >= 0.8
            ? 'The model explains most of the variation in the target on data it never saw during training.'
            : value >= 0.6
              ? 'The model captures the broad pattern but leaves real variation unexplained. Treat individual predictions as indicative rather than precise.'
              : 'The model explains little of the target. More informative features are likely needed before these predictions can support a decision.',
        tone: value >= 0.8 ? 'good' : value >= 0.6 ? 'warn' : 'bad',
      })
    }
    if (drivers.length) {
      const top = drivers[0]
      insights.push({
        title: `${top.feature} is the strongest driver`,
        body: `${top.meaning} It ${top.direction === 'positive' ? 'pushes the prediction up' : 'pulls the prediction down'} more than any other feature.`,
        tone: 'neutral',
      })
    }
    if (comparison.length > 1) {
      insights.push({
        title: `${comparison.length} models were compared`,
        body: `${selection.champion_name ?? 'The champion'} won on the validation split and was then measured once on the untouched test split. The test data was never used for tuning.`,
        tone: 'neutral',
      })
    }
  } else {
    insights.push({
      title: 'This project has not been analysed yet',
      body: 'It is registered, but no model selection artifact exists. Run the analysis to fill this page in.',
      tone: 'warn',
    })
  }

  const figures = (await listFigures(slug)).map((figure) => ({
    name: figure.name,
    caption: captionFor(figure.name),
    url: `/api/figure/${slug}/${figure.rel.split(path.sep).map(encodeURIComponent).join('/')}`,
  }))

  return {
    slug,
    displayName: descriptor.displayName,
    createdAt: selection?.generated_at ?? '',
    task: descriptor.task,
    target: descriptor.target,
    rows: preview?.totalRows ?? 0,
    columns: preview?.columns.length ?? 0,
    headline,
    champion: {
      name: selection?.champion_name ?? '',
      version: selection?.champion_version ?? '',
      metrics: championMetrics,
      rationale: selection?.selection_rationale ?? '',
    },
    comparison,
    drivers,
    insights,
    figures,
    reportMarkdown: await readReport(slug),
    extracted: {
      columns: preview?.columns ?? [],
      rows: preview?.rows ?? [],
    },
  }
}
