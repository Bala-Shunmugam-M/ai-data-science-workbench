/**
 * The four building blocks every stage page is assembled from.
 *
 * Nine pages of bespoke layout would drift within a week, so each stage is
 * composed from these instead: a titled table, a figure gallery, a key-value
 * card, and a "this has not run" placeholder. The stage components then contain
 * only the part that is genuinely specific to them - which artifacts to read
 * and what the numbers mean.
 */
import Image from 'next/image'

import type { ArtifactFigure, ArtifactTable } from '@/lib/artifacts'
import { tableTitle } from '@/lib/artifacts'
import { Card, DataTable, EmptyState } from '@/components/ui'
import { formatNumber } from '@/lib/format'

export function StageSection({
  title,
  hint,
  children,
}: {
  title: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <section className="flex flex-col gap-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-h2 text-text">{title}</h2>
        {hint && <p className="text-caption text-text-tertiary">{hint}</p>}
      </div>
      {children}
    </section>
  )
}

/** A CSV artifact rendered as a table, with its source path shown. */
export function TableCard({
  table,
  title,
  maxHeight = 420,
}: {
  table: ArtifactTable
  title?: string
  maxHeight?: number
}) {
  const shown = table.rows.length
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-h3 text-text">{title ?? tableTitle(table.source)}</h3>
        <p className="text-label text-text-tertiary">
          {shown < table.totalRows
            ? `${formatNumber(shown)} of ${formatNumber(table.totalRows)} rows · `
            : `${formatNumber(table.totalRows)} rows · `}
          <code>{table.source}</code>
        </p>
      </div>
      <Card padded={false}>
        <DataTable
          columns={table.columns.map((column) => ({ key: column, label: column }))}
          rows={table.rows.map((row) =>
            Object.fromEntries(table.columns.map((column, index) => [column, row[index] ?? ''])),
          )}
          maxHeight={maxHeight}
          zebra
        />
      </Card>
    </div>
  )
}

export function FigureGallery({
  figures,
  columns = 2,
}: {
  figures: ArtifactFigure[]
  columns?: 1 | 2 | 3
}) {
  if (!figures.length) return null
  const gridClass =
    columns === 1 ? 'grid-cols-1' : columns === 3 ? 'md:grid-cols-3' : 'md:grid-cols-2'

  return (
    <div className={`grid gap-6 ${gridClass}`}>
      {figures.map((figure) => (
        <Card key={figure.url} className="flex flex-col gap-3">
          <Image
            src={figure.url}
            alt={figure.label}
            width={880}
            height={560}
            unoptimized
            className="h-auto w-full rounded-md border border-border bg-white"
          />
          <p className="text-caption text-text-secondary">{figure.label}</p>
        </Card>
      ))}
    </div>
  )
}

/** Label/value pairs, for the small summary cards each stage opens with. */
export function FactGrid({
  facts,
  columns = 4,
}: {
  facts: { label: string; value: React.ReactNode; hint?: string }[]
  columns?: 2 | 3 | 4
}) {
  const gridClass =
    columns === 2 ? 'sm:grid-cols-2' : columns === 3 ? 'sm:grid-cols-3' : 'sm:grid-cols-2 xl:grid-cols-4'
  return (
    <div className={`grid gap-4 ${gridClass}`}>
      {facts.map((fact) => (
        <Card key={fact.label} className="flex flex-col gap-1">
          <p className="text-label uppercase tracking-wide text-text-tertiary">{fact.label}</p>
          <p className="text-h3 tabular-nums text-text">{fact.value}</p>
          {fact.hint && <p className="text-caption text-text-secondary">{fact.hint}</p>}
        </Card>
      ))}
    </div>
  )
}

/** Plain-text artifact (a .txt summary the pipeline wrote). */
export function TextCard({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-h3 text-text">{title}</h3>
      <Card padded={false}>
        <pre className="max-h-[420px] overflow-auto p-5 text-caption leading-relaxed text-text-secondary">
          <code>{body.trim()}</code>
        </pre>
      </Card>
    </div>
  )
}

export function StageNotRun({ title, blurb }: { title: string; blurb: string }) {
  return (
    <EmptyState
      title={`${title} has not run for this project`}
      body={`${blurb} Nothing has been written to disk for this stage yet, so there is nothing to show. Run the pipeline for this project and it will appear here.`}
    />
  )
}
