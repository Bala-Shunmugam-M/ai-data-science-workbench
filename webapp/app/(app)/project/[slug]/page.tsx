import { ArrowRight, Check, Minus } from 'lucide-react'
import Link from 'next/link'
import { notFound } from 'next/navigation'

import { PageHeader } from '@/components/shell/page-header'
import { Badge, Card, buttonVariants, cn } from '@/components/ui'
import { formatDate, formatNumber } from '@/lib/format'
import { stageHasRun } from '@/lib/artifacts'
import { STAGES } from '@/lib/stages'
import { findProject, headlineMetricFor, rawDataFile, readCsvPreview, readSelection } from '@/lib/workspace'

export const dynamic = 'force-dynamic'

/**
 * The journey overview: what happened to this dataset, as nine numbered steps.
 *
 * This is the screen the processing view hands off to. It exists so the answer
 * to "what did it actually do?" is one glance rather than nine clicks.
 */
export default async function ProjectOverview({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const project = await findProject(slug)
  if (!project) notFound()

  const selection = await readSelection(slug)
  const headline = headlineMetricFor(selection)
  const raw = await rawDataFile(project)
  const preview = raw ? await readCsvPreview(raw, 1) : null

  const stages = await Promise.all(
    STAGES.map(async (stage) => ({ ...stage, hasRun: await stageHasRun(slug, stage) })),
  )
  const completed = stages.filter((stage) => stage.hasRun).length
  const firstIncomplete = stages.find((stage) => !stage.hasRun)

  return (
    <>
      <PageHeader
        title={project.displayName}
        description={
          <>
            {project.task === 'classification' ? 'Classification' : 'Regression'} on{' '}
            <span className="text-text">{project.target}</span>
            {preview ? ` · ${formatNumber(preview.totalRows)} rows` : ''}
            {selection?.generated_at ? ` · analysed ${formatDate(selection.generated_at)}` : ''}
          </>
        }
        actions={
          <Link
            href={`/project/${slug}/dashboard`}
            className={buttonVariants({ size: 'sm' })}
          >
            Start at step 1
            <ArrowRight className="size-4" />
          </Link>
        }
      />

      {/* Progress summary: one honest sentence plus a bar. */}
      <Card className="mb-10 flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-h3 text-text">
              {completed} of {stages.length} stages produced artifacts
            </p>
            <p className="mt-1 text-caption text-text-secondary">
              {/* The old copy blamed this on uploads, which is wrong for a CLI
                  project like churn: its EDA is absent because the governed
                  semantic schema describes the housing dataset only, so the
                  shorter pipeline skips that stage. Naming a cause we cannot
                  verify from disk sends the reader looking in the wrong place. */}
              {firstIncomplete
                ? `Nothing on disk yet for “${firstIncomplete.title}”. Not every project runs every stage — some are schema-bound or dataset-specific.`
                : 'Every stage of the pipeline wrote output for this project.'}
            </p>
          </div>
          {headline && (
            <div className="text-right">
              <p className="text-label uppercase tracking-wide text-text-tertiary">
                {headline.label}
              </p>
              <p className="text-h2 tabular-nums text-text">{headline.value}</p>
            </div>
          )}
        </div>
        <div
          className="h-2 w-full overflow-hidden rounded-full bg-bg-subtle"
          role="progressbar"
          aria-valuenow={completed}
          aria-valuemin={0}
          aria-valuemax={stages.length}
          aria-label="Stages complete"
        >
          <div
            className="h-full rounded-full bg-brand transition-[width] duration-slow ease-standard"
            style={{ width: `${(completed / stages.length) * 100}%` }}
          />
        </div>
      </Card>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {stages.map((stage, index) => (
          <Link
            key={stage.id}
            href={`/project/${slug}/${stage.id}`}
            className="animate-fade-in stagger-item rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring"
            style={{ '--stagger-index': index } as React.CSSProperties}
          >
            <Card
              interactive
              className={cn('flex h-full flex-col gap-3', !stage.hasRun && 'opacity-70')}
            >
              <div className="flex items-start justify-between gap-3">
                <span
                  className={cn(
                    'grid size-9 shrink-0 place-items-center rounded-full border text-body tabular-nums',
                    stage.hasRun
                      ? 'border-brand bg-brand-subtle text-brand'
                      : 'border-border bg-bg-subtle text-text-tertiary',
                  )}
                >
                  {stage.number}
                </span>
                {stage.hasRun ? (
                  <Badge tone="good">
                    <Check className="size-3" />
                    Ready
                  </Badge>
                ) : (
                  <Badge tone="neutral">
                    <Minus className="size-3" />
                    Not run
                  </Badge>
                )}
              </div>
              <div>
                <h2 className="text-h3 text-text">{stage.title}</h2>
                <p className="mt-1 text-caption text-text-secondary">{stage.blurb}</p>
              </div>
            </Card>
          </Link>
        ))}
      </div>
    </>
  )
}
