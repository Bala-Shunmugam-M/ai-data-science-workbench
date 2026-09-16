import { ArrowRight, Check } from 'lucide-react'
import Link from 'next/link'

import { UploadFlow } from '@/components/upload/upload-flow'
import { Badge, Card, buttonVariants } from '@/components/ui'
import { formatDate, formatNumber } from '@/lib/format'
import { STAGES } from '@/lib/stages'
import { projectSummaries } from '@/lib/workspace'

// The workspaces directory changes whenever an analysis finishes.
export const dynamic = 'force-dynamic'

/**
 * Step 1, and the whole of the front door.
 *
 * The drop zone is the page, not a feature of it. Everything else - what the
 * nine stages will produce, and what has been analysed before - sits below the
 * fold as reassurance, not as competition for the primary action.
 */
export default async function HomePage() {
  const projects = await projectSummaries()

  return (
    <div className="flex flex-col gap-16">
      <section className="pt-4 text-center">
        <p className="text-label uppercase tracking-[0.14em] text-brand">Step 1 of 10</p>
        <h1 className="mx-auto mt-3 max-w-3xl text-display text-text">
          Drop in a dataset. Get a governed analysis.
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-body-lg text-text-secondary">
          One CSV in. The workbench profiles it, validates it, explores it, engineers features,
          trains a field of models, picks a champion and explains why &mdash; then hands you all
          nine stages of evidence.
        </p>

        <div className="mt-10">
          <UploadFlow />
        </div>
      </section>

      <section>
        <div className="mb-6 text-center">
          <h2 className="text-h2 text-text">What happens after you upload</h2>
          <p className="mt-2 text-body text-text-secondary">
            Nine stages, each writing real artifacts you can open and check.
          </p>
        </div>
        <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {STAGES.map((stage) => (
            <li
              key={stage.id}
              className="flex items-start gap-3 rounded-lg border border-border bg-surface p-4"
            >
              <span className="grid size-7 shrink-0 place-items-center rounded-full border border-border-strong bg-bg-subtle text-label tabular-nums text-text-secondary">
                {stage.number}
              </span>
              <div className="min-w-0">
                <p className="text-body font-medium text-text">{stage.title}</p>
                <p className="mt-0.5 text-caption text-text-secondary">{stage.blurb}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      {projects.length > 0 && (
        <section>
          <div className="mb-6 flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-h2 text-text">Previously analysed</h2>
            <p className="text-caption text-text-tertiary">
              {projects.length} project{projects.length === 1 ? '' : 's'} on this machine
            </p>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {projects.map((project) => (
              <Link
                key={project.slug}
                href={`/project/${project.slug}`}
                className="rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring"
              >
                <Card interactive className="flex h-full flex-col gap-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h3 className="truncate text-h3 text-text">{project.displayName}</h3>
                      <p className="mt-1 text-caption text-text-tertiary">
                        {project.task === 'classification' ? 'Classification' : 'Regression'}
                        {project.target ? ` · ${project.target}` : ''}
                      </p>
                    </div>
                    {project.status === 'ready' ? (
                      <Badge tone="good">
                        <Check className="size-3" />
                        Ready
                      </Badge>
                    ) : (
                      <Badge tone="warn">No results</Badge>
                    )}
                  </div>

                  {project.headlineMetric && (
                    <div>
                      <p className="text-label uppercase tracking-wide text-text-tertiary">
                        {project.headlineMetric.label}
                      </p>
                      <p className="text-h2 tabular-nums text-text">
                        {project.headlineMetric.value}
                      </p>
                    </div>
                  )}

                  <p className="mt-auto text-caption text-text-tertiary">
                    {project.rows ? `${formatNumber(project.rows)} rows` : 'rows unknown'}
                    {project.createdAt ? ` · ${formatDate(project.createdAt)}` : ''}
                  </p>
                </Card>
              </Link>
            ))}
          </div>
        </section>
      )}

      <section className="rounded-xl border border-border bg-bg-subtle p-8 text-center">
        <h2 className="text-h3 text-text">Prefer the command line?</h2>
        <p className="mx-auto mt-2 max-w-xl text-body text-text-secondary">
          Everything here reads artifacts the Python pipeline writes. The same run happens with{' '}
          <code className="rounded bg-surface px-1.5 py-0.5 text-caption">python main.py all</code>.
        </p>
        <Link
          href="/settings"
          className={`mt-5 ${buttonVariants({ variant: 'secondary', size: 'sm' })}`}
        >
          See where files are stored
          <ArrowRight className="size-4" />
        </Link>
      </section>
    </div>
  )
}
