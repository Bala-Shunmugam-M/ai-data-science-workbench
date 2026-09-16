import { ArrowLeft, ArrowRight } from 'lucide-react'
import Link from 'next/link'
import { notFound } from 'next/navigation'

import { StageBody } from '@/components/journey/stages'
import { StageNotRun } from '@/components/journey/artifact-blocks'
import { buttonVariants } from '@/components/ui'
import { stageHasRun } from '@/lib/artifacts'
import { STAGES, getStage, neighbours } from '@/lib/stages'
import { findProject } from '@/lib/workspace'

// Every stage reads the disk, and the slug is not known until a request
// arrives, so there is nothing worth prerendering here.
export const dynamic = 'force-dynamic'

export default async function StagePage({
  params,
}: {
  params: Promise<{ slug: string; stage: string }>
}) {
  const { slug, stage: stageId } = await params

  const stage = getStage(stageId)
  if (!stage) notFound()
  const project = await findProject(slug)
  if (!project) notFound()

  const hasRun = await stageHasRun(slug, stage)
  const { previous, next } = neighbours(stage.id)

  return (
    <div className="flex flex-col gap-8">
      <div>
        <p className="text-label uppercase tracking-wide text-text-tertiary">
          Step {stage.number} of {STAGES.length}
        </p>
        <h1 className="mt-1 text-h1 text-text">{stage.title}</h1>
        <p className="mt-2 max-w-2xl text-body text-text-secondary">{stage.blurb}</p>
      </div>

      {hasRun ? (
        <StageBody id={stage.id} slug={slug} />
      ) : (
        <StageNotRun title={stage.title} blurb={stage.blurb} />
      )}

      <nav
        aria-label="Stage navigation"
        className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-6"
      >
        {previous ? (
          <Link
            href={`/project/${slug}/${previous.id}`}
            className={buttonVariants({ variant: 'secondary' })}
          >
            <ArrowLeft className="size-4" />
            {previous.number}. {previous.title}
          </Link>
        ) : (
          <Link href={`/project/${slug}`} className={buttonVariants({ variant: 'ghost' })}>
            <ArrowLeft className="size-4" />
            Overview
          </Link>
        )}

        {next ? (
          <Link href={`/project/${slug}/${next.id}`} className={buttonVariants()}>
            {next.number}. {next.title}
            <ArrowRight className="size-4" />
          </Link>
        ) : (
          <Link href="/" className={buttonVariants()}>
            Analyse another dataset
            <ArrowRight className="size-4" />
          </Link>
        )}
      </nav>
    </div>
  )
}
