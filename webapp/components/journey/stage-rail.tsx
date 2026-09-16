'use client'

/**
 * The numbered stage rail: the spine of the whole journey.
 *
 * Upload is step 0 and is always complete by the time this renders - showing it
 * is what makes the sequence read as "upload, then these nine", which is the
 * mental model the pipeline actually has.
 *
 * A stage with no artifacts on disk is dimmed and labelled, not hidden. Hiding
 * it would make the pipeline look shorter than it is; the honest version is a
 * visible step that says it has not run.
 */
import { Check, Minus } from 'lucide-react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

import { cn } from '@/components/ui'
import type { StageDefinition, StageId } from '@/lib/stages'

export interface RailStage extends StageDefinition {
  hasRun: boolean
}

export function StageRail({
  slug,
  stages,
  current,
  className,
}: {
  slug: string
  stages: RailStage[]
  /**
   * Optional override. Normally left unset: the rail is rendered by the
   * project *layout*, whose params contain only `slug` — the `[stage]` segment
   * belongs to a nested route and is invisible there, so a passed-in value was
   * always `undefined` and no step was ever marked current.
   */
  current?: StageId
  className?: string
}) {
  // Read the stage from the URL instead. Same approach the app shell already
  // uses for its own nav.
  const pathname = usePathname()
  const active = current ?? (pathname.split('/')[3] as StageId | undefined)
  return (
    <nav aria-label="Pipeline stages" className={cn('flex flex-col gap-1', className)}>
      <Link
        href="/"
        className="group flex items-center gap-3 rounded-md px-3 py-2 text-caption text-text-tertiary transition-colors duration-fast hover:bg-surface-hover hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring"
      >
        <span className="grid size-6 shrink-0 place-items-center rounded-full border border-success bg-success-subtle text-success">
          <Check className="size-3.5" />
        </span>
        <span>Upload</span>
      </Link>

      {stages.map((stage) => {
        const isCurrent = stage.id === active
        return (
          <Link
            key={stage.id}
            href={`/project/${slug}/${stage.id}`}
            aria-current={isCurrent ? 'page' : undefined}
            className={cn(
              'group flex items-center gap-3 rounded-md px-3 py-2 text-caption transition-colors duration-fast',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring',
              isCurrent
                ? 'bg-brand-subtle font-medium text-brand'
                : stage.hasRun
                  ? 'text-text-secondary hover:bg-surface-hover hover:text-text'
                  : 'text-text-tertiary hover:bg-surface-hover',
            )}
          >
            <span
              className={cn(
                'grid size-6 shrink-0 place-items-center rounded-full border text-label tabular-nums',
                isCurrent
                  ? 'border-brand bg-brand text-brand-fg'
                  : stage.hasRun
                    ? 'border-border-strong bg-surface text-text-secondary'
                    : 'border-border bg-bg-subtle text-text-tertiary',
              )}
            >
              {stage.hasRun ? stage.number : <Minus className="size-3" />}
            </span>
            <span className="truncate">{stage.title}</span>
            {!stage.hasRun && <span className="ml-auto text-label text-text-tertiary">—</span>}
          </Link>
        )
      })}
    </nav>
  )
}
