'use client'

/**
 * The processing screen, which is mostly an exercise in not lying.
 *
 * Two honest details drive the design. First, the steps are real: each one
 * corresponds to a stage the Python process actually announces, so a step that
 * says "done" is done. Second, the wait is genuinely about a minute and most of
 * it is Python's cold import (measured at ~39s of a ~52s run on a small file,
 * and it barely grows with row count) - so the copy says so instead of showing
 * a progress bar that would have to invent its own position.
 *
 * The skeleton on the right is the results layout, so the shape of what is
 * coming is visible while it is computed.
 */
import { RotateCcw } from 'lucide-react'
import { useEffect, useState } from 'react'

import type { StepView } from '@/hooks/use-analysis'
import { Button, Card, ProgressSteps, Skeleton } from '@/components/ui'

function Elapsed({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [])

  const seconds = Math.max(0, Math.round((now - startedAt) / 1000))
  const minutes = Math.floor(seconds / 60)
  const label = minutes ? `${minutes}m ${String(seconds % 60).padStart(2, '0')}s` : `${seconds}s`

  return (
    <span className="tabular-nums" aria-live="off">
      {label}
    </span>
  )
}

function ResultsSkeleton() {
  return (
    <Card aria-hidden className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-4">
        {[0, 1, 2, 3].map((index) => (
          <div key={index} className="flex flex-col gap-2">
            <Skeleton w="60%" h={12} />
            <Skeleton w="80%" h={32} />
          </div>
        ))}
      </div>
      <div className="flex flex-col gap-3">
        <Skeleton w="40%" h={16} />
        <Skeleton h={180} radius="lg" />
      </div>
      <div className="flex flex-col gap-2">
        {[0, 1, 2, 3, 4].map((index) => (
          <Skeleton key={index} h={14} w={`${95 - index * 6}%`} />
        ))}
      </div>
    </Card>
  )
}

export function ProcessingView({
  fileName,
  steps,
  startedAt,
  onRetry,
}: {
  fileName: string
  steps: StepView[]
  startedAt: number
  onRetry: () => void
}) {
  const failed = steps.find((step) => step.state === 'failed')
  const active = steps.find((step) => step.state === 'active')

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,420px)_1fr]">
      <div className="flex flex-col gap-6">
        <div>
          <h2 className="text-h2 text-text">
            {failed ? 'Analysis stopped' : 'Analysing your dataset'}
          </h2>
          <p className="mt-2 text-body text-text-secondary">
            {failed ? (
              <>
                <span className="text-text">{fileName}</span> got as far as{' '}
                <span className="text-text">{failed.label.toLowerCase()}</span>.
              </>
            ) : (
              <>
                <span className="text-text">{fileName}</span> &middot; you can leave this open.
              </>
            )}
          </p>
        </div>

        <Card>
          {/* Vertical: this column is ~420px, and five horizontal steps there
              give each label ~84px, which makes them run into each other. */}
          <ProgressSteps steps={steps} activeId={active?.id} orientation="vertical" />
        </Card>

        {failed ? (
          <div className="flex flex-col gap-4">
            <div className="rounded-lg border border-danger bg-danger-subtle p-4">
              <p className="text-body font-medium text-danger">{failed.label} failed</p>
              <p className="mt-1 break-words text-caption text-danger">
                {failed.note ?? 'The workbench did not say why.'}
              </p>
            </div>
            <Button icon={<RotateCcw className="size-4" />} onClick={onRetry}>
              Try again
            </Button>
          </div>
        ) : (
          <p className="text-caption text-text-secondary">
            Elapsed <Elapsed startedAt={startedAt} /> &middot; usually about a minute. Most of that
            is Python loading its libraries, so a larger file is barely slower.
          </p>
        )}
      </div>

      <div className="hidden lg:block">
        <p className="mb-3 text-label uppercase tracking-wide text-text-tertiary">
          Your results will look like this
        </p>
        <ResultsSkeleton />
      </div>
    </div>
  )
}
