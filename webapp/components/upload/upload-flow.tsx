'use client'

import { Check, Lock, RotateCcw } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { useCallback, useEffect, useState } from 'react'

import { useAnalysis } from '@/hooks/use-analysis'
import { Button, Card, DropZone, ProgressSteps } from '@/components/ui'

import { ConfirmStep } from './confirm-step'
import { ProcessingView } from './processing-view'

/** Real slices of the two built-in datasets, cut to 600 rows each. */
const SAMPLES = [
  {
    id: 'california-housing-sample.csv',
    name: 'California housing',
    hint: '600 rows · regression',
  },
  {
    id: 'telco-churn-sample.csv',
    name: 'Telco churn',
    hint: '600 rows · classification',
  },
]

export function UploadFlow() {
  const router = useRouter()
  const { phase, scan, analyze, reset } = useAnalysis()
  const [dropError, setDropError] = useState<string | null>(null)

  useEffect(() => {
    if (phase.kind === 'done') {
      // The overview is a server component reading the disk, so it has to be
      // told the disk changed before we navigate to it.
      router.refresh()
      router.push(`/project/${phase.slug}`)
    }
  }, [phase, router])

  const handleFile = useCallback(
    (file: File) => {
      setDropError(null)
      void scan(file)
    },
    [scan],
  )

  const handleSample = useCallback(
    async (id: string) => {
      setDropError(null)
      try {
        const response = await fetch(`/samples/${id}`)
        if (!response.ok) throw new Error(`sample returned ${response.status}`)
        const blob = await response.blob()
        void scan(new File([blob], id, { type: 'text/csv' }))
      } catch (error) {
        setDropError(
          `That sample could not be loaded (${error instanceof Error ? error.message : 'unknown error'}).`,
        )
      }
    },
    [scan],
  )

  if (phase.kind === 'scanning') {
    return (
      <div className="mx-auto max-w-xl">
        <Card className="flex flex-col gap-5">
          <div>
            <h2 className="text-h3 text-text">{phase.file.name}</h2>
            <p className="mt-1 text-caption text-text-secondary">Reading the file and profiling its columns.</p>
          </div>
          <ProgressSteps steps={phase.steps} />
        </Card>
      </div>
    )
  }

  if (phase.kind === 'confirm') {
    return (
      <ConfirmStep
        file={phase.file}
        detection={phase.detection}
        onCancel={reset}
        onAnalyze={(setup) => void analyze(phase.detection, setup, phase.file)}
      />
    )
  }

  if (phase.kind === 'running') {
    return (
      <ProcessingView
        fileName={phase.file.name}
        steps={phase.steps}
        startedAt={phase.startedAt}
        onRetry={reset}
      />
    )
  }

  if (phase.kind === 'done') {
    return (
      <div className="mx-auto max-w-xl">
        <Card className="flex flex-col items-center gap-3 py-10 text-center">
          <span className="grid size-14 place-items-center rounded-full bg-success-subtle text-success">
            <Check className="size-7" />
          </span>
          <h2 className="text-h2 text-text">Analysis complete</h2>
          <p className="text-body text-text-secondary">Opening the nine-stage breakdown…</p>
        </Card>
      </div>
    )
  }

  if (phase.kind === 'error') {
    return (
      <div className="mx-auto max-w-xl">
        <Card className="flex flex-col gap-4">
          <div>
            <h2 className="text-h3 text-danger">{phase.message}</h2>
            {phase.hint && <p className="mt-2 text-body text-text-secondary">{phase.hint}</p>}
          </div>
          <div>
            <Button icon={<RotateCcw className="size-4" />} onClick={reset}>
              Start over
            </Button>
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div className="mx-auto flex max-w-xl flex-col gap-5">
      <DropZone
        onFile={handleFile}
        accept=".csv"
        maxBytes={50 * 1024 * 1024}
        sampleFiles={SAMPLES}
        onSampleSelect={(id) => void handleSample(id)}
        onError={setDropError}
        error={dropError}
      />

      <p className="flex items-start gap-2 text-caption text-text-secondary">
        <Lock className="mt-0.5 size-4 shrink-0 text-text-tertiary" aria-hidden />
        <span>
          Files never leave this machine. Your dataset is written to a folder on this computer and
          stays there until you delete it.
        </span>
      </p>
    </div>
  )
}
