'use client'

/**
 * The upload -> scan -> analyse state machine.
 *
 * One discriminated union rather than a scatter of booleans, because the
 * screen's whole job is to show exactly one truthful state at a time and
 * `isLoading && !isError && hasFile` combinations are how that goes wrong.
 *
 * The five UI steps do not map one-to-one onto network calls: `upload` and
 * `scan` both complete inside `POST /api/detect`, and `extract`, `analytics`
 * and `insights` stream from `POST /api/analyze`. The step list is therefore
 * driven partly by local knowledge and partly by server events.
 */
import { useCallback, useRef, useState } from 'react'

import type { DetectResponse, StageEvent, StageId, Task } from '@/lib/types'

export interface StepView {
  id: StageId
  label: string
  state: 'pending' | 'active' | 'done' | 'failed'
  note?: string
}

const STEP_LABELS: Record<StageId, string> = {
  upload: 'Uploading',
  scan: 'Scanning document',
  extract: 'Extracting data',
  analytics: 'Running analytics',
  insights: 'Generating insights',
}

const STEP_ORDER: StageId[] = ['upload', 'scan', 'extract', 'analytics', 'insights']

function buildSteps(
  states: Partial<Record<StageId, StepView['state']>>,
  notes: Partial<Record<StageId, string>> = {},
): StepView[] {
  const steps = STEP_ORDER.map((id) => ({
    id,
    label: STEP_LABELS[id],
    state: states[id] ?? 'pending',
    note: notes[id],
  }))

  // Once a step fails the run is over, so everything after it never starts.
  // Leaving those as "Pending" would suggest they are still coming.
  const failedAt = steps.findIndex((step) => step.state === 'failed')
  if (failedAt >= 0) {
    for (const step of steps.slice(failedAt + 1)) {
      if (step.state === 'pending') step.note = 'Not run'
    }
  }
  return steps
}

export interface FileMeta {
  name: string
  sizeBytes: number
}

export type Phase =
  | { kind: 'idle' }
  | { kind: 'scanning'; file: FileMeta; steps: StepView[] }
  | { kind: 'confirm'; file: FileMeta; detection: DetectResponse }
  | { kind: 'running'; file: FileMeta; steps: StepView[]; startedAt: number }
  | { kind: 'done'; slug: string }
  | { kind: 'error'; message: string; hint?: string }

export interface ConfirmedSetup {
  displayName: string
  target: string
  task: Task
  positiveClass: string | null
}

export function useAnalysis() {
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' })
  const abortRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setPhase({ kind: 'idle' })
  }, [])

  /** Steps 1-2: send the file and get the scan back. */
  const scan = useCallback(async (file: File) => {
    const meta: FileMeta = { name: file.name, sizeBytes: file.size }
    setPhase({
      kind: 'scanning',
      file: meta,
      steps: buildSteps({ upload: 'active' }),
    })

    const form = new FormData()
    form.append('file', file)

    let response: Response
    try {
      response = await fetch('/api/detect', { method: 'POST', body: form })
    } catch (error) {
      setPhase({
        kind: 'error',
        message: 'Could not reach the workbench',
        hint: error instanceof Error ? error.message : undefined,
      })
      return
    }

    // The upload itself is finished the moment the server answers.
    setPhase({
      kind: 'scanning',
      file: meta,
      steps: buildSteps({ upload: 'done', scan: 'active' }),
    })

    let payload: unknown
    try {
      payload = await response.json()
    } catch {
      setPhase({ kind: 'error', message: 'The workbench returned an unreadable response' })
      return
    }

    if (!response.ok) {
      const body = payload as { error?: string; hint?: string }
      setPhase({
        kind: 'error',
        message: body.error ?? 'That file could not be read',
        hint: body.hint,
      })
      return
    }

    setPhase({ kind: 'confirm', file: meta, detection: payload as DetectResponse })
  }, [])

  /** Steps 3-5: stream the pipeline. */
  const analyze = useCallback(
    async (detection: DetectResponse, setup: ConfirmedSetup, file: FileMeta) => {
      const startedAt = Date.now()
      const states: Partial<Record<StageId, StepView['state']>> = {
        upload: 'done',
        scan: 'done',
        extract: 'active',
      }
      const notes: Partial<Record<StageId, string>> = {}
      setPhase({ kind: 'running', file, steps: buildSteps(states, notes), startedAt })

      const controller = new AbortController()
      abortRef.current = controller

      let response: Response
      try {
        response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            fileId: detection.fileId,
            displayName: setup.displayName,
            target: setup.target,
            task: setup.task,
            positiveClass: setup.positiveClass,
          }),
          signal: controller.signal,
        })
      } catch (error) {
        setPhase({
          kind: 'error',
          message: 'Could not start the analysis',
          hint: error instanceof Error ? error.message : undefined,
        })
        return
      }

      if (!response.ok || !response.body) {
        const body = (await response.json().catch(() => ({}))) as { error?: string }
        setPhase({ kind: 'error', message: body.error ?? 'The analysis could not be started' })
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let slug: string | undefined

      const handle = (event: StageEvent) => {
        if (event.slug) slug = event.slug
        if (event.status === 'started') {
          states[event.stage] = 'active'
        } else if (event.status === 'done') {
          states[event.stage] = 'done'
        } else {
          states[event.stage] = 'failed'
          notes[event.stage] = event.message ?? 'This step failed.'
        }
        setPhase({ kind: 'running', file, steps: buildSteps(states, notes), startedAt })
      }

      try {
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })

          // SSE frames are separated by a blank line.
          const frames = buffer.split('\n\n')
          buffer = frames.pop() ?? ''
          for (const frame of frames) {
            for (const line of frame.split('\n')) {
              if (!line.startsWith('data:')) continue
              try {
                handle(JSON.parse(line.slice(5).trim()) as StageEvent)
              } catch {
                /* ignore a partial or malformed frame */
              }
            }
          }
        }
      } catch (error) {
        if (controller.signal.aborted) return
        setPhase({
          kind: 'error',
          message: 'The connection to the analysis dropped',
          hint: error instanceof Error ? error.message : undefined,
        })
        return
      }

      const failed = STEP_ORDER.find((id) => states[id] === 'failed')
      if (failed) {
        // Stay on the running view: the failed step is the message, and moving
        // to a generic error screen would hide which stage actually broke.
        return
      }
      if (slug) setPhase({ kind: 'done', slug })
      else
        setPhase({
          kind: 'error',
          message: 'The analysis finished but did not report a project name',
        })
    },
    [],
  )

  return { phase, scan, analyze, reset }
}

export { STEP_LABELS, STEP_ORDER }
