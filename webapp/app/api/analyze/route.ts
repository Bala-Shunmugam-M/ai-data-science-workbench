/**
 * Steps 3-5 of the journey, streamed as Server-Sent Events.
 *
 * SSE rather than polling because the work is one long subprocess that already
 * emits progress: polling would mean inventing a job store and a status
 * endpoint to carry information the child is happy to push. Each `data:` frame
 * is one `StageEvent`.
 *
 * The stream stays open for the whole run (about a minute — most of which is
 * Python's cold import, measured, not guessed), so nothing here may buffer.
 */
import { existsSync } from 'node:fs'
import path from 'node:path'

import { PythonError, lastMeaningfulLine, runBridgeStream } from '@/lib/python'
import type { StageEvent } from '@/lib/types'
import { UPLOAD_DIR } from '@/lib/uploads'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
// The default 30s would kill a run that legitimately takes ~60s.
export const maxDuration = 900

interface AnalyzeRequest {
  fileId?: string
  displayName?: string
  target?: string
  task?: string
  positiveClass?: string | null
}

const STAGE_IDS = new Set(['upload', 'scan', 'extract', 'analytics', 'insights'])

function isStageEvent(value: unknown): value is StageEvent {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return (
    typeof candidate.stage === 'string' &&
    STAGE_IDS.has(candidate.stage) &&
    typeof candidate.status === 'string'
  )
}

export async function POST(request: Request) {
  let body: AnalyzeRequest
  try {
    body = await request.json()
  } catch {
    return Response.json({ error: 'Malformed request body' }, { status: 400 })
  }

  const { fileId, displayName, target, task, positiveClass } = body
  if (!fileId || !displayName || !target || !task) {
    return Response.json(
      { error: 'fileId, displayName, target and task are all required' },
      { status: 400 },
    )
  }
  if (!/^[a-f0-9-]{36}$/i.test(fileId)) {
    return Response.json({ error: 'Invalid fileId' }, { status: 400 })
  }

  const csv = path.join(UPLOAD_DIR, `${fileId}.csv`)
  if (!existsSync(csv)) {
    return Response.json(
      { error: 'That upload has expired. Please choose the file again.' },
      { status: 410 },
    )
  }

  const args = [
    'analyze',
    '--csv',
    csv,
    '--name',
    displayName,
    '--target',
    target,
    '--task',
    task,
  ]
  if (positiveClass) args.push('--positive-class', positiveClass)

  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let closed = false
      const send = (event: StageEvent) => {
        if (closed) return
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
      }

      // The bridge reports the stage that actually broke. Only synthesise a
      // failure if it never got the chance to — otherwise a single error is
      // shown twice, on two different steps, and the second one is a lie about
      // where it happened.
      let reportedFailure = false

      try {
        const { code, stderr } = await runBridgeStream(args, (value) => {
          if (!isStageEvent(value)) return
          if (value.status === 'failed') reportedFailure = true
          send(value)
        })

        if (code !== 0 && !reportedFailure) {
          // A crash before the bridge could emit its own event (an import
          // error, a kill signal) would otherwise end the stream silently.
          send({
            stage: 'extract',
            status: 'failed',
            message: lastMeaningfulLine(stderr) || `Analysis exited with code ${code}`,
          })
        }
      } catch (error) {
        // Same rule as above: if the bridge already named the broken step, do
        // not append a second failure. `finally` owns closing the stream, so
        // there is no early return here.
        if (!reportedFailure) {
          send({
            stage: 'extract',
            status: 'failed',
            message:
              error instanceof PythonError
                ? error.message
                : error instanceof Error
                  ? error.message
                  : String(error),
          })
        }
      } finally {
        closed = true
        controller.close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      // Belt and braces against any proxy that would buffer the stream.
      'X-Accel-Buffering': 'no',
    },
  })
}
