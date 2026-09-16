/**
 * Step 1-2 of the journey: receive the file, then scan it.
 *
 * The upload is written to `webapp/.uploads/` and kept until `analyze` consumes
 * it — the bridge needs a real path on disk, and holding a 50 MB CSV in memory
 * across two requests would be worse.
 */
import { randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'

import { NextResponse } from 'next/server'

import { PythonError, runBridgeJson } from '@/lib/python'
import type { DetectResponse } from '@/lib/types'
import { MAX_UPLOAD_BYTES, UPLOAD_DIR } from '@/lib/uploads'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

function bad(error: string, hint: string) {
  return NextResponse.json({ error, hint }, { status: 400 })
}

/**
 * Re-scan an upload that is already on disk, for a different target column.
 *
 * The task (regression vs classification) is a property of the *target*, not
 * of the file, so changing the target has to re-derive it. Without this the
 * confirm step keeps whatever task was guessed for the originally-detected
 * column, which silently trains a classifier on a continuous target.
 *
 * Reuses the stored upload rather than making the browser send the CSV again.
 */
async function rescan(fileId: string, target: string) {
  if (!/^[a-f0-9-]{36}$/i.test(fileId)) {
    return bad('Invalid file reference', 'Choose the file again.')
  }
  const csv = path.join(UPLOAD_DIR, `${fileId}.csv`)
  if (!existsSync(csv)) {
    return NextResponse.json(
      { error: 'That upload has expired', hint: 'Please choose the file again.' },
      { status: 410 },
    )
  }

  try {
    const scanned = await runBridgeJson<
      Omit<DetectResponse, 'fileId' | 'fileName' | 'sizeBytes'>
    >(['detect', '--csv', csv, '--target', target])
    return NextResponse.json({ fileId, fileName: '', sizeBytes: 0, ...scanned })
  } catch (error) {
    return NextResponse.json(
      {
        error: `Could not analyse '${target}'`,
        hint: error instanceof PythonError ? error.message : String(error),
      },
      { status: 400 },
    )
  }
}

export async function POST(request: Request) {
  // JSON body means "re-scan the file I already sent, for this target".
  if (request.headers.get('content-type')?.includes('application/json')) {
    let body: { fileId?: string; target?: string }
    try {
      body = await request.json()
    } catch {
      return bad('Malformed request', 'Send {"fileId": "...", "target": "..."}.')
    }
    if (!body.fileId || !body.target) {
      return bad('fileId and target are both required', 'Send both fields.')
    }
    return rescan(body.fileId, body.target)
  }

  let form: FormData
  try {
    form = await request.formData()
  } catch {
    return bad('That upload could not be read', 'Send the file as multipart/form-data.')
  }

  const file = form.get('file')
  if (!(file instanceof File)) {
    return bad('No file was attached', 'Choose a CSV and try again.')
  }
  if (!file.name.toLowerCase().endsWith('.csv')) {
    return bad(
      `${file.name} isn't a CSV`,
      'This workbench reads comma-separated files. Export your sheet as CSV and retry.',
    )
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return bad(
      `${file.name} is ${(file.size / 1024 / 1024).toFixed(1)} MB`,
      'The limit is 50 MB. Try a sample of the data instead.',
    )
  }
  if (file.size === 0) {
    return bad(`${file.name} is empty`, 'The file has no contents.')
  }

  const fileId = randomUUID()
  const target = path.join(UPLOAD_DIR, `${fileId}.csv`)
  try {
    await mkdir(UPLOAD_DIR, { recursive: true })
    await writeFile(target, Buffer.from(await file.arrayBuffer()))
  } catch (error) {
    return NextResponse.json(
      {
        error: 'Could not save the upload',
        hint: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    )
  }

  const explicitTarget = form.get('target')
  const args = ['detect', '--csv', target]
  if (typeof explicitTarget === 'string' && explicitTarget) {
    args.push('--target', explicitTarget)
  }

  try {
    const scanned = await runBridgeJson<Omit<DetectResponse, 'fileId' | 'fileName' | 'sizeBytes'>>(
      args,
    )

    if (scanned.columns < 2) {
      return bad(
        `${file.name} has ${scanned.columns} column`,
        'A dataset needs a target column and at least one predictor.',
      )
    }
    if (scanned.rows < 10) {
      return bad(
        `${file.name} has only ${scanned.rows} rows`,
        'There is not enough data to train and evaluate a model.',
      )
    }

    const payload: DetectResponse = {
      fileId,
      fileName: file.name,
      sizeBytes: file.size,
      ...scanned,
    }
    return NextResponse.json(payload)
  } catch (error) {
    if (error instanceof PythonError) {
      return NextResponse.json(
        { error: 'That file could not be read as a table', hint: error.message },
        { status: 400 },
      )
    }
    return NextResponse.json(
      { error: 'Scanning failed', hint: error instanceof Error ? error.message : String(error) },
      { status: 500 },
    )
  }
}
