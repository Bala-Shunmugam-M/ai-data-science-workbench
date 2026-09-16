/**
 * Download the report or the source data for one project.
 *
 * Both are files the pipeline already wrote, so this streams bytes rather than
 * generating anything. `kind` is a closed set, not a path fragment.
 */
import { readFile } from 'node:fs/promises'

import { findProject, rawDataFile, readReport } from '@/lib/workspace'

export const dynamic = 'force-dynamic'

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ slug: string; kind: string }> },
) {
  const { slug, kind } = await params

  const descriptor = await findProject(slug)
  if (!descriptor) return new Response('Not found', { status: 404 })

  if (kind === 'report') {
    const markdown = await readReport(slug)
    if (!markdown) return new Response('This project has no report', { status: 404 })
    return new Response(markdown, {
      headers: {
        'Content-Type': 'text/markdown; charset=utf-8',
        'Content-Disposition': `attachment; filename="${slug}-report.md"`,
      },
    })
  }

  if (kind === 'data') {
    const file = await rawDataFile(descriptor)
    if (!file) return new Response('This project has no source data', { status: 404 })
    try {
      const bytes = await readFile(file)
      return new Response(new Uint8Array(bytes), {
        headers: {
          'Content-Type': 'text/csv; charset=utf-8',
          'Content-Disposition': `attachment; filename="${slug}.csv"`,
        },
      })
    } catch {
      return new Response('Not found', { status: 404 })
    }
  }

  return new Response('Unknown download', { status: 400 })
}
