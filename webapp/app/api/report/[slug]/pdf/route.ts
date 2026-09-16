/**
 * Download the project report as a real PDF.
 *
 * Rendering is done by a locally installed Chrome or Edge (see `lib/pdf.ts`)
 * and cached beside the HTML, so a repeat download is a file read.
 */
import { readFile } from 'node:fs/promises'

import { PdfUnavailableError, reportPdfPath } from '@/lib/pdf'
import { findProject } from '@/lib/workspace'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
// A cold render of a large report takes a few seconds; the default would cut
// it off mid-way and return a truncated file.
export const maxDuration = 180

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ slug: string }> },
) {
  const { slug } = await params

  const project = await findProject(slug)
  if (!project) return new Response('Unknown project', { status: 404 })

  try {
    const pdf = await reportPdfPath(slug)
    const bytes = await readFile(pdf)
    return new Response(new Uint8Array(bytes), {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': `attachment; filename="${slug}-report.pdf"`,
        // The PDF is regenerated whenever the report changes, so a cached copy
        // in the browser could easily be stale.
        'Cache-Control': 'no-cache',
      },
    })
  } catch (error) {
    if (error instanceof PdfUnavailableError) {
      // 409: the request is valid, the server just cannot satisfy it in its
      // current state. The message is written to be shown to a person.
      return new Response(error.message, {
        status: 409,
        headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      })
    }
    return new Response('PDF generation failed.', { status: 500 })
  }
}
