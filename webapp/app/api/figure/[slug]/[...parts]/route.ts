/**
 * Serve a PNG the pipeline produced.
 *
 * The figure path arrives from the URL, so containment is the point: the
 * resolved file must sit inside this project's `artifacts/` directory and must
 * be a PNG. `resolveFigure` returns null for anything else — `..` traversal,
 * an absolute path, a different project, a non-image.
 */
import { readFile } from 'node:fs/promises'

import { resolveFigure } from '@/lib/workspace'

export const dynamic = 'force-dynamic'

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ slug: string; parts: string[] }> },
) {
  const { slug, parts } = await params

  const relative = parts.map(decodeURIComponent).join('/')
  const file = resolveFigure(slug, relative)
  if (!file) {
    return new Response('Not found', { status: 400 })
  }

  try {
    const bytes = await readFile(file)
    return new Response(new Uint8Array(bytes), {
      headers: {
        'Content-Type': 'image/png',
        // Artifacts are rewritten by a re-analysis, so no long-lived cache.
        'Cache-Control': 'no-cache',
      },
    })
  } catch {
    return new Response('Not found', { status: 404 })
  }
}
