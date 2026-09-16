/**
 * Serve any readable artifact from inside one project's workspace.
 *
 * Two gates, both required. `resolveInWorkspace` guarantees the resolved path
 * stays inside the workspace, so `..` traversal and absolute paths are refused.
 * The extension allow-list then limits what can be read at all: the workspace
 * also holds `.joblib` model files, and there is no reason to hand those out
 * over HTTP.
 */
import { readFile } from 'node:fs/promises'
import path from 'node:path'

import { resolveInWorkspace } from '@/lib/artifacts'

export const dynamic = 'force-dynamic'

const CONTENT_TYPES: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.html': 'text/html; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.pdf': 'application/pdf',
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ slug: string; parts: string[] }> },
) {
  const { slug, parts } = await params
  const relative = parts.map(decodeURIComponent).join('/')

  const target = resolveInWorkspace(slug, relative)
  if (!target) return new Response('Bad path', { status: 400 })

  const contentType = CONTENT_TYPES[path.extname(target).toLowerCase()]
  if (!contentType) return new Response('Unsupported artifact type', { status: 400 })

  try {
    const bytes = await readFile(target)
    const download = new URL(request.url).searchParams.has('download')
    return new Response(new Uint8Array(bytes), {
      headers: {
        'Content-Type': contentType,
        // Artifacts are rewritten by a re-run, so never cache them.
        'Cache-Control': 'no-cache',
        ...(download
          ? { 'Content-Disposition': `attachment; filename="${path.basename(target)}"` }
          : {}),
      },
    })
  } catch {
    return new Response('Not found', { status: 404 })
  }
}
