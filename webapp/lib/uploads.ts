import path from 'node:path'

/**
 * Where an upload waits between `POST /api/detect` and `POST /api/analyze`.
 *
 * These live here rather than in the route module because Next.js only allows a
 * route file to export HTTP handlers and a fixed set of config keys; exporting
 * a constant from one fails the production type check.
 */
export const UPLOAD_DIR = path.join(process.cwd(), '.uploads')

/** Matches the limit stated on the upload screen. Keep the two in step. */
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024
