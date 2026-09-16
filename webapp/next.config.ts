import type { NextConfig } from 'next'

/**
 * `distDir` is overridable so a verification build cannot clobber a running
 * dev server.
 *
 * Both `next dev` and `next build` write to `.next` by default. Building while
 * dev is live replaces the chunks dev is serving, and the browser then dies
 * with `Cannot find module './873.js'` — an error that points at webpack
 * internals and says nothing about the actual cause. This bit twice during
 * development, which is once more than a comment is worth.
 *
 * `npm run verify` sets NEXT_DIST_DIR=.next-verify, so the two never share a
 * directory. A plain `npm run build` still produces `.next` for deployment.
 */
const nextConfig: NextConfig = {
  distDir: process.env.NEXT_DIST_DIR || '.next',
}

export default nextConfig
