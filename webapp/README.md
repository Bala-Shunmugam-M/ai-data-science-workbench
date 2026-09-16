# Workbench web app

A Next.js UI for the workbench's upload -> scan -> extract -> analytics ->
results journey. Full spec, decisions, and work-package breakdown live in
[`../docs/WEBAPP_PLAN.md`](../docs/WEBAPP_PLAN.md) — read that first for
anything not covered here.

## Running it

```bash
npm install
npm run dev
```

Opens on **http://localhost:3000**.

```bash
npm run build && npm run start   # production build
npx tsc --noEmit                  # type-check only
```

## Localhost-only, by design

This app spawns the repo's existing Python pipeline (`src/automl`,
`src/pipelines`) as a child process from its API route handlers and reads the
artifacts it writes under `workspaces/<slug>/`. That means:

- It must run **on the same machine** as this Python repo — there is no
  network call to a remote model service.
- It is **not deployable to Vercel (or any host without a local Python
  environment) as-is**. It's a local tool, not a hosted product.
- Uploaded files are written to `webapp/.uploads/` on disk and stay there;
  nothing is encrypted-in-transit or auto-deleted, because nothing leaves
  the machine in the first place.

## Stack

Next.js 15 (App Router) + TypeScript + Tailwind v4 + shadcn/ui + Recharts.
Design tokens (color, spacing, radii, shadows, type scale, motion) are
defined once in `app/globals.css` and exposed to Tailwind via `@theme
inline` — see §2 of the plan for the authoritative token table.

Dark mode is `class`-based (`.dark` on `<html>`), driven by the theme
provider in `lib/theme.tsx` (`useTheme()` returns `{ theme, setTheme,
resolvedTheme }`), persisted to `localStorage`, and applied by an inline
pre-hydration script in `app/layout.tsx` so a dark-mode reload never
flashes white.
