# Workbench Web App — Implementation Plan

**Deliverable:** a professional, SaaS-grade product UI for the workbench's
upload → scan → extract → analytics → results journey.
**Stack:** Next.js 15 (App Router) + TypeScript + Tailwind v4 + shadcn/ui + Recharts.
**Scope:** product app only. No marketing/landing page.

This document is the single source of truth for the implementers. Do not invent
tokens, component APIs or route shapes — they are all specified here. If
something is genuinely missing, choose the option most consistent with what is
written and note it in your final report.

---

## 1. Decisions

### 1.1 Location — `webapp/`, and `app_web/` is left alone

The new app lives at `ai-data-science-workbench/webapp/`.

`app_web/index.html` + `app_web/app_data.json` stay exactly where they are and
are **not deleted**. That page is a working, self-contained model-card/governance
artifact with a client-side Ridge predictor; deleting it to free a directory name
buys nothing and loses a working thing. Once the new app is verified, retiring
`app_web/` is the owner's call, not this build's.

### 1.2 Architecture — real Python bridge, not mocked JSON

The Next.js app **spawns the existing Python pipeline** through route handlers
and reads the artifacts it writes.

The trade-off: a static UI over pre-generated JSON would ship faster and deploy
anywhere, but the entire described journey — upload, staged scanning, extraction,
analytics — would be theatre over a fixed file, which is the exact opposite of
"feels intelligent and transparent". A local Node process spawning a local Python
process is cheap and makes every screen show something true.

Consequence: this app runs **on localhost next to the Python repo**. It is not
deployable to Vercel as-is. That is the correct trade for a workbench.

### 1.3 The five progress steps map onto real work

`main.py autorun` is four sequential Python calls (verified in `main.py::_cmd_autorun`):

```
generic_pipeline.run_prep() → training_pipeline.run() → evaluation_pipeline.run() → auto_report.run()
```

Mapped to the required UI steps:

| # | UI step             | What actually happens                                    |
|---|---------------------|----------------------------------------------------------|
| 1 | Uploading           | Node receives the file, writes it to `webapp/.uploads/`   |
| 2 | Scanning document   | `src.automl.detect.detect(df)` — profile, task, target    |
| 3 | Extracting data     | `register_project(...)` + `generic_pipeline.run_prep()`   |
| 4 | Running analytics   | `training_pipeline.run()` + `evaluation_pipeline.run()`   |
| 5 | Generating insights | `src.automl.report.run()`                                 |

Steps 3–5 stream from one Python process; step 2 is its own fast call so the
user can confirm the detected target before anything trains.

### 1.4 Routing

```
webapp/app/
  layout.tsx                 root: fonts, theme provider, <html>
  (app)/layout.tsx           shell: sidebar + topbar
  (app)/page.tsx             Dashboard / History
  (app)/upload/page.tsx      Upload + Processing (one route, staged states)
  (app)/results/[slug]/page.tsx   Results
  (app)/results/[slug]/preview/page.tsx   File preview (side-by-side)
  (app)/settings/page.tsx    Settings
  api/detect/route.ts        POST multipart -> detection JSON
  api/analyze/route.ts       POST JSON -> SSE stream of stage events
  api/projects/route.ts      GET -> history list
  api/results/[slug]/route.ts GET -> results payload
  api/figure/[slug]/[name]/route.ts GET -> PNG bytes
```

### 1.5 State

No state library. Server Components fetch where possible; the upload/processing
flow is one client component holding a discriminated-union state:

```ts
type Phase =
  | { kind: 'idle' }
  | { kind: 'uploading'; file: FileMeta; pct: number }
  | { kind: 'detecting'; file: FileMeta }
  | { kind: 'confirm'; file: FileMeta; detection: Detection; preview: PreviewTable }
  | { kind: 'running'; file: FileMeta; steps: StepState[]; slug?: string }
  | { kind: 'done'; slug: string }
  | { kind: 'error'; message: string; recoverable: boolean }
```

### 1.6 Charts — Recharts

Composable React primitives, restrained by default, no canvas/WebGL ceremony.
**Only three chart types are permitted:** horizontal bar, line, donut. No
gradients on data marks, no 3D, no drop shadows on series, no more than 6
categorical colours. Axis labels and a one-line plain-language caption under
every chart are mandatory.

### 1.7 Dark mode

`class`-based (`.dark` on `<html>`), toggled by a provider that reads
`localStorage` then `prefers-color-scheme`, with an inline pre-hydration script
to prevent a flash. Every token below has a dark value; components reference
**semantic** CSS variables only, never a raw hex or a Tailwind palette number.

---

## 2. Design tokens — authoritative

Defined once in `webapp/app/globals.css` as CSS custom properties, exposed to
Tailwind v4 via `@theme inline`. **Implementers must not add colours.**

### 2.1 Colour

| Semantic token        | Light     | Dark      |
|-----------------------|-----------|-----------|
| `--bg`                | `#FFFFFF` | `#0B0D12` |
| `--bg-subtle`         | `#F7F8FA` | `#171B24` |
| `--surface`           | `#FFFFFF` | `#12151C` |
| `--surface-hover`     | `#F7F8FA` | `#1A1F29` |
| `--border`            | `#E4E7EC` | `#262B36` |
| `--border-strong`     | `#D0D5DD` | `#343A46` |
| `--text`              | `#101828` | `#F2F4F7` |
| `--text-secondary`    | `#475467` | `#98A2B3` |
| `--text-tertiary`     | `#667085` | `#667085` |
| `--brand`             | `#1F4AD1` | `#5B82F0` |
| `--brand-hover`       | `#1A3CAB` | `#7398F4` |
| `--brand-subtle`      | `#EFF4FF` | `#151C33` |
| `--brand-fg`          | `#FFFFFF` | `#0B0D12` |
| `--success`           | `#067647` | `#75E0A7` |
| `--success-subtle`    | `#ECFDF3` | `#053321` |
| `--warning`           | `#B54708` | `#FEC84B` |
| `--warning-subtle`    | `#FFFAEB` | `#4E1D09` |
| `--danger`            | `#B42318` | `#FDA29B` |
| `--danger-subtle`     | `#FEF3F2` | `#55160C` |
| `--focus-ring`        | `#1F4AD1` | `#5B82F0` |

**One brand accent: a deep, slightly cool blue (`#1F4AD1`).** Blue reads as
trust/infrastructure and is what Stripe, Linear and Google all resolve to for
primary actions; the deeper value keeps it from looking like default Tailwind
blue, and it deliberately avoids the "AI purple" the brief calls out. Semantic
green/amber/red are for *data status only* — never for UI chrome.

Chart categorical ramp (max 6, in order):
`#1F4AD1`, `#0E9384`, `#B54708`, `#7A5AF8`, `#DD2590`, `#475467`.
Dark equivalents: `#5B82F0`, `#2ED3B7`, `#FEC84B`, `#9B8AFB`, `#F670C7`, `#98A2B3`.

### 2.2 Spacing — 4px base

`1=4 2=8 3=12 4=16 5=20 6=24 8=32 10=40 12=48 16=64 20=80`
Card padding `24`. Section gap `32`. Page gutter `32` desktop / `16` mobile.
Content max width `1280`.

### 2.3 Radii

`sm 6 · md 8 · lg 12 · xl 16 · full 9999`.
Buttons/inputs `md`. Cards `lg`. Drop zone / modals `xl`.

### 2.4 Shadow

```
xs  0 1px 2px rgba(16,24,40,.05)
sm  0 1px 3px rgba(16,24,40,.10), 0 1px 2px rgba(16,24,40,.06)
md  0 4px 8px -2px rgba(16,24,40,.10), 0 2px 4px -2px rgba(16,24,40,.06)
lg  0 12px 16px -4px rgba(16,24,40,.08), 0 4px 6px -2px rgba(16,24,40,.03)
```
Dark mode: replace shadows with `--border` outlines; shadows on near-black read as dirt.

### 2.5 Type — Inter via `next/font/google`

| Role        | Size/Line | Weight | Notes                     |
|-------------|-----------|--------|---------------------------|
| metric      | 48/56     | 700    | `tabular-nums`, `-0.02em` |
| display     | 36/44     | 700    | `-0.02em`                 |
| h1          | 30/38     | 600    | `-0.01em`                 |
| h2          | 24/32     | 600    |                           |
| h3          | 20/30     | 600    |                           |
| body-lg     | 18/28     | 400    |                           |
| body        | 16/24     | 400    | default                   |
| sm          | 14/20     | 400    | table cells, captions     |
| xs          | 12/18     | 500    | labels, badges, uppercase |

Numeric table columns and all metrics use `font-variant-numeric: tabular-nums`.

### 2.6 Motion

`fast 120ms · base 180ms · slow 280ms`, easing `cubic-bezier(.2,.8,.2,1)`.
Card stagger `60ms` per item, capped at 8 items. File-drop entry:
`opacity 0→1` + `scale .96→1` over `base`.
**Every animation must be wrapped in `@media (prefers-reduced-motion: no-preference)`.**

---

## 3. Data contracts

Types live in `webapp/lib/types.ts` and are the shared vocabulary.

```ts
export type Task = 'regression' | 'classification'

export interface ColumnProfile {
  name: string; dtype: string; missing: number; missingPct: number
  unique: number; example: string
}

export interface Detection {
  target: string; task: Task; positiveClass: string | null
  categoricalColumns: string[]; dropColumns: string[]
  selectionMetric: string; nClasses: number | null; warnings: string[]
}

export interface PreviewTable { columns: string[]; rows: (string | number | null)[][] }

export interface DetectResponse {
  fileId: string; fileName: string; sizeBytes: number
  rows: number; columns: number
  detection: Detection; profile: ColumnProfile[]; preview: PreviewTable
}

export type StageId = 'upload' | 'scan' | 'extract' | 'analytics' | 'insights'

export interface StageEvent {
  stage: StageId; status: 'started' | 'done' | 'failed'
  message?: string; slug?: string; elapsedMs?: number
}

export interface Metric { label: string; value: string; tone?: 'neutral' | 'good' | 'warn' | 'bad'; hint?: string }

export interface Insight { title: string; body: string; tone: 'good' | 'warn' | 'bad' | 'neutral' }

export interface ResultsPayload {
  slug: string; displayName: string; createdAt: string
  task: Task; target: string; rows: number; columns: number
  headline: Metric[]              // top summary bar
  champion: { name: string; version: string; metrics: Metric[]; rationale: string }
  comparison: { rank: number; model: string; version: string; metrics: Record<string, number> }[]
  drivers: { rank: number; feature: string; coefficient: number; direction: 'positive' | 'negative'; meaning: string }[]
  insights: Insight[]
  figures: { name: string; caption: string; url: string }[]
  reportMarkdown: string | null
  extracted: PreviewTable
}

export interface ProjectSummary {
  slug: string; displayName: string; createdAt: string
  task: Task; target: string; rows: number
  status: 'ready' | 'running' | 'failed'
  headlineMetric: Metric | null
}
```

### 3.1 `POST /api/detect`

`multipart/form-data` with `file`. Writes the CSV to `webapp/.uploads/<uuid>.csv`,
shells `python webapi/bridge.py detect --csv <path>`, returns `DetectResponse`.

Errors → `400` with `{ error, hint }`:
non-CSV extension, >50 MB, unparseable CSV, fewer than 2 columns, 0 rows.

### 3.2 `POST /api/analyze` → SSE

Body: `{ fileId, displayName, target, task, positiveClass? }`.
Responds `text/event-stream`, one `data: <StageEvent JSON>` per line, terminating
with `stage:'insights', status:'done', slug`. On failure emits
`status:'failed'` with a message and closes — the client must render the failure
against the step that broke, never a generic toast.

### 3.3 `GET /api/results/[slug]` → `ResultsPayload`

Reads from `workspaces/<slug>/`:
`dataset.json`, `artifacts/final_model_selection.json`,
`artifacts/reports/auto_report.md`, `artifacts/auto_eda/*.png`,
`artifacts/evaluation/figures/*.png`, `data/raw/data.csv` (first 50 rows).
`404` if the workspace has no `dataset.json`.

### 3.4 `GET /api/figure/[slug]/[name]` → `image/png`

**Must** resolve the path and verify it stays inside
`workspaces/<slug>/artifacts/`. Reject `..` and absolute paths with `400`.

### 3.5 Truthful trust copy

The upload screen states what is actually true — do not invent encryption or
retention claims:

> Files never leave this machine. Your dataset is written to
> `workspaces/<name>/` on this computer and stays until you delete it.

---

## 4. Component inventory

`webapp/components/ui/` — primitives (shadcn where it exists, hand-rolled otherwise).
Every interactive component implements **hover, focus-visible, active, disabled**
and is keyboard operable.

| Component | Props | Notes |
|---|---|---|
| `Button` | `variant: primary\|secondary\|ghost\|destructive`, `size: sm\|md\|lg`, `loading`, `icon` | Primary = filled `--brand`. Loading shows a spinner and keeps width. |
| `Card` | `padded`, `interactive` | `--surface`, `lg` radius, `sm` shadow (light) / border (dark) |
| `MetricTile` | `label`, `value`, `tone`, `hint`, `trend?` | metric type scale, tabular-nums |
| `Badge` | `tone: neutral\|good\|warn\|bad`, `size` | subtle bg + solid fg |
| `DataTable` | `columns`, `rows`, `maxHeight`, `zebra`, `stickyHeader` | numeric cols right-aligned + tabular-nums; horizontal scroll never on `<body>` |
| `EmptyState` | `icon`, `title`, `body`, `action` | never a bare "no data" |
| `Skeleton` | `w`, `h`, `radius` | shimmer; the default loading affordance |
| `ProgressSteps` | `steps: {id,label,state}[]`, `activeId` | vertical on mobile, horizontal on desktop |
| `DropZone` | `onFile`, `accept`, `maxBytes`, `sampleFiles` | full keyboard support; `aria-describedby` for formats |
| `Toast` | `tone`, `title`, `body` | polite live region |
| `Chart{Bar,Line,Donut}` | `data`, `xKey`, `series`, `caption` | Recharts, tokens only, caption required |
| `ThemeToggle` | — | light / dark / system |
| `Sidebar`, `Topbar` | — | collapsible, `aria-current` on active nav |

---

## 5. Screens

### 5.1 Upload — `/upload`

Centred, max-width `640`. Drop zone `xl` radius, dashed `--border-strong`,
`--bg-subtle` fill, min-height `280`. Copy: **"Drag & drop your CSV here or
click to browse"**. Under it, in `sm`/`--text-tertiary`: "CSV up to 50 MB".
Three sample-file chips that load instantly. Primary button **"Upload & Analyze"**.
Trust line (§3.5) directly beneath, with a lock icon.

On drop: file card fades+scales in showing name, size, row/col count once known,
and a determinate progress bar. Then the **confirm step** appears inline (not a
new page, not a wizard): detected target (select, overridable), detected task
(segmented control), project name (text), and a collapsed column profile +
20-row preview. Warnings from `detection.warnings` render as amber inline notes.

Empty state = the drop zone itself. Error state = inline red note under the drop
zone naming the fix ("This file has 1 column — a CSV needs at least a target and
one predictor").

### 5.2 Processing — same route, `running` phase

`ProgressSteps` with the five steps, each animating to `done` as its
`StageEvent` arrives. Elapsed time counter and "Usually takes 20–60 seconds"
under the list. **Right of the steps: a skeleton of the results page** — grey
metric tiles, chart placeholders, table rows — so progress is visible.
Never a bare spinner. On `failed`, the broken step turns red, the remaining
steps grey out, and the error message plus a "Try again" button appear;
the console output is available behind a collapsed "Show log".

### 5.3 Results — `/results/[slug]`

- **Top summary bar**: 3–4 `MetricTile`s in metric type scale — champion model,
  the headline accuracy metric (R² or ROC-AUC), rows analysed, features used.
  Tone-coloured. This is above the fold, always.
- **Actions top-right**: Download Report · Export CSV · Re-analyze · Share
  (Share copies the local URL and says so).
- **Insights** — 2–4 `Card`s, one sentence each, plain language, tone-coloured
  left border. Highest-value insight first.
- **Analytics** — 2-up chart grid: driver importance (horizontal bar) and one
  EDA figure; each with its mandatory caption.
- **Extracted data** — `DataTable`, first 50 rows, sticky header, own
  `overflow-x` container.
- **Model detail** — champion card + comparison table + coefficient/driver list.
  This is where the old `app_web` model-card content is preserved, but rendered
  from live artifacts instead of a frozen JSON snapshot.
- Cards fade/slide in staggered `60ms`.
- Loading = skeletons in the same layout. Empty (`404`) = `EmptyState` pointing
  back to `/upload`.

### 5.4 Dashboard / History — `/`

Card grid (or list on mobile) of past uploads: display name, task badge, target,
row count, date, status badge, headline metric, and quick actions (Open,
Re-analyze, Delete — Delete asks first). Empty state: a short "Upload your first
dataset" with the primary CTA. This is the app's index route.

### 5.5 File preview — `/results/[slug]/preview`

Two panes, 50/50 desktop, stacked mobile: left = original CSV as delivered
(monospace, `sm`, first 200 rows, virtualised or capped); right = the extracted /
prepared table. Synchronised horizontal scroll is **not** required. Column count
and row count shown per pane.

### 5.6 Settings — `/settings`

Grouped `Card`s, one concern each: Appearance (theme), Analysis defaults
(sample rows, max upload size), Storage (workspace path, "Reveal in Explorer",
per-project delete). No dense forms; label + control + one-line helper.

---

## 6. Work packages

**No file appears in two packages.** Each agent owns its list exclusively.

### P1 — Python bridge *(no dependencies; runs in parallel with P2)*

Owns: `ai-data-science-workbench/webapi/__init__.py`, `webapi/bridge.py`,
`webapi/README.md`.

`bridge.py` is a CLI with two subcommands that print **JSON to stdout only**
(all logging to stderr):

- `detect --csv <path> [--target <col>]` → one JSON object matching
  `DetectResponse` minus `fileId`/`fileName`/`sizeBytes`.
- `analyze --csv <path> --name <display> --target <col> --task <task>
  [--positive-class <v>]` → one JSON line per stage transition
  (`{"stage":...,"status":...,"message":...,"slug":...,"elapsedMs":...}`),
  flushed immediately, covering `extract`, `analytics`, `insights`. It calls
  `register_project`, then `generic_pipeline.run_prep()`,
  `training_pipeline.run()`, `evaluation_pipeline.run()`, `auto_report.run()`
  in-process with `WORKBENCH_PROJECT` set, catching exceptions per stage and
  emitting `status:"failed"` with the message.

Must not modify any existing Python file. Import from `src.automl.detect`,
`src.automl.register`, `src.pipelines.*`, `src.automl.report` as they are.

Acceptance:
`python webapi/bridge.py detect --csv workspaces/<any>/data/raw/data.csv` prints
valid JSON; piping it through `python -m json.tool` succeeds.

### P2 — Scaffold, tokens, theme *(depends on nothing; blocks P3–P6)*

Owns: `webapp/package.json`, `webapp/next.config.ts`, `webapp/tsconfig.json`,
`webapp/postcss.config.mjs`, `webapp/components.json`, `webapp/.gitignore`,
`webapp/app/globals.css`, `webapp/app/layout.tsx`, `webapp/lib/theme.tsx`,
`webapp/lib/types.ts`, `webapp/lib/utils.ts`, `webapp/lib/format.ts`,
`webapp/README.md`.

Scaffold Next.js 15 + TS + Tailwind v4 + shadcn/ui. Encode §2 tokens verbatim in
`globals.css`. Implement the theme provider + anti-flash script. `lib/types.ts`
is §3 verbatim. `lib/format.ts` holds number/byte/date formatters (tabular,
locale-stable).

Acceptance: `npm run build` succeeds; `npm run dev` serves a themed blank page
that respects the OS theme and survives a reload without flashing.

### P3 — Component library *(depends on P2)*

Owns: everything under `webapp/components/ui/`.

Every component in §4 except `Sidebar`/`Topbar`. Plus
`webapp/app/(dev)/kitchen-sink/page.tsx` — a single page rendering every
component in every state, light and dark. That page is the acceptance check.

Acceptance: `/kitchen-sink` renders all components in all states; keyboard tab
order is visible and correct; no raw hex anywhere in `components/ui/`.

### P4 — Shell, Dashboard, Settings *(depends on P2, P3; parallel with P5, P6)*

Owns: `webapp/components/shell/**`, `webapp/app/(app)/layout.tsx`,
`webapp/app/(app)/page.tsx`, `webapp/app/(app)/settings/page.tsx`,
`webapp/app/api/projects/route.ts`.

Collapsible sidebar (Dashboard, Upload, Settings), topbar with global search
placeholder, theme toggle and breadcrumb/page title. `GET /api/projects` scans
`workspaces/*/dataset.json` and returns `ProjectSummary[]` sorted newest first.

Acceptance: sidebar collapses and persists; `/` lists real workspaces; empty
state renders when none exist; usable at 375px.

### P5 — Upload + Processing *(depends on P2, P3; parallel with P4, P6)*

Owns: `webapp/app/(app)/upload/page.tsx`, `webapp/components/upload/**`,
`webapp/hooks/use-analysis.ts`, `webapp/app/api/detect/route.ts`,
`webapp/app/api/analyze/route.ts`, `webapp/lib/python.ts`,
`webapp/public/samples/*`.

`lib/python.ts` is the single place that spawns Python (resolve the interpreter,
set `WORKBENCH_PROJECT`, cwd = repo root, stream stdout lines). §5.1 and §5.2 in
full, against P1's contract.

Acceptance: dropping a real CSV shows the confirm step with a correctly detected
target, and "Upload & Analyze" drives all five steps to done and lands on
`/results/<slug>`. Killing the Python process mid-run shows the failure on the
correct step.

### P6 — Results + File preview *(depends on P2, P3; parallel with P4, P5)*

Owns: `webapp/app/(app)/results/[slug]/page.tsx`,
`webapp/app/(app)/results/[slug]/preview/page.tsx`,
`webapp/components/results/**`, `webapp/app/api/results/[slug]/route.ts`,
`webapp/app/api/figure/[slug]/[name]/route.ts`.

§5.3 and §5.5. Reads real artifacts. Charts are Recharts, restricted to §1.6.
Path traversal guard on the figure route is mandatory.

Acceptance: `/results/california-housing` renders headline metrics, champion
card, comparison table, drivers, at least one figure and the extracted table,
in both themes, without a horizontal scrollbar on `<body>` at 375/768/1280.

---

## 7. Verification

```bash
cd D:/AI/ai-data-science-workbench/webapp && npm run build
```

```bash
cd D:/AI/ai-data-science-workbench/webapp && npm run dev
```

Dev server on **port 3000**. Click-through: `/` → Upload → drop
`workspaces/california-housing/data/raw/*.csv` → confirm → watch five steps →
Results → Preview → Settings → toggle dark → reload (no flash).

- **Accessibility**: tab the whole flow with no mouse; visible focus everywhere;
  drop zone reachable and activatable by keyboard; charts have text alternatives;
  body text ≥16px; contrast ≥4.5:1 both themes.
- **Responsive**: 375, 768, 1280. Wide tables/charts scroll inside their own
  container — the page body never scrolls horizontally.
- **States**: every screen checked with data, empty, loading and error.

---

## 7b. Second pass — the nine-stage journey

The flat product UI in §5 was replaced by an upload-first, sequential journey.
The brief: "upload is step 1, then the process", modelled on iLovePDF's
upload → progress → result flow.

**New information architecture**

```
/                             step 1 — the drop zone IS the page,
                              plus what-happens-next and past projects
  ↓ scan → confirm → stream
/project/[slug]               journey overview: 9 numbered stage cards + progress bar
/project/[slug]/[stage]       one stage, with a persistent numbered rail and prev/next
/settings
```

The nine stages come straight from the Streamlit sidebar, in pipeline order:
Dashboard, Data, EDA, Features, Modeling, Evaluation, Explainability,
Governance, Reports. `lib/stages.ts` is the single source of truth — the rail,
the overview cards, the prev/next footer and route validation all read it.

**Stage completeness is detected, not assumed.** Each stage declares `probes`
(artifacts that prove it ran). A stage with none of them renders "has not run"
and is dimmed in the rail rather than hidden — hiding it would make the
pipeline look shorter than it is.

**New files**: `lib/stages.ts`, `lib/artifacts.ts`, `components/journey/*`,
`app/(app)/project/[slug]/{layout,page}.tsx`,
`app/(app)/project/[slug]/[stage]/page.tsx`,
`app/api/artifact/[slug]/[...parts]/route.ts`.
`/upload` and `/results/*` became redirects rather than being deleted.

**Autorun**: `start-webapp.bat` (installs on first run, starts the server,
polls the port, opens the browser) and `stop-webapp.bat`.

---

## 7f. Real PDF download

The earlier "Save as PDF" opened a print dialog. This renders an actual file.

`lib/pdf.ts` drives a **locally installed Chrome or Edge** in headless mode
(`--headless=new --print-to-pdf`). The report is self-contained HTML with its
own CSS and base64 figures, so faithful output needs a real layout engine —
and every library option costs more than it gives: Puppeteer/Playwright pull a
~300 MB Chromium, WeasyPrint needs GTK on Windows, wkhtmltopdf needs its own
binary. Windows 11 ships Edge, so the dependency is already there.

- `GET /api/report/[slug]/pdf` renders on demand and caches the result at
  `artifacts/reports/project_report.pdf`, re-rendering only when the HTML is
  newer. Cold ~3 s, warm ~0.2 s.
- Discovery order: `WORKBENCH_CHROME` env var, then Edge, then Chrome, with
  POSIX paths for non-Windows.
- No browser, or no report yet → **409 with a sentence written for a person**,
  surfaced inline by the button rather than navigating to an error page.
- `.pdf` added to the artifact route's extension allow-list so the deliverable
  row can serve it too.

Verified: `%PDF-1.4`, 8 pages, 1.9 MB, valid EOF marker; regenerates when the
source HTML changes; 409 on the two projects that have no `project_report.html`.

---

## 7e. Drivers moved into the domain layer

The generic driver extraction started life inside `webapi/bridge.py`, which
made a transport shim ~200 lines of model introspection and left the logic
reachable only from the web upload path.

It now lives in `src/explainability/drivers.py` as
`write_driver_artifacts(*, top_n=12, force=False)`, exposed by a new CLI
command:

```
python main.py drivers            # any project's champion
python main.py drivers --force    # replace an `explain` artifact
```

`bridge.py` is a two-line call again (351 lines, down from ~440), and the
module now also writes `top_drivers.png`, which the bridge version never did.

**The filename collision that surfaced.** `explain` and `drivers` both write
`coefficient_interpretation.{csv,json}`, so running `drivers` on the housing
project silently replaced the *graded* artifact — curated per-feature meanings
and dollar wording swapped for the generic table. `drivers` now refuses to
overwrite an artifact produced by `interpret_champion`, detected by the absence
of the `kind` key (which only the generic writer emits). `--force` overrides.
Uploaded projects are unaffected: their artifacts carry `kind`, so re-running
is free.

---

## 7d. Explainability for uploaded projects

Stage 7 was empty for every uploaded dataset — the most valuable screen, blank
on the app's main path, because only the graded housing pipeline wrote
`artifacts/explainability/`.

`interpret_champion()` could not simply be called for uploads: its executive
briefing is hard-coded to California house prices ("roughly 20,600 census
blocks", coastal caveats, dollar figures), so running it on an arbitrary
dataset would emit confident prose about the wrong domain.

`webapi/bridge.py::_write_explainability` therefore reuses only the
dataset-agnostic parts — the champion bundle, its coefficient/importance
vector and its feature names — and writes the coefficient table and JSON with
no briefing. It runs inside the existing `insights` stage, so the five-step
progress contract is unchanged.

Two model shapes, distinguished by a `kind` field in the JSON so the UI cannot
imply something false:

- `coefficient` — Linear/Ridge/Lasso/Logistic. Signed, so direction is real.
- `importance` — DecisionTree/RandomForest. Unsigned magnitudes; the chart
  caption explicitly says they carry no direction.

Multiclass logistic is skipped rather than collapsed: one row of coefficients
per class cannot be reduced to a single effect without inventing one.

Wording is domain-neutral ("increases the prediction", not "increases value" —
a logistic coefficient is log-odds), and the curated housing meanings are still
used for housing features, with `Model input '<name>'.` elsewhere.

---

## 7c. Third pass — parity, downloads, and page weight

**Reports.** Real Download / Save as PDF / Open buttons, plus every deliverable
(executive briefing, auto report, final model selection, evaluation report)
with its path and a download button. "Save as PDF" opens the report in its own
window and calls `print()` — the browser's own HTML-to-PDF, rather than adding
a headless-Chrome renderer to duplicate it.

**Detail parity with Streamlit.** The stage bodies were reading CSV artifacts
and ignoring the richer JSON ones. Now added: the semantic schema (via a new
`bridge.py schema` subcommand, since `SEMANTIC_SCHEMA` is a Python constant
with no on-disk form), `feature_proposal.json` in full, `model_proposal.json`
(scope, rationale, validation strategy, candidate catalogue, multicollinearity
flags), and lineage nodes with inputs/outputs/params.

**The "letter collision".** A programmatic overlap scan found zero real
overlaps once false positives from clipping containers were excluded. The
mechanism was `position: sticky` headers inside a `border-collapse: collapse`
table: the collapsed-border model hands painting to the table, the sticky `th`
background does not reliably paint, and scrolling rows show through it. Fixed
with `border-separate border-spacing-0`, an explicit opaque header background,
and row rules moved onto the cells (a `<tr>` cannot draw a border in the
separated model).

**Page weight.** Measure in a production build — the dev server's
instrumentation inflates pages roughly 4x and made it look as though whole CSVs
were being embedded. Two fixes, measured on the Data stage (2,354 KB → 406 KB):
tabbed dataset previews so one table renders instead of six, and the invariant
table-cell styling hoisted into `.dt-cell` / `.dt-head` in `globals.css`
instead of ~100 characters of repeated utilities on each of 7,000+ cells.

---

## 7a. As built — deviations from this plan

All six packages landed. What differs from the plan above, and why:

| Plan said | As built | Why |
|---|---|---|
| type roles `text-sm` / `text-xs` | `text-caption` / `text-label` | collided with Tailwind's core utilities |
| spacing scale redefined | Tailwind v4 defaults used as-is | they already equal the 4px base; redefining would duplicate truth |
| `lib/utils.ts` `cn()` | extended with a `font-size` class group | stock `tailwind-merge` silently drops `text-metric` when merged with a colour — `lib/utils.test.ts` guards it |
| `buttonVariants` exported from `button.tsx` | moved to `button-variants.ts` | `button.tsx` is `'use client'`; a client export cannot be *called* from a Server Component |
| `MAX_UPLOAD_BYTES` in the detect route | `lib/uploads.ts` | Next route files may only export handlers + fixed config keys |
| own icon set | `lucide-react` | already a dependency via `Button`; two icon sets is one too many |
| results payload built in the route | `lib/results.ts`, used by both | the page and the API route must not drift |
| `Re-analyze` action | **not built** | `bridge.py analyze` always registers a *new* project, so "re-analyze" would silently create a duplicate. Needs a bridge subcommand that re-runs an existing workspace. |
| `Share` action | built, labelled "Copy link" | it can only copy a localhost URL; the label says so |
| Toast used on screens | not used | nothing yet fires a message from outside its own render tree |

Measured, replacing the plan's guess: a full `analyze` takes **~26 s** through the
warm dev server (~52 s cold from the CLI), and is dominated by Python's import
cost rather than row count. The UI copy says "about a minute" and explains why a
bigger file is barely slower.

Known gap: uploaded projects have no `explainability/coefficient_interpretation.json`,
so the "What drives the prediction" section is absent for them. It renders for the
two built-ins. Generating it in the auto pipeline is a Python-side change.

---

## 8. Risks and dissent

1. **Not deployable.** Spawning Python ties this to localhost. Correct for a
   workbench; wrong if it ever needs hosting. Flagged, not hidden.
2. **`autorun` duration is unmeasured.** "Usually takes 20–60 seconds" is a
   guess — **UNVERIFIED**. P5 must measure one real run and correct the copy.
3. **Two UIs now.** Streamlit and this app will drift. This build does not
   touch Streamlit; consolidation is a later decision.
4. **The brief's "encrypted and deleted after X days" does not apply.** Files
   stay on the local disk forever. §3.5 says what is true instead. Shipping the
   original wording would be a false security claim.
5. **Sharing is local-only.** A "Share" button that copies a `localhost` URL is
   near-useless. It is included because the brief asks, but it must be honest
   ("Link copied — works on this machine"). Recommend cutting it.
6. **Multiclass results.** `report.run()` handles multiclass, but the results
   layout is specified around one headline metric. P6 should fall back to
   accuracy and note the gap rather than render a broken card.
