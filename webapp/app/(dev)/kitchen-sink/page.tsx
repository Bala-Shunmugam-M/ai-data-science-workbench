'use client'

// Dev-only acceptance page for P3. Density over beauty: every component,
// every state, on one screen, in whichever theme is active.

import {
  Download,
  FileSearch,
  Plus,
  Trash2,
} from 'lucide-react'
import { useState, type ReactNode } from 'react'
import {
  Badge,
  Button,
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
  ChartBar,
  ChartDonut,
  ChartLine,
  DataTable,
  DropZone,
  EmptyState,
  MetricTile,
  ProgressSteps,
  Skeleton,
  ThemeToggle,
  Toast,
  ToastRegion,
  type DataTableColumn,
  type ProgressStep,
} from '@/components/ui'
import { formatBytes, formatMetric, formatNumber, formatPercent } from '@/lib/format'

/* ------------------------------------------------------------------ */
/* sample data                                                         */
/* ------------------------------------------------------------------ */

const DRIVERS = [
  { feature: 'MedInc', importance: 0.512 },
  { feature: 'AveRooms', importance: 0.184 },
  { feature: 'Latitude', importance: 0.121 },
  { feature: 'HouseAge', importance: 0.078 },
  { feature: 'Population', importance: 0.061 },
  { feature: 'AveOccup', importance: 0.044 },
]

const LEARNING_CURVE = Array.from({ length: 12 }, (_, i) => ({
  size: (i + 1) * 1500,
  train: Number((0.94 - i * 0.008 + (i % 3) * 0.004).toFixed(4)),
  validation: Number((0.61 + i * 0.021 - (i % 4) * 0.006).toFixed(4)),
}))

const CLASS_MIX = [
  { label: 'Retained', count: 5163 },
  { label: 'Churned', count: 1869 },
]

const COMPARISON_COLUMNS: DataTableColumn[] = [
  { key: 'rank', label: '#', numeric: true, width: '3rem' },
  { key: 'model', label: 'Model' },
  { key: 'version', label: 'Version' },
  { key: 'r2', label: 'R²', numeric: true },
  { key: 'rmse', label: 'RMSE', numeric: true },
  { key: 'mae', label: 'MAE', numeric: true },
]

const COMPARISON_ROWS = [
  { rank: 1, model: 'GradientBoosting', version: 'v3', r2: 0.8448, rmse: 0.4551, mae: 0.3129 },
  { rank: 2, model: 'RandomForest', version: 'v2', r2: 0.8106, rmse: 0.5028, mae: 0.3344 },
  { rank: 3, model: 'Ridge', version: 'v1', r2: 0.6062, rmse: 0.7256, mae: 0.5332 },
  { rank: 4, model: 'DecisionTree', version: 'v1', r2: 0.5983, rmse: 0.7331, mae: 0.4712 },
]

const WIDE_COLUMNS: DataTableColumn[] = [
  { key: 'row', label: 'Row', numeric: true },
  ...Array.from({ length: 24 }, (_, i) => ({
    key: `f${i}`,
    label: `feature_${i}`,
    numeric: i % 3 !== 0,
  })),
]

const WIDE_ROWS = Array.from({ length: 12 }, (_, r) => {
  const row: Record<string, string | number> = { row: r + 1 }
  for (let i = 0; i < 24; i += 1) {
    row[`f${i}`] = i % 3 === 0 ? `cat_${(r + i) % 5}` : Number(((r + 1) * (i + 1) * 0.137).toFixed(3))
  }
  return row
})

const STEP_LABELS = [
  'Uploading',
  'Scanning document',
  'Extracting data',
  'Running analytics',
  'Generating insights',
]

function steps(
  states: ProgressStep['state'][],
  note?: string,
): ProgressStep[] {
  return STEP_LABELS.map((label, i) => ({
    id: String(i),
    label,
    state: states[i],
    note: states[i] === 'failed' ? note : undefined,
  }))
}

/* ------------------------------------------------------------------ */
/* layout helpers                                                      */
/* ------------------------------------------------------------------ */

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-3 border-t border-border pt-6">
      <h2 className="text-h2 text-text">{title}</h2>
      {children}
    </section>
  )
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="w-28 shrink-0 text-label uppercase tracking-wide text-text-tertiary">
        {label}
      </span>
      {children}
    </div>
  )
}

/* ------------------------------------------------------------------ */

export default function KitchenSinkPage() {
  const [toasts, setToasts] = useState<
    { id: number; tone: 'neutral' | 'good' | 'warn' | 'bad'; title: string; body: string }[]
  >([
    { id: 1, tone: 'good', title: 'Analysis complete', body: 'california-housing finished in 38s.' },
    { id: 2, tone: 'warn', title: 'Column dropped', body: '“id” looked like an identifier and was excluded.' },
  ])
  const [nextToast, setNextToast] = useState(3)
  const [droppedFile, setDroppedFile] = useState<string | null>(null)

  return (
    <main className="mx-auto flex max-w-content flex-col gap-8 p-4 md:p-8">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-h1 text-text">Kitchen sink</h1>
          <p className="text-caption text-text-secondary">
            P3 component library — every component, every state.
          </p>
        </div>
        <ThemeToggle />
      </header>

      {/* ---------------- Buttons ---------------- */}
      <Section title="Button">
        {(['primary', 'secondary', 'ghost', 'destructive'] as const).map(
          (variant) => (
            <div key={variant} className="flex flex-col gap-2">
              <Row label={variant}>
                {(['sm', 'md', 'lg'] as const).map((size) => (
                  <Button key={size} variant={variant} size={size}>
                    {size}
                  </Button>
                ))}
                <Button variant={variant} icon={<Plus />}>
                  With icon
                </Button>
                <Button variant={variant} disabled>
                  Disabled
                </Button>
                <Button variant={variant} loading>
                  Loading
                </Button>
              </Row>
            </div>
          ),
        )}
      </Section>

      {/* ---------------- Badges ---------------- */}
      <Section title="Badge">
        <Row label="sm">
          {(['neutral', 'good', 'warn', 'bad'] as const).map((tone) => (
            <Badge key={tone} tone={tone}>
              {tone}
            </Badge>
          ))}
        </Row>
        <Row label="md">
          {(['neutral', 'good', 'warn', 'bad'] as const).map((tone) => (
            <Badge key={tone} tone={tone} size="md">
              {tone}
            </Badge>
          ))}
        </Row>
      </Section>

      {/* ---------------- MetricTile ---------------- */}
      <Section title="MetricTile">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricTile
            label="R² score"
            value={formatMetric(0.8448)}
            tone="good"
            hint="Explains 84% of price variance"
            trend={{ direction: 'up', label: '+0.04 vs Ridge' }}
          />
          <MetricTile
            label="Missing cells"
            value={formatPercent(0.073)}
            tone="warn"
            hint="Imputed with column medians"
          />
          <MetricTile
            label="Failed folds"
            value="2 of 5"
            tone="bad"
            trend={{ direction: 'down', label: 'worse than last run' }}
          />
          <MetricTile
            label="Rows analysed"
            value={formatNumber(20640)}
            hint="8 features used"
          />
        </div>
      </Section>

      {/* ---------------- Card ---------------- */}
      <Section title="Card">
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle>Default card</CardTitle>
              <CardDescription>padded, non-interactive</CardDescription>
            </CardHeader>
            <p className="text-body text-text-secondary">
              Surface background, 12px radius, sm shadow in light and a 1px
              border in dark.
            </p>
          </Card>
          <Card interactive tabIndex={0}>
            <CardHeader>
              <CardTitle>Interactive card</CardTitle>
              <CardDescription>hover, focus-visible, active</CardDescription>
            </CardHeader>
            <p className="text-body text-text-secondary">
              Tab to me — the focus ring should be plainly visible.
            </p>
          </Card>
          <Card padded={false}>
            <div className="border-b border-border p-4 text-label uppercase tracking-wide text-text-tertiary">
              padded={'{false}'}
            </div>
            <p className="p-4 text-body text-text-secondary">
              For cards that own their own internal spacing.
            </p>
          </Card>
        </div>
      </Section>

      {/* ---------------- Skeleton ---------------- */}
      <Section title="Skeleton">
        <div className="grid gap-4 md:grid-cols-3">
          <Card className="flex flex-col gap-3">
            <Skeleton w="40%" h={12} />
            <Skeleton w="70%" h={40} />
            <Skeleton w="55%" h={12} />
          </Card>
          <Card className="flex flex-col gap-2">
            {[100, 92, 96, 80, 88].map((w, i) => (
              <Skeleton key={i} w={`${w}%`} h={16} radius="sm" />
            ))}
          </Card>
          <Card className="flex items-center gap-3">
            <Skeleton w={48} h={48} radius="full" />
            <div className="flex flex-1 flex-col gap-2">
              <Skeleton w="60%" h={14} />
              <Skeleton w="90%" h={14} />
            </div>
          </Card>
        </div>
      </Section>

      {/* ---------------- ProgressSteps ---------------- */}
      <Section title="ProgressSteps">
        {(
          [
            ['all pending', steps(['pending', 'pending', 'pending', 'pending', 'pending'])],
            ['in progress', steps(['done', 'done', 'active', 'pending', 'pending'])],
            ['all done', steps(['done', 'done', 'done', 'done', 'done'])],
            [
              'failed on step 4',
              steps(
                ['done', 'done', 'done', 'failed', 'pending'],
                'training_pipeline.run() raised ValueError: target has 1 unique value',
              ),
            ],
          ] as const
        ).map(([label, value]) => (
          <Card key={label} className="flex flex-col gap-4">
            <span className="text-label uppercase tracking-wide text-text-tertiary">
              {label}
            </span>
            <ProgressSteps steps={value} activeId="2" />
          </Card>
        ))}

        {/*
          Regression case. Five horizontal steps inside a 420px column gave
          each label ~84px, so "Scanning document" wrapped into its neighbour
          and the labels visually collided — the bug reported from the real
          processing screen. The container width is invisible to a media
          query, so the caller passes orientation explicitly.
        */}
        <Card className="flex flex-col gap-4">
          <span className="text-label uppercase tracking-wide text-text-tertiary">
            narrow column (420px) — vertical, as the processing view uses it
          </span>
          <div className="w-[420px] max-w-full rounded-md border border-dashed border-border-strong p-4">
            <ProgressSteps
              orientation="vertical"
              activeId="3"
              steps={steps(
                ['done', 'done', 'failed', 'pending', 'pending'],
                "Cannot split on 'label': 1 of 3 classes have fewer than 7 rows - 'rare' (1). A stratified 70/15/15 split needs at least 7 rows per class.",
              )}
            />
          </div>
        </Card>
      </Section>

      {/* ---------------- DataTable ---------------- */}
      <Section title="DataTable">
        <div className="flex flex-col gap-4">
          <div>
            <p className="mb-2 text-caption text-text-secondary">
              6 columns, zebra, sticky header, capped height
            </p>
            <DataTable
              columns={COMPARISON_COLUMNS}
              rows={COMPARISON_ROWS}
              zebra
              maxHeight={200}
              caption="Model comparison"
            />
          </div>
          <div>
            <p className="mb-2 text-caption text-text-secondary">
              25 columns — scrolls inside its own container, never the body
            </p>
            <DataTable
              columns={WIDE_COLUMNS}
              rows={WIDE_ROWS}
              maxHeight={240}
              caption="Wide extracted table"
            />
          </div>
          <div>
            <p className="mb-2 text-caption text-text-secondary">empty</p>
            <DataTable
              columns={COMPARISON_COLUMNS}
              rows={[]}
              emptyMessage="No models have been trained for this dataset yet."
            />
          </div>
        </div>
      </Section>

      {/* ---------------- Charts ---------------- */}
      <Section title="Charts">
        <div className="grid gap-4 lg:grid-cols-3">
          <Card>
            <ChartBar
              data={DRIVERS}
              xKey="feature"
              series={[{ key: 'importance', label: 'Importance' }]}
              caption="MedInc drives the prediction more than the next three features combined."
            />
          </Card>
          <Card>
            <ChartLine
              data={LEARNING_CURVE}
              xKey="size"
              series={[
                { key: 'train', label: 'Train R²' },
                { key: 'validation', label: 'Validation R²' },
              ]}
              caption="Validation score is still climbing at 18,000 rows — more data would help."
            />
          </Card>
          <Card>
            <ChartDonut
              data={CLASS_MIX}
              xKey="label"
              series={[{ key: 'count', label: 'Customers' }]}
              caption="About one customer in four churned, so accuracy alone would be misleading."
            />
          </Card>
        </div>
      </Section>

      {/* ---------------- EmptyState ---------------- */}
      <Section title="EmptyState">
        <EmptyState
          icon={<FileSearch />}
          title="No datasets yet"
          body="Upload a CSV and the workbench will profile it, pick a target and train a model against it."
          action={<Button icon={<Plus />}>Upload your first dataset</Button>}
        />
      </Section>

      {/* ---------------- DropZone ---------------- */}
      <Section title="DropZone">
        <div className="grid gap-6 lg:grid-cols-2">
          <div>
            <p className="mb-2 text-caption text-text-secondary">
              live — drag a file, click, or tab to it and press Enter
              {droppedFile ? ` · accepted: ${droppedFile}` : ''}
            </p>
            <DropZone
              accept=".csv"
              maxBytes={50 * 1024 * 1024}
              onFile={(f) => setDroppedFile(`${f.name} (${formatBytes(f.size)})`)}
              sampleFiles={[
                { name: 'california-housing.csv', hint: '20,640 rows' },
                { name: 'telco-churn.csv', hint: '7,032 rows' },
                { name: 'wine-quality.csv', hint: '1,599 rows' },
              ]}
              onSampleSelect={(id) => setDroppedFile(`sample: ${id}`)}
            />
          </div>
          <div>
            <p className="mb-2 text-caption text-text-secondary">
              rejected file · disabled
            </p>
            <DropZone
              accept=".csv"
              maxBytes={50 * 1024 * 1024}
              onFile={() => {}}
              error="“quarterly-report.xlsx” isn’t an accepted file type. This upload takes CSV files up to 50 MB."
            />
            <div className="mt-4">
              <DropZone accept=".csv" onFile={() => {}} disabled />
            </div>
          </div>
        </div>
      </Section>

      {/* ---------------- Toast ---------------- */}
      <Section title="Toast">
        <p className="text-caption text-text-secondary">
          Rendered inline here so all four tones are visible at once; the live
          stack is bottom-right and auto-dismisses (hover to pause).
        </p>
        <div className="flex flex-col gap-2">
          <Toast
            tone="neutral"
            title="Workspace opened"
            body="Reading artifacts from workspaces/california-housing."
            duration={0}
          />
          <Toast
            tone="good"
            title="Report generated"
            body="auto_report.md written."
            duration={0}
          />
          <Toast
            tone="warn"
            title="Small dataset"
            body="1,599 rows — cross-validated scores will be noisy."
            duration={0}
          />
          <Toast
            tone="bad"
            title="Analysis failed"
            body="training_pipeline.run() exited with code 1."
            duration={0}
            onDismiss={() => {}}
          />
        </div>
        <Row label="live stack">
          <Button
            variant="secondary"
            icon={<Download />}
            onClick={() => {
              setToasts((t) => [
                ...t,
                {
                  id: nextToast,
                  tone: 'good',
                  title: `Toast #${nextToast}`,
                  body: 'Auto-dismisses in 5s. Hover to pause.',
                },
              ])
              setNextToast((n) => n + 1)
            }}
          >
            Push a toast
          </Button>
          <Button
            variant="ghost"
            icon={<Trash2 />}
            onClick={() => setToasts([])}
          >
            Clear
          </Button>
        </Row>
      </Section>

      <ToastRegion>
        {toasts.map((t) => (
          <Toast
            key={t.id}
            tone={t.tone}
            title={t.title}
            body={t.body}
            onDismiss={() =>
              setToasts((all) => all.filter((x) => x.id !== t.id))
            }
          />
        ))}
      </ToastRegion>
    </main>
  )
}
