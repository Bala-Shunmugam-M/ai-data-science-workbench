/**
 * The ten stage bodies.
 *
 * All in one file on purpose: each is the same shape - read some artifacts,
 * render them through the shared blocks - and keeping them adjacent is what
 * stops nine pages drifting into nine different layouts. The moment one grows
 * real logic it should move out.
 *
 * Every one is an async Server Component. They read the disk directly rather
 * than fetching an API route, because there is no client that needs the data.
 */
import { Badge, Card, DataTable, EmptyState, Tabs } from '@/components/ui'
import { DriversChart } from '@/components/results/drivers-chart'
import { DeliverableRow, ReportActions } from '@/components/journey/report-actions'
import { formatBytes, formatDate, formatNumber } from '@/lib/format'
import {
  FactGrid,
  FigureGallery,
  StageSection,
  TableCard,
  TextCard,
} from '@/components/journey/artifact-blocks'
import {
  artifactSize,
  readFigures,
  readJsonArtifact,
  readJsonLines,
  readTable,
  readTablesIn,
  readTextArtifact,
} from '@/lib/artifacts'
import {
  findProject,
  headlineMetricFor,
  metricLabel,
  rawDataFile,
  readCsvPreview,
  readSelection,
  toMetric,
} from '@/lib/workspace'
import { semanticSchema } from '@/lib/schema'
import type { StageId } from '@/lib/stages'

interface StageProps {
  slug: string
}

// ---------------------------------------------------------------------------
// 1. Dashboard
// ---------------------------------------------------------------------------

interface WorkflowEntry {
  last_run?: string
  duration_s?: number
  status?: string
  outputs?: string[]
}

async function DashboardStage({ slug }: StageProps) {
  const selection = await readSelection(slug)
  const workflow = await readJsonArtifact<Record<string, WorkflowEntry>>(
    slug,
    'artifacts/workflow_status.json',
  )
  const headline = headlineMetricFor(selection)

  const stages = Object.entries(workflow ?? {})

  return (
    <div className="flex flex-col gap-10">
      <FactGrid
        facts={[
          {
            label: 'Champion',
            value: selection
              ? `${selection.champion_name ?? '—'} ${selection.champion_version ?? ''}`.trim()
              : 'Not selected',
          },
          { label: headline?.label ?? 'Headline metric', value: headline?.value ?? '—' },
          { label: 'Task', value: selection?.task ?? '—' },
          { label: 'Selection metric', value: selection?.selection_metric ?? '—' },
        ]}
      />

      {selection?.test_metrics && (
        <StageSection title="Test metrics" hint="Measured once, on the untouched test split">
          <FactGrid
            columns={4}
            facts={Object.entries(selection.test_metrics).map(([key, value]) => {
              const metric = toMetric(key, value)
              return { label: metric.label, value: metric.value }
            })}
          />
        </StageSection>
      )}

      {stages.length > 0 && (
        <StageSection title="Pipeline stage status" hint={`${stages.length} stages recorded`}>
          <Card padded={false}>
            <DataTable
              columns={[
                { key: 'stage', label: 'Stage' },
                { key: 'status', label: 'Status' },
                { key: 'last_run', label: 'Last run' },
                { key: 'duration', label: 'Duration', numeric: true },
                { key: 'outputs', label: 'Outputs', numeric: true },
              ]}
              rows={stages.map(([name, entry]) => ({
                stage: name,
                status: (
                  <Badge tone={entry.status === 'ok' ? 'good' : 'bad'}>
                    {entry.status ?? 'unknown'}
                  </Badge>
                ),
                last_run: entry.last_run ? formatDate(entry.last_run) : '—',
                duration: entry.duration_s ? `${entry.duration_s.toFixed(2)}s` : '—',
                outputs: entry.outputs?.length ?? 0,
              }))}
              maxHeight={480}
            />
          </Card>
        </StageSection>
      )}

      {selection?.selection_rationale && (
        <StageSection title="Why this model won">
          <Card>
            <p className="whitespace-pre-line text-body text-text-secondary">
              {selection.selection_rationale}
            </p>
          </Card>
        </StageSection>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 2. Data
// ---------------------------------------------------------------------------

async function DataStage({ slug }: StageProps) {
  const validationStatus = await readTextArtifact(slug, 'results/validation/validation_status.txt')
  const validationSummary = await readTable(slug, 'results/validation/validation_summary.csv')
  const validationIssues = await readTable(slug, 'results/validation/validation_issues.csv', 100)
  const profile = await readTable(slug, 'results/data_profile/data_profile.csv', 60)
  const profileSummary = await readTextArtifact(slug, 'results/data_profile/profile_summary.txt')

  // 25 rows per preview, not 100. Six tables of 100 wide rows rendered ~13 MB
  // of HTML into a page whose job is orientation, not data delivery — the full
  // CSVs are one click away on the Reports stage.
  const PREVIEW = 25

  const descriptor = await findProject(slug)
  const rawFile = descriptor ? await rawDataFile(descriptor) : null
  const raw = rawFile ? await readCsvPreview(rawFile, PREVIEW) : null
  const processed =
    (await readTable(slug, 'data/processed/housing_clean.csv', PREVIEW)) ??
    (await readTable(slug, 'data/processed/dataset_clean.csv', PREVIEW))

  const splits = await Promise.all(
    (['train', 'validation', 'test'] as const).map(async (name) => ({
      name,
      table: await readTable(slug, `data/splits/${name}.csv`, PREVIEW),
    })),
  )
  const engineered = await readTable(slug, 'data/engineered/train.csv', PREVIEW)
  const schema = await semanticSchema()

  const totalRows = splits.reduce((sum, split) => sum + (split.table?.totalRows ?? 0), 0)

  return (
    <div className="flex flex-col gap-10">
      {validationStatus && (
        <Card className="flex flex-wrap items-center gap-3">
          <span className="text-label uppercase tracking-wide text-text-tertiary">
            Validation gate
          </span>
          <Badge tone={/fail/i.test(validationStatus) ? 'bad' : 'good'}>
            {validationStatus.trim().split(/\r?\n/)[0]}
          </Badge>
        </Card>
      )}

      <StageSection title="Dataset size" hint="Row counts across the pipeline">
        <FactGrid
          columns={4}
          facts={[
            {
              label: 'Total rows',
              value: totalRows ? formatNumber(totalRows) : raw ? formatNumber(raw.totalRows) : '—',
              hint: 'train + validation + test',
            },
            ...splits.map((split) => ({
              label: `${split.name} rows`,
              value: split.table ? formatNumber(split.table.totalRows) : '—',
              hint: split.table ? `${split.table.columns.length} columns` : 'not written',
            })),
          ]}
        />
      </StageSection>

      <StageSection
        title="Dataset previews"
        hint={`first ${PREVIEW} rows of each — the full CSVs are downloadable on stage 9`}
      >
        <Tabs
          items={[
            raw && {
              id: 'raw',
              label: 'Raw',
              hint: formatNumber(raw.totalRows),
              content: (
                <TableCard
                  table={{
                    columns: raw.columns,
                    rows: raw.rows,
                    totalRows: raw.totalRows,
                    source: 'data/raw',
                  }}
                  title="Raw dataset"
                />
              ),
            },
            processed && {
              id: 'processed',
              label: 'Processed',
              hint: formatNumber(processed.totalRows),
              content: <TableCard table={processed} title="Processed dataset" />,
            },
            ...splits.map((split) =>
              split.table
                ? {
                    id: split.name,
                    label: `${split.name[0].toUpperCase()}${split.name.slice(1)}`,
                    hint: formatNumber(split.table.totalRows),
                    content: (
                      <TableCard
                        table={split.table}
                        title={`${split.name[0].toUpperCase()}${split.name.slice(1)} split`}
                      />
                    ),
                  }
                : null,
            ),
            engineered && {
              id: 'engineered',
              label: 'Engineered',
              hint: formatNumber(engineered.totalRows),
              content: <TableCard table={engineered} title="Engineered training data" />,
            },
          ].filter((item): item is NonNullable<typeof item> => Boolean(item))}
        />
      </StageSection>

      {schema.length > 0 && (
        <StageSection
          title="Human-approved semantic schema"
          hint={`${schema.length} columns · the contract every stage validates against`}
        >
          <Card padded={false}>
            <DataTable
              columns={[
                { key: 'column', label: 'Column' },
                { key: 'role', label: 'Role' },
                { key: 'semantic_type', label: 'Semantic type' },
                { key: 'expected_storage_type', label: 'Storage' },
                { key: 'nullable', label: 'Nullable' },
                { key: 'minimum', label: 'Min', numeric: true },
                { key: 'maximum', label: 'Max', numeric: true },
                { key: 'whole_number', label: 'Whole' },
                { key: 'preprocessing_group', label: 'Group' },
                { key: 'missing_treatment', label: 'Missing treatment' },
              ]}
              rows={schema.map((row) => ({
                column: row.column,
                role: row.role ?? '—',
                semantic_type: row.semantic_type ?? '—',
                expected_storage_type: row.expected_storage_type ?? '—',
                nullable: row.nullable ? 'yes' : 'no',
                minimum: row.minimum ?? '—',
                maximum: row.maximum ?? '—',
                whole_number: row.whole_number ? 'yes' : 'no',
                preprocessing_group: row.preprocessing_group ?? '—',
                missing_treatment: row.missing_treatment ?? '—',
              }))}
              maxHeight={440}
              zebra
            />
          </Card>
        </StageSection>
      )}

      {profile && <TableCard table={profile} title="Structural data profile" />}
      {profileSummary && <TextCard title="Profile summary" body={profileSummary} />}
      {validationSummary && <TableCard table={validationSummary} title="Validation summary" />}
      {validationIssues ? (
        <TableCard table={validationIssues} title="Validation issues" />
      ) : (
        validationStatus && (
          <p className="text-caption text-text-secondary">No validation issues were recorded.</p>
        )
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 3. EDA
// ---------------------------------------------------------------------------

async function EdaStage({ slug }: StageProps) {
  const understanding = await readTablesIn(slug, 'results/data_understanding')
  const eda = await readTablesIn(slug, 'results/eda')
  const summary = await readTablesIn(slug, 'results/eda_analysis_summary')
  const figures = [
    ...(await readFigures(slug, 'results/eda/figures')),
    ...(await readFigures(slug, 'artifacts/auto_eda')),
  ]
  const conclusion = await readTextArtifact(slug, 'results/eda_analysis_summary/eda_conclusion.txt')
  const edaReport = await readTextArtifact(slug, 'results/eda/eda_report.txt')

  if (!understanding.length && !eda.length && !figures.length) {
    return (
      <EmptyState
        title="No exploratory analysis on disk"
        body="This project has no results/eda or results/data_understanding artifacts yet."
      />
    )
  }

  return (
    <div className="flex flex-col gap-10">
      {figures.length > 0 && (
        <StageSection title="Figure gallery" hint={`${figures.length} figures`}>
          <FigureGallery figures={figures} columns={3} />
        </StageSection>
      )}

      {conclusion && <TextCard title="EDA conclusion" body={conclusion} />}

      {understanding.length > 0 && (
        <StageSection title="Data understanding" hint={`${understanding.length} tables`}>
          <div className="flex flex-col gap-8">
            {understanding.map((table) => (
              <TableCard key={table.source} table={table} maxHeight={320} />
            ))}
          </div>
        </StageSection>
      )}

      {eda.length > 0 && (
        <StageSection title="Exploratory tables" hint={`${eda.length} tables`}>
          <div className="flex flex-col gap-8">
            {eda.map((table) => (
              <TableCard key={table.source} table={table} maxHeight={320} />
            ))}
          </div>
        </StageSection>
      )}

      {summary.length > 0 && (
        <StageSection title="Analytical summary" hint="Screens that fed the feature plan">
          <div className="flex flex-col gap-8">
            {summary.map((table) => (
              <TableCard key={table.source} table={table} maxHeight={320} />
            ))}
          </div>
        </StageSection>
      )}

      {edaReport && <TextCard title="EDA report" body={edaReport} />}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 4. Features
// ---------------------------------------------------------------------------

interface FeatureProposalFile {
  total_proposed?: number
  automatic_count?: number
  features?: {
    feature_id?: string
    feature_name?: string
    feature_group?: string
    feature_subtype?: string
    formula?: string
    business_meaning?: string
    automatic?: boolean
    leakage_risk?: string
    priority?: string
    analyst_decision?: string
  }[]
}

interface FeatureApprovalFile {
  approved_features?: string[]
  approved_count?: number
  approved_predictor_columns?: string[]
}

async function FeaturesStage({ slug }: StageProps) {
  const proposal = await readJsonArtifact<FeatureProposalFile>(
    slug,
    'governance/approvals/feature_proposal.json',
  )
  const approval = await readJsonArtifact<FeatureApprovalFile>(
    slug,
    'governance/approvals/feature_approval.json',
  )
  const plan = await readTable(slug, 'results/feature_engineering/feature_plan.csv', 60)
  const engineered = await readTable(slug, 'data/engineered/train.csv', 10)

  const approvedNames = new Set((approval?.approved_features ?? []).map(String))
  const features = proposal?.features ?? []
  const predictorColumns = approval?.approved_predictor_columns ?? []

  // Groups are the five C17 buckets; showing the split is the point of the
  // governed plan, so count them even when the plan CSV is absent.
  const groups = new Map<string, number>()
  for (const feature of features) {
    const group = feature.feature_group ?? 'unknown'
    groups.set(group, (groups.get(group) ?? 0) + 1)
  }

  return (
    <div className="flex flex-col gap-10">
      <FactGrid
        columns={4}
        facts={[
          { label: 'Total proposed', value: proposal?.total_proposed ?? features.length ?? '—' },
          {
            label: 'Automatic',
            value: proposal?.automatic_count ?? '—',
            hint: 'executed without a decision',
          },
          {
            label: 'Approved',
            value: approval?.approved_count ?? approvedNames.size ?? '—',
          },
          {
            label: 'Engineered columns',
            value: engineered ? engineered.columns.length : '—',
            hint: engineered ? 'in data/engineered/train.csv' : 'not built',
          },
        ]}
      />

      {groups.size > 0 && (
        <StageSection title="By feature group" hint="The five governed buckets">
          <div className="flex flex-wrap gap-3">
            {[...groups.entries()].map(([group, count]) => (
              <Card key={group} className="min-w-[10rem] flex-1">
                <p className="text-label uppercase tracking-wide text-text-tertiary">{group}</p>
                <p className="text-h2 tabular-nums text-text">{count}</p>
              </Card>
            ))}
          </div>
        </StageSection>
      )}

      {features.length > 0 && (
        <StageSection
          title="Feature plan"
          hint={`${features.length} proposals, with their approval status`}
        >
          <Card padded={false}>
            <DataTable
              columns={[
                { key: 'id', label: 'ID' },
                { key: 'name', label: 'Feature' },
                { key: 'group', label: 'Group' },
                { key: 'subtype', label: 'Subtype' },
                { key: 'formula', label: 'Formula' },
                { key: 'meaning', label: 'Business meaning', wrap: true },
                { key: 'automatic', label: 'Automatic' },
                { key: 'leakage', label: 'Leakage risk' },
                { key: 'priority', label: 'Priority' },
                { key: 'approved', label: 'Status' },
              ]}
              rows={features.map((feature) => {
                const name = String(feature.feature_name ?? '')
                const isApproved = approvedNames.has(name)
                return {
                  id: feature.feature_id ?? '—',
                  name,
                  group: feature.feature_group ?? '—',
                  subtype: feature.feature_subtype ?? '—',
                  formula: feature.formula ?? '—',
                  meaning: feature.business_meaning ?? '—',
                  automatic: feature.automatic ? 'yes' : 'no',
                  leakage: feature.leakage_risk ?? '—',
                  priority: feature.priority ?? '—',
                  approved: (
                    <Badge tone={isApproved ? 'good' : 'neutral'}>
                      {isApproved ? 'Approved' : 'Not approved'}
                    </Badge>
                  ),
                }
              })}
              maxHeight={560}
              zebra
            />
          </Card>
        </StageSection>
      )}

      {predictorColumns.length > 0 && (
        <StageSection
          title="Approved predictor columns"
          hint={`${predictorColumns.length} columns feed the modelling matrix`}
        >
          <Card className="flex flex-wrap gap-2">
            {predictorColumns.map((column) => (
              <Badge key={column} tone="good">
                {column}
              </Badge>
            ))}
          </Card>
        </StageSection>
      )}

      {plan && <TableCard table={plan} title="Feature plan (CSV artifact)" maxHeight={420} />}
      {engineered && <TableCard table={engineered} title="Engineered training data (preview)" />}

      {!features.length && !plan && (
        <EmptyState
          title="No feature plan on disk"
          body="Neither governance/approvals/feature_proposal.json nor results/feature_engineering/feature_plan.csv exists for this project."
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 5. Modeling
// ---------------------------------------------------------------------------

interface RegistryModel {
  name?: string
  version?: string
  params?: Record<string, unknown>
  validation_metrics?: Record<string, number>
  registered_at?: string
}

interface ModelProposalFile {
  modelling_scope?: string
  rationale?: string
  recommended_models?: string[]
  validation_strategy?: Record<string, unknown>
  candidate_catalog?: Record<string, unknown>[]
  multicollinearity_flags?: Record<string, unknown>[]
}

/** Generic object[] -> DataTable, for artifacts whose columns are not fixed. */
function RecordTable({
  records,
  maxHeight = 420,
  maxColumns = 10,
}: {
  records: Record<string, unknown>[]
  maxHeight?: number
  maxColumns?: number
}) {
  if (!records.length) return null
  // Union of keys, not just the first record's: an experiment log can gain
  // fields between runs and dropping them silently would hide real history.
  const keys: string[] = []
  for (const record of records) {
    for (const key of Object.keys(record)) if (!keys.includes(key)) keys.push(key)
  }
  const shown = keys.slice(0, maxColumns)

  return (
    <Card padded={false}>
      <DataTable
        columns={shown.map((key) => ({ key, label: key.replace(/_/g, ' ') }))}
        rows={records.map((record) =>
          Object.fromEntries(
            shown.map((key) => {
              const value = record[key]
              if (value === null || value === undefined) return [key, '—']
              if (typeof value === 'object') return [key, JSON.stringify(value)]
              if (typeof value === 'number') {
                return [
                  key,
                  Number.isInteger(value)
                    ? formatNumber(value)
                    : Math.abs(value) >= 1000
                      ? formatNumber(Math.round(value))
                      : value.toFixed(4),
                ]
              }
              return [key, String(value)]
            }),
          ),
        )}
        maxHeight={maxHeight}
        zebra
      />
    </Card>
  )
}

async function ModelingStage({ slug }: StageProps) {
  const registry = await readJsonArtifact<{
    models?: RegistryModel[]
    champion?: { name?: string; version?: string }
  }>(slug, 'models/model_registry.json')
  const proposal = await readJsonArtifact<ModelProposalFile>(
    slug,
    'governance/approvals/model_proposal.json',
  )
  const approval = await readJsonArtifact<{ approved_models?: string[] }>(
    slug,
    'governance/approvals/model_approval.json',
  )
  const experiments = await readJsonLines<Record<string, unknown>>(
    slug,
    'artifacts/experiments/experiments.jsonl',
  )

  const models = registry?.models ?? []
  const champion = registry?.champion ?? {}
  const approvedModels = new Set((approval?.approved_models ?? []).map(String))
  const recommended = proposal?.recommended_models ?? [...approvedModels]

  const metricKeys = models.length ? Object.keys(models[0].validation_metrics ?? {}) : []
  const sorted = [...models].sort((a, b) => {
    const key = metricKeys.includes('rmse') ? 'rmse' : metricKeys[0]
    if (!key) return 0
    const av = a.validation_metrics?.[key] ?? Number.POSITIVE_INFINITY
    const bv = b.validation_metrics?.[key] ?? Number.POSITIVE_INFINITY
    // rmse/mae: lower is better. roc_auc/r2: higher. Sort accordingly.
    return ['rmse', 'mae', 'mape'].includes(key) ? av - bv : bv - av
  })

  return (
    <div className="flex flex-col gap-10">
      <FactGrid
        columns={4}
        facts={[
          { label: 'Recommended', value: recommended.length || '—' },
          { label: 'Approved to train', value: approvedModels.size || '—' },
          { label: 'Registered versions', value: models.length || '—' },
          { label: 'Experiments logged', value: experiments.length || '—' },
        ]}
      />

      {proposal && (
        <StageSection title="Model proposal" hint="What was recommended, and why">
          <Card className="flex flex-col gap-4">
            {proposal.modelling_scope && (
              <div>
                <p className="text-label uppercase tracking-wide text-text-tertiary">Scope</p>
                <p className="mt-1 text-body text-text">{proposal.modelling_scope}</p>
              </div>
            )}
            {proposal.rationale && (
              <div>
                <p className="text-label uppercase tracking-wide text-text-tertiary">Rationale</p>
                <p className="mt-1 text-body text-text-secondary">{proposal.rationale}</p>
              </div>
            )}
            {proposal.validation_strategy && (
              <div>
                <p className="text-label uppercase tracking-wide text-text-tertiary">
                  Validation strategy
                </p>
                <dl className="mt-2 grid gap-2 sm:grid-cols-2">
                  {Object.entries(proposal.validation_strategy).map(([key, value]) => (
                    <div key={key} className="rounded-md border border-border bg-bg-subtle p-3">
                      <dt className="text-label uppercase tracking-wide text-text-tertiary">
                        {key.replace(/_/g, ' ')}
                      </dt>
                      <dd className="mt-0.5 text-caption text-text">
                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
            )}
          </Card>
        </StageSection>
      )}

      {recommended.length > 0 && (
        <StageSection title="Approval status" hint="Nothing outside the approved set may train">
          <Card padded={false}>
            <DataTable
              columns={[
                { key: 'model', label: 'Model' },
                { key: 'status', label: 'Status' },
              ]}
              rows={recommended.map((name) => ({
                model: name,
                status: approvedModels.has(name) ? (
                  <Badge tone="good">Approved</Badge>
                ) : (
                  <Badge tone="neutral">Not approved</Badge>
                ),
              }))}
            />
          </Card>
        </StageSection>
      )}

      {(proposal?.candidate_catalog?.length ?? 0) > 0 && (
        <StageSection title="Candidate catalogue">
          <RecordTable records={proposal!.candidate_catalog!} />
        </StageSection>
      )}

      {(proposal?.multicollinearity_flags?.length ?? 0) > 0 && (
        <StageSection
          title="Multicollinearity flags"
          hint={`${proposal!.multicollinearity_flags!.length} flagged pairs`}
        >
          <RecordTable records={proposal!.multicollinearity_flags!} />
        </StageSection>
      )}

      {models.length > 0 ? (
        <StageSection
          title="Model registry"
          hint={`${models.length} versions, best validation score first`}
        >
          <Card padded={false}>
            <DataTable
              columns={[
                { key: 'name', label: 'Model' },
                { key: 'version', label: 'Version' },
                { key: 'champion', label: 'Champion' },
                ...metricKeys.map((key) => ({
                  key,
                  label: metricLabel(key).replace(/^Test /, ''),
                  numeric: true,
                })),
                { key: 'params', label: 'Params' },
                { key: 'registered', label: 'Registered' },
              ]}
              rows={sorted.map((model) => ({
                name: model.name ?? '—',
                version: model.version ?? '—',
                champion:
                  model.name === champion.name && model.version === champion.version ? (
                    <Badge tone="good">Champion</Badge>
                  ) : (
                    ''
                  ),
                ...Object.fromEntries(
                  metricKeys.map((key) => {
                    const value = model.validation_metrics?.[key]
                    return [
                      key,
                      typeof value === 'number'
                        ? value >= 1000
                          ? formatNumber(Math.round(value))
                          : value.toFixed(4)
                        : '—',
                    ]
                  }),
                ),
                params:
                  model.params && Object.keys(model.params).length
                    ? JSON.stringify(model.params)
                    : '—',
                registered: model.registered_at ? formatDate(model.registered_at) : '—',
              }))}
              maxHeight={440}
              zebra
            />
          </Card>
        </StageSection>
      ) : (
        <EmptyState title="No model registry" body="No models/model_registry.json on disk." />
      )}

      {experiments.length > 0 && (
        <StageSection title="Experiment log" hint={`${experiments.length} runs, append-only`}>
          <RecordTable records={experiments.slice(-100)} maxHeight={440} maxColumns={12} />
        </StageSection>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 6. Evaluation
// ---------------------------------------------------------------------------

async function EvaluationStage({ slug }: StageProps) {
  const comparison = await readTable(slug, 'artifacts/evaluation/model_comparison.csv', 50)
  const report = await readJsonArtifact<Record<string, unknown>>(
    slug,
    'artifacts/evaluation/model_evaluation_report.json',
  )
  const figures = await readFigures(slug, 'artifacts/evaluation/figures')
  const selection = await readSelection(slug)

  return (
    <div className="flex flex-col gap-10">
      {selection?.test_metrics && (
        <StageSection title="Champion card" hint="The single measurement taken on the test split">
          <Card className="flex flex-col gap-5">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-h3 text-text">
                {selection.champion_name} {selection.champion_version}
              </h3>
              <Badge tone="good">Champion</Badge>
            </div>
            <FactGrid
              columns={4}
              facts={Object.entries(selection.test_metrics).map(([key, value]) => {
                const metric = toMetric(key, value)
                return { label: metric.label, value: metric.value }
              })}
            />
          </Card>
        </StageSection>
      )}

      {comparison ? (
        <TableCard table={comparison} title="Model comparison (validation split)" />
      ) : (
        <EmptyState
          title="No comparison artifact"
          body="artifacts/evaluation/model_comparison.csv is not on disk for this project."
        />
      )}

      {figures.length > 0 && (
        <StageSection title="Residual diagnostics" hint={`${figures.length} figures`}>
          <FigureGallery figures={figures} />
        </StageSection>
      )}

      {report && (
        <StageSection title="Evaluation report">
          <Card padded={false}>
            <pre className="max-h-[420px] overflow-auto p-5 text-caption leading-relaxed text-text-secondary">
              <code>{JSON.stringify(report, null, 2)}</code>
            </pre>
          </Card>
        </StageSection>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 7. Explainability
// ---------------------------------------------------------------------------

interface CoefficientRecord {
  rank: number
  feature: string
  coefficient: number
  meaning: string
}

async function ExplainabilityStage({ slug }: StageProps) {
  const coefficients = await readTable(
    slug,
    'artifacts/explainability/coefficient_interpretation.csv',
    100,
  )
  const structured = await readJsonArtifact<{
    coefficients?: CoefficientRecord[]
    kind?: 'coefficient' | 'importance'
    champion?: string
  }>(slug, 'artifacts/explainability/coefficient_interpretation.json')
  // Housing's own artifact predates the `kind` field; it is always signed
  // coefficients, so treating a missing value as "coefficient" is correct.
  const kind = structured?.kind ?? 'coefficient'
  const figures = await readFigures(slug, 'artifacts/explainability')
  const briefing = await readTextArtifact(slug, 'artifacts/reports/executive_briefing.md')

  const drivers = (structured?.coefficients ?? []).slice(0, 12).map((entry) => ({
    rank: entry.rank,
    feature: entry.feature,
    coefficient: entry.coefficient,
    direction: (entry.coefficient >= 0 ? 'positive' : 'negative') as 'positive' | 'negative',
    meaning: entry.meaning,
  }))

  if (!coefficients && !figures.length && !briefing) {
    return (
      <EmptyState
        title="No explainability artifacts"
        body="This project has no artifacts/explainability output. The auto pipeline used for uploaded datasets does not generate coefficient interpretations yet."
      />
    )
  }

  return (
    <div className="flex flex-col gap-10">
      {drivers.length > 0 && (
        <StageSection
          title="Top drivers"
          hint={
            kind === 'importance'
              ? 'Split importance — magnitudes only, no direction'
              : `Standardized coefficients${structured?.champion ? ` · ${structured.champion}` : ''}`
          }
        >
          <Card>
            <DriversChart drivers={drivers} kind={kind} />
          </Card>
        </StageSection>
      )}
      {figures.length > 0 && (
        <StageSection title="Explainability figures">
          <FigureGallery figures={figures} columns={1} />
        </StageSection>
      )}
      {coefficients && (
        <TableCard table={coefficients} title="Standardized coefficients" maxHeight={520} />
      )}
      {briefing && <TextCard title="Executive briefing" body={briefing} />}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 8. Governance
// ---------------------------------------------------------------------------

interface AuditEvent {
  timestamp?: string
  actor?: string
  event_type?: string
  payload?: Record<string, unknown>
}

interface LineageNode {
  stage?: string
  timestamp?: string
  script?: string
  inputs?: unknown[]
  outputs?: unknown[]
  params?: Record<string, unknown>
}

async function GovernanceStage({ slug }: StageProps) {
  const audit = await readJsonLines<AuditEvent>(slug, 'governance/audit/audit_log.jsonl')
  const lineage = await readJsonArtifact<{ nodes?: LineageNode[]; edges?: unknown[] }>(
    slug,
    'governance/lineage/lineage.json',
  )
  const decisions = await readJsonArtifact<{
    decisions?: Record<string, { decision?: string; rationale?: string; key?: string }>
  }>(slug, 'governance/decisions.json')

  const decisionList = Object.values(decisions?.decisions ?? {})

  return (
    <div className="flex flex-col gap-10">
      <FactGrid
        columns={3}
        facts={[
          { label: 'Audit events', value: audit.length || '—' },
          { label: 'Lineage nodes', value: lineage?.nodes?.length ?? '—' },
          { label: 'Standing decisions', value: decisionList.length || '—' },
        ]}
      />

      {decisionList.length > 0 && (
        <StageSection title="Standing decisions" hint="The load-bearing choices, with reasons">
          <div className="flex flex-col gap-4">
            {decisionList.map((decision, index) => (
              <Card key={decision.key ?? index} className="flex flex-col gap-2">
                <h3 className="text-h3 text-text">{decision.decision}</h3>
                {decision.rationale && (
                  <p className="text-body text-text-secondary">{decision.rationale}</p>
                )}
              </Card>
            ))}
          </div>
        </StageSection>
      )}

      {(lineage?.nodes?.length ?? 0) > 0 && (
        <StageSection
          title="Data and model lineage"
          hint={lineage!.nodes!.map((node) => node.stage ?? '?').join(' → ')}
        >
          <div className="flex flex-col gap-4">
            {lineage!.nodes!.map((node, index) => (
              <Card key={`${node.stage}-${index}`} className="flex flex-col gap-3">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h3 className="text-h3 text-text">{node.stage ?? `Node ${index + 1}`}</h3>
                  <p className="text-caption text-text-tertiary">
                    {node.timestamp ? formatDate(node.timestamp) : ''}
                  </p>
                </div>
                {node.script && (
                  <code className="block truncate text-label text-text-tertiary">
                    {node.script}
                  </code>
                )}
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <p className="text-label uppercase tracking-wide text-text-tertiary">
                      Inputs ({node.inputs?.length ?? 0})
                    </p>
                    <ul className="mt-1 flex flex-col gap-1">
                      {(node.inputs ?? []).map((input, i) => (
                        <li key={i} className="break-all text-caption text-text-secondary">
                          {typeof input === 'string' ? input : JSON.stringify(input)}
                        </li>
                      ))}
                      {!node.inputs?.length && (
                        <li className="text-caption text-text-tertiary">none recorded</li>
                      )}
                    </ul>
                  </div>
                  <div>
                    <p className="text-label uppercase tracking-wide text-text-tertiary">
                      Outputs ({node.outputs?.length ?? 0})
                    </p>
                    <ul className="mt-1 flex flex-col gap-1">
                      {(node.outputs ?? []).map((output, i) => (
                        <li key={i} className="break-all text-caption text-text-secondary">
                          {typeof output === 'string' ? output : JSON.stringify(output)}
                        </li>
                      ))}
                      {!node.outputs?.length && (
                        <li className="text-caption text-text-tertiary">none recorded</li>
                      )}
                    </ul>
                  </div>
                </div>
                {node.params && Object.keys(node.params).length > 0 && (
                  <div>
                    <p className="text-label uppercase tracking-wide text-text-tertiary">Params</p>
                    <pre className="mt-1 overflow-x-auto rounded-md border border-border bg-bg-subtle p-3 text-label text-text-secondary">
                      <code>{JSON.stringify(node.params, null, 2)}</code>
                    </pre>
                  </div>
                )}
              </Card>
            ))}
          </div>
        </StageSection>
      )}

      {audit.length > 0 ? (
        <StageSection title="Audit trail" hint={`${audit.length} events, newest first`}>
          <Card padded={false}>
            <DataTable
              columns={[
                { key: 'timestamp', label: 'When' },
                { key: 'actor', label: 'Actor' },
                { key: 'event', label: 'Event' },
                { key: 'payload', label: 'Payload', wrap: true },
              ]}
              rows={[...audit]
                .reverse()
                .slice(0, 200)
                .map((event, index) => ({
                  timestamp: event.timestamp ? formatDate(event.timestamp) : `#${index + 1}`,
                  actor: event.actor ?? '—',
                  event: event.event_type ?? '—',
                  payload: JSON.stringify(event.payload ?? {}),
                }))}
              maxHeight={560}
              zebra
            />
          </Card>
        </StageSection>
      ) : (
        <EmptyState
          title="No audit trail"
          body="governance/audit/audit_log.jsonl is not on disk for this project."
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 9. Reports
// ---------------------------------------------------------------------------

async function ReportsStage({ slug }: StageProps) {
  const artifact = (relative: string) => `/api/artifact/${slug}/${relative}`

  // Size, not contents: the report is megabytes and is rendered by an iframe
  // that fetches it separately. Reading it here would double the page weight
  // for the sake of one label.
  const htmlSize = await artifactSize(slug, 'artifacts/reports/project_report.html')
  const briefing = await readTextArtifact(slug, 'artifacts/reports/executive_briefing.md')
  const auto = await readTextArtifact(slug, 'artifacts/reports/auto_report.md')
  const selectionSize = await artifactSize(slug, 'artifacts/final_model_selection.json')
  const evaluationSize = await artifactSize(
    slug,
    'artifacts/evaluation/model_evaluation_report.json',
  )
  const html = htmlSize !== null

  if (!html && !briefing && !auto) {
    return (
      <EmptyState
        title="No reports yet"
        body="Nothing has been written to artifacts/reports for this project."
      />
    )
  }

  const pdfSize = await artifactSize(slug, 'artifacts/reports/project_report.pdf')

  const deliverables: {
    label: string
    relative: string
    fileName: string
    available: boolean
    /** Overrides the artifact route for files served by a dedicated endpoint. */
    href?: string
  }[] = [
    {
      // Rendered on demand by /api/report/[slug]/pdf, so it is available
      // whenever the HTML report is - reporting "not generated yet" for a file
      // one click would create read as a missing feature rather than a lazy one.
      label: `Project report (PDF)${pdfSize ? ` — ${formatBytes(pdfSize)}` : ' — rendered on download'}`,
      relative: 'artifacts/reports/project_report.pdf',
      fileName: 'project_report.pdf',
      available: html,
      href: `/api/report/${slug}/pdf`,
    },
    {
      label: 'Executive briefing (Markdown)',
      relative: 'artifacts/reports/executive_briefing.md',
      fileName: 'executive_briefing.md',
      available: Boolean(briefing),
    },
    {
      label: 'Automated report (Markdown)',
      relative: 'artifacts/reports/auto_report.md',
      fileName: 'auto_report.md',
      available: Boolean(auto),
    },
    {
      label: 'Final model selection (JSON)',
      relative: 'artifacts/final_model_selection.json',
      fileName: 'final_model_selection.json',
      available: selectionSize !== null,
    },
    {
      label: 'Model evaluation report (JSON)',
      relative: 'artifacts/evaluation/model_evaluation_report.json',
      fileName: 'model_evaluation_report.json',
      available: evaluationSize !== null,
    },
  ]

  return (
    <div className="flex flex-col gap-10">
      {html ? (
        <StageSection
          title="Project report"
          hint={`${formatBytes(htmlSize ?? 0)} · self-contained HTML`}
        >
          <Card className="flex flex-col gap-5">
            <ReportActions
              src={artifact('artifacts/reports/project_report.html')}
              pdfHref={`/api/report/${slug}/pdf`}
              fileName={`${slug}-project-report.html`}
              pdfFileName={`${slug}-project-report.pdf`}
            />
            <code className="block truncate text-label text-text-tertiary">
              artifacts/reports/project_report.html
            </code>
            {/* Sandboxed: the report is generated locally, but it is still a
                whole third-party document and needs no access to this app. */}
            <iframe
              title="Project report preview"
              src={artifact('artifacts/reports/project_report.html')}
              sandbox=""
              className="h-[820px] w-full rounded-md border border-border bg-white"
            />
          </Card>
        </StageSection>
      ) : (
        <EmptyState
          title="No project report"
          body="artifacts/reports/project_report.html has not been generated for this project. Uploaded datasets produce a Markdown auto-report instead."
        />
      )}

      <StageSection title="Other deliverables" hint="Every file this stage produced">
        <Card padded={false}>
          <div className="divide-y divide-border px-5">
            {deliverables.map((item) => (
              <DeliverableRow
                key={item.relative}
                label={item.label}
                path={item.relative}
                href={item.href ?? artifact(item.relative)}
                available={item.available}
                fileName={item.fileName}
              />
            ))}
          </div>
        </Card>
      </StageSection>

      {auto && <TextCard title="Automated report" body={auto} />}
      {briefing && <TextCard title="Executive briefing" body={briefing} />}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 9. Trust
// ---------------------------------------------------------------------------

interface FairnessGaps {
  n_groups_compared?: number
  n_groups_compared_per_metric?: Record<string, number>
  selection_rate_ratio?: number | null
  selection_amplification?: number | null
  [gap: string]: unknown
}

interface FairnessAudit {
  column: string
  metric_set: 'binary_classification' | 'multiclass_classification' | 'regression'
  n_groups: number
  gaps: FairnessGaps
  excluded_from_gaps?: { group: string; n: number; reason: string }[]
}

interface FairnessArtifact {
  champion?: string
  split?: string
  n_rows?: number
  min_group_size?: number
  max_levels?: number
  columns_audited?: string[]
  audits?: FairnessAudit[]
}

interface ImportanceArtifact {
  measured_against?: string
  n_repeats?: number
  n_features?: number
  n_distinguishable_from_noise?: number
  split?: string
  subsampled?: boolean
  n_rows_used?: number
}

interface CalibrationArtifact {
  assessed?: boolean
  reason?: string
  brier_score?: number
  ece?: number | null
  mce?: number | null
  mean_predicted?: number
  base_rate?: number
  global_bias?: number
  positive_label?: string
  n_bins_empty?: number
}

/** Which gap leads for each metric set — mirrors `_RANK_BY` in src/trust/fairness.py. */
const HEADLINE_GAP: Record<FairnessAudit['metric_set'], string> = {
  binary_classification: 'selection_rate_gap',
  multiclass_classification: 'accuracy_gap',
  regression: 'mae_gap',
}

/**
 * Plain-English reading of `selection_amplification`.
 *
 * Mirrors the thresholds in `model_card._fairness_section`. A large raw gap is
 * often the model being accurate — this is the part it adds beyond the difference
 * that is genuinely in the outcomes.
 */
function amplificationReading(value: number | null | undefined): string {
  if (typeof value !== 'number') return '—'
  if (value > 0.02) return 'model widens the real gap'
  if (value < -0.02) return 'model narrows the real gap'
  return 'tracks the real gap'
}

async function TrustStage({ slug }: StageProps) {
  const fairness = await readJsonArtifact<FairnessArtifact>(
    slug,
    'artifacts/trust/subgroup_fairness.json',
  )
  const importance = await readJsonArtifact<ImportanceArtifact>(
    slug,
    'artifacts/trust/permutation_importance.json',
  )
  const calibration = await readJsonArtifact<CalibrationArtifact>(
    slug,
    'artifacts/trust/calibration.json',
  )
  const card = await readTextArtifact(slug, 'artifacts/trust/model_card.md')
  const figures = await readFigures(slug, 'artifacts/trust/figures')
  const importanceTable = await readTable(slug, 'artifacts/trust/permutation_importance.csv', 60)
  const fairnessTable = await readTable(slug, 'artifacts/trust/subgroup_fairness.csv', 200)

  if (!fairness && !importance && !calibration && !card) {
    return (
      <EmptyState
        title="No trust artifacts"
        body="This project has no artifacts/trust output. Run `python main.py trust` to audit the champion for subgroup fairness, permutation importance and calibration."
      />
    )
  }

  const audits = fairness?.audits ?? []
  // "Not applicable" and "not assessed" are different claims, and the artifact
  // distinguishes them, so the tile must too.
  const calibrationValue = !calibration
    ? 'Not assessed'
    : calibration.assessed
      ? typeof calibration.ece === 'number'
        ? formatNumber(calibration.ece, 4)
        : 'n/a'
      : 'N/A'

  return (
    <div className="flex flex-col gap-10">
      <FactGrid
        columns={4}
        facts={[
          {
            label: 'Champion audited',
            value: fairness?.champion ?? importance?.measured_against ?? '—',
            hint: `on the ${fairness?.split ?? importance?.split ?? 'test'} split`,
          },
          {
            label: 'Subgroups audited',
            value: formatNumber(audits.length),
            hint:
              audits.length > 0
                ? `${formatNumber(fairness?.n_rows ?? 0)} rows · groups under ${fairness?.min_group_size ?? 30} excluded from gaps`
                : 'no column had 2–10 distinct non-float values',
          },
          {
            label: 'Features above noise',
            value:
              importance
                ? `${formatNumber(importance.n_distinguishable_from_noise ?? 0)} / ${formatNumber(importance.n_features ?? 0)}`
                : '—',
            hint: importance ? `by permutation, ${importance.n_repeats ?? 10} shuffles` : undefined,
          },
          {
            label: 'Calibration (ECE)',
            value: calibrationValue,
            hint:
              calibration && !calibration.assessed
                ? calibration.reason
                : calibration?.assessed && typeof calibration.global_bias === 'number'
                  ? `predicted mean ${formatNumber(calibration.mean_predicted ?? 0, 3)} vs base rate ${formatNumber(calibration.base_rate ?? 0, 3)}`
                  : undefined,
          },
        ]}
      />

      {audits.length > 0 && (
        <StageSection
          title="Subgroup gaps"
          hint="Largest first. A gap is not a verdict."
        >
          <Card padded={false}>
            <DataTable
              columns={[
                { key: 'column', label: 'Column' },
                { key: 'groups', label: 'Groups' },
                { key: 'gap', label: 'Headline gap' },
                { key: 'ratio', label: 'Selection ratio' },
                { key: 'amplification', label: 'Amplification' },
                { key: 'reading', label: 'Reading' },
              ]}
              rows={audits.map((audit) => {
                const headline = audit.gaps?.[HEADLINE_GAP[audit.metric_set]]
                const ratio = audit.gaps?.selection_rate_ratio
                const amplification = audit.gaps?.selection_amplification
                return {
                  column: audit.column,
                  groups: `${audit.n_groups} (${audit.gaps?.n_groups_compared ?? 0} compared)`,
                  gap: typeof headline === 'number' ? formatNumber(headline, 4) : 'not measurable',
                  ratio:
                    typeof ratio === 'number'
                      ? ratio === 0
                        ? '0 — a group is never flagged'
                        : formatNumber(ratio, 3)
                      : '—',
                  amplification:
                    typeof amplification === 'number' ? formatNumber(amplification, 4) : '—',
                  reading: amplificationReading(amplification),
                }
              })}
              maxHeight={480}
              zebra
            />
          </Card>
          <p className="text-caption text-text-secondary">
            Every low-cardinality column is audited, including columns the model uses as features on
            purpose — so a large gap is often the model being accurate. Amplification is the part it
            adds beyond the difference that is genuinely in the outcomes. Whether any gap is
            acceptable depends on the domain and the decision it feeds, which this pipeline does not
            know.
          </p>
        </StageSection>
      )}

      {figures.length > 0 && (
        <StageSection
          title="Trust figures"
          hint="Greyed bars are groups or features the sample size cannot support"
        >
          <FigureGallery figures={figures} columns={1} />
        </StageSection>
      )}

      {calibration?.assessed && (
        <StageSection
          title="Probability calibration"
          hint={`positive class ${calibration.positive_label ?? '—'}`}
        >
          <FactGrid
            columns={4}
            facts={[
              { label: 'Brier score', value: formatNumber(calibration.brier_score ?? 0, 4) },
              {
                label: 'ECE',
                value: formatNumber(calibration.ece ?? 0, 4),
                hint: 'average gap between promised and observed',
              },
              {
                label: 'Worst band (MCE)',
                value: formatNumber(calibration.mce ?? 0, 4),
                hint: 'the average hides this',
              },
              {
                label: 'Global bias',
                value: formatNumber(calibration.global_bias ?? 0, 4),
                hint:
                  (calibration.global_bias ?? 0) > 0
                    ? 'optimistic — promises the positive class too often'
                    : 'pessimistic — promises it too rarely',
              },
            ]}
          />
          <p className="text-caption text-text-secondary">
            ROC-AUC measures whether the model <em>ranks</em> correctly; this measures whether its
            probabilities mean what they say. Anything that multiplies a predicted probability by a
            cost — the retention simulator included — depends on this number, not on the AUC.
          </p>
        </StageSection>
      )}

      {importance && importanceTable && (
        <StageSection
          title="Permutation importance"
          hint={`drop in ${importance.measured_against} when a column is shuffled${importance.subsampled ? ` · ${formatNumber(importance.n_rows_used ?? 0)} sampled rows` : ''}`}
        >
          <TableCard table={importanceTable} title="Measured on held-out data" maxHeight={480} />
          <p className="text-caption text-text-secondary">
            Measured by shuffling held-out columns rather than reading the model&apos;s own
            coefficients, so it works for any estimator and reflects unseen data. Dummy columns are
            shuffled independently, so a categorical&apos;s total importance is not the sum of its
            dummies, and correlated columns share credit — read the table in groups.
          </p>
        </StageSection>
      )}

      {fairnessTable && (
        <TableCard table={fairnessTable} title="Per-group metrics" maxHeight={520} />
      )}

      {card && <TextCard title="Model card" body={card} />}
    </div>
  )
}

// ---------------------------------------------------------------------------

const STAGE_COMPONENTS: Record<StageId, (props: StageProps) => Promise<React.ReactElement>> = {
  dashboard: DashboardStage,
  data: DataStage,
  eda: EdaStage,
  features: FeaturesStage,
  modeling: ModelingStage,
  evaluation: EvaluationStage,
  explainability: ExplainabilityStage,
  governance: GovernanceStage,
  trust: TrustStage,
  reports: ReportsStage,
}

export async function StageBody({ id, slug }: { id: StageId; slug: string }) {
  const Component = STAGE_COMPONENTS[id]
  return <Component slug={slug} />
}
