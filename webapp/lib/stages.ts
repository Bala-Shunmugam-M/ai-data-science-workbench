/**
 * The ten pipeline stages, in the order the work actually happens.
 *
 * This is the single source of truth for the journey: the rail, the overview
 * cards, the prev/next footer and the route validation all read it. Adding a
 * stage means adding one entry here and one component.
 *
 * `probes` are the artifacts that prove a stage produced something. A stage
 * with none of its probes on disk renders an honest "not run yet" rather than
 * an empty page pretending to be a result.
 */
export type StageId =
  | 'dashboard'
  | 'data'
  | 'eda'
  | 'features'
  | 'modeling'
  | 'evaluation'
  | 'explainability'
  | 'governance'
  | 'trust'
  | 'reports'

export interface StageDefinition {
  id: StageId
  /** 1-10, shown in the rail. Upload is step 0 and lives outside this list. */
  number: number
  title: string
  /** One line, used on the overview card and under the page title. */
  blurb: string
  /** Files (relative to the workspace root) that prove this stage ran. */
  probes: string[]
}

export const STAGES: StageDefinition[] = [
  {
    id: 'dashboard',
    number: 1,
    title: 'Dashboard',
    blurb: 'Project KPIs, the champion model, and which pipeline stages have run.',
    probes: ['artifacts/final_model_selection.json', 'artifacts/workflow_status.json'],
  },
  {
    id: 'data',
    number: 2,
    title: 'Data',
    blurb: 'Raw, processed and split previews, the semantic schema, and the validation report.',
    probes: ['data/raw', 'results/validation/validation_summary.csv', 'results/data_profile/data_profile.csv'],
  },
  {
    id: 'eda',
    number: 3,
    title: 'EDA',
    blurb: 'Data-understanding tables and the exploratory figure gallery.',
    probes: ['results/eda', 'results/data_understanding', 'artifacts/auto_eda'],
  },
  {
    id: 'features',
    number: 4,
    title: 'Features',
    blurb: 'The feature plan and its governance approval status.',
    probes: ['results/feature_engineering/feature_plan.csv', 'governance/approvals/feature_approval.json'],
  },
  {
    id: 'modeling',
    number: 5,
    title: 'Modeling',
    blurb: 'Model catalog, approvals, the registry, and the experiment log.',
    probes: ['models/model_registry.json', 'artifacts/experiments/experiments.jsonl'],
  },
  {
    id: 'evaluation',
    number: 6,
    title: 'Evaluation',
    blurb: 'Model comparison, the champion card, and residual diagnostics.',
    probes: ['artifacts/evaluation/model_comparison.csv', 'artifacts/evaluation/model_evaluation_report.json'],
  },
  {
    id: 'explainability',
    number: 7,
    title: 'Explainability',
    blurb: 'Standardized coefficients, the top drivers, and the executive briefing.',
    probes: [
      'artifacts/explainability/coefficient_interpretation.json',
      'artifacts/reports/executive_briefing.md',
    ],
  },
  {
    id: 'governance',
    number: 8,
    title: 'Governance',
    blurb: 'The audit trail, data and model lineage, and the standing decisions.',
    probes: ['governance/audit/audit_log.jsonl', 'governance/lineage/lineage.json', 'governance/decisions.json'],
  },
  {
    id: 'trust',
    number: 9,
    title: 'Trust',
    blurb:
      'Subgroup fairness, permutation importance, probability calibration, and the model card.',
    // The model card is listed first because it is the artifact a reader wants;
    // the other two cover a partial run where the card was not reached.
    probes: [
      'artifacts/trust/model_card.md',
      'artifacts/trust/subgroup_fairness.json',
      'artifacts/trust/permutation_importance.json',
    ],
  },
  {
    id: 'reports',
    number: 10,
    title: 'Reports',
    blurb: 'The self-contained HTML project report, ready to download.',
    probes: ['artifacts/reports/project_report.html', 'artifacts/reports/auto_report.md'],
  },
]

export const STAGE_IDS = STAGES.map((stage) => stage.id)

export function getStage(id: string): StageDefinition | undefined {
  return STAGES.find((stage) => stage.id === id)
}

export function neighbours(id: StageId): {
  previous: StageDefinition | null
  next: StageDefinition | null
} {
  const index = STAGES.findIndex((stage) => stage.id === id)
  return {
    previous: index > 0 ? STAGES[index - 1] : null,
    next: index >= 0 && index < STAGES.length - 1 ? STAGES[index + 1] : null,
  }
}
