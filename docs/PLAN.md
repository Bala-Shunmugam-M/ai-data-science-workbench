# AI Data Science Workbench — Master Implementation Plan

**Course:** Machine Learning (23BA045E), Amrita School of Business, Term IV 2026
**Instructor:** Dr. Prashobhan Palakkeel
**Assessment targets:** ML Project (25 marks) + Project Presentation & Model Defence (10 marks)
**Showcase project:** California Housing — median house value prediction (professor's starter pipeline)

---

## 1. What we are building

A governed, reproducible, end-to-end machine learning platform ("AI Data Science Workbench")
implementing the 8-stage lifecycle from the architecture diagrams:

1. Project Management
2. Data Ingestion
3. Data Understanding
4. Preprocessing
5. Feature Engineering
6. Model Proposal
7. Model Training
8. Model Evaluation

with a cross-cutting governance layer (approvals, audit trail, lineage, versioned artifacts,
stage-wise reports), a workflow orchestrator, and a Streamlit GUI.

The platform is **dataset-agnostic** (configured via `config/datasets.yaml`); the first
registered project is the professor's California Housing pipeline.

## 2. Non-negotiable conventions (from professor's starter code)

These come directly from the provided scripts and MUST be preserved:

- **Split early; test data untouched** until final evaluation. Stratified split on
  `income_category` (binned `median_income`, bins [0, 1.5, 3.0, 4.5, 6.0, inf]), seed 42,
  train/validation/test.
- **All fitted transformations (imputation, scaling, encoding) are fit on train only**
  — e.g. `total_bedrooms` median imputation happens *after* the split.
- **Human-approved semantic schema** (`SEMANTIC_SCHEMA` dict: role, semantic_type,
  expected_storage_type, nullable, min/max, whole_number, preprocessing_group) drives
  validation and preprocessing.
- **Feature engineering is a governed two-step**: first a formal *plan* (what, why, source
  variables, automatic/optional, validation required), then execution of *approved* features
  only.
- **Stage outputs are files** under structured `data/` and `results/`/`artifacts/` folders;
  every stage is runnable standalone and re-runnable deterministically.
- Data source: `kagglehub.dataset_download("harrywang/housing")` → `housing.csv`.

## 3. Modelling scope

Linear family only (per team decision): **LinearRegression, Ridge, Lasso**
(+ optional polynomial/interaction features through the approved feature plan).
Evaluation: RMSE, MAE, R² on validation for selection; final metrics on test once.
Interpretation: standardized coefficients, per-feature contribution narrative for the
managerial report (executive briefing style).

## 4. Project structure (root: `ai-data-science-workbench/`)

```
main.py                      # CLI entry point (run stages / pipelines / app)
README.md  requirements.txt  pyproject.toml  .env.example  .gitignore  LICENSE
config/                      # paths.py, constants.py, settings.yaml,
                             # datasets.yaml, models.yaml, pipeline.yaml
src/
  project_manager/           # project init, metadata, registry (projects.json)
  environment_manager/       # dependency + environment checks
  data_manager/              # data access facade, dataset registry, validator
  connectors/                # kaggle_connector, file_connector
  domain/                    # dataset.py, schema.py (semantic schema), data_profile.py
  eda/                       # explorer, profiler, visualizer, report
  preprocessing/             # pipeline, imputer, encoder, scaler, splitter
  feature_engineering/       # discovery, proposal, transformers, pipeline, evaluator
  model_proposal/            # proposal, model_catalog, scoring
  model_training/            # factory, trainer, metrics, callbacks
  model_evaluation/          # evaluator, metrics, comparator, reporter
  explainability/            # coefficient interpretation (+ SHAP-ready stub)
  governance/                # governance_manager, audit_logger, decision_tracker,
                             # lineage_tracker, policies (yaml)
  artifacts/                 # artifact_store, artifact_metadata, model_registry,
                             # experiment_tracker
  reporting/                 # report_generator, html_reporter
  workflow/                  # orchestrator, pipeline_builder
  pipelines/                 # ingestion / preprocessing / feature / training / evaluation
  utils/                     # file, logging, time, data utils, common
app/                         # Streamlit GUI: main.py, pages/ (dashboard, data,
                             # eda, features, modeling, evaluation, governance, reports)
data/  raw/ processed/ splits/ engineered/
models/                      # trained model files + registry
artifacts/                   # generated reports, evaluation outputs, experiments
governance/                  # approvals/*.json, audit log, lineage
results/                     # stage analytical outputs (professor's convention)
docs/                        # PLAN.md, methodology.md, professor_clarifications.md,
                             # testing_and_demo.md
tests/                       # smoke tests for each pipeline stage
```

## 5. Milestones & agent assignment

| Milestone | Content | Agent |
|---|---|---|
| M1 | Scaffold, config, utils, data management: ingestion → schema → validation → preprocess → split → data understanding → EDA → feature plan & execution (adapt professor's 11 scripts into modules) | Opus |
| M2 | Model proposal, training (linear/ridge/lasso), evaluation, explainability, governance (approvals, audit, lineage), artifact store, model registry, experiment tracking, report generation | Opus |
| M3 | Workflow orchestrator + runnable pipelines + Streamlit GUI + README + docs + end-to-end smoke run | Sonnet |

## 6. Key generated artifacts (deliverables for grading)

- `governance/approvals/feature_approval.json` — approved features list
- `governance/approvals/model_approval.json` — approved models list
- `artifacts/reports/…` — data quality, EDA, training, evaluation reports (HTML/CSV)
- `artifacts/evaluation/model_evaluation_report.json` — final test metrics
- `artifacts/final_model_selection.json` — selected model + rationale
- `models/` — versioned trained models
- Managerial report: factors affecting property prices (executive briefing)

## 7. Flags for professor clarification

1. Is a GUI expected for the term project, or is a CLI-driven governed pipeline sufficient?
2. Should the AutoML comparison (session 21) be included inside the project deliverable?
3. Required format/length of the team project report and citation style.
4. Whether model families beyond regression are required for the final submission
   (current build is linear-family by team decision; the platform's model catalog is
   extensible so ensembles can be added later without structural change).

## 8. Testing & demo strategy (summary — full version in docs/testing_and_demo.md)

- Smoke test: `python main.py run-all` executes ingestion → evaluation on a fresh clone.
- Stage tests: each pipeline stage validates its input contract and writes its outputs.
- Demo script for Model Defence: dashboard walk-through → lineage of one feature from
  proposal → approval → trained model → evaluation → managerial recommendation.
