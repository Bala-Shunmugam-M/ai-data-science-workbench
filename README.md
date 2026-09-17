# AI Data Science Workbench

A governed, reproducible, end-to-end machine learning platform built for the
**Machine Learning (23BA045E)** course at **Amrita School of Business** (Term
IV, 2026). The showcase project is the professor's **California Housing**
starter dataset: predicting `median_house_value` from census-block features.

**Course context**

| | |
|---|---|
| Course | Machine Learning, 23BA045E |
| Programme | Amrita School of Business (MBA) |
| Instructor | Dr. Prashobhan Palakkeel |
| Assessment | ML Project (25 marks) + Project Presentation & Model Defence (10 marks) |
| Team | _fill in team member names here_ |

The platform is **dataset-agnostic** (see `config/datasets.yaml`); the
California Housing pipeline is the first registered project and the one
graded for this course.

## Open the site

Double-click **`Launch-Workbench.bat`**. It installs dependencies on first run,
builds the site if anything changed, starts the production server and opens the
browser. A warm start takes a couple of seconds; the first ever run takes a
minute or two while it installs and builds.

| Command | What it does |
|---|---|
| `Launch-Workbench.bat` | Build if needed, start, open the browser |
| `Launch-Workbench.bat /rebuild` | Force a fresh build first |
| `Launch-Workbench.bat /stop` | Stop the running site |
| `start-webapp.bat` | Dev server instead, with hot reload (slower pages) |

Needs **Node.js** to serve the site and **Python** on PATH to analyse an upload
— the analysis runs through the same Python pipeline the CLI uses.

---

## Deployed surfaces

Three ways to reach this project, with different trade-offs. The split is not a
preference: it follows from what each host can actually run.

| Surface | Host | Interactive? |
|---|---|---|
| **Report site** | GitHub Pages | No — generated reports, model cards and fairness audits |
| **Streamlit app** | Streamlit Community Cloud | Yes — all 11 pages, including upload and the retention simulator |
| **Next.js console** | local only | Yes — the full journey UI |

### Why the console is not hosted

Its API routes spawn the pipeline as a child process (`webapi/bridge.py`), so a
host must run **Node and Python in the same container**. That rules out GitHub
Pages, Vercel and Netlify outright. Container hosts can do it — a working
`deploy/render/Dockerfile` and `render.yaml` are in the repository — but every
remaining free tier (Render, Fly.io, Railway, Koyeb) now requires a payment
card, and Hugging Face Spaces bills for any Space that runs Python:

    docker    BLOCKED - PRO subscription required
    gradio    BLOCKED - PRO subscription required
    streamlit BLOCKED - PRO subscription required
    static    ALLOWED

So the console runs locally, and the Streamlit app carries the live interactive
demo. Both drive the identical pipeline.

### Deploying

**Report site** — automatic. `.github/workflows/pages.yml` publishes on every
push to `main`, then verifies the live URL actually serves that build rather
than trusting the deploy step's exit code.

**Streamlit app** — at [share.streamlit.io](https://share.streamlit.io), point a
new app at this repository, branch `main`, main file `app/main.py`. Free, no
card. On first load it seeds two demo projects by running the real pipeline on
the committed 600-row samples (`src/automl/demo_seed.py`); the repository ships
the pipeline's code but not its output, so a bare clone would otherwise show
empty dashboards.

**Console on a container host** — `render.yaml` is a ready Blueprint if you
have a host with a card on file.

---

## 0. Two projects, one workbench

The workbench now hosts **two projects side by side**, isolated from each other:

| Project | `WORKBENCH_PROJECT` | Task | Champion | Headline |
|---|---|---|---|---|
| California Housing | `california-housing` (default) | regression | Ridge | test R² 0.667 |
| Customer Churn (Telco) | `churn` | classification | LogisticRegression | test ROC-AUC 0.845 |

The active project is chosen by the `WORKBENCH_PROJECT` environment variable.
Housing keeps the legacy flat layout at the repo root (its graded artifacts live
there); every other project is isolated under `workspaces/<slug>/` so nothing
overwrites the graded showcase. Task, target, metric, categorical columns, and
split strategy are declared per project in `config/datasets.yaml` and resolved by
`config/active.py`; the modelling stages (train / evaluate / explain) branch on
the active task, so one code path serves both regression and classification.

**Run the churn project end to end** (ingest → clean → stratified split → train
Logistic / Decision Tree / Random Forest → evaluate, champion by ROC-AUC):

```bash
# Windows PowerShell
$env:WORKBENCH_PROJECT="churn"; python main.py churn-all
# bash
WORKBENCH_PROJECT=churn python main.py churn-all
```

Then launch the GUI and open **🎯 Retention Simulator** (page 10) — the
interactive "what-if" tool: move the risk-threshold, discount, offer-effectiveness,
and horizon sliders and watch customers-retained, campaign cost, net value, and
ROI update live. The simulator is self-contained (always reads the churn
workspace), so it works even when the app is launched as the housing project.

```bash
streamlit run app/main.py        # use the sidebar "Project" switcher to toggle projects live
```

## 0.1 Bring your own dataset (upload any CSV)

The GUI's **➕ New Project** page turns any CSV into a full project with no
hand-configuration:

1. **Upload** a CSV.
2. The app **profiles** every column and **auto-detects** the task
   (regression vs binary classification), **guesses the target** and positive
   class, flags identifier / high-cardinality columns to drop — you can
   override the target and task before running.
3. **Analyze & Train** registers it as `workspaces/<slug>/` and runs
   `autorun` (auto-clean → task-appropriate split → train the right model
   family → evaluate → auto-EDA + report).
4. The new project appears in the **sidebar switcher** and its results
   (champion metrics, target distribution, top correlations, diagnostics,
   report) render inline.

From a terminal the same thing is: register a `workspaces/<slug>/dataset.json`
(+ raw CSV) then `WORKBENCH_PROJECT=<slug> python main.py autorun`. Detection
lives in `src/automl/detect.py`; the generic pipeline in
`src/pipelines/generic_pipeline.py`.

**Supported tasks:** regression, binary classification, and **multiclass**
classification. The task layer adapts automatically:

| | binary classification | multiclass classification |
|---|---|---|
| Target handling | encoded to 0/1 on the detected positive class | native labels (strings or codes) |
| precision / recall / F1 | positive class | **macro-averaged** over classes |
| ROC-AUC | positive-class scores | **one-vs-rest, macro-averaged** |
| Diagnostics | 2×2 confusion matrix + ROC | N×N confusion matrix + per-class OvR ROC |
| Report associations | Pearson correlation with target | **ANOVA F** (class separation) |

Verified end to end on 3-class (iris, wine) and 10-class (digits, test accuracy
0.970 / macro F1 0.970) uploads; the binary and regression numbers above are
unchanged by the multiclass work.

---

The GUI has a **live project switcher** in the sidebar (`app/project_state.py`):
pick *California Housing* or *Customer Churn* and every page re-resolves in place —
no relaunch, no env var. The champion KPIs, evaluation diagnostics, and the
Dashboard "Run stage" control are task-aware (RMSE/R² + residuals for regression;
ROC-AUC/accuracy + confusion-matrix/ROC and a `churn-all` button for
classification). It works by reloading `config.paths`/`config.active` from the
selected project on each run, so Streamlit's module cache can't pin one workspace.

---

## 1. Architecture overview

The workbench implements the 8-stage lifecycle from the course's
architecture diagrams, plus a cross-cutting governance layer and a
Milestone-3 workflow orchestrator + Streamlit GUI (see `docs/PLAN.md` for the
full design rationale):

```
1. Project Management     (config/, project registry conventions)
2. Data Ingestion          -> data/raw/housing.csv
3. Data Understanding      -> results/data_profile/, results/data_understanding/
4. Preprocessing           -> data/processed/, data/splits/ (train/val/test, split EARLY)
5. Feature Engineering     -> governance/approvals/feature_proposal.json -> data/engineered/
6. Model Proposal          -> governance/approvals/model_proposal.json
7. Model Training          -> models/<name>/<version>/, models/model_registry.json
8. Model Evaluation        -> artifacts/evaluation/, artifacts/final_model_selection.json
```

Cross-cutting layers:

- **Governance** (`src/governance/`): append-only audit trail
  (`governance/audit/audit_log.jsonl`), the feature/model approval workflow
  that gates training (`governance/approvals/*.json`), a data/model lineage
  graph (`governance/lineage/lineage.json`), and a standing-decisions
  registry (`governance/decisions.json`).
- **Workflow orchestrator** (`src/workflow/`): a light DAG runner over the
  same `run()` functions the CLI calls, with dependency checking and status
  tracking (`artifacts/workflow_status.json`). See §4.
- **Streamlit GUI** (`app/`): a read-only dashboard over every artifact above.
  See §5.

Every stage is runnable standalone, re-runnable deterministically, and writes
its outputs as files under `data/`, `results/`, `artifacts/`, `governance/`,
or `models/` - there is no hidden state.

---

## 2. Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline end to end (ingest -> report)
python main.py all

# 4. Launch the read-only Streamlit dashboard
streamlit run app/main.py
```

`python main.py all` downloads (or reuses the cached) California Housing
dataset via `kagglehub`, then runs every stage through to the self-contained
HTML report at `artifacts/reports/project_report.html`. A first run needs
network access for the Kaggle download; subsequent runs use the local cache.

Run the automated tests:

```bash
pytest
```

See `docs/testing_and_demo.md` for the full test/demo guide, including a
10-minute Model Defence walkthrough script.

---

## 3. CLI reference

Every subcommand is `python main.py <command> [options]`.

| Command | What it does |
|---|---|
| `ingest [--dataset NAME] [--download]` | Download/cache the dataset, structurally profile it, run semantic validation. |
| `validate [--dataset NAME]` | Re-run semantic-schema validation only. |
| `preprocess` | Clean -> stratified split -> fit train-only transforms (impute/encode/scale). |
| `split` | Split the already-cleaned dataset only (standalone). |
| `understand` | Write the data-understanding report (structure/quality, train split only). |
| `eda` | Run EDA + the analytical summary (figures + tables, train split only). |
| `features` | Write the governed feature-engineering plan and execute the automatic features. |
| `propose` | Write the governed model proposal (candidate models + validation strategy). |
| `approve [--features] [--models]` | Approve features and/or models (gates training). No flag = approve both. |
| `train` | Train every approved model; refuses any unapproved model. |
| `evaluate` | Compare models on validation, promote the champion, evaluate it once on the untouched test split. |
| `explain` | Champion standardized coefficients + top-drivers chart + executive briefing. |
| `drivers [--force]` | Champion driver table only, dataset-agnostic (works for any uploaded project). |
| `trust [--split test\|validation] [--groups COL ...]` | Subgroup fairness, permutation importance, probability calibration, model card. |
| `report` | Assemble the self-contained HTML project report. |
| `all [--dataset NAME] [--download]` | Run every stage end to end via the orchestrator (always reruns; the CLI's original all-in-one behavior). |
| `pipeline [--from STAGE] [--to STAGE] [--force] [--dataset NAME] [--download]` | Run an orchestrated stage range. Defaults to resuming (skips a stage whose declared outputs already exist); `--force` reruns unconditionally. |

Examples:

```bash
python main.py ingest --download                     # force a fresh Kaggle download
python main.py approve --models                       # approve models only
python main.py pipeline --from propose --to report     # re-run everything downstream of the model proposal
python main.py pipeline --from train --to evaluate --force   # force-retrain and re-evaluate only
python main.py trust                                  # audit the champion, write the model card
python main.py trust --groups gender SeniorCitizen    # audit only these subgroup columns
```

### The `trust` stage

Runs after `evaluate` has frozen the champion, and writes to `artifacts/trust/`:

| Artifact | What it answers |
|---|---|
| `subgroup_fairness.{json,csv}` | Does accuracy hold across subgroups, or is the headline an average hiding a group the model fails? Audits every column with 2-10 distinct non-float values, ranked largest gap first. |
| `permutation_importance.{json,csv}` | What does the model actually depend on, measured by shuffling held-out columns rather than trusting `coef_` or `feature_importances_`? |
| `calibration.json` | When it says 0.8, does that happen 80% of the time? Binary classifiers only; regression and multiclass record why they were skipped. |
| `model_card.{md,json}` | All of the above plus the registry, evaluation report, approvals, audit log and lineage, in one document. |

Two things it deliberately does not do. It never labels a column a protected
attribute or judges whether a gap is acceptable — it reports the spread and the
group sizes behind it. And it never authors an intended-use statement, because
that is a claim about which decisions the model may inform, which no artifact
here can supply.

Because every low-cardinality column is audited, a large gap is often the model
being *accurate* — month-to-month contracts really do churn more. The
`selection_amplification` figure separates the two: it is the part of the gap the
model adds on top of the difference that is genuinely in the outcomes.

---

## 4. Workflow orchestrator (Milestone 3)

`src/workflow/pipeline_builder.py` declares one `StageSpec` per pipeline
stage (in canonical order: `ingest -> preprocess -> understand -> eda ->
features -> propose -> approve -> train -> evaluate -> explain -> report`),
each wrapping the *same* `run()` function the CLI already calls, plus its
declared input files and output files.

`src/workflow/orchestrator.py` reads that registry to:

- **Refuse to run a stage whose declared inputs are missing**, naming the
  prerequisite stage(s) in the error (e.g. running `train` before `approve`
  fails with a message pointing at `approve`).
- **Track status** per stage in `artifacts/workflow_status.json`: `last_run`
  (UTC), `duration_s`, `status` (`ok` / `failed` / `skipped`), and `outputs`.
- **Resume** by default: a stage whose declared outputs already exist on disk
  is skipped (no `--force`).
- **Force a full re-run** with `--force`.

`main.py all` and `main.py pipeline` are both thin wrappers over
`orchestrator.run_pipeline(...)`; no stage's internal behavior changed for
Milestone 3.

---

## 5. Streamlit GUI (Milestone 3)

```bash
streamlit run app/main.py
```

A **read-only** dashboard over every artifact above. The only control that
changes project state is the **"Run stage"** section on the Dashboard page,
which shells out to `python main.py pipeline ...` and streams its output -
it never calls pipeline code in-process.

| Page | Contents |
|---|---|
| 1 Dashboard | KPIs (rows, split sizes, feature/model counts, champion + test metrics), pipeline stage-status table, "Run stage" control. |
| 2 Data | Raw/processed/split previews, the semantic schema table, the validation report. |
| 3 EDA | Data-understanding tables + the EDA figure gallery. |
| 4 Features | The feature plan with approval-status badges. |
| 5 Modeling | Model proposal/catalog, approvals, registry metrics, experiment log. |
| 6 Evaluation | Validation comparison table, champion card, residual diagnostics. |
| 7 Explainability | Standardized coefficients, top-drivers chart, rendered executive briefing. |
| 8 Governance | Audit-log viewer, lineage graph, standing decisions. |
| 9 Reports | Preview/download the self-contained HTML project report and other deliverables. |

Every page degrades gracefully: a stage that hasn't been run yet shows an
info banner naming the CLI command to run, instead of crashing.

---

## 6. Project structure

```
main.py                      # CLI entry point
requirements.txt  pyproject.toml  .env.example  .gitignore  LICENSE
config/                       # paths.py, constants.py, settings.yaml, datasets.yaml
src/
  connectors/                 # kaggle_connector, file_connector
  data_manager/                # data access facade, dataset registry, validator
  domain/                      # dataset.py, schema.py (semantic schema), data_profile.py
  eda/                         # explorer, understanding, analysis_summary
  preprocessing/                # pipeline, imputer, encoder, scaler, splitter
  feature_engineering/          # feature_proposal, pipeline, transformers
  model_proposal/               # proposal, model_catalog
  model_training/                # factory, trainer, metrics, design_matrix
  model_evaluation/               # evaluator, comparator
  explainability/                  # interpretation, shap_explainer
  governance/                       # approvals, audit_logger, decision_tracker, lineage_tracker
  artifacts/                        # artifact_store, model_registry, experiment_tracker
  reporting/                        # report_generator, html_reporter
  workflow/                         # orchestrator, pipeline_builder     <- Milestone 3
  pipelines/                        # one *_pipeline.py per stage, called by main.py
  utils/                             # logging_utils, file_utils, data_utils, common
app/                          # Streamlit GUI: main.py, pages/1..9_*.py   <- Milestone 3
data/  raw/ processed/ splits/ engineered/
models/                       # trained model files + model_registry.json
artifacts/                    # generated reports, evaluation outputs, experiments, workflow_status.json
governance/                   # approvals/*.json, audit log, lineage, decisions.json
results/                      # stage analytical outputs (professor's convention)
docs/                         # PLAN.md, methodology.md, testing_and_demo.md, professor_clarifications.md
tests/                        # pytest suite (pipeline stages + orchestrator)
```

---

## 7. Key results (California Housing showcase)

- **Champion model:** Ridge regression, `alpha = 100`
- **Validation:** RMSE $64,787 | R² 0.680
- **Test (evaluated once, untouched until final evaluation):** RMSE $66,681 | R² 0.667
- **Top value drivers** (standardized coefficients, champion model):
  1. `median_income` (+) - dominant driver
  2. `latitude` (-)
  3. `longitude` (-)
  4. `log_population` (-)
  5. `ocean_proximity_INLAND` (-)

Full narrative in `artifacts/reports/executive_briefing.md` and the complete
report in `artifacts/reports/project_report.html`.

---

## 8. Team

_Fill in before submission:_

| Role | Name |
|---|---|
| Team member | |
| Team member | |
| Team member | |
| Team member | |

See `docs/professor_clarifications.md` for open questions to raise before
the Model Defence, and `docs/testing_and_demo.md` for the demo script.
