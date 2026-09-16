# Multi-project + Classification extension — build plan

Adds a **second project** (Customer Churn, classification) alongside the existing
**California Housing** (regression) showcase, plus an interactive **retention
simulator**, without disturbing the graded housing artifacts.

## Design decisions

- **Housing stays put.** `california-housing` keeps the legacy flat layout
  (`data/`, `models/`, `artifacts/`, ...) because its registry stores absolute
  `model_path`s and its graded report already lives there. Moving it would break
  those paths. Every *other* project gets an isolated `workspaces/<slug>/` tree.
- **Active project = env var.** `WORKBENCH_PROJECT` (default `california-housing`)
  selects the workspace. CLI: `WORKBENCH_PROJECT=churn python main.py all`.
- **Task is data-driven.** Each `datasets.yaml` entry declares `task`
  (`regression`|`classification`), `target`, `categorical_columns`, `stratify`
  (`income`|`target`|`none`), `selection_metric`, and `positive_class`.
  Stages branch on the active dataset's task instead of hard-coded globals.
- **Parallel classification path, not a rewrite.** Regression stays byte-for-byte;
  classification adds classifier catalog entries, classification metrics, and
  task branches in trainer/evaluator/explain.

## Phases

1. **Backbone** — `config/paths.py` project-aware workspace; `config/active.py`
   task-config resolver; `datasets.yaml` gains task fields + a `churn` entry.
2. **Classification modeling** — metrics (acc/prec/rec/F1/ROC-AUC), catalog
   (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier),
   task-aware trainer tuning + evaluator (champion by ROC-AUC; confusion-matrix
   + ROC figures) + explain (odds ratios, tree importances/rules).
3. **Churn data** — Telco Customer Churn ingestion + schema + stratify-by-target.
4. **Retention simulator** — `src/simulation/retention.py` + a Streamlit page:
   sliders (risk threshold, discount %, acceptance, monthly revenue) → live
   customers-saved / cost / net-value / ROI.
5. **Tests + end-to-end** — churn runs green; housing regression unchanged.

## Dataset

Telco Customer Churn — kaggle `blastchar/telco-customer-churn`,
file `WA_Fn-UseC_-Telco-Customer-Churn.csv`, target `Churn`, positive class `Yes`.
Requires network + Kaggle auth on first download (cached thereafter), same as
housing.
