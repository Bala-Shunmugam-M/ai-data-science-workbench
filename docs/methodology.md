# Methodology

**Project:** AI Data Science Workbench - California Housing median-value regression
**Course:** Machine Learning (23BA045E), Amrita School of Business

This document narrates the pipeline in CRISP-DM terms, and cites the course
conventions (James, Witten, Hastie & Tibshirani, *An Introduction to
Statistical Learning*, ISLR; Géron, *Hands-On Machine Learning with
Scikit-Learn, Keras & TensorFlow*; the CRISP-DM process model) where they
motivated a specific design decision. Every claim below is backed by a stage
that writes its output to disk under `results/`, `artifacts/`, or
`governance/` - nothing here is asserted without an artifact a reviewer can
open.

---

## 1. Business understanding

**Question:** which measurable characteristics of a California census block
most affect its median home value, and can we predict that value reliably
enough to support portfolio, pricing, and market-entry decisions?

**Framing choice:** the team restricted the model catalogue to the **linear
family** (LinearRegression, Ridge, Lasso) rather than tree ensembles or
black-box models. This is a managerial-communication decision, not a
capability limit - CRISP-DM's business-understanding phase asks for a model
the *audience* can act on. A linear model's coefficients translate directly
into "a one-standard-deviation increase in median income raises predicted
value by about $74,600, holding other factors fixed" - a sentence a
non-technical stakeholder can use. ISLR (ch. 3, ch. 6) and Géron (ch. 4)
both frame regularized linear regression as the right first model precisely
because of this interpretability property, and both treat it as a strong
baseline against which more flexible models should be justified, not
assumed. The model catalogue (`src/model_proposal/model_catalog.py`) stays
extensible so ensembles can be added later without structural change - see
`docs/professor_clarifications.md` Q4.

---

## 2. Data understanding

**Source:** the professor's starter dataset, California census housing data
(`kagglehub.dataset_download("harrywang/housing")`), ~20,640 census blocks
described by location, housing age, room/bedroom counts, population,
households, median income, and coastal proximity (`ocean_proximity`), with
`median_house_value` as the target.

**Structural profiling** (`src/domain/data_profile.py`, stage: `ingest`)
runs first and makes *no* semantic assumptions - it reports observed dtypes,
missingness, cardinality, and candidate flags only, exactly the "get to know
your data before you touch it" step CRISP-DM prescribes before any cleaning
decision.

**Semantic validation** (`src/data_manager/data_validator.py`, stage:
`validate`) then checks the data against a **human-approved semantic
schema** (`src/domain/schema.py`'s `SEMANTIC_SCHEMA`): expected storage
type, nullability, numeric bounds, whole-number requirements, and allowed
categorical levels. This schema is the governed contract every downstream
stage assumes; a `FAILED` status is a hard gate that blocks preprocessing.

**Deep data understanding + EDA** (`src/eda/`, stages: `understand`, `eda`)
run on the **training split only** (see §3) and produce: a per-variable
summary with skew/kurtosis, missing-value and categorical-frequency tables,
logical-consistency checks (e.g. households should not exceed population), a
correlation matrix and target-correlation ranking, an IQR + modified-Z
extreme-value screen, and a full figure set (target distribution,
per-variable histograms/boxplots, the correlation heatmap, predictor-vs-target
scatter plots, and a geographic scatter colored by value). No observation is
removed, capped, or transformed at this stage - EDA in this platform is
strictly diagnostic, per CRISP-DM's separation of understanding from
preparation.

---

## 3. Leakage-safe preparation

This is the section the professor's starter code is most prescriptive
about, and the platform preserves every rule exactly.

### Split early; test untouched

The dataset is split into **train / validation / test (70% / 15% / 15%)**
immediately after cleaning (`src/preprocessing/splitter.py`, stage:
`preprocess`), using a **stratified split on `income_category`** - median
income binned into `[0, 1.5, 3.0, 4.5, 6.0, inf]` - so all three splits keep
the same income distribution as the population. This is the textbook
stratified-sampling argument from ISLR/Géron: median income is the strongest
predictor of the target, so an unstratified random split risks a validation
or test set that is not representative on exactly the variable that matters
most.

The **test split is then read exactly once**, at the final evaluation step
(§5) - never for EDA, never for tuning, never for feature selection. This is
the professor's explicit doctrine, and it is enforced in code
(`src/model_evaluation/evaluator.py` loads the test CSV a single time and
scores only the champion against it) and verified by a dedicated test
(`tests/test_model_pipeline.py::test_evaluation_touches_test_only_for_champion`,
which instruments the loader and asserts the test path is read exactly
once).

### Train-only fitted transformations

Every fitted transform - `total_bedrooms` median imputation, `StandardScaler`
on numeric predictors, `OneHotEncoder` on `ocean_proximity` - is fit on the
training split only, *after* the split, and reused (never refit) on
validation/test. This is the single most common source of optimistic bias in
student ML projects (ISLR ch. 5's cross-validation chapter is built around
exactly this failure mode: any statistic touching held-out data before
prediction time inflates reported performance) and is why
`median_imputation_after_split` is recorded as a standing governance
decision (`governance/decisions.json`).

### Governed feature engineering

Feature engineering is a **two-step governed process**
(`src/feature_engineering/feature_proposal.py`, stage: `features`):

1. A formal **proposal**: every candidate feature (six deterministic ratios,
   four log transforms of skewed counts, two optional interaction terms, a
   domain-grouping option, and the scaling/encoding steps already covered by
   preprocessing) is recorded with its source columns, formula, business
   meaning, rationale, leakage risk, and whether it requires a training-only
   fit - *before* any feature is computed.
2. **Execution of only the `automatic=True` features** (the six ratios and
   four log transforms - all leakage-free, deterministic row-wise
   transforms) into `data/engineered/{train,validation,test}.csv`. Optional
   features (interaction terms, category grouping) are flagged for
   analyst/domain approval and are not executed automatically.

The **approval workflow** (`src/governance/approvals.py`, `main.py approve`)
then marks which proposed features are approved and computes the final
`approved_predictor_columns` list the trainer is allowed to use - training
refuses to run without this file.

---

## 4. Modelling: proposal, training, tuning

**Model proposal** (`src/model_proposal/proposal.py`, stage: `propose`)
inspects the engineered training data (row/predictor counts, a
high-pairwise-correlation screen at |r| >= 0.80) and recommends
LinearRegression, Ridge, and Lasso, with the rationale that Ridge's L2
shrinkage stabilizes the collinear room/bedroom/population/household count
predictors and Lasso's L1 selection can surface a sparser driver list for
the executive briefing.

**Training** (`src/model_training/trainer.py`, stage: `train`) fits every
**approved** model (governance refuses any model absent from
`model_approval.json`) on the engineered training split. Ridge and Lasso's
`alpha` is **tuned on the dedicated validation split** - a 25-point
log-spaced grid, each candidate scored by validation RMSE - deliberately
*not* via k-fold cross-validation over train+validation. See
`docs/testing_and_demo.md` for the full "why not k-fold" answer; in short,
introducing k-fold here would let validation-fold information leak into
model selection in a project whose explicit doctrine is a single, disjoint
validation set, and it would blur the train/validation/test boundary the
professor's starter code establishes.

Every trained model version is registered
(`models/<name>/<version>/{model.joblib,metadata.json}`,
`models/model_registry.json`) and logged to the append-only experiment
history (`artifacts/experiments/experiments.jsonl`), so every run - not just
the winner - is reproducible and auditable.

---

## 5. Evaluation on the untouched test set

**Evaluation** (`src/model_evaluation/evaluator.py`, stage: `evaluate`)
compares every registered model on the validation split, selects the
champion by lowest validation RMSE, and evaluates **only that champion**,
**once**, on the untouched test split. Champion selection is written to
`artifacts/final_model_selection.json` with a plain-language selection
narrative, and residual diagnostics (residuals-vs-predicted, residual
histogram, predicted-vs-actual) are generated for the champion only.

Current result: **Ridge (alpha=100)** - validation RMSE $64,787 / R² 0.680,
test RMSE $66,681 / R² 0.667.

---

## 6. Interpretation

**Explainability** (`src/explainability/interpretation.py`, stage:
`explain`) exploits the fact that every numeric predictor is standardized
before fitting: each fitted coefficient is already a *standardized effect* -
the dollar change in predicted value per one-standard-deviation increase in
that predictor, holding the others fixed (one-hot coefficients are the
dollar effect of category membership versus the baseline). This produces,
without any additional modelling: a ranked coefficient table, a top-drivers
chart, and a one-page, non-mathematical executive briefing
(`artifacts/reports/executive_briefing.md`) - the "managerial report: factors
affecting property prices" deliverable named in the course brief.

Top drivers of predicted value: `median_income` (+), `latitude` (-),
`longitude` (-), `log_population` (-), `ocean_proximity_INLAND` (-).

**Caveats the briefing surfaces explicitly:** multicollinearity among the
room/bedroom/population/household counts (their individual coefficients
should be read as a group), the linear model's inability to capture the
sharp, nonlinear coastal-proximity premium, census-block-level aggregation
(conclusions apply to neighbourhoods, not individual houses), and the
**$500,001 target cap** in the source census data, which compresses the
model's ability to distinguish the most expensive blocks (see
`docs/testing_and_demo.md` for the examiner Q&A on this point).

---

## 7. Governance as a first-class layer

Every governed stage writes to one of four append-only or versioned stores:

- **Audit trail** (`governance/audit/audit_log.jsonl`) - one JSON record per
  governance event (proposal created, approval granted, training run,
  evaluation run, final selection), timestamped and attributed to an actor.
- **Approvals** (`governance/approvals/*.json`) - the binding gate: training
  raises `GovernanceError` for any unapproved model, and the modelling
  matrix excludes any predictor not in `approved_predictor_columns`.
- **Lineage** (`governance/lineage/lineage.json`) - a node per stage
  (training, evaluation, explainability) recording input/output file paths
  and sha256 hashes, so any evaluated model can be traced back to the exact
  data and transformations behind it.
- **Decisions** (`governance/decisions.json`) - the standing team choices
  (linear-family-only, split-early, train-only imputation, validation-set
  tuning, governed approvals) with their rationale, keyed so a reviewer sees
  *why* the platform behaves as it does without reading the code.

This governance layer is what the Milestone-3 Streamlit GUI's Governance
page renders, and what the Model Defence demo script (§4 of
`docs/testing_and_demo.md`) traces end to end for a single feature.
