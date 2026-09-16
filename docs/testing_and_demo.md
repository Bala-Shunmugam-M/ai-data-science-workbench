# Testing and Demo Guide

## 1. Running the tests

```bash
pip install -r requirements.txt
pytest
```

The suite (`tests/`) is entirely synthetic-data and `tmp_path`-isolated - it
never touches the network, the Kaggle download, or the project's real
`data/`, `models/`, `artifacts/`, or `governance/` directories. As of
Milestone 3 it covers:

| File | What it checks |
|---|---|
| `test_schema.py` | Semantic-schema validation rules and statuses. |
| `test_splitter.py` | Stratified split proportions, determinism, and income-distribution preservation. |
| `test_imputer.py` | Train-only median imputation (no leakage from validation/test). |
| `test_features.py` | Feature-proposal structure and automatic-feature execution. |
| `test_model_pipeline.py` | Governance refusal of unapproved models, target exclusion from predictors, validation-set alpha tuning, model-registry round trip, metric correctness, and the test-split-touched-once guarantee. |
| `test_orchestrator.py` | Orchestrator dependency refusal, status-file persistence, resume/force-rerun semantics, and stage-range execution. |

Run a single file or test with `pytest tests/test_orchestrator.py -k force`.

## 2. Running the pipeline

```bash
python main.py all                                    # full run, always reruns every stage
python main.py pipeline --from propose --to report      # orchestrated partial re-run, resumable
streamlit run app/main.py                               # read-only dashboard
```

`python main.py all` is the smoke test named in `docs/PLAN.md` §8: on a
fresh clone it should execute ingestion through reporting without manual
intervention (a Kaggle download happens once; subsequent runs reuse the
cache). Every stage's declared inputs/outputs are dependency-checked by the
orchestrator - running a downstream stage before its prerequisite produces a
clear error naming the missing stage rather than a confusing traceback deep
inside pandas or scikit-learn.

## 3. Model Defence demo script (~10 minutes)

A suggested walkthrough for the Project Presentation & Model Defence.

**0. Setup (before the session starts):** have `python main.py all` already
completed once, and have `streamlit run app/main.py` running in a browser
tab.

**1. Dashboard walkthrough (2 min).** Open the Dashboard page. Point at the
KPI row (20,640 rows total, 70/15/15 split sizes), the champion card (Ridge,
test RMSE $66,681 / R² 0.667), and the pipeline stage-status table - every
stage `ok`, with timing. This is the single-screen proof the whole chain ran
end to end and is reproducible.

**2. Trace one feature from proposal to briefing (5 min).** This is the
governance narrative examiners look for.

   - **Features page:** find `log_population` in the feature plan - group
     "Statistical", formula `log1p(population)`, `automatic = True`,
     `approved = ✅`.
   - **Governance page:** filter the audit trail for
     `feature_approval_granted`; show `log_population` in the payload's
     `approved_predictor_columns`. Open the lineage graph's `model_training`
     node and show the engineered training file's hash as an input.
   - **Modeling page:** show the Ridge registry entry - `alpha = 100`,
     validation RMSE $64,787 - and that Ridge is `is_champion = true`.
   - **Evaluation page:** show the champion card's test metrics (the *one*
     time the test split was touched) and the residual diagnostics.
   - **Explainability page:** find `log_population` in the coefficient table
     - rank 4, coefficient about -$48,747, "decreases value" - and read the
     matching line from the executive briefing: a one-standard-deviation
     increase in a district's (log-scaled) population is associated with
     roughly $48,700 lower predicted value, holding other factors fixed.

   This closes the loop: **proposal -> approval -> engineered data ->
   trained model -> test evaluation -> plain-English interpretation**, each
   step backed by a file a reviewer can open.

**3. Managerial takeaway (2 min).** Read the executive briefing's top-5
drivers and recommendations aloud (Explainability page, or
`artifacts/reports/executive_briefing.md`): median income and coastal
proximity are the primary levers; the model is best used for *relative*
neighbourhood ranking, not single-home valuation; and the caveats
(multicollinearity, coastal nonlinearity, block-level aggregation, the
$500k value cap) are stated up front rather than discovered by the examiner.

**4. Close (1 min).** Open the Reports page and show the single
self-contained `project_report.html` - the artifact a grader can open
without running any code.

## 4. Likely examiner questions and answers

**Q: Why Ridge over Lasso, if Lasso was in the candidate set?**
A: Both scored within 100 RMSE of each other on the validation split (Ridge
64,787 vs Lasso 64,887); the selection metric was validation RMSE and Ridge
won, narrowly. The two behave differently on the collinear count predictors
(rooms/bedrooms/population/households): Ridge shrinks correlated
coefficients together, Lasso can zero one out somewhat arbitrarily among
near-duplicates. Given the model is used for coefficient-based managerial
interpretation, Ridge's smoother shrinkage is also the more defensible
choice, not just the marginally lower-RMSE one.

**Q: Why tune on a dedicated validation set instead of k-fold
cross-validation?**
A: The professor's starter doctrine is an explicit three-way split with the
test set held out for a single final measurement. k-fold over train+validation
would blur that boundary and effectively let validation-fold rows influence
each other's out-of-fold score - a different (and for this course, not the
specified) methodology. A dedicated validation split keeps train, tune, and
test strictly separated and matches exactly what was asked for. (This is a
genuine methodological trade-off, not a free lunch: k-fold would use the data
more efficiently and give a less noisy alpha estimate; we chose fidelity to
the assigned doctrine over that efficiency gain.)

**Q: Why is the test split touched only once?**
A: Every time a model or hyperparameter choice is evaluated against a
held-out set, that set stops being "unseen" for the purposes of an honest
final performance estimate - even without explicitly refitting, repeated
peeking lets you implicitly select for what happens to score well on that
particular sample. The platform enforces this in code (the evaluator loads
the test CSV exactly once, scores only the already-selected champion) and a
dedicated test
(`tests/test_model_pipeline.py::test_evaluation_touches_test_only_for_champion`)
asserts it by instrumenting the file loader.

**Q: What leakage safeguards are in place?**
A: (1) The split happens before any statistic is computed from the data.
(2) `total_bedrooms` median imputation, the `StandardScaler`, and the
`OneHotEncoder` are all fit on the training split only and reused (never
refit) on validation/test. (3) The stratification column (`income_category`)
is a temporary artifact, dropped from the saved splits so it can never leak
into the model as a feature. (4) The approval workflow enumerates the exact
predictor columns the trainer may use, computed from the engineered training
data's actual columns - so an approved-but-absent feature name can't
silently leak through either.

**Q: What is the $500,000 cap caveat, and why does it matter?**
A: The source census data caps `median_house_value` at $500,001 - any block
whose true median value exceeded that is recorded at the cap. This means the
model cannot learn to distinguish the most expensive blocks from one
another, and will systematically under-predict at the top of the market
(the model has never seen a "true" label above the cap to learn from). The
executive briefing states this explicitly as a caveat rather than leaving it
for the examiner to discover, and recommends treating model output as a
*relative* ranking tool rather than a precise valuation, especially for
high-value coastal segments.
