# Model Card: California Housing

*Generated 2026-08-09T14:20:54+00:00 — Ridge v001*

## Model details

- Family: **Ridge**
- Version: **v001**, registered 2026-07-20T17:23:18+00:00
- Task: **regression**, target **median_house_value**
- Hyperparameters: `alpha=100.0`
- Training-data fingerprint: `3120788253c6558524af37ab52893dd5216febfd36f74cfafeb6f41a0590293d`

## Intended use

**Not authored.** This card records what the model is and how it behaves. It does not state which decisions the model may be used for, because that is not derivable from any artifact in this pipeline — it is a judgement about the domain and the consequences of a wrong prediction. Whoever deploys this model owns that statement.

## Data

- Train: 14,448 rows
- Validation: 3,096 rows
- Test: 3,096 rows (used once, after the champion was chosen)

## Performance

Champion selected by **rmse** on the validation split, from 6 candidates.

**Validation:**
- rmse: 64,787.48
- mae: 46,246.34
- r2: 0.6801
- mape: 28.2869

**Test (held out until the champion was fixed):**
- rmse: 66,680.81
- mae: 48,731.42
- r2: 0.6665
- mape: 28.3237

> 6 candidate model(s) were compared on the untouched validation split using RMSE as the selection metric (lower is better).
  - Ridge (v001): RMSE 64,787, MAE 46,246, R2 0.680.
  - Ridge (v002): RMSE 64,787, MAE 46,246, R2 0.680.
  - Lasso (v002): RMSE 64,887, MAE 46,301, R2 0.679.
  - Lasso (v001): RMSE 64,887, MAE 46,301, R2 0.679.
  - LinearRegression (v002): RMSE 64,887, MAE 46,301, R2 0.679.
  - LinearRegression (v001): RMSE 64,887, MAE 46,301, R2 0.679.
Ridge (v001) had the best validation RMSE and was promoted to champion.
The champion was then evaluated once on the held-out test split: RMSE 66,681, MAE 48,731, R2 0.667. The test split was used only for this single final measurement, never for tuning or selection.

## What the model depends on

**From the model's own internals** (effect on the prediction):

1. `median_income` — 74,630 (increases the prediction)
2. `latitude` — -51,647 (decreases the prediction)
3. `longitude` — -49,816 (decreases the prediction)
4. `log_population` — -48,747 (decreases the prediction)
5. `ocean_proximity_INLAND` — -27,388 (decreases the prediction)
6. `log_total_rooms` — 25,134 (increases the prediction)
7. `bedrooms_per_room` — 24,798 (increases the prediction)
8. `bedrooms_per_household` — -18,900 (decreases the prediction)

**By permutation** (drop in neg_root_mean_squared_error when a column is shuffled, 10 shuffles, measured on the test split):

1. `median_income` — 59,074 ± 854
2. `latitude` — 33,632 ± 905
3. `longitude` — 29,640 ± 841
4. `log_population` — 29,168 ± 732
5. `log_total_rooms` — 11,240 ± 444
6. `bedrooms_per_room` — 8,649 ± 231
7. `log_households` — 4,926 ± 282
8. `households` — 3,360 ± 172

21 of 23 features show a dependence larger than the measurement's own noise.

Where the two lists disagree, the disagreement is the finding: a feature with a large coefficient and no permutation importance is one the model cannot actually use on unseen data.

## Subgroup performance

Measured on the test split (3,096 rows). Groups smaller than 30 rows are reported but excluded from the gap arithmetic.

### `ocean_proximity` — 5 groups, 4 compared

- mean error gap: **10,156**
- mae gap: **29,104**
- rmse gap: **34,708**
- Excluded from the gaps: `ISLAND` (n=1, fewer than 30 rows)

A gap is not a verdict. Whether one is acceptable depends on the domain, the decision the prediction feeds, and the law that applies — none of which this pipeline knows.

## Probability calibration

**Not applicable.** Calibration is a property of probability forecasts; this is a regression model. The equivalent regression diagnostics are the residual plots written by the evaluation stage.

## Governance trail

- Approvals on file: `feature_approval`, `model_approval`
- Audit events recorded: 34
- Training runs logged: 6
- Lineage nodes: 4

## Limitations

- Every number here is measured on one dataset and one split. They are estimates with sampling error, not guarantees.
- The drivers and importances are associations the model found. None of them is evidence that changing the input would change the outcome.
- Nothing in this pipeline monitors the model after this point. These numbers describe the data as it was when the model was trained; they say nothing about how it will behave once that distribution moves.

## Sources

Every artifact this card looked for. A section reporting "not assessed" next to a `found` source means the artifact exists but is empty or unreadable.

- `evaluation_report`: found
- `drivers`: found
- `permutation_importance`: found
- `subgroup_fairness`: found
- `calibration`: found
- `feature_approval`: found
- `model_approval`: found
- `audit_log`: found
- `lineage`: found
- `experiments`: found
