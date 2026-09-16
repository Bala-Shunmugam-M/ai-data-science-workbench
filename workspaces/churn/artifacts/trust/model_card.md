# Model Card: Customer Churn (Telco)

*Generated 2026-08-09T14:23:51+00:00 — LogisticRegression v001*

## Model details

- Family: **LogisticRegression**
- Version: **v001**, registered 2026-07-26T11:06:26+00:00
- Task: **classification**, target **Churn**
- Hyperparameters: `max_iter=2000`, `solver=liblinear`, `C=2.154434690031882`
- Training-data fingerprint: `41cdaed48c20190eb70e9afcce0138d763889304479b0bc696d14bee58bb0fb5`

## Intended use

**Not authored.** This card records what the model is and how it behaves. It does not state which decisions the model may be used for, because that is not derivable from any artifact in this pipeline — it is a judgement about the domain and the consequences of a wrong prediction. Whoever deploys this model owns that statement.

## Data

- Train: 4,930 rows
- Validation: 1,056 rows
- Test: 1,057 rows (used once, after the champion was chosen)

## Performance

Champion selected by **roc_auc** on the validation split, from 6 candidates.

**Validation:**
- accuracy: 0.8078
- precision: 0.6498
- recall: 0.5964
- f1: 0.6220
- roc_auc: 0.8455

**Test (held out until the champion was fixed):**
- accuracy: 0.8108
- precision: 0.6866
- recall: 0.5302
- f1: 0.5984
- roc_auc: 0.8448

> 6 candidate model(s) were compared on the untouched validation split using ROC-AUC as the selection metric (higher is better).
  - LogisticRegression (v001): ROC-AUC 0.845, ACCURACY 0.808, PRECISION 0.650, RECALL 0.596, F1 0.622.
  - LogisticRegression (v002): ROC-AUC 0.845, ACCURACY 0.808, PRECISION 0.650, RECALL 0.596, F1 0.622.
  - RandomForestClassifier (v002): ROC-AUC 0.845, ACCURACY 0.736, PRECISION 0.501, RECALL 0.836, F1 0.627.
  - RandomForestClassifier (v001): ROC-AUC 0.845, ACCURACY 0.736, PRECISION 0.501, RECALL 0.836, F1 0.627.
  - DecisionTreeClassifier (v001): ROC-AUC 0.839, ACCURACY 0.750, PRECISION 0.519, RECALL 0.775, F1 0.622.
  - DecisionTreeClassifier (v002): ROC-AUC 0.839, ACCURACY 0.750, PRECISION 0.519, RECALL 0.775, F1 0.622.
LogisticRegression (v001) had the best validation ROC-AUC and was promoted to champion.
The champion was then evaluated once on the held-out test split: ROC-AUC 0.845, ACCURACY 0.811, PRECISION 0.687, RECALL 0.530, F1 0.598. The test split was used only for this single final measurement, never for tuning or selection.

## What the model depends on

**From the model's own internals** (effect on the prediction):

1. `tenure` — -1.342 (decreases the prediction)
2. `MonthlyCharges` — -0.7961 (decreases the prediction)
3. `Contract_Two year` — -0.7606 (decreases the prediction)
4. `InternetService_Fiber optic` — 0.6688 (increases the prediction)
5. `InternetService_DSL` — -0.6637 (decreases the prediction)
6. `TotalCharges` — 0.6633 (increases the prediction)
7. `Contract_Month-to-month` — 0.5756 (increases the prediction)
8. `PaperlessBilling_No` — -0.3761 (decreases the prediction)

**By permutation** (drop in roc_auc when a column is shuffled, 10 shuffles, measured on the test split):

1. `tenure` — 0.1915 ± 0.0175
2. `MonthlyCharges` — 0.04995 ± 0.00707
3. `TotalCharges` — 0.02011 ± 0.00603
4. `InternetService_DSL` — 0.01258 ± 0.00338
5. `Contract_Two year` — 0.0115 ± 0.00328
6. `InternetService_Fiber optic` — 0.01096 ± 0.00379
7. `Contract_Month-to-month` — 0.007747 ± 0.00321
8. `SeniorCitizen_0` — 0.002481 ± 0.000899

18 of 46 features show a dependence larger than the measurement's own noise.

Where the two lists disagree, the disagreement is the finding: a feature with a large coefficient and no permutation importance is one the model cannot actually use on unseen data.

## Subgroup performance

Measured on the test split (1,057 rows). Groups smaller than 30 rows are reported but excluded from the gap arithmetic.

### `InternetService` — 3 groups, 3 compared

- base rate gap: **0.3145**
- selection rate gap: **0.4354**
- selection rate ratio: **0** — **one group is never predicted positive at all.** Whatever the positive prediction triggers, that group cannot receive it
- tpr gap: **0.7158**
- fpr gap: **0.2364**
- precision gap: **0.03771**
- accuracy gap: **0.1558**
- selection amplification: **+0.1208** — the model spreads these groups further apart than their actual outcomes are

### `OnlineSecurity` — 3 groups, 3 compared

- base rate gap: **0.3142**
- selection rate gap: **0.3973**
- selection rate ratio: **0** — **one group is never predicted positive at all.** Whatever the positive prediction triggers, that group cannot receive it
- tpr gap: **0.662**
- fpr gap: **0.2098**
- precision gap: **0.09082**
- accuracy gap: **0.1625**
- selection amplification: **+0.08314** — the model spreads these groups further apart than their actual outcomes are

### `OnlineBackup` — 3 groups, 3 compared

- base rate gap: **0.3099**
- selection rate gap: **0.3823**
- selection rate ratio: **0** — **one group is never predicted positive at all.** Whatever the positive prediction triggers, that group cannot receive it
- tpr gap: **0.6474**
- fpr gap: **0.1978**
- precision gap: **0.04491**
- accuracy gap: **0.1609**
- selection amplification: **+0.07234** — the model spreads these groups further apart than their actual outcomes are

### `Contract` — 3 groups, 3 compared

- base rate gap: **0.4053**
- selection rate gap: **0.3794**
- selection rate ratio: **0** — **one group is never predicted positive at all.** Whatever the positive prediction triggers, that group cannot receive it
- tpr gap: **0.6132**
- fpr gap: **0.2067**
- accuracy gap: **0.2637**
- Not measurable here: precision gap — fewer than two groups had both a defined value and enough rows.
- selection amplification: **-0.02592** — the model under-separates groups that genuinely differ

### `TechSupport` — 3 groups, 3 compared

- base rate gap: **0.3048**
- selection rate gap: **0.3792**
- selection rate ratio: **0** — **one group is never predicted positive at all.** Whatever the positive prediction triggers, that group cannot receive it
- tpr gap: **0.6514**
- fpr gap: **0.1938**
- precision gap: **0.1576**
- accuracy gap: **0.1561**
- selection amplification: **+0.0744** — the model spreads these groups further apart than their actual outcomes are

### `PaymentMethod` — 4 groups, 4 compared

- base rate gap: **0.309**
- selection rate gap: **0.378**
- selection rate ratio: **0.1703** — below the conventional four-fifths (80%) reference point for adverse impact
- tpr gap: **0.4522**
- fpr gap: **0.2258**
- precision gap: **0.1984**
- accuracy gap: **0.1572**
- selection amplification: **+0.06899** — the model spreads these groups further apart than their actual outcomes are

### `DeviceProtection` — 3 groups, 3 compared

- base rate gap: **0.2751**
- selection rate gap: **0.3581**
- selection rate ratio: **0** — **one group is never predicted positive at all.** Whatever the positive prediction triggers, that group cannot receive it
- tpr gap: **0.6453**
- fpr gap: **0.1853**
- precision gap: **0.04015**
- accuracy gap: **0.1485**
- selection amplification: **+0.08295** — the model spreads these groups further apart than their actual outcomes are

### `StreamingTV` — 3 groups, 3 compared

- base rate gap: **0.2398**
- selection rate gap: **0.2854**
- selection rate ratio: **0** — **one group is never predicted positive at all.** Whatever the positive prediction triggers, that group cannot receive it
- tpr gap: **0.5906**
- fpr gap: **0.128**
- precision gap: **0.04096**
- accuracy gap: **0.1233**
- selection amplification: **+0.04562** — the model spreads these groups further apart than their actual outcomes are

### `StreamingMovies` — 3 groups, 3 compared

- base rate gap: **0.2352**
- selection rate gap: **0.2766**
- selection rate ratio: **0** — **one group is never predicted positive at all.** Whatever the positive prediction triggers, that group cannot receive it
- tpr gap: **0.5878**
- fpr gap: **0.1231**
- precision gap: **0.06048**
- accuracy gap: **0.1173**
- selection amplification: **+0.04146** — the model spreads these groups further apart than their actual outcomes are

### `SeniorCitizen` — 2 groups, 2 compared

- base rate gap: **0.2066**
- selection rate gap: **0.2268**
- selection rate ratio: **0.4296** — below the conventional four-fifths (80%) reference point for adverse impact
- tpr gap: **0.1386**
- fpr gap: **0.1397**
- precision gap: **0.02339**
- accuracy gap: **0.1065**
- selection amplification: **+0.02014** — the model spreads these groups further apart than their actual outcomes are

### `PaperlessBilling` — 2 groups, 2 compared

- base rate gap: **0.1456**
- selection rate gap: **0.2104**
- selection rate ratio: **0.2864** — below the conventional four-fifths (80%) reference point for adverse impact
- tpr gap: **0.3355**
- fpr gap: **0.09431**
- precision gap: **0.06674**
- accuracy gap: **0.05087**
- selection amplification: **+0.06483** — the model spreads these groups further apart than their actual outcomes are

### `Dependents` — 2 groups, 2 compared

- base rate gap: **0.1565**
- selection rate gap: **0.1877**
- selection rate ratio: **0.2881** — below the conventional four-fifths (80%) reference point for adverse impact
- tpr gap: **0.2731**
- fpr gap: **0.08575**
- precision gap: **0.05271**
- accuracy gap: **0.07613**
- selection amplification: **+0.03124** — the model spreads these groups further apart than their actual outcomes are

### `Partner` — 2 groups, 2 compared

- base rate gap: **0.1264**
- selection rate gap: **0.1792**
- selection rate ratio: **0.3868** — below the conventional four-fifths (80%) reference point for adverse impact
- tpr gap: **0.178**
- fpr gap: **0.1082**
- precision gap: **0.07471**
- accuracy gap: **0.08358**
- selection amplification: **+0.05279** — the model spreads these groups further apart than their actual outcomes are

### `MultipleLines` — 3 groups, 3 compared

- base rate gap: **0.0148**
- selection rate gap: **0.09826**
- selection rate ratio: **0.6115** — below the conventional four-fifths (80%) reference point for adverse impact
- tpr gap: **0.1588**
- fpr gap: **0.08685**
- precision gap: **0.1969**
- accuracy gap: **0.04196**
- selection amplification: **+0.08346** — the model spreads these groups further apart than their actual outcomes are

### `PhoneService` — 2 groups, 2 compared

- base rate gap: **0.002416**
- selection rate gap: **0.05578**
- selection rate ratio: **0.7349** — below the conventional four-fifths (80%) reference point for adverse impact
- tpr gap: **0.03333**
- fpr gap: **0.06545**
- precision gap: **0.1934**
- accuracy gap: **0.03807**
- selection amplification: **+0.05336** — the model spreads these groups further apart than their actual outcomes are

### `gender` — 2 groups, 2 compared

- base rate gap: **0.04834**
- selection rate gap: **0.02777**
- selection rate ratio: **0.8736**
- tpr gap: **0.0991**
- fpr gap: **0.04475**
- precision gap: **0.09621**
- accuracy gap: **0.07753**
- selection amplification: **-0.02056** — the model under-separates groups that genuinely differ

A gap is not a verdict. Whether one is acceptable depends on the domain, the decision the prediction feeds, and the law that applies — none of which this pipeline knows.

Every low-cardinality column is audited, including columns the model uses as features on purpose. A large gap across such a column is usually the model being accurate; `selection amplification` is the part it adds beyond the difference that is genuinely there.

## Probability calibration

- Brier score: **0.1362**
- Expected calibration error (ECE): **0.0301** — about 3 points off on average
- Worst band (MCE): **0.1806**
- Mean predicted probability 0.2566 vs actual base rate 0.2658 (bias -0.0093)

Calibration is separate from ranking quality. ROC-AUC measures whether the model orders cases correctly; this measures whether its probabilities mean what they say. Anything that multiplies a predicted probability by a cost depends on this number, not on the AUC.

## Governance trail

- Approvals on file: `feature_approval`, `model_approval`
- Audit events recorded: 22
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
