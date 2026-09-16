# Executive Briefing: Customer Churn (Telco)

*Prepared 2026-08-02T13:46:09+00:00 - champion model: LogisticRegression v001*

## What was built

A classification model predicting **Churn**. Candidate models were compared on a validation split held out from training; the best was then measured once on a test split that was never used for tuning or selection, so the reported accuracy reflects unseen data.

## How well it predicts

- accuracy: 0.8078
- precision: 0.6498
- recall: 0.5964
- f1: 0.6220
- roc_auc: 0.8455

## Strongest drivers (by effect on the prediction)

1. **tenure** - lowers the prediction (-1.342).
2. **MonthlyCharges** - lowers the prediction (-0.7961).
3. **Contract_Two year** - lowers the prediction (-0.7606).
4. **InternetService_Fiber optic** - raises the prediction (0.6688).
5. **InternetService_DSL** - lowers the prediction (-0.6637).

## How to read this

These are standardized coefficients: each is the change in the prediction for a one-standard-deviation move in that input, holding the others fixed. A positive value pushes the prediction up.

## Limits

- This briefing is generated from the model and its metrics alone. It makes no claim about what the target means in your domain, nor about cause and effect - the drivers are associations the model found, not levers proven to change the outcome.
- Correlated inputs share credit unpredictably; read the top drivers as a group rather than as independent factors.
