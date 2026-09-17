---
title: AI Data Science Workbench
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: A governed ML pipeline - upload a CSV, get an audited model
---

# AI Data Science Workbench

A governed, task-agnostic machine learning pipeline. Upload a CSV and it runs
twelve stages end to end — profiling, leakage-safe preparation, feature
proposal, an approval gate, training with hold-out tuning, champion selection,
explainability, and a trust audit — then hands back a model card that states
what it does not know.

**The claim is not that the models are accurate.** It is that each result is
reproducible, each model choice is documented, leakage is structurally
prevented, and each model's own bias is measured and published beside its
accuracy.

## Try it

Two demo projects are pre-built, so there is something to browse immediately:

| Project | Task | What to look at |
|---|---|---|
| **Telco Churn Demo** | Classification | The Trust page — calibration, subgroup fairness, selection amplification |
| **California Housing Demo** | Regression | Explainability — coefficients converted back to dollar effects |

Switch between them with the sidebar selector. Then use **New Project** to
upload your own CSV: the detector guesses the target and task, shows you its
reasoning, and lets you override both before anything trains.

The **Retention Simulator** turns predicted churn probabilities into a campaign
business case. It multiplies a probability by a cost, which is precisely why
the calibration numbers on the Trust page matter more than the AUC.

## What you are looking at

- **Twelve governed stages**, each declaring its inputs and outputs. The
  orchestrator refuses to run a stage whose prerequisites are missing.
- **An approval gate.** `require_model_approved()` raises before any estimator
  is fitted, so an unreviewed model cannot reach the results.
- **Structural leakage prevention.** Every fitted statistic is learned from the
  training split only, and the fitted preprocessor is pickled *inside* the model
  bundle so validation and test transform identically. `split_x_y()` raises if
  the target appears among the predictors — leakage becomes a crash, not a
  flattering score.
- **A trust audit.** Subgroup fairness, selection amplification (the part of a
  gap the model *added*, rather than the part that was really there),
  permutation importance measured against the selection metric rather than
  impurity, and probability calibration.
- **A governance trail.** An append-only audit log, a lineage graph hashing
  every input and output file, and a model registry naming the champion.

## Honest notes

The demo projects are built from **600-row samples**, not the full datasets, so
their scores are not the project's headline results. The graded runs — Ridge at
test R² 0.667 on California housing, Logistic Regression at test ROC-AUC 0.8448
on Telco churn — live in the repository and on the report site.

Nothing here is faked: the demos are produced at image build time by running the
same pipeline the code implements.

## Source

- **Repository:** https://github.com/Bala-Shunmugam-M/ai-data-science-workbench
- **Report site:** https://bala-shunmugam-m.github.io/ai-data-science-workbench/

Built as an MBA Machine Learning project at Amrita School of Business.
