# Questions for the Professor

Four open items flagged during the build (see `docs/PLAN.md` §7), phrased
ready to ask in office hours or before the Model Defence.

---

### 1. Is a GUI expected, or is a CLI-driven governed pipeline sufficient?

We built a Streamlit dashboard (`streamlit run app/main.py`) on top of the
governed CLI pipeline for Milestone 3, on the assumption that a visual
walkthrough is useful for the Model Defence presentation. Could you confirm
whether a GUI is actually expected for grading, or whether the CLI plus the
generated artifacts (`artifacts/reports/project_report.html`, the governance
files, the evaluation report) would have been sufficient on its own? If the
GUI is not required for grading, we'd like to know so we can calibrate how
much presentation time to spend on it versus the modelling/governance
content.

### 2. Should the AutoML comparison (session 21) be included in the deliverable?

The current build is restricted to the linear family (LinearRegression,
Ridge, Lasso) by team decision, on the grounds that coefficient
interpretability best serves the managerial-report requirement. Session 21
covered an AutoML comparison sweep - should that broader model comparison be
included as part of the graded project deliverable (e.g. as an appendix
showing where linear models stand relative to a wider search), or is it
acceptable to reference it as a possible extension without including the
run itself?

### 3. What is the required format/length of the team project report, and what citation style should we use?

We produced a self-contained HTML report (`artifacts/reports/project_report.html`)
assembling every stage's artifacts, plus a separate one-page executive
briefing (`artifacts/reports/executive_briefing.md`) written for a
non-technical audience. Is this the expected report format, or is a
separate written document (Word/PDF, a specific page count) required for
submission? Relatedly, `docs/methodology.md` cites ISLR, Géron, and CRISP-DM
informally inline - should the final submission use a specific citation
style (APA, etc.) and a formal references section?

### 4. Are model families beyond regression required for the final submission?

The current build is linear-family only (LinearRegression, Ridge, Lasso) by
team decision, justified by interpretability for the managerial audience.
The model catalogue (`src/model_proposal/model_catalog.py`) is deliberately
structured so tree-based or ensemble models could be added without changing
the governance, training, or evaluation architecture. Is the linear-only
scope acceptable for the final submission as-is, or is at least one
non-linear model (e.g. a decision tree or random forest, for a direct
interpretability-vs-accuracy comparison in the report) expected?
