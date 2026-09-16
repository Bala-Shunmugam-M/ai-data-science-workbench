# ML pipeline traps — a checklist for anyone (human or AI) changing this code

Every item below was a **real defect in this repository**, found by audit rather
than by a failing test. They share one property: each produces a plausible,
confident, wrong answer instead of an error. None of them crashed, and none was
caught by the 49 tests that existed at the time.

They are written generically, because none is specific to this project.

---

## 1. NaN wins a `max()` / `min()` selection

Every comparison against NaN is `False`, so `max()` never replaces a NaN once it
is the running best.

```python
>>> max([{'m': float('nan'), 'n': 'BAD'}, {'m': 0.55}, {'m': 0.95, 'n': 'BEST'}],
...     key=lambda r: r['m'])
{'m': nan, 'n': 'BAD'}      # not BEST
```

Metrics here return NaN **by design** when undefined — `roc_auc` when a class is
absent from a split, `mape` when an actual is zero — so this is reachable on any
small or imbalanced dataset.

**It is ordering-dependent.** NaN first breaks; NaN in the middle works. That is
why it survives casual testing.

The same bug appears in a comparator: `is_better(metric, 0.95, nan)` is `False`
under a plain `>`, so a real score can never displace a NaN incumbent and the
first candidate evaluated wins forever.

> Guard with `value is None or math.isnan(value)`. Guarding `None` alone is not
> enough. See `src/model_training/metrics.py::is_better` and
> `src/model_evaluation/evaluator.py::select_champion`, plus
> `tests/test_metric_selection.py`.

## 2. Task must be re-derived when the target changes

Whether a problem is regression or classification is a property of the **target
column**, not of the file. A UI that detects the task once, then lets the user
change the target, will carry the stale task forward.

Best case it fails loudly (stratifying a continuous column). Worst case a numeric
column with few distinct values trains a classifier and reports a meaningless
ROC-AUC that looks entirely normal.

> `POST /api/detect` accepts `{fileId, target}` to re-scan without re-uploading;
> the confirm step calls it on every target change.

## 3. Stratified splits need more rows per class than the obvious minimum

sklearn requires 2 rows per class. For a **two-stage** 70/15/15 split that is not
the binding constraint — the second split is. A class of size `c` must satisfy:

```
c * TEMP_TEST_SIZE >= 2      # ~7 rows per class at TEMP_TEST_SIZE = 0.30
```

An earlier guard here hard-coded 4, passed the first split, and let sklearn raise
on the second — the exact failure it existed to prevent. Derive the threshold
from the split constants; do not guess it.

> `src/preprocessing/splitter.py::_require_stratifiable`. Name the column and the
> offending classes: sklearn's own message names neither.

## 4. Degenerate columns reach the estimator

- An **all-null column**: median/mean/std are all NaN, and a `std > 0` guard does
  not catch it because `NaN > 0` is `False`. The NaN flows into the design matrix
  and surfaces as `Input X contains NaN`, far from the cause.
- A **constant target**: detection may emit a warning, but a warning stops
  nothing. Registration succeeded, prep succeeded, and the run died inside
  `.fit()`.

> Drop predictors with `nunique(dropna=True) == 0`; refuse a single-class target
> at registration, where the message can name the column.

## 5. `positive_class` silently binarises a multiclass target

Applying a positive-class label to a 3+ class target merges every other class
into "rest", while the stored `n_classes` still says multiclass. The metrics and
the metadata then describe two different problems, and nothing errors.

> Reject the flag unless the target is genuinely binary, and check the class
> actually appears in the data.

## 6. Module-level path constants freeze the active project

`config/paths.py` reads `WORKBENCH_PROJECT` **at import time** into module-level
constants. Setting the variable after importing anything that pulls in
`config.paths` is a silent no-op, and every subsequent write targets the wrong
workspace.

> Compute the project identifier first, set the env var, and only then import
> path-dependent modules. `webapi/bridge.py` documents and follows this order;
> it is the reason the slug is computed locally rather than imported.

## 7. Two callers of one pipeline will drift

`main.py autorun` and the web upload path each kept their own list of calls. They
drifted: the web path generated the driver table and the HTML report; the CLI
command did not. Same dataset, different deliverables depending on launch method.

> Define the phases once (`src/pipelines/generic_pipeline.py`:
> `run_prep` / `run_modelling` / `run_insights` / `run_full`) and compose them.
> Add new work to a phase, never to a caller.

## 8. Generated prose is domain-bound; do not reuse it

`src/explainability/interpretation.py` writes an executive briefing hard-coded to
California house prices — "roughly 20,600 census blocks", coastal caveats, dollar
amounts. Running it on an uploaded dataset would emit fluent, confident analysis
about the wrong domain.

> `src/explainability/drivers.py` is the dataset-agnostic sibling: it reuses the
> parts that are true for any tabular model (coefficients, importances, feature
> names) and states nothing about what the target *means*.

---

## Testing lessons

- **A test that mirrors the implementation tests nothing.** The first version of
  `tests/test_metric_selection.py` re-implemented the selection logic; it would
  have passed forever while the real code regressed. `select_champion` was
  extracted to module level so the test binds to the shipping function.
- **Prove a regression test fails.** Revert the fix, confirm the test goes red,
  restore. 10 of 20 failed against the original bug — that is what makes them
  worth keeping.
- **Metrics that can return NaN need a matching selection test.** The failure is
  invisible: it produces a plausible champion and a plausible number.
