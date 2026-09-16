# webapi — Python bridge

A small argparse CLI that a Node.js process spawns so the Next.js web app
(`webapp/`) can drive the existing workbench pipeline and stream real
progress, without any changes to `main.py`, `src/`, or the Streamlit app.

**stdout carries JSON and nothing else.** All logging, warnings, and any
stray `print()` from third-party code (sklearn, matplotlib, pandas) are kept
off stdout; `python -m json.tool` (or a Node line-by-line `JSON.parse`) can
consume it directly.

Run with the repo's global interpreter (no venv) from the repo root:

```bash
python webapi/bridge.py <command> ...
```

## `detect`

```bash
python webapi/bridge.py detect --csv workspaces/california-housing/data/raw/housing.csv
python webapi/bridge.py detect --csv path/to/data.csv --target price
```

Reads the CSV, runs `src.automl.detect.detect(df, target=...)`, and prints
**one** JSON object to stdout:

```json
{
  "rows": 20640,
  "columns": 10,
  "detection": {
    "target": "median_house_value", "task": "regression", "positiveClass": null,
    "categoricalColumns": ["ocean_proximity"], "dropColumns": [],
    "selectionMetric": "rmse", "nClasses": null, "warnings": []
  },
  "profile": [
    {"name": "longitude", "dtype": "float64", "missing": 0, "missingPct": 0.0, "unique": 844, "example": "-122.23, -122.22"}
  ],
  "preview": {"columns": ["longitude", "..."], "rows": [[-122.23, "..."], ["..."]]}
}
```

`--target` is optional; omit it to let `detect()` guess the target column.

## `analyze`

```bash
python webapi/bridge.py analyze --csv path/to/data.csv --name "My Dataset" \
  --target price --task regression
```

Streams **one JSON object per line**, flushed immediately, one line per
stage transition, covering `extract`, `analytics`, `insights`:

```
{"stage":"extract","status":"started"}
{"stage":"extract","status":"done","slug":"my-dataset","elapsedMs":1234}
{"stage":"analytics","status":"started"}
{"stage":"analytics","status":"done","slug":"my-dataset","elapsedMs":9876}
{"stage":"insights","status":"started"}
{"stage":"insights","status":"done","slug":"my-dataset","elapsedMs":543}
```

`slug` is set by `register_project(...)` during `extract` and included on
every event from that point on, so the client can navigate to
`/results/<slug>` as soon as it has one. On failure the current stage emits
`{"stage":..., "status":"failed", "message":"..."}`, a traceback is printed to
stderr, and the process exits with code 1 — the remaining stages never run.

Work performed, in order:

1. **extract** — re-detects the dataset from the raw CLI args, applies the
   `--task`/`--positive-class` overrides, calls `register_project(...)`, sets
   `WORKBENCH_PROJECT` in the environment, then `generic_pipeline.run_prep()`.
2. **analytics** — `training_pipeline.run()` then `evaluation_pipeline.run()`.
3. **insights** — `src.automl.report.run()`.

`--positive-class` only applies when `--task classification`.

## Notes

- `WORKBENCH_PROJECT` must be set in the environment *before* any workbench
  module that reads `config.paths` is imported — that module freezes the
  active workspace path at import time. `bridge.py` computes the project
  slug and sets the env var itself before importing `register_project` or
  any pipeline module; see the docstring in `bridge.py` for why.
- Non-zero exit code + stderr traceback is the general error signal for both
  commands; `analyze` additionally reports which named stage failed.
