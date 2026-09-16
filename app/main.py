"""
app.main
========

PURPOSE
-------
Streamlit entry point for the AI Data Science Workbench GUI (Milestone 3).

    streamlit run app/main.py

This is a READ-ONLY dashboard over the artifacts every CLI stage
(``python main.py <stage>``) already writes to ``data/``, ``results/``,
``artifacts/``, ``governance/``, and ``models/``. The one exception is the
"Run stage" section on the Pipeline page, which shells out to
``python main.py pipeline ...`` rather than mutating state in-process.

Use the sidebar to navigate:
    1 Dashboard | 2 Data | 3 EDA | 4 Features | 5 Modeling | 6 Evaluation |
    7 Explainability | 8 Governance | 9 Reports
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app import project_state  # noqa: E402

_PROJECT = project_state.apply_project()  # activate selected project BEFORE path import

from app.common import PAGE_ICON, safe_read_json  # noqa: E402
from config.paths import FINAL_MODEL_SELECTION_PATH  # noqa: E402

st.set_page_config(
    page_title="AI Data Science Workbench",
    page_icon=PAGE_ICON,
    layout="wide",
)
project_state.render_selector(_PROJECT)

st.title("AI Data Science Workbench")
st.caption(
    "Machine Learning (23BA045E), Amrita School of Business — "
    f"active project: {project_state.project_label(_PROJECT)}"
)

st.markdown(
    """
This is a **read-only** dashboard over the artifacts the CLI pipeline writes
to disk. It never mutates project state itself, with one exception: the
**Pipeline** page's "Run stage" control, which shells out to
`python main.py pipeline ...` and streams its output.

Use the sidebar to open a page:

| Page | What it shows |
|---|---|
| **1 Dashboard** | Project KPIs, champion model, pipeline stage status |
| **2 Data** | Raw/processed/split previews, semantic schema, validation report |
| **3 EDA** | Data-understanding tables + the EDA figure gallery |
| **4 Features** | Feature plan and its governance approval status |
| **5 Modeling** | Model catalog, approvals, registry metrics, experiment log |
| **6 Evaluation** | Model comparison, champion card, residual diagnostics |
| **7 Explainability** | Standardized coefficients, top drivers, executive briefing |
| **8 Governance** | Audit trail, data/model lineage, standing decisions |
| **9 Reports** | The self-contained HTML project report, ready to download |

Run the pipeline from a terminal first if a page reports a stage as not run yet:

```
python main.py all
```
"""
)

final_selection = safe_read_json(FINAL_MODEL_SELECTION_PATH)
if final_selection:
    st.success(project_state.champion_summary_line(final_selection))
else:
    run_hint = (
        "WORKBENCH_PROJECT=churn python main.py churn-all"
        if _PROJECT == "churn"
        else "python main.py all"
    )
    st.warning(f"No champion model selected yet for **{_PROJECT}**. Run `{run_hint}`.")
