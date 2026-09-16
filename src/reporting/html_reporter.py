"""
src.reporting.html_reporter
===========================

PURPOSE
-------
Render the report context from :mod:`src.reporting.report_generator` into a
single self-contained, print-friendly HTML file. All figures are base64-embedded
so the report is one portable file with no external dependencies, and the CSS is
minimal and inline.

PIPELINE POSITION
-----------------
    explainability -> [reporting] -> deliverable

OUTPUTS
-------
    artifacts/reports/project_report.html
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

from config.paths import PROJECT_REPORT_PATH, ensure_dir
from src.reporting.report_generator import build_report_context
from src.utils.file_utils import save_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       color: #1a1a1a; line-height: 1.5; max-width: 960px; margin: 0 auto;
       padding: 2rem 1.5rem; background: #fff; }
h1 { font-size: 1.9rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.3rem; margin-top: 2.2rem; border-bottom: 2px solid #2a4d7a;
     padding-bottom: 0.3rem; color: #2a4d7a; }
h3 { font-size: 1.05rem; margin-top: 1.3rem; color: #333; }
.subtitle { color: #666; font-size: 1rem; margin-top: 0; }
.meta { color: #888; font-size: 0.85rem; }
table { border-collapse: collapse; width: 100%; margin: 0.8rem 0; font-size: 0.85rem; }
th, td { border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }
th { background: #eef2f8; }
tr:nth-child(even) td { background: #f8f9fb; }
figure { margin: 1rem 0; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid #ddd; }
figcaption { font-size: 0.82rem; color: #666; margin-top: 0.3rem; }
.badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
         font-size: 0.8rem; font-weight: 600; }
.badge-ok { background: #dff3e4; color: #1d6b34; }
.badge-warn { background: #fff3d6; color: #8a6100; }
.card { background: #f6f8fb; border: 1px solid #dce3ee; border-radius: 6px;
        padding: 0.8rem 1rem; margin: 0.8rem 0; }
.pos { color: #1d6b34; } .neg { color: #b1332f; }
pre { white-space: pre-wrap; background: #f6f8fb; padding: 0.8rem; border-radius: 6px;
      font-size: 0.82rem; overflow-x: auto; }
.kpi-row { display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 0.8rem 0; }
.kpi { flex: 1 1 130px; background: #2a4d7a; color: #fff; border-radius: 6px;
       padding: 0.7rem 0.9rem; }
.kpi .v { font-size: 1.3rem; font-weight: 700; }
.kpi .l { font-size: 0.75rem; opacity: 0.85; }
@media print { body { max-width: none; } h2 { page-break-after: avoid; }
                figure, table { page-break-inside: avoid; } }
"""


def _b64_image(path: str | Path, caption: str) -> str:
    """Return a ``<figure>`` with a base64-embedded PNG, or a placeholder note."""

    p = Path(path)
    if not p.exists():
        return f'<p class="meta">[figure not available: {html.escape(str(p.name))}]</p>'
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return (
        f'<figure><img alt="{html.escape(caption)}" '
        f'src="data:image/png;base64,{data}"/>'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Render a list of dicts as an HTML table using (key, header) column specs."""

    if not rows:
        return '<p class="meta">No data available.</p>'
    head = "".join(f"<th>{_esc(header)}</th>" for _, header in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_esc(row.get(key))}</td>" for key, _ in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _fmt_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    parts = []
    for key in ("rmse", "mae", "r2", "mape"):
        if key in metrics and metrics[key] is not None:
            val = metrics[key]
            parts.append(f"{key.upper()}={val:,.2f}" if isinstance(val, (int, float)) else f"{key.upper()}={val}")
    return ", ".join(parts)


def render_html(context: dict[str, Any]) -> str:
    """Render the full report context into one HTML document string."""

    ov = context["overview"]
    parts: list[str] = []
    parts.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append(f"<title>{_esc(ov['title'])}</title><style>{_CSS}</style></head><body>")

    # --- Header ---
    parts.append(f"<h1>{_esc(ov['title'])}</h1>")
    parts.append(f"<p class='subtitle'>{_esc(ov['subtitle'])}</p>")
    parts.append(f"<p class='meta'>Generated {_esc(context['generated_at'])} - "
                 f"target: <code>{_esc(context['target_column'])}</code></p>")

    # --- 1. Project overview ---
    parts.append("<h2>1. Project overview</h2>")
    parts.append(f"<div class='card'>{_esc(ov['scope'])}</div>")

    # --- 2. Data validation summary ---
    val = context["validation"]
    status = val.get("status", "n/a")
    badge = "badge-ok" if "PASSED" in status.upper() and "WARN" not in status.upper() else "badge-warn"
    parts.append("<h2>2. Data validation summary</h2>")
    parts.append(f"<p>Status: <span class='badge {badge}'>{_esc(status)}</span></p>")
    if val.get("summary"):
        parts.append(_table([val["summary"]], [(k, k) for k in val["summary"].keys()]))

    # --- 3. Key EDA figures ---
    parts.append("<h2>3. Key EDA figures</h2>")
    if context["eda_figures"]:
        for fig in context["eda_figures"]:
            parts.append(_b64_image(fig["path"], fig["caption"]))
    else:
        parts.append('<p class="meta">No EDA figures found.</p>')

    # --- 4. Feature plan and approvals ---
    feats = context["features"]
    parts.append("<h2>4. Feature plan and approvals</h2>")
    parts.append(f"<p>{_esc(feats['total_proposed'])} features proposed; "
                 f"{len(feats['approved_features'])} approved for modelling.</p>")
    parts.append(_table(
        feats["plan"],
        [("feature_name", "Feature"), ("formula", "Formula"),
         ("business_meaning", "Business meaning"),
         ("automatic", "Automatic"), ("approved", "Approved")],
    ))

    # --- 5. Model comparison ---
    models = context["models"]
    parts.append("<h2>5. Model proposal, approval and comparison</h2>")
    parts.append(f"<p>Recommended: {_esc(', '.join(models['recommended_models']))}. "
                 f"Approved: {_esc(', '.join(models['approved_models']))}.</p>")
    parts.append("<h3>Validation comparison (selection by RMSE)</h3>")
    parts.append(_table(
        models["comparison"],
        [("rank", "Rank"), ("model", "Model"), ("version", "Version"),
         ("validation_rmse", "Val RMSE"), ("validation_mae", "Val MAE"),
         ("validation_r2", "Val R2")],
    ))

    # --- 6. Champion evaluation ---
    champ = models.get("champion", {})
    parts.append("<h2>6. Champion evaluation (untouched test set)</h2>")
    if champ:
        test_m = champ.get("test_metrics", {})
        val_m = champ.get("validation_metrics", {})
        parts.append(f"<p>Champion: <strong>{_esc(champ.get('name'))} "
                     f"{_esc(champ.get('version'))}</strong></p>")
        parts.append("<div class='kpi-row'>")
        for label, value in (
            ("Test RMSE", test_m.get("rmse")),
            ("Test MAE", test_m.get("mae")),
            ("Test R2", test_m.get("r2")),
            ("Val RMSE", val_m.get("rmse")),
        ):
            vtxt = f"{value:,.2f}" if isinstance(value, (int, float)) else "n/a"
            parts.append(f"<div class='kpi'><div class='v'>{vtxt}</div>"
                         f"<div class='l'>{_esc(label)}</div></div>")
        parts.append("</div>")
    if models.get("selection_narrative"):
        parts.append(f"<pre>{_esc(models['selection_narrative'])}</pre>")
    for _, fig_path in (models.get("residual_figures") or {}).items():
        parts.append(_b64_image(fig_path, Path(fig_path).stem.replace("_", " ").title()))

    # --- 7. Coefficient interpretation ---
    coeffs = context["coefficients"]
    parts.append("<h2>7. Coefficient interpretation</h2>")
    if coeffs.get("top_drivers_png"):
        parts.append(_b64_image(coeffs["top_drivers_png"], "Top standardized value drivers."))
    parts.append(_table(
        coeffs["coefficients"][:15],
        [("rank", "Rank"), ("feature", "Feature"), ("coefficient", "Coefficient"),
         ("direction", "Direction"), ("meaning", "Plain-English meaning")],
    ))

    # --- 8. Governance summary ---
    gov = context["governance"]
    audit = gov.get("audit_summary", {})
    lineage = gov.get("lineage", {})
    parts.append("<h2>8. Governance summary</h2>")
    parts.append("<div class='kpi-row'>")
    parts.append(f"<div class='kpi'><div class='v'>{_esc(audit.get('total_events', 0))}</div>"
                 "<div class='l'>Audit events</div></div>")
    parts.append(f"<div class='kpi'><div class='v'>{_esc(lineage.get('node_count', 0))}</div>"
                 "<div class='l'>Lineage nodes</div></div>")
    parts.append(f"<div class='kpi'><div class='v'>{len(gov.get('decisions', []))}</div>"
                 "<div class='l'>Recorded decisions</div></div>")
    parts.append("</div>")
    parts.append("<h3>Audit events by type</h3>")
    parts.append(_table(
        [{"event_type": k, "count": v} for k, v in audit.get("by_event_type", {}).items()],
        [("event_type", "Event type"), ("count", "Count")],
    ))
    parts.append("<h3>Lineage stages</h3>")
    parts.append(f"<p>{_esc(' -> '.join(lineage.get('stages', []) or ['none']))}</p>")

    # --- 9. Managerial recommendations ---
    parts.append("<h2>9. Managerial recommendations</h2>")
    briefing = context.get("executive_briefing_md", "")
    if briefing:
        parts.append(f"<pre>{_esc(briefing)}</pre>")
    else:
        parts.append('<p class="meta">Executive briefing not available.</p>')

    parts.append("</body></html>")
    return "".join(parts)


def write_report(*, path: Path = PROJECT_REPORT_PATH) -> Path:
    """Build the context, render HTML, and write the self-contained report."""

    ensure_dir(path.parent)
    context = build_report_context()
    document = render_html(context)
    save_text(document, path)
    logger.info("Project report written -> %s (%d bytes)", path, len(document))
    return path
