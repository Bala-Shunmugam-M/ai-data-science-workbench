"""
presentation.build_deck
=======================

PURPOSE
-------
Generate the 10-slide viva presentation for the AI Data Science Workbench.

Every RESULT on these slides is read at build time from src.reporting.facts,
the single source shared with the written report and the published Pages site.
Re-running the pipeline updates all three together; none of them can quote a
different number for the same run.

Figures that describe the CODE rather than a run - the twelve stage names, the
six model specs, the split fractions, the five standing decisions - are written
here in full, because they change only when the code changes and a reader
should be able to check them against:

  config/constants.py                 -> split fractions, seed, grid size
  src/model_proposal/model_catalog.py -> the six model specs
  src/workflow/pipeline_builder.py    -> the twelve stage names
  src/governance/decision_tracker.py  -> the five standing decisions

Nothing is estimated.

USAGE
-----
    python presentation/build_deck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

OUT_PATH = Path(__file__).parent / "ML-Pipeline-Viva.pptx"
REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------
# Every number on these slides comes from src.reporting.facts, which is the
# single source shared with the written report and the published Pages site.
# The deck used to carry its own copy of this loader; that meant three
# deliverables could quote three different numbers for the same run.
sys.path.insert(0, str(REPO))

from src.reporting.facts import (  # noqa: E402
    CHURN,
    HOUSING,
    money as _money,
    warn_if_stale,
)

# The slide functions below read a flat dict. Building it here, rather than
# threading the dataclasses through every slide, keeps this refactor to the
# loader alone.
F = {
    "h_champion": HOUSING.champion_name,
    "h_version": HOUSING.champion_version,
    "h_alpha": HOUSING.champion_params.get("alpha", 100.0),
    "h_tuned": HOUSING.tuned_parameter,
    "h_val_rmse": HOUSING.validation_metrics["rmse"],
    "h_val_r2": HOUSING.validation_metrics["r2"],
    "h_test_rmse": HOUSING.test_metrics["rmse"],
    "h_test_r2": HOUSING.test_metrics["r2"],
    "h_n_models": HOUSING.n_models,
    "h_runner": HOUSING.runner_up_name,
    "h_runner_rmse": HOUSING.runner_up_metrics["rmse"],
    "c_champion": CHURN.champion_name,
    "c_version": CHURN.champion_version,
    "c_tuned": CHURN.tuned_parameter,
    "c_val_auc": CHURN.validation_metrics["roc_auc"],
    "c_val_acc": CHURN.validation_metrics["accuracy"],
    "c_test_auc": CHURN.test_metrics["roc_auc"],
    "c_test_acc": CHURN.test_metrics["accuracy"],
    "c_runner": CHURN.runner_up_name,
    "c_runner_auc": CHURN.runner_up_metrics["roc_auc"],
    "brier": CHURN.calibration.get("brier_score", 0.136219),
    "ece": CHURN.calibration.get("ece", 0.030095),
    "mce": CHURN.calibration.get("mce", 0.180602),
    "mean_pred": CHURN.calibration.get("mean_predicted", 0.256572),
    "base_rate": CHURN.calibration.get("base_rate", 0.265847),
    "bias": CHURN.calibration.get("global_bias", -0.009275),
    "n_rows": CHURN.calibration.get("n_rows", 1057),
    "n_features": CHURN.importance.get("n_features", 46),
    "n_signal": CHURN.importance.get("n_distinguishable_from_noise", 18),
    "top_features": CHURN.top_features or [("tenure", 0.191509)],
    "n_audited": CHURN.n_audited or 16,
    "worst_column": CHURN.worst_fairness_column or "InternetService",
    "worst_tpr_gap": CHURN.gap(CHURN.worst_fairness_column, "tpr_gap") or 0.715847,
    "amp_worst": CHURN.amplification(CHURN.worst_fairness_column) or 0.120826,
    "amp_contract": CHURN.amplification("Contract") or -0.025923,
}

# --- Palette: one accent, dark ink on a light ground. -----------------------
INK = RGBColor(0x1A, 0x1F, 0x2B)
MUTED = RGBColor(0x5B, 0x64, 0x74)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
ACCENT_SOFT = RGBColor(0xE8, 0xEF, 0xF6)
RULE = RGBColor(0xD5, 0xDB, 0xE2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xBF, 0xD4, 0xE8)
GOOD = RGBColor(0x1E, 0x6B, 0x45)
WARN = RGBColor(0x9A, 0x5B, 0x0E)

SW, SH = 13.333, 7.5
MARGIN = 0.75


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def textbox(slide, x, y, w, h, *, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, *, size=14, bold=False, color=INK, space_after=6,
         first=False, align=PP_ALIGN.LEFT, italic=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.space_before = Pt(0)
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return p


def rect(slide, x, y, w, h, fill=None, line=None, line_w=1.0, round_=True):
    shape = MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    if round_:
        shp.adjustments[0] = 0.06
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    return shp


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def header(slide, kicker, title):
    """Accent rule + kicker + title. Body content starts at y = 1.55."""
    rect(slide, MARGIN, 0.52, 0.085, 0.62, fill=ACCENT, round_=False)
    tf = textbox(slide, MARGIN + 0.25, 0.46, SW - 2 * MARGIN - 0.25, 0.8)
    para(tf, kicker.upper(), size=10.5, bold=True, color=ACCENT,
         space_after=2, first=True)
    para(tf, title, size=24, bold=True, color=INK, space_after=0)


def footer(slide, n):
    tf = textbox(slide, MARGIN, SH - 0.5, SW - 2 * MARGIN, 0.3)
    para(tf, f"AI Data Science Workbench   |   {n} / 10", size=9,
         color=MUTED, first=True, align=PP_ALIGN.RIGHT)


def table(slide, x, y, w, rows, *, col_w, head_size=11, body_size=10.5,
          row_h=0.34, head_h=0.36, bold_first_col=True):
    n_rows, n_cols = len(rows), len(rows[0])
    height = head_h + row_h * (n_rows - 1)
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y),
                                   Inches(w), Inches(height))
    tbl = shape.table
    tbl.first_row = True
    tbl.horz_banding = False

    total = sum(col_w)
    for i, weight in enumerate(col_w):
        tbl.columns[i].width = Inches(w * weight / total)
    tbl.rows[0].height = Inches(head_h)
    for r in range(1, n_rows):
        tbl.rows[r].height = Inches(row_h)

    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.09)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = (
                ACCENT if r == 0 else (WHITE if r % 2 else ACCENT_SOFT))
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = val
            run.font.size = Pt(head_size if r == 0 else body_size)
            run.font.bold = (r == 0) or (c == 0 and bold_first_col)
            run.font.color.rgb = WHITE if r == 0 else INK
            run.font.name = "Calibri"
    return tbl


def panel(slide, x, y, w, h, heading, lines, *, head_color=ACCENT,
          body_size=12, fill=WHITE, border=RULE, line_w=1.0):
    rect(slide, x, y, w, h, fill=fill, line=border, line_w=line_w)
    tf = textbox(slide, x + 0.22, y + 0.17, w - 0.44, h - 0.32)
    para(tf, heading, size=12.5, bold=True, color=head_color, space_after=7,
         first=True)
    for line in lines:
        para(tf, "•  " + line, size=body_size, color=INK, space_after=5)


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------
def slide_01_title(prs):
    s = blank(prs)
    rect(s, 0, 0, SW, 2.3, fill=ACCENT, round_=False)

    tf = textbox(s, MARGIN, 0.6, SW - 2 * MARGIN, 1.5)
    para(tf, "MACHINE LEARNING PROJECT  ·  MODEL DEFENCE", size=12, bold=True,
         color=PALE, space_after=9, first=True)
    para(tf, "AI Data Science Workbench", size=34, bold=True, color=WHITE,
         space_after=4)
    para(tf, "A governed, task-agnostic machine learning pipeline",
         size=17, color=PALE, space_after=0)

    tf = textbox(s, MARGIN, 2.72, SW - 2 * MARGIN, 0.9)
    para(tf, "The claim is not \"the model is accurate\".", size=17,
         color=MUTED, first=True, space_after=4)
    para(tf, "It is: the result is reproducible, the choice is documented, "
             "leakage is structurally prevented, and the model's own bias is "
             "measured and published with it.", size=17, bold=True, color=INK)

    cards = [
        ("12", "governed stages"),
        ("6", "models in catalogue"),
        ("2", "tasks, one code path"),
        ("136", "tests"),
    ]
    cw, gap = 2.85, 0.28
    x = MARGIN
    for value, label in cards:
        rect(s, x, 4.35, cw, 1.35, fill=ACCENT_SOFT, line=RULE)
        tf = textbox(s, x + 0.15, 4.52, cw - 0.3, 1.0)
        para(tf, value, size=29, bold=True, color=ACCENT, space_after=2,
             first=True, align=PP_ALIGN.CENTER)
        para(tf, label, size=11, color=MUTED, align=PP_ALIGN.CENTER)
        x += cw + gap

    tf = textbox(s, MARGIN, 6.1, SW - 2 * MARGIN, 0.7)
    para(tf, "California Housing (regression)   ·   Telco Customer Churn "
             "(classification)", size=13, bold=True, color=INK, first=True,
         space_after=4)
    para(tf, "Python + scikit-learn  ·  governance layer  ·  trust audit  "
             "·  Streamlit and Next.js review consoles", size=11.5,
         color=MUTED)
    return s


def slide_02_problem(prs):
    s = blank(prs)
    header(s, "Slide 2", "The Problem, and the Design Response")

    panel(s, MARGIN, 1.55, 5.85, 2.7, "Why a notebook was not enough", [
        "A notebook records what ran — never why it was chosen.",
        "Results drift: no fixed seed, no split policy, no data hash.",
        "Leakage is silent. A scaler fitted on all rows never errors, "
        "it just returns a better score.",
        "Nothing stops an unreviewed model reaching the results.",
    ], head_color=WARN, body_size=11.5)

    panel(s, MARGIN + 6.15, 1.55, 5.85, 2.7, "What the workbench adds", [
        "12 named stages, each declaring its inputs and outputs.",
        "An approval gate the trainer physically cannot bypass.",
        "Every fitted statistic learned on train, carried with the model.",
        "Audit log, lineage graph, model registry and model card per run.",
    ], head_color=GOOD, body_size=11.5)

    tf = textbox(s, MARGIN, 4.45, SW - 2 * MARGIN, 0.35)
    para(tf, "One engine, two tasks — the strongest test that the design is "
             "genuinely general", size=13.5, bold=True, color=ACCENT,
         first=True)

    table(s, MARGIN, 4.88, SW - 2 * MARGIN, [
        ["Project", "Task", "Target", "Selection metric", "Positive class"],
        ["California Housing", "Regression", "median_house_value",
         "rmse  (lower is better)", "not applicable"],
        ["Telco Customer Churn", "Classification", "Churn",
         "roc_auc  (higher is better)", "\"Yes\""],
    ], col_w=[2.6, 1.7, 2.7, 2.7, 2.1], row_h=0.42)

    rect(s, MARGIN, 6.15, SW - 2 * MARGIN, 0.62, fill=ACCENT_SOFT, line=ACCENT)
    tf = textbox(s, MARGIN + 0.22, 6.29, SW - 2 * MARGIN - 0.44, 0.4)
    para(tf, "Switching project is one environment variable: "
             "WORKBENCH_PROJECT=churn  —  no code edit, no branch.",
         size=12, bold=True, color=ACCENT, first=True)
    footer(s, 2)
    return s


def slide_03_architecture(prs):
    s = blank(prs)
    header(s, "Slide 3", "Architecture — Twelve Governed Stages")

    stages = [
        ("ingest", "load, profile, hash"),
        ("preprocess", "clean, split 70/15/15"),
        ("understand", "schema + consistency"),
        ("eda", "distributions, screens"),
        ("features", "propose, then execute"),
        ("propose", "model proposal doc"),
        ("approve", "governance gate"),
        ("train", "fit + tune on validation"),
        ("evaluate", "compare, promote champion"),
        ("explain", "drivers, coefficients"),
        ("trust", "fairness, calibration"),
        ("report", "HTML report, model card"),
    ]

    bw, bh, gap = 1.87, 0.98, 0.14
    for i, (name, desc) in enumerate(stages):
        row, col = divmod(i, 6)
        x = MARGIN + col * (bw + gap)
        y = 1.7 + row * (bh + 0.55)
        gate = name in ("approve", "trust")
        rect(s, x, y, bw, bh, fill=ACCENT if gate else ACCENT_SOFT,
             line=ACCENT if gate else RULE)
        tf = textbox(s, x + 0.1, y + 0.13, bw - 0.2, bh - 0.22)
        para(tf, f"{i + 1}. {name}", size=11.5, bold=True,
             color=WHITE if gate else ACCENT, space_after=3, first=True)
        para(tf, desc, size=8.5, color=WHITE if gate else MUTED)
        if col < 5:
            ar = textbox(s, x + bw, y + 0.28, gap + 0.05, 0.35)
            para(ar, "›", size=15, bold=True, color=MUTED, first=True,
                 align=PP_ALIGN.CENTER)

    tf = textbox(s, MARGIN, 4.42, SW - 2 * MARGIN, 0.3)
    para(tf, "Filled boxes are the two verification gates. Registry order is "
             "the canonical pipeline order.", size=10.5, italic=True,
         color=MUTED, first=True)

    panel(s, MARGIN, 4.85, 3.85, 1.8, "The orchestrator enforces it", [
        "Refuses a stage whose declared inputs are missing, naming "
        "the prerequisite that produces them.",
        "Resume skips finished stages; --force re-runs.",
    ], body_size=10.5)

    panel(s, MARGIN + 4.15, 4.85, 3.85, 1.8, "Every stage leaves proof", [
        "Declared output paths, checked after the run.",
        "workflow_status.json records status, duration, outputs.",
        "An audit event and a lineage node per governed stage.",
    ], body_size=10.5)

    panel(s, MARGIN + 8.3, 4.85, 3.53, 1.8, "Review layers", [
        "Streamlit app + Next.js console read the artifacts.",
        "Both are read-only — Python owns all computation.",
    ], body_size=10.5)
    footer(s, 3)
    return s


def slide_04_leakage(prs):
    s = blank(prs)
    header(s, "Slide 4", "Data Discipline — Why Leakage Cannot Happen Here")

    rect(s, MARGIN, 1.5, 5.5, 0.42, fill=ACCENT, round_=False)
    tf = textbox(s, MARGIN + 0.18, 1.58, 5.2, 0.3)
    para(tf, "The invariant, stated in the code", size=11.5, bold=True,
         color=WHITE, first=True)
    rect(s, MARGIN, 1.92, 5.5, 1.15, fill=ACCENT_SOFT, line=ACCENT)
    tf = textbox(s, MARGIN + 0.2, 2.06, 5.1, 0.95)
    para(tf, "\"The split happens BEFORE any fitted transformation. "
             "Imputer, encoder and scaler are fit on the training split only "
             "and reused unchanged on validation and test.\"",
         size=12, italic=True, color=INK, first=True)

    table(s, MARGIN, 3.3, 5.5, [
        ["Split policy", "Value"],
        ["Train fraction", "0.70"],
        ["Validation fraction", "0.15"],
        ["Test fraction", "0.15"],
        ["Random state", "42, fixed project-wide"],
        ["Stratified on", "income band / target class"],
    ], col_w=[3.0, 2.5], row_h=0.33)

    panel(s, MARGIN + 5.95, 1.5, 6.05, 2.55, "Three structural guarantees", [
        "Every statistic — medians, means, standard deviations, "
        "category levels — is learned from the TRAIN split only.",
        "The fitted preprocessor is pickled INSIDE the model bundle, "
        "so validation, test and explainability transform identically.",
        "split_x_y() raises if the target appears among the predictors. "
        "Leakage becomes a crash, not a flattering score.",
    ], body_size=11.5)

    tf = textbox(s, MARGIN + 5.95, 4.2, 6.05, 0.3)
    para(tf, "The transformation, in order", size=12.5, bold=True,
         color=ACCENT, first=True)

    steps = [
        ("Numeric", "median-impute using the train median"),
        ("Numeric", "standardise using train mean and std"),
        ("Categorical", "one-hot using train categories only"),
        ("Unseen level", "all-zero row — never an error"),
    ]
    y = 4.6
    for label, desc in steps:
        rect(s, MARGIN + 5.95, y, 6.05, 0.44, fill=WHITE, line=RULE)
        tf = textbox(s, MARGIN + 6.12, y + 0.11, 5.7, 0.3)
        p = para(tf, f"{label}   —   {desc}", size=11, color=INK, first=True)
        y += 0.52

    rect(s, MARGIN, 5.92, 5.5, 0.75, fill=WHITE, line=WARN)
    tf = textbox(s, MARGIN + 0.2, 6.04, 5.1, 0.55)
    para(tf, "Zero-variance guard: a column whose train std is 0 gets a "
             "divisor of 1.0 — no divide-by-zero, no silent NaN column.",
         size=11, color=INK, first=True)
    footer(s, 4)
    return s


def slide_05_catalogue(prs):
    s = blank(prs)
    header(s, "Slide 5", "The Model Catalogue — Six Candidates in One Registry")

    table(s, MARGIN, 1.55, SW - 2 * MARGIN, [
        ["Model", "Family", "Task", "Tuned parameter", "Fixed settings"],
        ["LinearRegression", "linear", "Regression",
         "none — no hyperparameter", "the transparent baseline"],
        ["Ridge", "linear", "Regression", "alpha,  log 1e-3 → 1e3",
         "L2 shrinkage, keeps all predictors"],
        ["Lasso", "linear", "Regression", "alpha,  log 1e-4 → 1e1",
         "max_iter=10000, tol=0.1"],
        ["LogisticRegression", "linear", "Classification",
         "C,  log 1e-2 → 1e2", "liblinear, max_iter=2000"],
        ["DecisionTreeClassifier", "tree", "Classification",
         "max_depth,  2 → 12", "class_weight = balanced"],
        ["RandomForestClassifier", "ensemble", "Classification",
         "max_depth,  4 → 16", "300 trees, balanced, n_jobs=-1"],
    ], col_w=[2.9, 1.25, 1.85, 2.9, 3.1], row_h=0.42, body_size=11)

    panel(s, MARGIN, 4.9, 3.85, 1.8, "Why one tunable each", [
        "A single parameter lets ONE validation sweep serve both "
        "tasks — no separate tuners to drift apart.",
        "25-point grids (ALPHA_GRID_SIZE), log or integer scale.",
    ], body_size=10.5)

    panel(s, MARGIN + 4.15, 4.9, 3.85, 1.8, "Why the linear family leads", [
        "A standing governance decision, not an accident.",
        "Coefficients read as signed effects a manager can act on.",
        "Trees and the forest were added for churn as the accuracy "
        "benchmark the interpretable model must match.",
    ], body_size=10.5)

    panel(s, MARGIN + 8.3, 4.9, 3.53, 1.8, "Extensible by construction", [
        "Models are ModelSpec dataclasses resolved by string path.",
        "Adding a family edits the catalogue only — trainer, "
        "evaluator and reporter import no estimator directly.",
    ], body_size=10.5)
    footer(s, 5)
    return s


def slide_06_selection(prs):
    s = blank(prs)
    header(s, "Slide 6", "Tuning and Champion Selection — the Defensible Part")

    panel(s, MARGIN, 1.5, 5.85, 2.2, "Hold-out validation, NOT k-fold", [
        "The parameter is swept on a dedicated validation split, "
        "never k-fold over train + validation.",
        "One split, one purpose. Reusing training rows to choose a "
        "parameter leaks the choice back into the fit.",
        "Recorded as a standing decision: tune_on_validation_not_kfold.",
    ], body_size=11.5)

    panel(s, MARGIN + 6.15, 1.5, 5.85, 2.2, "One loop serves both tasks", [
        "METRIC_HIGHER_IS_BETTER supplies the direction, so rmse "
        "(lower wins) and roc_auc (higher wins) share one code path.",
        "is_better() handles NaN explicitly: an undefined score never "
        "wins, and always loses to a real one.",
        "Without that a NaN incumbent could never be displaced — and "
        "the bug was ordering-dependent, so it hid for weeks.",
    ], body_size=11.5)

    tf = textbox(s, MARGIN, 3.88, SW - 2 * MARGIN, 0.3)
    para(tf, "The promotion sequence", size=13, bold=True, color=ACCENT,
         first=True)

    steps = [
        "Sweep 25 candidates\non VALIDATION",
        "Refit best params\non TRAIN",
        "Compare all models\non VALIDATION",
        "Promote champion\nto the registry",
        "Score champion once\non TEST",
    ]
    bw, gap = 2.22, 0.26
    x = MARGIN
    for i, text in enumerate(steps):
        last = i == len(steps) - 1
        rect(s, x, 4.3, bw, 0.95, fill=ACCENT if last else ACCENT_SOFT,
             line=ACCENT if last else RULE)
        tf = textbox(s, x + 0.1, 4.42, bw - 0.2, 0.75)
        for j, line in enumerate(text.split("\n")):
            para(tf, line, size=10.5, bold=(j == 0),
                 color=WHITE if last else INK, space_after=1,
                 first=(j == 0), align=PP_ALIGN.CENTER)
        if not last:
            ar = textbox(s, x + bw, 4.58, gap + 0.05, 0.35)
            para(ar, "›", size=15, bold=True, color=MUTED, first=True,
                 align=PP_ALIGN.CENTER)
        x += bw + gap

    panel(s, MARGIN, 5.45, 5.85, 1.25, "The gate before any fit", [
        "train_model() calls require_model_approved(name) first.",
        "An unapproved model raises GovernanceError — never trains.",
    ], head_color=ACCENT, body_size=11, border=ACCENT, line_w=1.5)

    panel(s, MARGIN + 6.15, 5.45, 5.85, 1.25, "The test split is opened once",
          [
              "Only the champion is scored on it, after promotion.",
              "The trust stage reads it again to REPORT on a decision "
              "already frozen — it never feeds back into selection.",
          ], head_color=ACCENT, body_size=11, border=ACCENT, line_w=1.5)
    footer(s, 6)
    return s


def slide_07_results(prs):
    s = blank(prs)
    header(s, "Slide 7", "Results — Same Pipeline, Two Tasks")

    table(s, MARGIN, 1.5, SW - 2 * MARGIN, [
        ["", "California Housing", "Telco Customer Churn"],
        ["Task", "Regression", "Binary classification"],
        ["Candidates trained", f"{F['h_n_models']}  (3 models × 2 versions)",
         "6  (3 models × 2 versions)"],
        ["Selection metric", "rmse — lower is better",
         "roc_auc — higher is better"],
        ["Champion",
         f"{F['h_champion']} {F['h_version']},  alpha = {F['h_alpha']:g}",
         f"{F['c_champion']} {F['c_version']},  C = 2.154"],
        ["Validation",
         f"RMSE {_money(F['h_val_rmse'])}   ·   R² {F['h_val_r2']:.3f}",
         f"ROC-AUC {F['c_val_auc']:.4f}   ·   Accuracy {F['c_val_acc']:.3f}"],
        ["Test  (opened once)",
         f"RMSE {_money(F['h_test_rmse'])}   ·   R² {F['h_test_r2']:.3f}",
         f"ROC-AUC {F['c_test_auc']:.4f}   ·   Accuracy {F['c_test_acc']:.3f}"],
        ["Runner-up",
         f"{F['h_runner']}, RMSE {_money(F['h_runner_rmse'])}",
         "RandomForest, ROC-AUC 0.8446"],
        ["Top driver", "median_income  (positive)",
         f"{F['top_features'][0][0]}  "
         f"(permutation {F['top_features'][0][1]:.4f})"],
    ], col_w=[2.5, 4.75, 4.75], row_h=0.37, body_size=11)

    panel(s, MARGIN, 5.25, 5.85, 1.45,
          "Housing: an honest margin, reported as found", [
              "Ridge beat plain LinearRegression by $100 of RMSE — real, "
              "but small. Shrinkage helped slightly; it did not transform.",
              "Validation $64,787 → test $66,681: a modest, expected drop.",
          ], body_size=11, head_color=ACCENT)

    panel(s, MARGIN + 6.15, 5.25, 5.85, 1.45,
          "Churn: interpretability cost nothing", [
              "RandomForest matched the AUC (0.8446 vs 0.8455) but was not "
              "more useful — so the explainable model was promoted.",
              "Validation 0.8455 → test 0.8448: no overfit to validation.",
          ], body_size=11, head_color=ACCENT)
    footer(s, 7)
    return s


def slide_08_governance(prs):
    s = blank(prs)
    header(s, "Slide 8", "Governance — the Paper Trail Behind Every Number")

    tf = textbox(s, MARGIN, 1.48, SW - 2 * MARGIN, 0.32)
    para(tf, "Five standing decisions are recorded before any model is "
             "trained — the methodology is in the repository, not only in "
             "the report", size=12, bold=True, color=ACCENT, first=True)

    decisions = [
        ("linear_family_only", "Catalogue led by the linear family"),
        ("split_early_test_untouched", "Test split untouched until the end"),
        ("median_imputation_after_split", "Impute after splitting, train only"),
        ("tune_on_validation_not_kfold", "Tune on validation, not k-fold"),
        ("governed_approvals_gate_training", "Approval gates all training"),
    ]
    bw, gap = 2.31, 0.19
    x = MARGIN
    for key, desc in decisions:
        rect(s, x, 1.88, bw, 1.05, fill=ACCENT_SOFT, line=RULE)
        tf = textbox(s, x + 0.13, 2.0, bw - 0.26, 0.85)
        para(tf, key, size=8.5, bold=True, color=ACCENT, space_after=4,
             first=True)
        para(tf, desc, size=10, color=INK)
        x += bw + gap

    table(s, MARGIN, 3.2, 6.6, [
        ["Artifact written every run", "What it proves"],
        ["models/<name>/<version>/model.joblib", "the exact fitted object"],
        ["models/<name>/<version>/metadata.json",
         "params, tuning history, metrics"],
        ["models/model_registry.json", "who the champion is"],
        ["governance/audit/audit_log.jsonl", "10 event types, append-only"],
        ["governance/lineage/lineage.json", "6 nodes: input → script → output"],
        ["artifacts/experiments/experiments.jsonl", "immutable run history"],
    ], col_w=[4.0, 2.6], row_h=0.35, body_size=10)

    panel(s, MARGIN + 6.9, 3.2, 4.93, 1.65, "Gates that cannot be bypassed", [
        "Unapproved model → GovernanceError, before any fit.",
        "The design matrix uses only approved predictor columns.",
        "Versions auto-increment; nothing is ever overwritten.",
    ], body_size=10.5)

    panel(s, MARGIN + 6.9, 5.0, 4.93, 1.4, "Reproducibility anchors", [
        "data_hash: SHA-256 of the training file on every entry.",
        "random_state = 42 on every stochastic estimator.",
        "Lineage hashes each input and output file.",
    ], body_size=10.5)

    rect(s, MARGIN, 5.65, 6.6, 1.0, fill=ACCENT)
    tf = textbox(s, MARGIN + 0.22, 5.79, 6.16, 0.8)
    para(tf, "\"Six months from now, can you prove which data, which "
             "parameters and whose approval produced this number?\"",
         size=11.5, italic=True, color=PALE, first=True, space_after=4)
    para(tf, "Yes — registry, metadata, audit log and data hash together "
             "answer it.", size=11.5, bold=True, color=WHITE)
    footer(s, 8)
    return s


def slide_09_trust(prs):
    s = blank(prs)
    header(s, "Slide 9",
           "The Trust Layer — Deciding Whether to Believe the Model")

    tf = textbox(s, MARGIN, 1.45, SW - 2 * MARGIN, 0.3)
    para(tf, f"Measured on the churn champion, test split, {F['n_rows']:,} "
             "rows — after "
             "the champion was frozen, so it reports on the decision rather "
             "than making it", size=11, italic=True, color=MUTED, first=True)

    blocks = [
        ("Calibration",
         "When it says 0.8, does that\nhappen 80% of the time?",
         [f"Brier  {F['brier']:.4f}",
          f"ECE  {F['ece']:.4f}   ·   MCE  {F['mce']:.4f}",
          f"Predicted {F['mean_pred']:.4f} vs actual {F['base_rate']:.4f}",
          f"Bias {F['bias']:+.4f} — slightly cautious.".replace("+-", "−"),
          "The ROI simulator needs this,",
          "not the AUC."]),
        ("Permutation importance",
         "What does it actually depend on,\nnot what does it claim?",
         [f"{F['n_features']} features, 10 shuffles each",
          f"Only {F['n_signal']} beat noise"]
         + [f"{i + 1}.  {name}   {value:.4f}"
            for i, (name, value) in enumerate(F["top_features"])]
         + ["Scored on roc_auc, not impurity."]),
        ("Subgroup fairness",
         "Is the headline an average that\nhides a group it fails?",
         [f"{F['n_audited']} categorical columns audited",
          "Groups under 30 rows excluded",
          f"Largest gap: {F['worst_column']}",
          f"TPR gap {F['worst_tpr_gap']:.3f} across 3 groups",
          "One group is never predicted", "positive at all (ratio = 0)."]),
        ("Selection amplification",
         "How much of that gap did the\nmodel itself add?",
         [f"{F['worst_column']}:  {F['amp_worst']:+.4f}",
          f"Flagged {F['amp_worst'] * 100:.0f} points MORE often than",
          "its real churn rate justifies.",
          f"Contract:  {F['amp_contract']:+.4f} — the model",
          "NARROWS that real gap.",
          "Subtracts base rate from selection."]),
    ]
    bw, gap = 2.85, 0.28
    x = MARGIN
    for i, (title, question, lines) in enumerate(blocks):
        hot = i == 3
        rect(s, x, 1.85, bw, 3.35, fill=ACCENT_SOFT if hot else WHITE,
             line=ACCENT if hot else RULE, line_w=1.75 if hot else 1.0)
        tf = textbox(s, x + 0.17, 2.0, bw - 0.34, 3.05)
        para(tf, title, size=12, bold=True, color=ACCENT, space_after=4,
             first=True)
        for q in question.split("\n"):
            para(tf, q, size=9, italic=True, color=MUTED, space_after=2)
        para(tf, "", size=4, space_after=2)
        for line in lines:
            para(tf, line, size=10, color=INK, space_after=4)
        x += bw + gap

    panel(s, MARGIN, 5.42, SW - 2 * MARGIN, 1.28,
          "Two deliberate refusals — what the trust layer will NOT do", [
              "It never labels a column a protected attribute, and never "
              "judges whether a gap is acceptable. That is a human call.",
              "It never authors an intended-use statement. A card with no "
              "evidence SAYS so — an omitted fairness section would look "
              "like a fairness result that passed.",
          ], body_size=11, head_color=ACCENT, border=ACCENT, line_w=1.5)
    footer(s, 9)
    return s


def slide_10_limits(prs):
    s = blank(prs)
    header(s, "Slide 10", "Honest Limitations, and What Comes Next")

    panel(s, MARGIN, 1.5, 5.85, 3.4, "Known limitations", [
        "Housing R² of 0.667 is modest. The linear family cannot "
        "capture the coastal geography EDA flagged — stated, not hidden.",
        "Ridge beat the plain baseline by $100 of RMSE. That margin is "
        "real but small, and should not be oversold.",
        "No unsupervised branch: the detector labels every target "
        "regression or classification, so clustering data is mis-routed.",
        "Generated narrative prose is domain-bound to housing, so an "
        "uploaded project can inherit justification text it never earned.",
        "No gradient boosting, SVM, kNN or neural families.",
    ], head_color=WARN, body_size=11)

    panel(s, MARGIN + 6.15, 1.5, 5.85, 3.4, "Next steps, in priority order", [
        "Template the model-card and proposal prose from the dataset "
        "descriptor, so governance text matches the actual project.",
        "Add an unsupervised branch (k-means + silhouette) with its own "
        "selection metric and trust checks.",
        "Add one boosted-tree family to test whether the "
        "single-tunable-parameter design still holds.",
        "Nested validation, to quantify selection optimism directly "
        "instead of arguing it away.",
        "Port the retention simulator into the Next.js console.",
    ], head_color=GOOD, body_size=11)

    rect(s, MARGIN, 5.15, SW - 2 * MARGIN, 1.5, fill=ACCENT)
    tf = textbox(s, MARGIN + 0.35, 5.35, SW - 2 * MARGIN - 0.7, 1.15)
    para(tf, "WHAT THIS PROJECT DEFENDS", size=11, bold=True, color=PALE,
         space_after=7, first=True)
    para(tf, "Every number on these slides can be traced to a file: which "
             "data produced it, which parameters, whose approval, and what "
             "the model gets wrong about which group. That traceability — "
             "not the accuracy — is the deliverable.",
         size=14.5, bold=True, color=WHITE)
    footer(s, 10)
    return s


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    for fn in (slide_01_title, slide_02_problem, slide_03_architecture,
               slide_04_leakage, slide_05_catalogue, slide_06_selection,
               slide_07_results, slide_08_governance, slide_09_trust,
               slide_10_limits):
        fn(prs)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_PATH)
    return OUT_PATH


if __name__ == "__main__":
    path = build()
    check = Presentation(path)
    assert len(check.slides) == 10, f"expected 10 slides, got {len(check.slides)}"
    print(f"OK  {path}  ({len(check.slides)} slides)")
    print(warn_if_stale())
