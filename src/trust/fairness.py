"""
src.trust.fairness
==================

PURPOSE
-------
Break the champion's headline metric apart by subgroup, so a number that looks
good on average cannot hide a group the model fails.

A single test ROC-AUC of 0.845 is compatible with a model that is excellent for
most customers and no better than a coin flip for one segment. Nothing earlier in
this pipeline can tell those two situations apart. This module can.

WHAT IT MEASURES, PER TASK
--------------------------
Binary classification, per group:
    ``n``               rows in the group
    ``base_rate``       share of the group whose ACTUAL label is positive
    ``selection_rate``  share the model PREDICTS positive - the group's exposure
                        to whatever the positive prediction triggers
    ``tpr`` / ``fpr``   recall, and the false-alarm rate
    ``precision``       of those predicted positive, the share that were
    ``accuracy``
Multiclass, per group: ``n``, ``accuracy``, ``macro_f1`` (averaged over the
    classes present in that group, so descriptive only - see
    :func:`_multiclass_group_metrics`), ``n_classes_present``.
Regression, per group: ``n``, ``mean_error`` (signed - the group's systematic
    over- or under-prediction), ``mae``, ``rmse``.

``r2`` is deliberately absent from the regression group table. R-squared is
measured against the *group's own* variance, so a group whose targets are all
similar scores terribly even when its errors are tiny. Reporting it per group
invites exactly the wrong conclusion.

GAPS
----
Each audited column gets a ``gaps`` block: the max-minus-min spread of each
metric across eligible groups, plus a ratio for ``selection_rate`` (the
"disparate impact ratio" - the smallest group selection rate over the largest).

A group is eligible for the gap arithmetic only when it has at least
``MIN_GROUP_SIZE`` rows and the metric is defined for it. A gap driven by a
four-row group is noise, and reporting it as a finding produces false alarms that
train the reader to ignore the report.

Both kinds of exclusion are visible, by different routes, because they are not
the same shape. Being too small excludes a group from *every* metric, so it is
listed once in ``excluded_from_gaps`` with its row count. Having an undefined
value excludes a group from *one* metric only, so it is counted in
``n_groups_compared_per_metric`` - which lists just the metrics that lost groups,
because a per-group-per-metric matrix is unreadable and the group rows already
carry every value.

WHAT THIS MODULE WILL NOT DO
----------------------------
It does not call any column a protected attribute, and it does not judge whether
a gap is acceptable. It audits every low-cardinality column it finds, reports the
spread and the group sizes, and stops. Whether a disparity is lawful, expected
from the domain, or disqualifying is a human decision this code has no context to
make - see the note in :mod:`src.trust`.

OUTPUTS
-------
    artifacts/trust/subgroup_fairness.json
    artifacts/trust/subgroup_fairness.csv
    artifacts/trust/figures/subgroup_<headline metric>.png
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless: figures are written, never displayed.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from config.paths import (  # noqa: E402
    FAIRNESS_CSV_PATH,
    FAIRNESS_JSON_PATH,
    TRUST_FIGURES_DIR,
    ensure_dir,
)
from src.trust.subject import TrustSubject  # noqa: E402
from src.utils.common import utc_timestamp  # noqa: E402
from src.utils.file_utils import save_csv, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

__all__ = [
    "MAX_LEVELS",
    "MIN_GROUP_SIZE",
    "candidate_subgroup_columns",
    "audit_column",
    "gaps_for",
    "run",
]

#: A column with more distinct values than this is not a subgroup, it is an
#: identifier or a continuous measure. Twelve months of the year fit; a postcode
#: does not.
MAX_LEVELS = 10

#: Below this many rows a group's metrics are too noisy to compare. The group is
#: still reported - suppressing it would hide a real segment - but it is excluded
#: from the gap arithmetic and flagged ``reliable: false``.
MIN_GROUP_SIZE = 30

#: Groups are labelled by string, so a null needs a name of its own. Missingness
#: is frequently the most interesting subgroup, so it is audited rather than
#: dropped.
_MISSING_LABEL = "<MISSING>"

_PRECISION = 6


def _round(value: float | None) -> float | None:
    """Round for readability, preserving None and NaN as JSON ``null``."""

    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else round(number, _PRECISION)


def candidate_subgroup_columns(
    frame: pd.DataFrame,
    target: str,
    *,
    max_levels: int = MAX_LEVELS,
) -> list[str]:
    """
    Columns worth auditing as subgroups, in frame order.

    A column qualifies when it is not the target, is not a float, has between 2 and
    ``max_levels`` distinct non-null values, and does not have a distinct value for
    every single row.

    That last rule is about small splits. The level count is measured on the split in
    front of us, so on an 8-row test split an identifier like ``Unnamed: 0`` or a
    continuous integer like ``Assault`` has 8 distinct values and sails through a
    "2 to 10 levels" filter. Every resulting group holds one row, nothing is
    comparable, and the card still reports the column as audited. A column with one
    distinct value per row is an identifier in this split whatever it is elsewhere.

    Floats are excluded because a continuous measure sliced into groups is a
    different analysis (binning) with different assumptions, and because the
    columns that matter here - ``gender``, ``SeniorCitizen``, ``ocean_proximity``,
    ``Partner`` - are strings, booleans or integer codes. A genuinely categorical
    float can still be audited by passing ``groups=[...]`` to :func:`run`.

    A single-valued column is excluded because there is nothing to compare it
    against, not because it is uninteresting.
    """

    chosen: list[str] = []
    rows = int(len(frame))
    for column in frame.columns:
        if column == target:
            continue
        series = frame[column]
        if pd.api.types.is_float_dtype(series.dtype):
            continue
        levels = int(series.nunique(dropna=True))
        if levels >= rows > 1:
            continue  # one distinct value per row: an identifier in this split
        if 2 <= levels <= max_levels:
            chosen.append(str(column))
    return chosen


def _binary_group_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[Any]
) -> dict[str, float | None]:
    """Confusion-derived rates for one group of a binary problem.

    ``labels`` is the GLOBAL sorted label pair, not the group's own labels. A
    group containing only negatives would otherwise produce a 1x1 confusion
    matrix and silently mislabel its single cell as a true positive.

    ``tpr`` is None when the group contains no actual positives and ``fpr`` when
    it contains no actual negatives - both are genuinely undefined, and returning
    0.0 would assert a measurement that was never taken.
    """

    negative, positive = labels[0], labels[1]
    matrix = confusion_matrix(y_true, y_pred, labels=[negative, positive])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    n = tn + fp + fn + tp

    actual_positive = tp + fn
    actual_negative = tn + fp
    predicted_positive = tp + fp

    return {
        "n": n,
        "base_rate": (actual_positive / n) if n else None,
        "selection_rate": (predicted_positive / n) if n else None,
        "tpr": (tp / actual_positive) if actual_positive else None,
        "fpr": (fp / actual_negative) if actual_negative else None,
        "precision": (tp / predicted_positive) if predicted_positive else None,
        "accuracy": ((tp + tn) / n) if n else None,
    }


def _multiclass_group_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[Any]
) -> dict[str, float | None]:
    """Accuracy and macro-F1 for one group of a multiclass problem.

    ``macro_f1`` is averaged over the classes PRESENT in the group, not over the
    global label set, and ``n_classes_present`` is reported alongside it.

    Averaging over global labels looks like the fair choice - every group scored by
    the same formula - but it is not. An absent class contributes F1 = 0 to the
    macro average, so a group holding 2 of 5 classes is capped at 0.4 even with
    perfect predictions. The number then measures class coverage as much as
    accuracy, and differences between groups are largely composition artifacts.

    The cost of averaging over present classes instead is that groups with
    different label sets are on different scales, so ``macro_f1`` is descriptive
    only - it is deliberately excluded from the gap set. ``accuracy`` is comparable
    across groups and is what the ranking uses.
    """

    n = int(len(y_true))
    if n == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None, "n_classes_present": None}
    present = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    return {
        "n": n,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
        ),
        "n_classes_present": len(present),
    }


def _regression_group_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float | None]:
    """Signed bias, MAE and RMSE for one group.

    ``mean_error`` is signed on purpose. MAE tells you the group is wrong by a
    lot; ``mean_error`` tells you the model consistently predicts *too low* for
    that group, which is the finding that changes a decision.
    """

    n = int(len(y_true))
    if n == 0:
        return {"n": 0, "mean_error": None, "mae": None, "rmse": None}
    errors = np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64")
    return {
        "n": n,
        "mean_error": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


#: Which metrics get a gap, and which one leads the chart, per metric set.
_GAP_METRICS: dict[str, tuple[str, ...]] = {
    "binary_classification": (
        "base_rate",
        "selection_rate",
        "tpr",
        "fpr",
        "precision",
        "accuracy",
    ),
    # ``macro_f1`` is absent on purpose: it is averaged over each group's PRESENT
    # classes, so groups with different label sets sit on different scales and a
    # max-minus-min spread across them would report composition, not fairness.
    "multiclass_classification": ("accuracy",),
    "regression": ("mean_error", "mae", "rmse"),
}

#: The gap each metric set is ranked by, so the largest disparity is read first.
#: Without this, a dataset with sixteen categorical columns produces sixteen
#: undifferentiated audits and the reader learns to skip all of them.
_RANK_BY: dict[str, str] = {
    "binary_classification": "selection_rate_gap",
    "multiclass_classification": "accuracy_gap",
    "regression": "mae_gap",
}

_HEADLINE_METRIC: dict[str, str] = {
    # Accuracy is defined for every group of every classification problem, which
    # a rate like TPR is not.
    "binary_classification": "accuracy",
    "multiclass_classification": "accuracy",
    # Signed, so the chart shows the direction of the bias, not just its size.
    "regression": "mean_error",
}


def gaps_for(
    groups: list[dict[str, Any]],
    metric_set: str,
    *,
    min_group_size: int = MIN_GROUP_SIZE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Max-minus-min spread per metric, plus the groups excluded and why.

    Returns ``(gaps, excluded)``. A metric's gap is ``None`` when fewer than two
    groups have both a defined value and enough rows - there is no spread to
    report across a single group, and saying "0.0" would read as "no disparity
    found" when the truth is "not measurable here".
    """

    excluded = [
        {"group": g["group"], "n": g["n"], "reason": f"fewer than {min_group_size} rows"}
        for g in groups
        if g["n"] < min_group_size
    ]
    eligible = [g for g in groups if g["n"] >= min_group_size]

    gaps: dict[str, Any] = {}
    # Per metric, because eligibility differs per metric: a group can have plenty of
    # rows and still have no defined tpr. Reporting only the size-based
    # ``n_groups_compared`` let ``tpr_gap`` be a two-group spread while the card
    # printed "6 groups, 6 compared" directly above it.
    compared: dict[str, int] = {}
    for metric in _GAP_METRICS[metric_set]:
        values = [
            g[metric]
            for g in eligible
            if g.get(metric) is not None and not math.isnan(float(g[metric]))
        ]
        compared[metric] = len(values)
        if len(values) < 2:
            gaps[f"{metric}_gap"] = None
            continue
        gaps[f"{metric}_gap"] = _round(max(values) - min(values))

        # The disparate impact ratio, reported only where it means something: the
        # ratio of the least- to the most-selected group. Undefined when no group
        # is selected at all, which is a degenerate model rather than a fair one.
        if metric == "selection_rate":
            largest = max(values)
            gaps["selection_rate_ratio"] = (
                _round(min(values) / largest) if largest > 0 else None
            )

    # How much of the selection gap is the model, and how much is the world?
    #
    # A large selection-rate gap is not evidence of bias on its own. If
    # month-to-month customers really do churn three times as often, a model that
    # flags them three times as often is being accurate. What matters is whether
    # the model spreads the groups FURTHER apart than the outcomes actually are.
    #
    # Positive amplification: the model exaggerates a real difference.
    # Near zero: the model tracks reality.
    # Negative: the model under-separates groups that genuinely differ.
    #
    # LIMITATION: these are two independent spreads, so the widest-apart pair for
    # selection rate need not be the widest-apart pair for base rate. This
    # compares dispersion, not a matched pair, and is a triage signal for which
    # column to look at first - not a bias test.
    if "selection_rate_gap" in gaps:
        selection_gap = gaps.get("selection_rate_gap")
        base_gap = gaps.get("base_rate_gap")
        gaps["selection_amplification"] = (
            _round(selection_gap - base_gap)
            if selection_gap is not None and base_gap is not None
            else None
        )

    gaps["n_groups_compared"] = len(eligible)
    #: Per-metric counts, so a gap computed over two groups cannot be read as one
    #: computed over six. Only the metrics that lost groups are listed.
    gaps["n_groups_compared_per_metric"] = {
        metric: count for metric, count in compared.items() if count != len(eligible)
    }
    return gaps, excluded


def rank_key(audit: dict[str, Any]) -> float:
    """Sort weight for one audit: the size of its headline gap.

    Module level so the ordering can be tested without running the whole stage -
    :func:`run` writes artifacts, and a test that calls it to check a sort would
    overwrite a real project's output.

    An unmeasurable gap returns -1.0 and therefore sorts last under a descending
    sort. It is the absence of evidence; putting it first would bury the findings
    that do have evidence.
    """

    value = audit["gaps"].get(_RANK_BY[audit["metric_set"]])
    return -1.0 if value is None else abs(float(value))


def audit_column(
    subject: TrustSubject,
    column: str,
    *,
    min_group_size: int = MIN_GROUP_SIZE,
) -> dict[str, Any] | None:
    """
    Audit one subgroup column. ``None`` when it has fewer than two groups here.

    Groups are the column's distinct values in this split, nulls included under
    ``<MISSING>``.
    """

    y_true = np.asarray(subject.y_true)
    y_pred = np.asarray(subject.y_pred)
    keys = subject.frame[column].astype("string").fillna(_MISSING_LABEL)

    if subject.is_classification:
        # Arity from the MODEL, never from the split's labels. A three-class model
        # whose split holds two labels would otherwise be scored as binary, and
        # ``_binary_group_metrics`` would take an arbitrary one of those two as
        # "positive" - making tpr, fpr, precision and selection_rate answers to a
        # question nobody asked, and feeding the four-fifths annotation in the card.
        metric_set = (
            "binary_classification"
            if subject.is_binary
            else "multiclass_classification"
        )
        if metric_set == "binary_classification":
            # Ordered (negative, positive) with the positive label taken from the
            # estimator via the same helper ``positive_proba`` uses.
            positive = subject.positive_label
            labels = [
                next(c for c in subject.model_classes if c != positive),
                positive,
            ]
        else:
            labels = list(subject.model_classes) or sorted(
                set(np.unique(y_true)) | set(np.unique(y_pred))
            )
    else:
        labels = []
        metric_set = "regression"

    groups: list[dict[str, Any]] = []
    for level in sorted(keys.dropna().unique().tolist()):
        mask = (keys == level).to_numpy()
        if not mask.any():
            continue
        group_true, group_pred = y_true[mask], y_pred[mask]
        if metric_set == "binary_classification":
            values = _binary_group_metrics(group_true, group_pred, labels)
        elif metric_set == "multiclass_classification":
            values = _multiclass_group_metrics(group_true, group_pred, labels)
        else:
            values = _regression_group_metrics(group_true, group_pred)

        row: dict[str, Any] = {"group": str(level), "n": int(values.pop("n"))}
        # ``len(keys)`` is 0 only for an empty split, which auto-detection cannot
        # produce (it needs nunique >= 2) but an explicit ``groups=[...]`` can.
        row["share"] = _round(row["n"] / len(keys)) if len(keys) else None
        row["reliable"] = row["n"] >= min_group_size
        row.update({name: _round(value) for name, value in values.items()})
        groups.append(row)

    if len(groups) < 2:
        return None

    gaps, excluded = gaps_for(groups, metric_set, min_group_size=min_group_size)
    return {
        "column": column,
        "metric_set": metric_set,
        "n_groups": len(groups),
        "groups": groups,
        "gaps": gaps,
        "excluded_from_gaps": excluded,
    }


def _figure(audits: list[dict[str, Any]], champion: str, max_panels: int = 6) -> Path | None:
    """One panel per audited column: the headline metric by group."""

    plottable = [
        audit
        for audit in audits
        if any(
            group.get(_HEADLINE_METRIC[audit["metric_set"]]) is not None
            for group in audit["groups"]
        )
    ][:max_panels]
    if not plottable:
        return None

    metric = _HEADLINE_METRIC[plottable[0]["metric_set"]]
    ensure_dir(TRUST_FIGURES_DIR)
    columns = min(2, len(plottable))
    rows = math.ceil(len(plottable) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6.5 * columns, 3.6 * rows), squeeze=False
    )

    for index, audit in enumerate(plottable):
        ax = axes[index // columns][index % columns]
        groups = audit["groups"]
        names = [g["group"] for g in groups]
        values = [0.0 if g.get(metric) is None else float(g[metric]) for g in groups]
        # Unreliable groups are drawn, but greyed: visible without inviting a
        # conclusion the sample size does not support.
        colors = ["#2a6f97" if g["reliable"] else "#b8bcc0" for g in groups]
        ax.bar(names, values, color=colors, edgecolor="white")
        if metric == "mean_error":
            ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{audit['column']}  (n={', '.join(str(g['n']) for g in groups)})", fontsize=9)
        ax.set_ylabel(metric.replace("_", " "))
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    for index in range(len(plottable), rows * columns):
        axes[index // columns][index % columns].axis("off")

    fig.suptitle(f"{metric.replace('_', ' ').title()} by subgroup - {champion}", fontsize=11)
    fig.tight_layout()
    path = TRUST_FIGURES_DIR / f"subgroup_{metric}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run(
    subject: TrustSubject,
    *,
    groups: list[str] | None = None,
    min_group_size: int = MIN_GROUP_SIZE,
    max_levels: int = MAX_LEVELS,
) -> dict[str, Any]:
    """
    Audit every subgroup column and write the artifacts.

    ``groups`` overrides the auto-detected column list - use it to audit a
    categorical float, or to restrict a wide dataset to the columns that matter.

    Always returns a payload, even when no column qualifies. "No subgroup column
    was found in this dataset" is a real, reportable finding; writing nothing
    would leave the model card unable to distinguish it from "not run".
    """

    columns = (
        candidate_subgroup_columns(subject.frame, subject.target, max_levels=max_levels)
        if groups is None
        else list(groups)
    )
    missing = [c for c in columns if c not in subject.frame.columns]
    if missing:
        raise KeyError(f"Subgroup columns not present in the {subject.split} split: {missing}")

    audits: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for column in columns:
        audit = audit_column(subject, column, min_group_size=min_group_size)
        if audit is None:
            skipped.append({"column": column, "reason": "fewer than 2 groups in this split"})
        else:
            # A column whose groups are all below ``min_group_size`` still belongs in
            # the audit: the per-group rates are informative even when no pair is
            # large enough to compare, and dropping it would discard the only view of
            # a small project's segments. What must not happen is a reader assuming
            # gaps were measured, so that is flagged rather than the column hidden.
            audit["gaps_measurable"] = (
                int(audit["gaps"].get("n_groups_compared") or 0) >= 2
            )
            audits.append(audit)

    audits.sort(key=rank_key, reverse=True)
    figure = _figure(audits, subject.label)

    payload: dict[str, Any] = {
        "artifact": "subgroup_fairness",
        "generated_at": utc_timestamp(),
        "champion": subject.label,
        "split": subject.split,
        "task": subject.task,
        "n_rows": int(len(subject.frame)),
        "min_group_size": min_group_size,
        "max_levels": max_levels,
        "columns_audited": [audit["column"] for audit in audits],
        "columns_skipped": skipped,
        "ranked_by": "the absolute value of each metric set's headline gap, largest first",
        "audits": audits,
        "figure": str(figure) if figure else None,
        "interpretation_limits": [
            "These are gaps, not verdicts. Whether a gap is acceptable depends on "
            "the domain, the decision the prediction feeds, and the law that "
            "applies - none of which this artifact knows.",
            "EVERY low-cardinality column is audited, including columns that are "
            "model features by design. A large gap across a service tier or a "
            "contract type is usually the model working, not failing - which is "
            "why 'selection_amplification' is reported: it is the part of the gap "
            "the model adds on top of the difference that is really there.",
            "A subgroup column may itself be a model input, or may be excluded "
            "from the model entirely. Either way the gap is real; the remedy is "
            "not the same, and removing the column does not remove the gap.",
            f"Groups with fewer than {min_group_size} rows are reported but "
            "excluded from the gap arithmetic; see 'excluded_from_gaps'.",
            "Gaps are measured on one split. They are estimates with sampling "
            "error, and small gaps between similarly sized groups may not "
            "replicate.",
        ],
    }

    ensure_dir(FAIRNESS_JSON_PATH.parent)
    save_json(payload, FAIRNESS_JSON_PATH)

    flat = [
        {"column": audit["column"], "metric_set": audit["metric_set"], **group}
        for audit in audits
        for group in audit["groups"]
    ]
    save_csv(pd.DataFrame(flat), FAIRNESS_CSV_PATH)

    if audits:
        logger.info(
            "Subgroup fairness: audited %s on the %s split.",
            ", ".join(audit["column"] for audit in audits),
            subject.split,
        )
    else:
        logger.info(
            "Subgroup fairness: no column had 2-%d distinct non-float values; "
            "nothing to compare.",
            max_levels,
        )
    return payload
