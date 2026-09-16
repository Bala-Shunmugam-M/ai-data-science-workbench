"""
src.trust.calibration
=====================

PURPOSE
-------
Check whether the champion's probabilities mean what they say.

ROC-AUC measures whether the model *ranks* correctly - whether the customers it
scores highest really do churn more often than the ones it scores low. It says
nothing about whether "0.8" corresponds to an 80% chance. A model can rank
perfectly (AUC 1.0) while every probability it emits is wrong by 30 points.

That distinction only matters when someone uses the number rather than the rank -
and the retention simulator in this very repo does exactly that: it multiplies a
predicted probability by a discount cost to compute an ROI. If the probabilities
run 20 points high, the ROI is 20 points of fiction. This module is what makes
that checkable.

WHAT IT REPORTS
---------------
``brier_score``     Mean squared error of the probabilities. Lower is better; it
                    bundles calibration and discrimination together, so it is a
                    summary rather than a diagnosis.
``ece``             Expected calibration error - the average gap between promised
                    and observed frequency, weighted by how many rows fall in each
                    probability band. The headline number: 0.05 means the
                    probabilities are off by about five points on average.
``mce``             The worst single band's gap. A small ECE with a large MCE
                    means the model is well behaved in the middle and unreliable
                    at one extreme, which the average hides.
``mean_predicted``  vs ``base_rate`` - whether the model is globally optimistic
                    or pessimistic, before any binning.
``reliability``     Per band: predicted mean, observed frequency, row count.

BINNING
-------
Ten equal-width bands over [0, 1], the standard ECE definition, so the number is
comparable to any other model scored the same way. Empty bands are dropped from
the reliability curve and counted in ``n_bins_empty`` - a model whose predictions
all cluster in two bands has a reliability curve with two points, and pretending
otherwise by switching to quantile bands would hide that clustering.

SCOPE
-----
Binary classification with probability output only. Regression and multiclass are
skipped with a stated reason rather than approximated: multiclass calibration
needs a reliability curve per class, and there is no single honest number to
report in its place.

OUTPUTS
-------
    artifacts/trust/calibration.json
    artifacts/trust/figures/calibration.png
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless: figures are written, never displayed.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config.paths import (  # noqa: E402
    CALIBRATION_JSON_PATH,
    TRUST_FIGURES_DIR,
    ensure_dir,
)
from src.trust.subject import TrustSubject  # noqa: E402
from src.utils.common import utc_timestamp  # noqa: E402
from src.utils.file_utils import save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

__all__ = ["DEFAULT_BINS", "reliability_table", "calibration_errors", "run"]

DEFAULT_BINS = 10

_PRECISION = 6


def reliability_table(
    y_binary: np.ndarray, y_score: np.ndarray, *, n_bins: int = DEFAULT_BINS
) -> list[dict[str, Any]]:
    """
    One row per non-empty equal-width probability band.

    Bands are ``[0, 0.1), [0.1, 0.2), ... [0.9, 1.0]`` - the last is closed so a
    prediction of exactly 1.0 lands in it rather than falling outside every band.
    """

    # ``np.digitize`` sends NaN to the highest index, so an unfiltered NaN silently
    # joins the top band - the one a reader scrutinises hardest. :func:`run`
    # filters earlier because it also needs finite values for the Brier score; this
    # guard exists so a direct caller of this function cannot be misled either.
    y_binary, y_score = np.asarray(y_binary), np.asarray(y_score, dtype="float64")
    finite = np.isfinite(y_score)
    if not finite.all():
        y_binary, y_score = y_binary[finite], y_score[finite]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # ``right=False`` gives half-open bands; clipping the top index folds an exact
    # 1.0 into the final band.
    indices = np.clip(np.digitize(y_score, edges[1:-1], right=False), 0, n_bins - 1)

    rows: list[dict[str, Any]] = []
    for band in range(n_bins):
        mask = indices == band
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin": band + 1,
                "range_low": round(float(edges[band]), 3),
                "range_high": round(float(edges[band + 1]), 3),
                "n": count,
                "mean_predicted": round(float(y_score[mask].mean()), _PRECISION),
                "observed_frequency": round(float(y_binary[mask].mean()), _PRECISION),
                "gap": round(
                    float(y_score[mask].mean() - y_binary[mask].mean()), _PRECISION
                ),
            }
        )
    return rows


def calibration_errors(
    rows: list[dict[str, Any]], n_total: int
) -> tuple[float | None, float | None]:
    """
    ``(ece, mce)`` from a reliability table - the weighted mean gap and the worst.

    Both are ``None`` for an empty table. Gaps are absolute: a band that promises
    too little is as miscalibrated as one that promises too much, and letting the
    two cancel would report a badly calibrated model as a good one.
    """

    if not rows or n_total == 0:
        return None, None

    # A non-finite gap must be dropped, not carried into ``max()``. Every
    # comparison against NaN is False, so one NaN gap becomes the reported worst
    # band and can never be displaced by a real one - lesson #1 of
    # docs/ML_PIPELINE_LESSONS.md, in its aggregation form.
    usable = [
        row
        for row in rows
        if isinstance(row.get("gap"), (int, float)) and math.isfinite(row["gap"])
    ]
    if not usable:
        return None, None

    gaps = [abs(row["gap"]) for row in usable]
    weights = [row["n"] / n_total for row in usable]
    ece = float(sum(gap * weight for gap, weight in zip(gaps, weights)))
    return round(ece, _PRECISION), round(float(max(gaps)), _PRECISION)


def _skip(reason: str, subject: TrustSubject) -> dict[str, Any]:
    """A written artifact recording that calibration was not assessed, and why.

    Written rather than omitted so the model card can distinguish "this model has
    no probabilities to check" from "the trust stage never ran".
    """

    payload = {
        "artifact": "calibration",
        "generated_at": utc_timestamp(),
        "champion": subject.label,
        "split": subject.split,
        "task": subject.task,
        "assessed": False,
        "reason": reason,
    }
    ensure_dir(CALIBRATION_JSON_PATH.parent)
    save_json(payload, CALIBRATION_JSON_PATH)
    logger.info("Calibration not assessed: %s", reason)
    return payload


def _figure(
    rows: list[dict[str, Any]],
    y_score: np.ndarray,
    champion: str,
    ece: float | None,
) -> Path:
    """Reliability diagram above a histogram of the predicted probabilities.

    The histogram is not decoration: a reliability curve that looks excellent
    across bands where almost no rows fall is not evidence of anything, and the
    row counts are the only way to see that.
    """

    ensure_dir(TRUST_FIGURES_DIR)
    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(6.5, 7), height_ratios=[3, 1], sharex=True
    )

    ax.plot([0, 1], [0, 1], color="crimson", linestyle="--", linewidth=1, label="Perfect")
    if rows:
        ax.plot(
            [row["mean_predicted"] for row in rows],
            [row["observed_frequency"] for row in rows],
            marker="o",
            color="#2a6f97",
            linewidth=2,
            label="Champion",
        )
    ax.set_ylabel("Observed frequency")
    title = f"Calibration - {champion}"
    if ece is not None:
        title += f"  (ECE {ece:.3f})"
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9)

    ax_hist.hist(y_score, bins=20, range=(0.0, 1.0), color="#8fb3cc", edgecolor="white")
    ax_hist.set_xlabel("Predicted probability of the positive class")
    ax_hist.set_ylabel("Rows")

    fig.tight_layout()
    path = TRUST_FIGURES_DIR / "calibration.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run(subject: TrustSubject, *, n_bins: int = DEFAULT_BINS) -> dict[str, Any]:
    """
    Assess the champion's probability calibration, or record why it cannot be.

    Always returns a payload with an ``assessed`` flag.
    """

    if not subject.is_classification:
        return _skip(
            "Calibration is a property of probability forecasts; this is a "
            "regression model. The equivalent regression diagnostics are the "
            "residual plots written by the evaluation stage.",
            subject,
        )
    if subject.y_score is None:
        return _skip(
            "The champion does not expose predict_proba, so there are no "
            "probabilities to calibrate.",
            subject,
        )
    if not subject.is_binary:
        return _skip(
            f"Only binary calibration is implemented; this model was fitted on "
            f"{subject.n_classes} classes. Assessing it properly needs one "
            "reliability curve per class, and no single number honestly "
            "summarises them.",
            subject,
        )

    # ``subject.is_binary`` now describes the MODEL, so a three-class model can no
    # longer reach here via a split that happens to hold two labels. This checks
    # the array shape as well, because the two travelling together is the whole
    # correctness argument: a wider matrix means the arity above is lying.
    score = np.asarray(subject.y_score, dtype="float64")
    if score.ndim > 1:
        if score.shape[1] != 2:
            return _skip(
                f"The champion reports {subject.n_classes} classes but produced a "
                f"{score.shape[1]}-column probability matrix. Refusing to guess "
                "which column is the positive class.",
                subject,
            )
        score = score[:, 1]

    # The positive label comes from the estimator, through the same helper
    # ``positive_proba`` uses to choose its column. Re-deriving it as
    # "the larger of the two labels" is the obvious guess and it is wrong: for
    # ``classes_ == [1, 2]`` the probability column is ``P(1)``, so pairing it with
    # ``truth == 2`` inverts the entire curve and lands ECE near 1.0 in silence.
    truth = np.asarray(subject.y_true)
    positive_label = subject.positive_label
    y_binary = (truth == positive_label).astype("int64")

    if not y_binary.any() or y_binary.all():
        return _skip(
            f"The {subject.split} split contains only one class "
            f"({'all' if y_binary.all() else 'no'} rows are the positive label "
            f"{positive_label!r}), so no observed frequency can be compared "
            "against a predicted one.",
            subject,
        )

    # NaN scores would each be assigned to the TOP band by ``np.digitize`` - the
    # band a reader scrutinises hardest - and would then turn every summary
    # statistic into NaN. ``nan > 0.10`` is False, so the ECE warning at the foot
    # of this function would go quiet on exactly the input that most needs it.
    finite = np.isfinite(score)
    n_dropped = int((~finite).sum())
    if n_dropped:
        logger.warning(
            "Dropping %d of %d rows whose predicted probability is not finite.",
            n_dropped,
            len(score),
        )
        score, y_binary = score[finite], y_binary[finite]
        if len(score) == 0:
            return _skip(
                "Every predicted probability was NaN or infinite, so there is "
                "nothing to calibrate.",
                subject,
            )

    rows = reliability_table(y_binary, score, n_bins=n_bins)
    ece, mce = calibration_errors(rows, int(len(y_binary)))
    brier = float(np.mean((score - y_binary) ** 2))
    mean_predicted = float(score.mean())
    base_rate = float(y_binary.mean())

    figure = _figure(rows, score, subject.label, ece)

    payload: dict[str, Any] = {
        "artifact": "calibration",
        "generated_at": utc_timestamp(),
        "champion": subject.label,
        "split": subject.split,
        "task": subject.task,
        "assessed": True,
        "positive_label": str(positive_label),
        "n_rows": int(len(y_binary)),
        "n_rows_dropped_not_finite": n_dropped,
        "n_bins": n_bins,
        "n_bins_empty": n_bins - len(rows),
        "brier_score": round(brier, _PRECISION),
        "ece": ece,
        "mce": mce,
        "mean_predicted": round(mean_predicted, _PRECISION),
        "base_rate": round(base_rate, _PRECISION),
        # Signed, so the direction is legible: positive means the model promises
        # the positive class more often than it actually happens.
        "global_bias": round(mean_predicted - base_rate, _PRECISION),
        "reliability": rows,
        "figure": str(figure),
        "interpretation_limits": [
            "Calibration is separate from accuracy. A well-calibrated model can "
            "rank poorly, and a model that ranks perfectly can be badly "
            "calibrated. ROC-AUC does not measure this.",
            "Bands with few rows have noisy observed frequencies. Read the "
            "reliability table next to its 'n' column, and see 'n_bins_empty' - "
            "an empty band is one the model never predicts into.",
            "This is measured on one split. Recalibrating a model against these "
            "numbers would need a separate held-out set, or the correction just "
            "fits this split's noise.",
        ],
    }
    # ``np.isfinite`` rather than a bare ``> 0.10``: every comparison against NaN
    # is False, so a NaN ECE would silence this warning on the input that most
    # needs it. The filter above should make that unreachable; this makes it so.
    if ece is not None and np.isfinite(ece) and ece > 0.10:
        payload["interpretation_limits"].append(
            f"ECE is {ece:.3f}: the probabilities are off by roughly "
            f"{ece * 100:.0f} points on average. Any downstream calculation that "
            "multiplies these probabilities by a cost - the retention simulator "
            "included - inherits that error."
        )

    ensure_dir(CALIBRATION_JSON_PATH.parent)
    save_json(payload, CALIBRATION_JSON_PATH)

    logger.info(
        "Calibration: Brier=%.4f ECE=%s MCE=%s (predicted mean %.3f vs base rate %.3f).",
        brier,
        "n/a" if ece is None else f"{ece:.4f}",
        "n/a" if mce is None else f"{mce:.4f}",
        mean_predicted,
        base_rate,
    )
    return payload
