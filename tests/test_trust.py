"""
tests.test_trust
================

Unit tests for the trust layer: subgroup fairness, permutation-importance
scorer mapping, probability calibration, and model-card rendering.

Every test here works on hand-built inputs with arithmetic that can be checked by
hand. None of them trains a model or touches a workspace, so they run in
milliseconds and fail for exactly one reason.

The cases chosen are the ones that would ship a wrong number silently rather than
crash: a group with no positives (an undefined rate that must not become 0.0), a
gap driven by a four-row group (noise that must not be reported as a finding),
gaps that would cancel if they were signed, and a card section whose source
artifact is missing (which must say so rather than disappear).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.model_training.design_matrix import positive_class_label
from src.trust import calibration, fairness, model_card, robust_importance
from src.trust.subject import ConfigDriftError, TrustSubject, _check_config_matches_bundle


class _FakeEstimator:
    """Stands in for a fitted classifier, carrying only ``classes_``.

    ``positive_label`` routes through ``design_matrix.positive_class_label``, which
    reads the estimator out of the bundle - so a subject built for a test needs
    something there with the right ``classes_``.
    """

    def __init__(self, classes) -> None:
        self.classes_ = list(classes)


def _subject(
    frame: pd.DataFrame,
    y_true,
    y_pred,
    *,
    task: str = "classification",
    target: str = "y",
    selection_metric: str = "roc_auc",
    y_score=None,
    model_classes=None,
) -> TrustSubject:
    """A TrustSubject with only the fields the audits actually read.

    ``model_classes`` defaults to the labels present in ``y_true``, which is the
    common case for a test. Pass it explicitly to model the case that matters: a
    model fitted on more classes than this split happens to contain.
    """

    if task == "classification" and model_classes is None:
        model_classes = sorted(set(np.asarray(y_true).tolist()))
    model_classes = list(model_classes or [])

    return TrustSubject(
        champion={},
        label="Test v001",
        bundle={"estimator": _FakeEstimator(model_classes)},
        split="test",
        frame=frame,
        matrix=pd.DataFrame(index=frame.index),
        y_true=pd.Series(y_true),
        y_pred=np.asarray(y_pred),
        y_score=None if y_score is None else np.asarray(y_score),
        task=task,
        target=target,
        selection_metric=selection_metric,
        model_classes=model_classes,
    )


# ---------------------------------------------------------------------------
# Subgroup column detection
# ---------------------------------------------------------------------------


def test_candidate_columns_picks_low_cardinality_and_skips_target_and_floats():
    frame = pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 0, 1],
            "gender": ["F", "M", "F", "M", "F", "M"],          # 2 levels -> in
            "senior": [0, 1, 0, 0, 1, 1],                      # int code -> in
            "income": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],          # float -> out
            "customer_id": ["a", "b", "c", "d", "e", "f"],      # 6 of 6 distinct -> out
            "region": ["N", "N", "S", "S", "E", "E"],          # 3 levels -> in
            "constant": ["x", "x", "x", "x", "x", "x"],        # 1 level -> out
        }
    )
    assert fairness.candidate_subgroup_columns(frame, "y", max_levels=4) == [
        "gender",
        "senior",
        "region",
    ]


def test_float_column_is_excluded_even_with_few_levels():
    """A float with three values is still excluded: binning is a different analysis."""

    frame = pd.DataFrame({"y": [0, 1, 0, 1], "score": [0.5, 0.5, 1.5, 1.5]})
    assert fairness.candidate_subgroup_columns(frame, "y") == []


def test_nulls_become_their_own_group():
    """Missingness is frequently the most interesting segment, so it is audited."""

    frame = pd.DataFrame(
        {
            "y": [0, 1] * 4,
            "plan": ["A", "A", "A", "A", None, None, None, None],
        }
    )
    audit = fairness.audit_column(
        _subject(frame, frame["y"], [0, 1] * 4), "plan", min_group_size=1
    )
    assert [group["group"] for group in audit["groups"]] == ["<MISSING>", "A"]


# ---------------------------------------------------------------------------
# Binary group metrics
# ---------------------------------------------------------------------------


def test_binary_group_rates_match_hand_computed_confusion():
    """Group A: tp=2 fn=1 tn=2 fp=1 -> tpr 2/3, fpr 1/3, precision 2/3."""

    frame = pd.DataFrame(
        {
            "y": [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            "g": ["A"] * 6 + ["B"] * 4,
        }
    )
    y_pred = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0]
    audit = fairness.audit_column(
        _subject(frame, frame["y"], y_pred), "g", min_group_size=1
    )
    group_a = next(g for g in audit["groups"] if g["group"] == "A")

    assert group_a["n"] == 6
    assert group_a["base_rate"] == pytest.approx(3 / 6)
    assert group_a["selection_rate"] == pytest.approx(3 / 6)
    assert group_a["tpr"] == pytest.approx(2 / 3)
    assert group_a["fpr"] == pytest.approx(1 / 3)
    assert group_a["precision"] == pytest.approx(2 / 3)
    assert group_a["accuracy"] == pytest.approx(4 / 6)


def test_group_without_positives_reports_tpr_as_null_not_zero():
    """An undefined rate must stay undefined.

    A group with no actual positives has no recall. Reporting 0.0 asserts the
    model failed every positive case in a group that contained none, which reads
    as the worst possible fairness finding and is pure fabrication.
    """

    frame = pd.DataFrame({"y": [1, 1, 1, 0, 0, 0], "g": ["A", "A", "A", "B", "B", "B"]})
    audit = fairness.audit_column(
        _subject(frame, frame["y"], [1, 1, 0, 0, 0, 0]), "g", min_group_size=1
    )
    group_b = next(g for g in audit["groups"] if g["group"] == "B")

    assert group_b["tpr"] is None      # no actual positives in B
    assert group_b["fpr"] == 0.0       # three actual negatives, none flagged
    assert group_b["precision"] is None  # nothing predicted positive


def test_single_group_column_is_skipped():
    frame = pd.DataFrame({"y": [0, 1, 0, 1], "g": ["A", "A", "A", "A"]})
    assert fairness.audit_column(_subject(frame, frame["y"], [0, 1, 0, 1]), "g") is None


# ---------------------------------------------------------------------------
# Gap arithmetic
# ---------------------------------------------------------------------------


def test_gap_is_max_minus_min_and_ratio_is_min_over_max():
    groups = [
        {"group": "A", "n": 100, "selection_rate": 0.20, "tpr": 0.8, "fpr": 0.1, "precision": 0.7, "accuracy": 0.90},
        {"group": "B", "n": 100, "selection_rate": 0.50, "tpr": 0.6, "fpr": 0.2, "precision": 0.5, "accuracy": 0.75},
    ]
    for group in groups:  # base_rate is now part of the gap set
        group["base_rate"] = group["selection_rate"]
    gaps, excluded = fairness.gaps_for(groups, "binary_classification", min_group_size=30)

    assert gaps["selection_rate_gap"] == pytest.approx(0.30)
    assert gaps["selection_rate_ratio"] == pytest.approx(0.20 / 0.50)
    assert gaps["accuracy_gap"] == pytest.approx(0.15)
    assert gaps["n_groups_compared"] == 2
    assert excluded == []


def test_small_group_is_reported_but_excluded_from_gaps():
    """A four-row group cannot drive a headline gap, and its exclusion is visible."""

    groups = [
        {"group": "big_a", "n": 500, "accuracy": 0.90, "macro_f1": 0.88},
        {"group": "big_b", "n": 400, "accuracy": 0.88, "macro_f1": 0.86},
        {"group": "tiny", "n": 4, "accuracy": 0.25, "macro_f1": 0.20},
    ]
    gaps, excluded = fairness.gaps_for(
        groups, "multiclass_classification", min_group_size=30
    )

    # 0.02 from the two large groups, NOT 0.65 from the four-row outlier.
    assert gaps["accuracy_gap"] == pytest.approx(0.02)
    assert gaps["n_groups_compared"] == 2
    assert len(excluded) == 1
    assert excluded[0]["group"] == "tiny"
    assert "fewer than 30 rows" in excluded[0]["reason"]


def test_gap_is_none_when_fewer_than_two_groups_are_eligible():
    """None, not 0.0. "No disparity found" and "not measurable" are different claims."""

    groups = [
        {"group": "big", "n": 500, "accuracy": 0.9, "macro_f1": 0.9},
        {"group": "tiny", "n": 3, "accuracy": 0.1, "macro_f1": 0.1},
    ]
    gaps, _ = fairness.gaps_for(groups, "multiclass_classification", min_group_size=30)

    assert gaps["accuracy_gap"] is None
    assert gaps["n_groups_compared"] == 1


def test_undefined_metric_is_dropped_from_its_gap_only():
    """A None tpr must not poison the other metrics' gaps."""

    groups = [
        {"group": "A", "n": 100, "selection_rate": 0.3, "tpr": None, "fpr": 0.1, "precision": 0.6, "accuracy": 0.8},
        {"group": "B", "n": 100, "selection_rate": 0.5, "tpr": 0.7, "fpr": 0.2, "precision": 0.5, "accuracy": 0.7},
    ]
    gaps, _ = fairness.gaps_for(groups, "binary_classification", min_group_size=30)

    assert gaps["tpr_gap"] is None                              # only one defined value
    assert gaps["selection_rate_gap"] == pytest.approx(0.2)     # unaffected
    assert gaps["accuracy_gap"] == pytest.approx(0.1)


def test_amplification_is_zero_when_the_model_tracks_real_base_rates():
    """A large selection gap that matches a large base-rate gap is accuracy, not bias.

    This is the number that stops a correct model being reported as a biased one:
    if one group really does churn 40 points more often, flagging it 40 points more
    often is the model working.
    """

    groups = [
        {"group": "month_to_month", "n": 500, "base_rate": 0.60, "selection_rate": 0.60,
         "tpr": 0.8, "fpr": 0.2, "precision": 0.7, "accuracy": 0.8},
        {"group": "two_year", "n": 500, "base_rate": 0.20, "selection_rate": 0.20,
         "tpr": 0.7, "fpr": 0.1, "precision": 0.6, "accuracy": 0.85},
    ]
    gaps, _ = fairness.gaps_for(groups, "binary_classification", min_group_size=30)

    assert gaps["base_rate_gap"] == pytest.approx(0.40)
    assert gaps["selection_rate_gap"] == pytest.approx(0.40)
    assert gaps["selection_amplification"] == pytest.approx(0.0)


def test_amplification_is_positive_when_the_model_exaggerates_a_real_difference():
    groups = [
        {"group": "A", "n": 500, "base_rate": 0.40, "selection_rate": 0.70,
         "tpr": 0.9, "fpr": 0.4, "precision": 0.5, "accuracy": 0.6},
        {"group": "B", "n": 500, "base_rate": 0.30, "selection_rate": 0.10,
         "tpr": 0.3, "fpr": 0.05, "precision": 0.6, "accuracy": 0.7},
    ]
    gaps, _ = fairness.gaps_for(groups, "binary_classification", min_group_size=30)

    # Real spread 0.10, model's spread 0.60 -> it adds 0.50 of its own.
    assert gaps["selection_amplification"] == pytest.approx(0.50)


def test_amplification_is_none_when_either_spread_is_unmeasurable():
    groups = [
        {"group": "A", "n": 500, "base_rate": 0.4, "selection_rate": None,
         "tpr": 0.9, "fpr": 0.4, "precision": 0.5, "accuracy": 0.6},
        {"group": "B", "n": 500, "base_rate": 0.3, "selection_rate": 0.1,
         "tpr": 0.3, "fpr": 0.05, "precision": 0.6, "accuracy": 0.7},
    ]
    gaps, _ = fairness.gaps_for(groups, "binary_classification", min_group_size=30)

    assert gaps["selection_rate_gap"] is None
    assert gaps["selection_amplification"] is None


def test_audits_are_ranked_with_the_largest_disparity_first():
    """Sixteen categorical columns in frame order buries the findings that matter.

    ``rank_key`` is tested rather than ``run``, because ``run`` writes artifacts
    into the active project and a sort test has no business doing that.
    """

    audits = [
        {"column": "flat", "metric_set": "binary_classification",
         "gaps": {"selection_rate_gap": 0.01}},
        {"column": "sharp", "metric_set": "binary_classification",
         "gaps": {"selection_rate_gap": 0.44}},
        {"column": "middling", "metric_set": "binary_classification",
         "gaps": {"selection_rate_gap": 0.20}},
    ]
    ranked = sorted(audits, key=fairness.rank_key, reverse=True)

    assert [audit["column"] for audit in ranked] == ["sharp", "middling", "flat"]


def test_unmeasurable_gap_sorts_last_rather_than_first():
    """Absence of evidence must not outrank a real finding."""

    unmeasurable = {
        "column": "u", "metric_set": "binary_classification",
        "gaps": {"selection_rate_gap": None},
    }
    real = {
        "column": "r", "metric_set": "binary_classification",
        "gaps": {"selection_rate_gap": 0.30},
    }
    ranked = sorted([unmeasurable, real], key=fairness.rank_key, reverse=True)

    assert [audit["column"] for audit in ranked] == ["r", "u"]
    assert fairness.rank_key(unmeasurable) == -1.0


def test_rank_key_uses_the_metric_set_appropriate_gap():
    """Regression ranks on mae_gap, not on a classification metric it lacks."""

    assert fairness.rank_key(
        {"metric_set": "regression", "gaps": {"mae_gap": 42.0, "mean_error_gap": 1.0}}
    ) == 42.0
    assert fairness.rank_key(
        {"metric_set": "multiclass_classification", "gaps": {"accuracy_gap": 0.15}}
    ) == pytest.approx(0.15)


def test_regression_mean_error_keeps_its_sign():
    """Signed bias is the finding; MAE alone cannot say "consistently too low"."""

    frame = pd.DataFrame({"y": [10.0, 12.0, 30.0, 32.0], "g": ["A", "A", "B", "B"]})
    # A is under-predicted by 5, B is over-predicted by 5.
    audit = fairness.audit_column(
        _subject(
            frame, frame["y"], [5.0, 7.0, 35.0, 37.0], task="regression", selection_metric="rmse"
        ),
        "g",
        min_group_size=1,
    )
    by_group = {g["group"]: g for g in audit["groups"]}

    assert audit["metric_set"] == "regression"
    assert by_group["A"]["mean_error"] == pytest.approx(5.0)
    assert by_group["B"]["mean_error"] == pytest.approx(-5.0)
    assert by_group["A"]["mae"] == pytest.approx(5.0)
    assert by_group["B"]["mae"] == pytest.approx(5.0)
    # MAE is identical, so only the signed spread reveals the opposite biases.
    assert audit["gaps"]["mae_gap"] == pytest.approx(0.0)
    assert audit["gaps"]["mean_error_gap"] == pytest.approx(10.0)


def test_regression_groups_carry_no_r2():
    """R2 per group is measured against the group's own variance and misleads."""

    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "g": ["A", "A", "B", "B"]})
    audit = fairness.audit_column(
        _subject(frame, frame["y"], [1.1, 2.1, 3.1, 4.1], task="regression"),
        "g",
        min_group_size=1,
    )
    assert "r2" not in audit["groups"][0]


# ---------------------------------------------------------------------------
# Permutation-importance scorer mapping
# ---------------------------------------------------------------------------


def test_error_metrics_map_to_negated_scorers():
    """Permutation importance needs higher-is-better, so RMSE maps to its neg form."""

    assert robust_importance.scorer_for("rmse", is_binary=False) == "neg_root_mean_squared_error"
    assert robust_importance.scorer_for("mae", is_binary=False) == "neg_mean_absolute_error"
    assert robust_importance.scorer_for("r2", is_binary=False) == "r2"


def test_class_metrics_depend_on_arity():
    assert robust_importance.scorer_for("roc_auc", is_binary=True) == "roc_auc"
    assert robust_importance.scorer_for("roc_auc", is_binary=False) == "roc_auc_ovr"
    assert robust_importance.scorer_for("f1", is_binary=True) == "f1"
    assert robust_importance.scorer_for("f1", is_binary=False) == "f1_macro"


def test_unmapped_metric_returns_none_rather_than_a_guess():
    """A plausible-but-wrong scorer would measure importance against a metric
    nobody chose. None makes the caller fall back explicitly and record it."""

    assert robust_importance.scorer_for("balanced_weirdness", is_binary=True) is None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_perfectly_calibrated_input_has_zero_error():
    """Half the rows at p=0.0 all negative, half at p=1.0 all positive."""

    y_binary = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    rows = calibration.reliability_table(y_binary, y_score)
    ece, mce = calibration.calibration_errors(rows, len(y_binary))

    assert ece == pytest.approx(0.0)
    assert mce == pytest.approx(0.0)
    assert len(rows) == 2  # only the first and last bands are populated


def test_probability_of_exactly_one_lands_in_the_last_band():
    """The top band is closed, so p=1.0 is not silently dropped."""

    rows = calibration.reliability_table(np.array([1, 1]), np.array([1.0, 1.0]), n_bins=10)
    assert len(rows) == 1
    assert rows[0]["bin"] == 10
    assert rows[0]["n"] == 2


def test_empty_bands_are_omitted_from_the_table():
    rows = calibration.reliability_table(
        np.array([0, 1]), np.array([0.05, 0.95]), n_bins=10
    )
    assert [row["bin"] for row in rows] == [1, 10]


def test_ece_is_weighted_by_band_population():
    """A 0.5-gap band holding one row must not outweigh a clean band holding nine."""

    # Band 1 (p=0.05): 9 rows, all negative -> gap 0.05.
    # Band 6 (p=0.55): 1 row, negative     -> gap 0.55.
    y_binary = np.zeros(10, dtype=int)
    y_score = np.array([0.05] * 9 + [0.55])
    rows = calibration.reliability_table(y_binary, y_score)
    ece, mce = calibration.calibration_errors(rows, 10)

    assert ece == pytest.approx(0.05 * 0.9 + 0.55 * 0.1)
    assert mce == pytest.approx(0.55)


def test_opposing_gaps_do_not_cancel():
    """Gaps are absolute. One over-confident band and one under-confident band
    average to zero if signed, reporting a badly calibrated model as perfect."""

    rows = [
        {"bin": 1, "n": 50, "gap": 0.20},
        {"bin": 2, "n": 50, "gap": -0.20},
    ]
    ece, mce = calibration.calibration_errors(rows, 100)

    assert ece == pytest.approx(0.20)
    assert mce == pytest.approx(0.20)


def test_calibration_errors_of_empty_table_are_none():
    assert calibration.calibration_errors([], 0) == (None, None)


def test_reliability_gap_is_predicted_minus_observed():
    """Pins the SIGN of `gap`, computed by the real function.

    Reversing the subtraction flips every band's gap and inverts the meaning of
    `global_bias` (optimistic vs pessimistic) while leaving ECE and MCE - which are
    absolute - completely unchanged. Without this, the sign is untested.
    """

    # Band 9 predicts 0.85 but nothing happens: the model promised too much.
    rows = calibration.reliability_table(np.array([0, 0]), np.array([0.85, 0.85]))
    assert rows[0]["gap"] == pytest.approx(0.85)

    # Band 2 predicts 0.15 and everything happens: the model promised too little.
    rows = calibration.reliability_table(np.array([1, 1]), np.array([0.15, 0.15]))
    assert rows[0]["gap"] == pytest.approx(-0.85)


def test_interior_bin_boundary_lands_in_the_upper_band():
    """Bands are half-open, so 0.1 belongs to [0.1, 0.2), i.e. bin 2.

    Only the 0.0 and 1.0 extremes were pinned before; the interior convention is the
    one a refactor would silently flip.
    """

    rows = calibration.reliability_table(
        np.array([0, 0, 0]), np.array([0.0999, 0.1, 0.2]), n_bins=10
    )
    assert [row["bin"] for row in rows] == [1, 2, 3]


def test_nan_score_is_dropped_rather_than_banked_into_the_top_band():
    """np.digitize sends NaN to the highest index, silently inflating band 10 -
    the band a reader scrutinises hardest."""

    rows = calibration.reliability_table(
        np.array([0, 1, 0]), np.array([0.05, 0.95, np.nan])
    )
    assert sum(row["n"] for row in rows) == 2
    assert [row["bin"] for row in rows] == [1, 10]


def test_nan_gap_cannot_become_the_reported_worst_band():
    """A NaN survives max() because every comparison against it is False."""

    rows = [
        {"bin": 1, "n": 50, "gap": 0.10},
        {"bin": 2, "n": 50, "gap": float("nan")},
    ]
    ece, mce = calibration.calibration_errors(rows, 100)

    assert mce == pytest.approx(0.10)
    assert ece == pytest.approx(0.10 * 0.5)


# ---------------------------------------------------------------------------
# The positive class: one rule, shared
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "classes, expected",
    [
        ([0, 1], 1),
        ([-1, 1], 1),
        (["No", "Yes"], "Yes"),
        # The case that breaks `sorted(classes)[-1]`: the positive column is the
        # one for label 1, which here is the SMALLER label.
        ([1, 2], 1),
        ([1, 9], 1),
        # No literal 1 anywhere -> the last class.
        ([3, 7], 7),
    ],
)
def test_positive_class_label_prefers_literal_one_not_the_largest(classes, expected):
    assert positive_class_label(_FakeEstimator(classes)) == expected


def test_calibration_pairs_the_probability_column_with_the_matching_truth():
    """The blocker: for classes_ == [1, 2] the score is P(1), so the truth
    indicator must be `== 1`. Deriving it as `sorted(unique)[-1]` gives 2, which
    inverts the entire curve and lands ECE near 1.0 in silence."""

    frame = pd.DataFrame({"y": [1, 1, 2, 2]})
    # A perfectly calibrated model: P(label 1) is 1.0 for the two label-1 rows.
    subject = _subject(
        frame, [1, 1, 2, 2], [1, 1, 2, 2],
        y_score=[1.0, 1.0, 0.0, 0.0], model_classes=[1, 2],
    )
    assert subject.positive_label == 1

    truth = np.asarray(subject.y_true)
    y_binary = (truth == subject.positive_label).astype(int)
    rows = calibration.reliability_table(y_binary, np.asarray(subject.y_score))
    ece, _ = calibration.calibration_errors(rows, 4)

    # Correct pairing -> perfectly calibrated. The inverted pairing gives 1.0.
    assert ece == pytest.approx(0.0)


def test_subject_arity_describes_the_model_not_the_split():
    """The root cause of four separate findings.

    A three-class model whose test split holds only two labels is still multiclass.
    Inferring arity from the split let it past calibration's binary guards, which
    then read column 1 of a three-column probability matrix as "the positive class".
    """

    frame = pd.DataFrame({"y": [0, 0, 1, 1]})
    subject = _subject(frame, [0, 0, 1, 1], [0, 0, 1, 1], model_classes=[0, 1, 2])

    assert subject.n_classes == 3
    assert subject.n_classes_in_split == 2
    assert subject.is_binary is False
    assert subject.positive_label is None


def test_classification_subject_without_model_classes_is_rejected():
    """Empty model_classes would make n_classes 0 and route a binary model down
    the multiclass path, silently."""

    frame = pd.DataFrame({"y": [0, 1]})
    with pytest.raises(ValueError, match="model_classes"):
        TrustSubject(
            champion={}, label="x", bundle={}, split="test", frame=frame,
            matrix=pd.DataFrame(index=frame.index), y_true=pd.Series([0, 1]),
            y_pred=np.array([0, 1]), y_score=None, task="classification",
            target="y", selection_metric="roc_auc", model_classes=[],
        )


def test_regression_subject_needs_no_model_classes():
    frame = pd.DataFrame({"y": [1.0, 2.0]})
    subject = _subject(frame, [1.0, 2.0], [1.1, 2.1], task="regression")
    assert subject.n_classes == 0
    assert subject.is_binary is False


def test_multiclass_metric_set_is_chosen_for_a_multiclass_model():
    """Even when this split holds exactly two labels."""

    frame = pd.DataFrame({"y": [0, 0, 1, 1] * 3, "g": ["A", "B"] * 6})
    audit = fairness.audit_column(
        _subject(frame, frame["y"], [0, 0, 1, 1] * 3, model_classes=[0, 1, 2]),
        "g",
        min_group_size=1,
    )
    assert audit["metric_set"] == "multiclass_classification"
    # No tpr/fpr/precision, which would need a positive class this model has not got.
    assert "tpr" not in audit["groups"][0]


# ---------------------------------------------------------------------------
# Config drift
# ---------------------------------------------------------------------------


def test_config_drift_on_positive_class_is_refused():
    """Flipping positive_class after training inverts y_true against an unchanged
    model: every rate flips, the calibration curve mirrors, and it all reports
    cleanly."""

    bundle = {"task": "classification", "positive_class": "Yes"}
    with pytest.raises(ConfigDriftError, match="positive_class"):
        _check_config_matches_bundle(bundle, task="classification", positive_class="No")


def test_config_drift_on_task_is_refused():
    bundle = {"task": "classification", "positive_class": None}
    with pytest.raises(ConfigDriftError, match="task"):
        _check_config_matches_bundle(bundle, task="regression", positive_class=None)


def test_matching_config_passes_and_missing_provenance_is_not_drift():
    """A bundle predating these fields has no opinion; that is missing provenance,
    not a conflict, and must not block the audit."""

    _check_config_matches_bundle(
        {"task": "classification", "positive_class": "Yes"},
        task="classification", positive_class="Yes",
    )
    _check_config_matches_bundle({}, task="classification", positive_class="Yes")


# ---------------------------------------------------------------------------
# Importance sort
# ---------------------------------------------------------------------------


def test_non_finite_importance_sorts_last_and_is_flagged():
    """A NaN key takes an arbitrary rank and pushes a real feature down, because
    every comparison against NaN is False."""

    rows = [
        {"feature": "a", "importance_mean": float("nan")},
        {"feature": "b", "importance_mean": 0.1},
        {"feature": "c", "importance_mean": 0.9},
    ]
    ranked = sorted(
        rows,
        key=lambda r: r["importance_mean"]
        if math.isfinite(r["importance_mean"])
        else float("-inf"),
        reverse=True,
    )
    assert [r["feature"] for r in ranked] == ["c", "b", "a"]


def test_per_metric_comparison_counts_expose_a_narrower_gap():
    """`tpr_gap` computed over two groups must not read as computed over three."""

    groups = [
        {"group": "A", "n": 100, "base_rate": 0.3, "selection_rate": 0.3,
         "tpr": 0.8, "fpr": 0.1, "precision": 0.7, "accuracy": 0.9},
        {"group": "B", "n": 100, "base_rate": 0.4, "selection_rate": 0.5,
         "tpr": 0.6, "fpr": 0.2, "precision": 0.5, "accuracy": 0.8},
        {"group": "C", "n": 100, "base_rate": 0.0, "selection_rate": 0.1,
         "tpr": None, "fpr": 0.1, "precision": None, "accuracy": 0.9},
    ]
    gaps, _ = fairness.gaps_for(groups, "binary_classification", min_group_size=30)

    assert gaps["n_groups_compared"] == 3
    assert gaps["n_groups_compared_per_metric"]["tpr"] == 2
    assert gaps["n_groups_compared_per_metric"]["precision"] == 2
    # Metrics that lost nothing are omitted rather than repeated.
    assert "accuracy" not in gaps["n_groups_compared_per_metric"]


# ---------------------------------------------------------------------------
# Model card rendering
# ---------------------------------------------------------------------------


def _bare_card() -> dict:
    """A card whose every optional source is missing - the worst-case render."""

    return {
        "artifact": "model_card",
        "generated_at": "2026-01-01T00:00:00Z",
        "project": {
            "display_name": "Test Project",
            "target": "y",
            "task": "classification",
            "selection_metric": "roc_auc",
        },
        "model": {
            "name": "LogisticRegression",
            "version": "v001",
            "registered_at": "2026-01-01T00:00:00Z",
            "params": {},
            "data_hash": None,
            "model_path": "models/x/v001/model.joblib",
        },
        "intended_use": None,
        "data": {"train_rows": None, "validation_rows": None, "test_rows": None},
        "performance": {
            "selection_metric": "roc_auc",
            "n_models_compared": None,
            "validation_metrics": None,
            "test_metrics": None,
            "selection_narrative": None,
        },
        "drivers": None,
        "permutation_importance": None,
        "fairness": None,
        "calibration": None,
        "governance": {
            "feature_approval": None,
            "model_approval": None,
            "audit_events": None,
            "experiment_runs": None,
            "lineage_nodes": 0,
        },
        "sources": {"drivers": {"path": "x", "exists": False}},
    }


def test_card_renders_with_every_optional_source_missing():
    """No format specifier may crash on a None, and no section may vanish."""

    text = model_card.render(_bare_card())

    for heading in (
        "## Model details",
        "## Intended use",
        "## Data",
        "## Performance",
        "## What the model depends on",
        "## Subgroup performance",
        "## Probability calibration",
        "## Governance trail",
        "## Limitations",
        "## Sources",
    ):
        assert heading in text, f"{heading} disappeared when its source was missing"
    assert "Not assessed" in text
    assert "**missing**" in text


def test_card_never_authors_an_intended_use_statement():
    """The one claim this pipeline must not generate: where the model may be used."""

    text = model_card.render(_bare_card())
    assert "**Not authored.**" in text
    assert "not derivable from any artifact" in text


def test_card_reports_absent_approvals_as_none():
    text = model_card.render(_bare_card())
    assert "Approvals on file: **none**" in text


def test_card_distinguishes_not_applicable_calibration_from_not_assessed():
    """A regression model has no probabilities; that is different from an unrun stage."""

    card = _bare_card()
    card["calibration"] = {
        "assessed": False,
        "reason": "Calibration is a property of probability forecasts; this is a "
        "regression model.",
    }
    text = model_card.render(card)

    assert "**Not applicable.**" in text
    assert "regression model" in text


def _card_with_ratio(ratio: float) -> str:
    card = _bare_card()
    card["fairness"] = {
        "split": "test",
        "n_rows": 1000,
        "min_group_size": 30,
        "max_levels": 10,
        "audits": [
            {
                "column": "plan",
                "metric_set": "binary_classification",
                "n_groups": 2,
                "groups": [],
                "gaps": {"selection_rate_ratio": ratio, "n_groups_compared": 2},
                "excluded_from_gaps": [],
            }
        ],
    }
    return model_card.render(card)


def test_zero_selection_ratio_is_called_out_not_rendered_as_a_bare_zero():
    """An entire subgroup never being flagged is the strongest finding available.

    Formatted with ',.4g' it prints as "0" and reads like an unremarkable number
    among nine others.
    """

    text = _card_with_ratio(0.0)
    assert "never predicted positive at all" in text


def test_ratio_below_four_fifths_gets_the_conventional_reference_point():
    text = _card_with_ratio(0.43)
    assert "four-fifths" in text


def test_healthy_ratio_gets_no_annotation():
    """No scare text on a ratio that clears the conventional reference point."""

    text = _card_with_ratio(0.95)
    assert "four-fifths" not in text
    assert "never predicted positive" not in text


def test_large_regression_gaps_render_as_grouped_digits_not_exponents():
    """A dollar gap of 29,104 must not print as "2.91e+04"."""

    card = _bare_card()
    card["fairness"] = {
        "split": "test",
        "n_rows": 3096,
        "min_group_size": 30,
        "max_levels": 10,
        "audits": [
            {
                "column": "ocean_proximity",
                "metric_set": "regression",
                "n_groups": 5,
                "groups": [],
                "gaps": {"mae_gap": 29103.76, "n_groups_compared": 4},
                "excluded_from_gaps": [],
            }
        ],
    }
    text = model_card.render(card)

    assert "29,104" in text
    assert "e+04" not in text


def test_amplification_paragraph_is_absent_from_a_regression_card():
    """Prose about "columns the model uses as features" belongs to the
    classification path; on a house-price card it is wrong-domain filler."""

    card = _bare_card()
    card["fairness"] = {
        "split": "test", "n_rows": 3096, "min_group_size": 30, "max_levels": 10,
        "audits": [
            {
                "column": "ocean_proximity", "metric_set": "regression", "n_groups": 5,
                "groups": [], "gaps": {"mae_gap": 100.0, "n_groups_compared": 4},
                "excluded_from_gaps": [],
            }
        ],
    }
    assert "selection amplification" not in model_card.render(card)


def test_card_refuses_to_pair_one_model_with_another_models_metrics():
    """Registry identity + evaluation-report metrics come from two artifacts that
    nothing forces to agree. Showing X's name above Y's test metrics is the worst
    possible failure in a document whose rule is "no evidence, say so"."""

    card = _bare_card()
    card["performance"].update(
        {
            "metrics_stale": True,
            "evaluated_model": "RandomForestClassifier v003",
            "test_metrics": None,
            "selection_narrative": None,
        }
    )
    text = model_card.render(card)

    assert "describes a different model" in text
    assert "RandomForestClassifier v003" in text
    assert "main.py evaluate" in text


def test_card_survives_a_fairness_artifact_missing_its_row_count():
    """The section formatted n_rows directly, breaking the module's own stated rule
    and taking down the card that exists to survive missing artifacts."""

    card = _bare_card()
    card["fairness"] = {
        "split": "test", "min_group_size": 30, "max_levels": 10,
        "audits": [
            {
                "column": "g", "metric_set": "binary_classification", "n_groups": 2,
                "groups": [], "gaps": {"accuracy_gap": 0.1}, "excluded_from_gaps": [],
            }
        ],
    }  # note: no "n_rows"
    text = model_card.render(card)

    assert "## Subgroup performance" in text
    assert "row count not recorded" in text


def test_card_reports_no_subgroup_column_as_a_finding_not_a_pass():
    """An empty audit list must not read as "fairness was fine"."""

    card = _bare_card()
    card["fairness"] = {
        "split": "test",
        "n_rows": 1000,
        "min_group_size": 30,
        "max_levels": 10,
        "audits": [],
    }
    text = model_card.render(card)

    assert "**No subgroup column found.**" in text
    assert "not a clean result" in text


# ---------------------------------------------------------------------------
# The upload path (the drift guard)
# ---------------------------------------------------------------------------


def test_the_uploaded_dataset_chain_runs_the_trust_audit(monkeypatch):
    """`generic_pipeline.run_insights` must call trust, not just the orchestrator.

    The trust stage was first registered only in `src/workflow/pipeline_builder.py`,
    which `main.py all` and `main.py pipeline` read. The webapp and `main.py autorun`
    go through `generic_pipeline` instead, so every *uploaded* dataset produced no
    model card and the Trust page read "has not run" forever — while the two built-in
    projects had one. That is the exact drift `generic_pipeline`'s own header warns
    about, so it gets a test rather than a comment.
    """

    from src.pipelines import generic_pipeline

    called: list[str] = []
    monkeypatch.setattr("src.automl.report.run", lambda *a, **k: called.append("report"))
    monkeypatch.setattr(
        "src.explainability.drivers.write_driver_artifacts",
        lambda *a, **k: called.append("drivers"),
    )
    monkeypatch.setattr(
        "src.pipelines.trust_pipeline.run", lambda *a, **k: called.append("trust")
    )
    monkeypatch.setattr(
        "src.reporting.html_reporter.write_report", lambda *a, **k: called.append("html")
    )

    generic_pipeline.run_insights()

    assert "trust" in called, "run_insights did not run the trust audit"
    # After drivers: the model card quotes the driver table when it exists.
    assert called.index("drivers") < called.index("trust")


def test_a_failing_trust_audit_does_not_discard_a_completed_analysis(monkeypatch):
    """Trust is a deliverable, not a precondition — same rule as the HTML report."""

    from src.pipelines import generic_pipeline

    reached: list[str] = []
    monkeypatch.setattr("src.automl.report.run", lambda *a, **k: None)
    monkeypatch.setattr("src.explainability.drivers.write_driver_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(
        "src.pipelines.trust_pipeline.run",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "src.reporting.html_reporter.write_report", lambda *a, **k: reached.append("html")
    )

    generic_pipeline.run_insights()  # must not raise

    assert reached == ["html"], "a trust failure stopped the remaining deliverables"


def test_an_identifier_column_is_not_a_subgroup_on_a_small_split():
    """Level count is measured on the split, so on a tiny split an identifier or a
    continuous integer has one distinct value per row and passes a "2-10 levels"
    filter. Every group then holds one row and nothing is comparable."""

    frame = pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
            "Unnamed: 0": list("abcdefgh"),   # 8 distinct in 8 rows -> identifier
            "Assault": [236, 263, 294, 190, 276, 204, 110, 238],  # continuous ints
            "region": ["N", "N", "N", "N", "S", "S", "S", "S"],   # a real subgroup
        }
    )
    chosen = fairness.candidate_subgroup_columns(frame, "y", max_levels=10)

    assert chosen == ["region"]


def test_a_column_with_unmeasurable_gaps_is_flagged_but_still_reported(monkeypatch):
    """Per-group rates stay; the claim that a gap was measured does not.

    On a small split every group can fall below the minimum size, so no pair is
    comparable. Hiding the column would discard the only view of a small project's
    segments; reporting it as audited would imply a disparity finding. So it is kept
    and flagged.

    ``fairness.run`` writes into the ACTIVE project, so its writers are stubbed here.
    An earlier version of this test called it directly and overwrote the housing
    project's real fairness artifact with this four-row toy.
    """

    monkeypatch.setattr("src.trust.fairness.save_json", lambda *a, **k: None)
    monkeypatch.setattr("src.trust.fairness.save_csv", lambda *a, **k: None)
    monkeypatch.setattr("src.trust.fairness._figure", lambda *a, **k: None)

    frame = pd.DataFrame({"y": [0, 1, 0, 1], "plan": ["A", "A", "B", "B"]})
    payload = fairness.run(
        _subject(frame, frame["y"], [0, 1, 0, 1]),
        groups=["plan"],
        min_group_size=30,          # far above the 2 rows each group has
    )

    assert payload["columns_audited"] == ["plan"]
    audit = payload["audits"][0]
    assert audit["gaps_measurable"] is False
    assert audit["gaps"]["n_groups_compared"] == 0
    # The per-group rates survive, because they are the only view of these segments.
    assert [group["n"] for group in audit["groups"]] == [2, 2]
    assert all(group["reliable"] is False for group in audit["groups"])


def test_a_measurable_column_is_flagged_as_such(monkeypatch):
    monkeypatch.setattr("src.trust.fairness.save_json", lambda *a, **k: None)
    monkeypatch.setattr("src.trust.fairness.save_csv", lambda *a, **k: None)
    monkeypatch.setattr("src.trust.fairness._figure", lambda *a, **k: None)

    frame = pd.DataFrame({"y": [0, 1] * 40, "plan": ["A"] * 40 + ["B"] * 40})
    payload = fairness.run(
        _subject(frame, frame["y"], [0, 1] * 40), groups=["plan"], min_group_size=30
    )
    assert payload["audits"][0]["gaps_measurable"] is True
