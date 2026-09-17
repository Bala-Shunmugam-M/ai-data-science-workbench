"""
tests.test_deliverable_facts
============================

The slide deck, the written report and the published Pages site all quote this
project's results. They used to hold three independent transcriptions of the
same numbers, so a pipeline re-run silently made all three stale and a hand
edit could make them contradict each other in front of a reader.

They now share ``src.reporting.facts``. These tests defend that arrangement:
the shared module must behave, and the generated documents must actually agree
with it. The failure mode being guarded against is a document that looks
authoritative and quotes a number nothing else in the repository supports.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.reporting import facts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "docs" / "PROJECT-REPORT.md"
SITE_INDEX = PROJECT_ROOT / "_site" / "index.html"


# ---------------------------------------------------------------------------
# The shared module itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("project", facts.PROJECTS, ids=lambda p: p.slug)
def test_every_project_has_a_champion(project):
    assert project.champion_name
    assert project.champion_version


@pytest.mark.parametrize("project", facts.PROJECTS, ids=lambda p: p.slug)
def test_runner_up_is_never_the_champion(project):
    """
    The registry holds a v002 of every model. Excluding only the champion ROW
    therefore lets the champion's own second version win the runner-up slot,
    and the deliverable reads "Champion: Ridge / Runner-up: Ridge" - which is
    nonsense that still renders perfectly.
    """
    assert project.runner_up_name != project.champion_name


@pytest.mark.parametrize("project", facts.PROJECTS, ids=lambda p: p.slug)
def test_runner_up_does_not_beat_the_champion(project):
    """
    Direction matters. Taking min() on a higher-is-better metric reports the
    WORST model as the runner-up, and nothing about the rendered output looks
    wrong - which is why this needs a test rather than a careful reading.
    """
    metric = project.selection_metric
    champion = project.validation_metrics[metric]
    runner = project.runner_up_metrics.get(metric)
    if runner is None:
        pytest.skip("runner-up carries no value for the selection metric")
    if metric in ("rmse", "mae", "mape"):
        assert runner >= champion
    else:
        assert runner <= champion


@pytest.mark.parametrize("project", facts.PROJECTS, ids=lambda p: p.slug)
def test_selection_metric_is_present_in_the_scores(project):
    assert project.selection_metric in project.validation_metrics
    assert project.selection_metric in project.test_metrics


def test_absent_metrics_say_so_rather_than_rendering_empty():
    """
    An empty string reads as "nobody thought to include this". "not available"
    reads as "this was looked for and is missing" - the same rule the model
    card follows.
    """
    assert facts.metric_line(None) == "not available"
    assert facts.metric_line({}) == "not available"
    assert facts.metric_line({"rmse": "n/a"}) == "not available"


def test_currency_scale_metrics_are_not_printed_as_ratios():
    assert facts.metric("rmse", 66680.81) == "RMSE $66,681"
    assert facts.metric("mae", 48731.42) == "MAE $48,731"
    # Below the currency threshold a small RMSE is a ratio, not dollars.
    assert facts.metric("rmse", 0.5492) == "RMSE 0.5492"


def test_metric_formatting_is_stable():
    """If the deck rounds to four places and the report to three, the two
    documents disagree on paper even when the value is identical."""
    assert facts.metric("r2", 0.666513) == "R2 0.667"
    assert facts.metric("roc_auc", 0.844778) == "ROC-AUC 0.8448"
    assert facts.metric("mape", 28.3237) == "MAPE 28.32%"


# ---------------------------------------------------------------------------
# The documents must agree with it
# ---------------------------------------------------------------------------
def _canonical_numbers() -> dict[str, str]:
    """The figures every deliverable is expected to quote."""
    return {
        "housing test rmse": facts.money(facts.HOUSING.test_metrics["rmse"]),
        "housing validation rmse": facts.money(
            facts.HOUSING.validation_metrics["rmse"]),
        "churn test roc_auc": f"{facts.CHURN.test_metrics['roc_auc']:.4f}",
        "churn validation roc_auc": f"{facts.CHURN.validation_metrics['roc_auc']:.4f}",
    }


@pytest.mark.skipif(not REPORT.exists(), reason="report not generated yet")
@pytest.mark.parametrize("label,value", list(_canonical_numbers().items()))
def test_report_quotes_the_shared_figures(label, value):
    text = REPORT.read_text(encoding="utf-8")
    assert value in text, (
        f"the report does not contain the current {label} ({value}). "
        f"Re-run: python docs/build_report.py")


@pytest.mark.skipif(not REPORT.exists(), reason="report not generated yet")
def test_report_has_no_unrendered_placeholders():
    """A published "{c_val_auc}" is worse than a stale number: it tells the
    reader the document was generated and not checked."""
    text = REPORT.read_text(encoding="utf-8")
    leftovers = re.findall(r"\{(?:h_|c_|cal_|imp_|amp_)\w+\}", text)
    assert not leftovers, f"unrendered placeholders: {sorted(set(leftovers))}"


@pytest.mark.skipif(not SITE_INDEX.exists(),
                    reason="pages site not built in this checkout")
@pytest.mark.parametrize("label,value", list(_canonical_numbers().items()))
def test_site_quotes_the_shared_figures(label, value):
    html = SITE_INDEX.read_text(encoding="utf-8")
    assert value in html, (
        f"the published site does not contain the current {label} ({value}). "
        f"Re-run: python .github/scripts/build_pages.py")


def test_facts_self_check_passes():
    """The module's own runnable check, exercised by the suite too."""
    facts.demo()
