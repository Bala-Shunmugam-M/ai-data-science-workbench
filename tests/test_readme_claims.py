"""
tests.test_readme_claims
========================

The README states facts about the code: how many stages there are and in what
order, how the data is split, how many models the catalogue holds, how many
candidate values a sweep tries. Those numbers were correct when written and have
no mechanism keeping them correct afterwards.

That is not hypothetical. A draft of the README claimed every model has
"exactly one tunable hyperparameter". The catalogue reports ``[0, 1]``:
LinearRegression has none, which is precisely what makes it the honest baseline.
The sentence was plausible, well-written and wrong, and it was caught by reading
the code by hand rather than by anything automatic.

These tests close that gap. They are deliberately about NUMBERS AND NAMES, not
prose: documentation should be free to be rewritten, and pinned only where it
asserts something the code can contradict.

A failure here means one of two things, and the fix differs:
  * the code changed and the README now lies -> update the README
  * the README was always wrong -> fix it, and ask what else it claims
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from config.constants import (
    ALPHA_GRID_SIZE,
    RANDOM_STATE,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from src.governance.decision_tracker import DEFAULT_DECISIONS
from src.model_proposal.model_catalog import MODEL_CATALOG
from src.workflow.pipeline_builder import STAGE_NAMES

README = Path(__file__).resolve().parents[1] / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def test_readme_states_the_real_number_of_stages(readme):
    """"Twelve stages" must not survive someone adding a thirteenth."""
    spelled = {
        10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen",
    }.get(len(STAGE_NAMES))
    assert spelled, f"no spelling for {len(STAGE_NAMES)} stages; extend the map"
    assert f"{spelled} stages" in readme, (
        f"the pipeline has {len(STAGE_NAMES)} stages, so the README should say "
        f"'{spelled} stages'")


def test_readme_pipeline_diagram_lists_every_stage_in_order(readme):
    """
    The diagram is the first thing a reader trusts, and a stale one is worse
    than none: it teaches an order the orchestrator will refuse to run.
    """
    blocks = re.findall(r"```\n(.*?)```", readme, flags=re.DOTALL)
    diagrams = [b for b in blocks if "ingest" in b and "→" in b]
    assert diagrams, "no pipeline diagram found in the README"

    diagram = diagrams[0]
    found = [name for name in STAGE_NAMES if re.search(rf"\b{name}\b", diagram)]
    missing = [name for name in STAGE_NAMES if name not in found]
    assert not missing, f"stages absent from the README diagram: {missing}"
    assert found == list(STAGE_NAMES), (
        f"the diagram lists stages in a different order than the registry.\n"
        f"  registry: {list(STAGE_NAMES)}\n"
        f"  diagram : {found}")


# ---------------------------------------------------------------------------
# Splits and tuning
# ---------------------------------------------------------------------------
def test_readme_quotes_the_configured_split(readme):
    percentages = [int(round(f * 100)) for f in
                   (TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION)]
    spaced = " / ".join(str(p) for p in percentages)     # "70 / 15 / 15"
    tight = "/".join(str(p) for p in percentages)        # "70/15/15"
    assert spaced in readme or tight in readme, (
        f"constants.py splits {spaced}; the README does not say so")


def test_readme_quotes_the_configured_grid_size(readme):
    assert f"{ALPHA_GRID_SIZE} candidate" in readme, (
        f"ALPHA_GRID_SIZE is {ALPHA_GRID_SIZE}; the README should say "
        f"'{ALPHA_GRID_SIZE} candidate values'")


def test_readme_does_not_contradict_the_random_seed(readme):
    """Any 'seed NN' claim must be the seed actually used."""
    claimed = {int(n) for n in re.findall(r"seed[^\d\n]{0,12}(\d+)", readme)}
    wrong = {n for n in claimed if n != RANDOM_STATE}
    assert not wrong, (
        f"README mentions seed {sorted(wrong)}; RANDOM_STATE is {RANDOM_STATE}")


# ---------------------------------------------------------------------------
# Catalogue and governance
# ---------------------------------------------------------------------------
def test_readme_states_the_real_number_of_models(readme):
    spelled = {4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}.get(
        len(MODEL_CATALOG))
    assert spelled, f"no spelling for {len(MODEL_CATALOG)} models; extend the map"
    assert f"{spelled} models" in readme, (
        f"the catalogue holds {len(MODEL_CATALOG)}, so the README should say "
        f"'{spelled} models'")


def test_readme_does_not_overclaim_the_tunable_count():
    """
    The regression this file exists for.

    'exactly one tunable hyperparameter' was false because LinearRegression has
    none. The assertion is on the CODE, so if a future model gains a second
    tunable parameter then 'at most one' becomes false too and this fails.
    """
    counts = {len(spec.search_space) for spec in MODEL_CATALOG.values()}
    assert counts <= {0, 1}, (
        f"a model now has more than one tunable hyperparameter ({sorted(counts)}); "
        f"the README's 'at most one' claim - and the single-sweep tuner that "
        f"relies on it - are both no longer true")


def test_readme_states_the_real_number_of_standing_decisions(readme):
    spelled = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}.get(
        len(DEFAULT_DECISIONS))
    assert spelled, f"no spelling for {len(DEFAULT_DECISIONS)}; extend the map"
    assert f"{spelled} standing" in readme, (
        f"{len(DEFAULT_DECISIONS)} decisions are seeded, so the README should "
        f"say '{spelled} standing'")


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------
def test_readme_defers_to_links_file(readme):
    """URLs live in one file; the README should point there, not duplicate."""
    links = README.parent / "LINKS.md"
    assert links.exists(), "LINKS.md is missing"
    assert "LINKS.md" in readme, "the README should link to LINKS.md"
