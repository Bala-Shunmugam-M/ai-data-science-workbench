"""
src.simulation.retention
=========================

PURPOSE
-------
The churn project's interactive **retention simulator** engine. It scores every
test customer's churn probability with the trained champion, then answers the
business question the Streamlit page makes playable:

    "If we offer an X% discount to every customer above risk threshold T, and a
     fraction A of the would-churn customers accept and stay, how many do we
     retain, what does the campaign cost, and what is the ROI?"

The engine is deliberately project-explicit (it always reads the ``churn``
workspace) so the simulator works from the dashboard regardless of which project
the app was launched with.

Model (simple, defensible, and stated so the Model Defence can poke at it):

    targeted            customers with predicted P(churn) >= threshold
    expected_churners   sum of P(churn) over the targeted customers
    retained            expected_churners * acceptance          (offer works A of the time)
    revenue_saved       retained * monthly_revenue * horizon_months
    campaign_cost       sum over targeted of monthly_revenue * discount * horizon_months
    net_value           revenue_saved - campaign_cost
    roi                 net_value / campaign_cost

``monthly_revenue`` is each customer's own ``MonthlyCharges``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

if __name__ == "__main__" and __package__ is None:
    # Running this file directly puts src/simulation/ on sys.path, not the
    # repository root, so `python src/simulation/retention.py` - the self-check
    # this module's docstring advertises - died on `import config`. Importing
    # it as a module (pytest, Streamlit) was always fine, which is why the
    # broken path went unnoticed.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import PROJECT_ROOT
from src.model_training.design_matrix import positive_proba

_WORKSPACES = PROJECT_ROOT / "workspaces"
_REVENUE_COLUMN = "MonthlyCharges"


def _usable(root: Path) -> bool:
    """
    True when this workspace can actually be simulated.

    Both files are required. The selection file alone is not enough: the
    repository publishes artifacts but NOT datasets or model pickles, so a
    fresh clone has a churn workspace whose final_model_selection.json exists
    and whose model and test split do not. Checking only the former is how the
    simulator ends up loading a path that is not there.
    """

    return (
        (root / "artifacts" / "final_model_selection.json").exists()
        and (root / "data" / "engineered" / "test.csv").exists()
    )


def _resolve_root() -> Path:
    """
    The churn workspace to simulate.

    ``workspaces/churn`` is preferred, so every existing local result is
    unchanged. When it is not usable - a fresh clone, or a deployment built
    from one - any other workspace carrying the revenue column is accepted, so
    a seeded demo project works instead of the page silently showing nothing.
    """

    preferred = _WORKSPACES / "churn"
    if _usable(preferred):
        return preferred

    if _WORKSPACES.is_dir():
        for candidate in sorted(_WORKSPACES.glob("*")):
            if not candidate.is_dir() or not _usable(candidate):
                continue
            # The revenue column is what makes a project simulatable at all: a
            # regression project has a champion and a test split but no notion
            # of a monthly bill to discount.
            try:
                with (candidate / "data" / "engineered" / "test.csv").open(
                        encoding="utf-8") as handle:
                    header = handle.readline()
            except OSError:
                continue
            if _REVENUE_COLUMN in header:
                return candidate

    return preferred


CHURN_ROOT: Path = _resolve_root()
_FINAL_SELECTION = CHURN_ROOT / "artifacts" / "final_model_selection.json"
_TEST_SPLIT = CHURN_ROOT / "data" / "engineered" / "test.csv"


def is_ready() -> bool:
    """True when the churn champion has been trained/evaluated and is loadable."""

    return _FINAL_SELECTION.exists() and _TEST_SPLIT.exists()


def score_customers() -> pd.DataFrame:
    """
    Return the test customers with a ``churn_probability`` and ``monthly_revenue``.

    Loads the champion bundle recorded in the churn workspace's
    ``final_model_selection.json`` and scores the untouched test split.
    """

    import json

    selection = json.loads(_FINAL_SELECTION.read_text(encoding="utf-8"))
    bundle = joblib.load(selection["model_path"])
    test_df = pd.read_csv(_TEST_SPLIT)

    predictor_columns = bundle["predictor_columns"]
    matrix = bundle["preprocessor"].transform(test_df[predictor_columns])
    proba = positive_proba(bundle["estimator"], matrix)

    out = test_df.copy()
    out["churn_probability"] = proba
    out["monthly_revenue"] = (
        pd.to_numeric(out[_REVENUE_COLUMN], errors="coerce").fillna(0.0)
        if _REVENUE_COLUMN in out.columns
        else 0.0
    )
    return out


@dataclass(frozen=True)
class SimulationResult:
    """Business outcome of one retention-campaign scenario."""

    n_customers: int
    n_targeted: int
    expected_churners: float
    retained: float
    revenue_saved: float
    campaign_cost: float
    net_value: float
    roi: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_customers": self.n_customers,
            "n_targeted": self.n_targeted,
            "expected_churners": self.expected_churners,
            "retained": self.retained,
            "revenue_saved": self.revenue_saved,
            "campaign_cost": self.campaign_cost,
            "net_value": self.net_value,
            "roi": self.roi,
        }


def simulate(
    scored: pd.DataFrame,
    *,
    threshold: float,
    discount: float,
    acceptance: float,
    horizon_months: int = 12,
) -> SimulationResult:
    """
    Simulate one retention campaign over the scored customers.

    ``threshold``   target customers with P(churn) at or above this (0-1).
    ``discount``    fraction of monthly revenue given up as the offer (0-1).
    ``acceptance``  fraction of would-churn targeted customers actually retained (0-1).
    ``horizon_months``  months over which revenue/cost accrue.
    """

    p = scored["churn_probability"].to_numpy()
    revenue = scored["monthly_revenue"].to_numpy()

    targeted_mask = p >= threshold
    n_targeted = int(targeted_mask.sum())

    expected_churners = float(p[targeted_mask].sum())
    retained = expected_churners * acceptance
    revenue_saved = float((revenue[targeted_mask] * p[targeted_mask]).sum()) * acceptance * horizon_months
    campaign_cost = float(revenue[targeted_mask].sum()) * discount * horizon_months
    net_value = revenue_saved - campaign_cost
    roi = (net_value / campaign_cost) if campaign_cost > 0 else 0.0

    return SimulationResult(
        n_customers=int(len(scored)),
        n_targeted=n_targeted,
        expected_churners=expected_churners,
        retained=retained,
        revenue_saved=revenue_saved,
        campaign_cost=campaign_cost,
        net_value=net_value,
        roi=roi,
    )


def demo() -> None:
    """Self-check: the simulation's accounting identities hold on synthetic data."""

    rng = np.random.default_rng(0)
    scored = pd.DataFrame(
        {"churn_probability": rng.uniform(0, 1, 500), "monthly_revenue": rng.uniform(20, 120, 500)}
    )

    # No one above threshold 1.01 -> everything zero, ROI zero (no divide-by-zero).
    empty = simulate(scored, threshold=1.01, discount=0.1, acceptance=0.5)
    assert empty.n_targeted == 0 and empty.campaign_cost == 0.0 and empty.roi == 0.0

    r = simulate(scored, threshold=0.5, discount=0.10, acceptance=0.4, horizon_months=12)
    assert r.n_targeted > 0
    assert 0.0 <= r.retained <= r.expected_churners  # retained can't exceed expected churners
    assert abs(r.net_value - (r.revenue_saved - r.campaign_cost)) < 1e-6
    # Higher acceptance never reduces net value (more churners saved, same cost).
    r_hi = simulate(scored, threshold=0.5, discount=0.10, acceptance=0.8, horizon_months=12)
    assert r_hi.net_value >= r.net_value
    print("retention.demo OK:", {k: round(v, 2) if isinstance(v, float) else v for k, v in r.as_dict().items()})


if __name__ == "__main__":
    demo()
