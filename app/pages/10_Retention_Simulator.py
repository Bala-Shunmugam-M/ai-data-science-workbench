"""
app.pages.10_Retention_Simulator
================================

PURPOSE
-------
The churn project's interactive "what-if" retention simulator. Move the sliders
and watch the campaign economics update live: who to target by churn risk, what
discount to offer, how effective it is, and the resulting customers retained,
cost, and ROI. Self-contained: it always reads the ``churn`` workspace, so it
works no matter which project the app was launched with.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from app import project_state  # noqa: E402

_PROJECT = project_state.apply_project()

from app.common import PAGE_ICON, page_header  # noqa: E402
from src.simulation import retention  # noqa: E402

st.set_page_config(page_title="Retention Simulator", page_icon="\U0001F3AF", layout="wide")
project_state.render_selector(_PROJECT)
page_header(
    "\U0001F3AF Retention Simulator (Churn)",
    "Target high-risk customers with a discount offer and see the campaign economics live.",
)

if not retention.is_ready():
    st.warning(
        "The churn champion has not been trained yet. Run it first:\n\n"
        "```\nWORKBENCH_PROJECT=churn python main.py churn-all\n```"
    )
    st.stop()


@st.cache_data(show_spinner="Scoring customers with the champion model...")
def _scored():
    return retention.score_customers()


scored = _scored()

# --- Controls -------------------------------------------------------------
with st.sidebar:
    st.header("Campaign levers")
    threshold = st.slider(
        "Risk threshold — target customers at/above this churn probability",
        0.0, 1.0, 0.50, 0.01,
    )
    discount = st.slider("Discount offered (% of monthly bill)", 0, 50, 10, 1) / 100.0
    acceptance = st.slider(
        "Offer effectiveness — share of would-churn targets actually retained",
        0, 100, 40, 5,
    ) / 100.0
    horizon = st.slider("Value horizon (months)", 1, 36, 12, 1)

result = retention.simulate(
    scored, threshold=threshold, discount=discount,
    acceptance=acceptance, horizon_months=horizon,
)

# --- Headline metrics -----------------------------------------------------
st.subheader("Campaign outcome")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers targeted", f"{result.n_targeted:,}", f"of {result.n_customers:,}")
c2.metric("Customers retained", f"{result.retained:,.0f}", f"{result.expected_churners:,.0f} would churn")
c3.metric("Net value", f"${result.net_value:,.0f}", f"ROI {result.roi:.2f}x")
c4.metric("Campaign cost", f"${result.campaign_cost:,.0f}", f"saves ${result.revenue_saved:,.0f}")

if result.net_value > 0:
    st.success(
        f"Profitable: targeting the {result.n_targeted:,} highest-risk customers is "
        f"expected to retain ~{result.retained:,.0f} of them for a net "
        f"${result.net_value:,.0f} over {horizon} months (ROI {result.roi:.2f}x)."
    )
else:
    st.error(
        f"Unprofitable at these settings: campaign cost ${result.campaign_cost:,.0f} "
        f"exceeds the ${result.revenue_saved:,.0f} of revenue saved. Try a higher "
        "risk threshold (target fewer, riskier customers) or a smaller discount."
    )

# --- Risk distribution with the threshold marked --------------------------
left, right = st.columns([3, 2])
with left:
    st.markdown("**Churn-risk distribution** (test customers) — bars right of the line are targeted.")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scored["churn_probability"], bins=40, color="#4C78A8", edgecolor="white")
    ax.axvline(threshold, color="crimson", linewidth=2, label=f"threshold {threshold:.2f}")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Customers")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("**Highest-risk targeted customers**")
    cols = [c for c in ("churn_probability", "monthly_revenue", "Contract", "tenure") if c in scored.columns]
    targeted = (
        scored[scored["churn_probability"] >= threshold]
        .sort_values("churn_probability", ascending=False)
        .head(15)[cols]
        .rename(columns={"churn_probability": "P(churn)", "monthly_revenue": "$/mo"})
    )
    st.dataframe(targeted, hide_index=True, use_container_width=True)

st.caption(
    "Model: targeted = P(churn) ≥ threshold; retained = (Σ P over targeted) × effectiveness; "
    "revenue saved = retained customers' monthly bills × horizon; cost = targeted bills × discount × horizon. "
    "Probabilities come from the trained champion; the test split is used for scoring only."
)
