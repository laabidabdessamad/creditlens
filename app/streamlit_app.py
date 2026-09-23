"""CreditLens - Streamlit front end. Talks to the FastAPI service over HTTP.

CREDITLENS_API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

DEFAULT_API = "https://creditlens-api.onrender.com"
ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "reports" / "figures"


def _api_url() -> str:
    try:
        url = st.secrets.get("CREDITLENS_API_URL")
    except Exception:  # no secrets file locally
        url = None
    return (url or os.getenv("CREDITLENS_API_URL") or DEFAULT_API).rstrip("/")


API = _api_url()

st.set_page_config(page_title="CreditLens", page_icon="🔍", layout="wide")

JOBS = {
    "Unskilled, non-resident": 0,
    "Unskilled, resident": 1,
    "Skilled employee": 2,
    "Highly skilled / management": 3,
}
NO_ACCOUNT = "No account / unknown"
PRESETS = {
    "Custom": dict(
        job="Skilled employee",
        housing="own",
        sav="little",
        chk="moderate",
        amount=4500,
        months=24,
        purpose="car",
    ),
    "Small, short loan - strong finances": dict(
        job="Skilled employee",
        housing="own",
        sav="rich",
        chk="rich",
        amount=900,
        months=8,
        purpose="radio/TV",
    ),
    "Large, long loan - weak finances": dict(
        job="Unskilled, resident",
        housing="rent",
        sav="little",
        chk="little",
        amount=11000,
        months=48,
        purpose="business",
    ),
}


# ---------------------------------------------------------------- API helpers
def wake_api(max_wait: int = 100) -> bool:
    """Render's free tier sleeps after inactivity; the first request can take ~30-60 s."""
    if st.session_state.get("api_ready"):
        return True
    start = time.time()
    with st.status(
        "Connecting to the scoring API (free hosting tier may need ~1 min to wake up)...",
        expanded=True,
    ) as status:
        while time.time() - start < max_wait:
            try:
                r = requests.get(f"{API}/health", timeout=8)
                if r.ok:
                    st.session_state.api_ready = True
                    status.update(label="API is ready", state="complete", expanded=False)
                    return True
            except requests.RequestException:
                pass
            st.write(f"Waiting for the API... {int(time.time() - start)}s")
            time.sleep(4)
        status.update(label="API unreachable", state="error")
    return False


@st.cache_data(ttl=3600, show_spinner=False)
def model_info() -> dict:
    return requests.get(f"{API}/model-info", timeout=20).json()


def call(endpoint: str, payload: dict) -> dict:
    r = requests.post(f"{API}/{endpoint}", json=payload, timeout=30)
    if r.status_code == 422:
        raise ValueError(r.json())
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ plots
def gauge(p: float, thr: float):
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.barh([0], [thr], color="#2e8b57", alpha=0.35)
    ax.barh([0], [1 - thr], left=thr, color="#c0392b", alpha=0.35)
    ax.plot([p], [0], marker="v", markersize=16, color="black")
    ax.axvline(thr, color="black", ls="--", lw=1)
    ax.text(thr, 0.62, f"threshold {thr:.0%}", ha="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.9)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Probability of default")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def contrib_chart(contribs: list[dict]):
    top = contribs[:7][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    colors = ["#c0392b" if c["contribution"] > 0 else "#2e8b57" for c in top]
    ax.barh(
        [f"{c['label']}\n({c['value']})" for c in top],
        [c["contribution"] for c in top],
        color=colors,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Effect on risk (log-odds):  <- lowers   |   raises ->")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- page
st.title("🔍 CreditLens")
st.caption(
    "Cost-sensitive, explainable credit-risk scoring - a portfolio project on the "
    "German Credit dataset. **Demo only, not for real lending decisions.**"
)

if not wake_api():
    st.error(f"Could not reach the API at {API}. Try again in a minute.")
    st.stop()

tab_assess, tab_perf, tab_fair = st.tabs(
    ["Assess an applicant", "Model performance", "Fairness & limitations"]
)

# ------------------------------------------------------------- tab 1: assess
with tab_assess:
    preset_name = st.selectbox("Start from an example", list(PRESETS))
    d = PRESETS[preset_name]
    left, right = st.columns([1, 1.15], gap="large")

    with left, st.form("applicant"):
        k = preset_name  # changing the preset re-keys the widgets so defaults refresh
        job = st.selectbox("Job level", list(JOBS), index=list(JOBS).index(d["job"]), key=f"j{k}")
        housing = st.selectbox(
            "Housing",
            ["own", "rent", "free"],
            index=["own", "rent", "free"].index(d["housing"]),
            key=f"h{k}",
        )
        sav_opts = [NO_ACCOUNT, "little", "moderate", "quite rich", "rich"]
        chk_opts = [NO_ACCOUNT, "little", "moderate", "rich"]
        sav = st.selectbox("Savings account", sav_opts, index=sav_opts.index(d["sav"]), key=f"s{k}")
        chk = st.selectbox(
            "Checking account", chk_opts, index=chk_opts.index(d["chk"]), key=f"c{k}"
        )
        amount = st.number_input("Credit amount (DM)", 250, 20000, d["amount"], 50, key=f"a{k}")
        months = st.slider("Duration (months)", 4, 72, d["months"], key=f"m{k}")
        purposes = [
            "car",
            "radio/TV",
            "furniture/equipment",
            "business",
            "education",
            "repairs",
            "domestic appliances",
            "vacation/others",
        ]
        purpose = st.selectbox("Purpose", purposes, index=purposes.index(d["purpose"]), key=f"p{k}")
        submitted = st.form_submit_button("Assess applicant", type="primary")

    payload = {
        "job": JOBS[job],
        "housing": housing,
        "saving_accounts": None if sav == NO_ACCOUNT else sav,
        "checking_account": None if chk == NO_ACCOUNT else chk,
        "credit_amount": int(amount),
        "duration": int(months),
        "purpose": purpose,
    }

    with right:
        if submitted or st.session_state.get("last_payload"):
            if submitted:
                st.session_state.last_payload = payload
            try:
                res = call("explain", st.session_state.last_payload)
            except (ValueError, requests.RequestException) as exc:
                st.error(f"Scoring failed: {exc}")
                st.stop()
            p, thr = res["probability_of_default"], res["decision_threshold"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Probability of default", f"{p:.1%}")
            c2.metric("Decision threshold", f"{thr:.0%}")
            (c3.error if res["decision"] == "decline" else c3.success)(
                f"**{res['decision'].upper()}**"
            )
            st.pyplot(gauge(p, thr), clear_figure=True)
            for w in res["warnings"]:
                st.warning(w)
            st.subheader("Why this score?")
            st.pyplot(contrib_chart(res["contributions"]), clear_figure=True)
            with st.expander("How to read this"):
                st.markdown(
                    "* The applicant is **declined when P(default) >= threshold**. The threshold "
                    "is low on purpose: approving a loan that defaults costs **5x** more than "
                    "declining a good one, so the lender should decline anything moderately risky.\n"
                    "* Red bars push the risk **up**, green bars push it **down** (exact TreeSHAP "
                    "values, in log-odds).\n"
                    "* The model **never sees sex or age**."
                )
        else:
            st.info("Fill in the applicant on the left and press **Assess applicant**.")

# ---------------------------------------------------------- tab 2: performance
with tab_perf:
    info = model_info()
    m = info["test_metrics"]
    a, b, c, e = st.columns(4)
    a.metric("Algorithm", info["algorithm"].replace("_", " ").title())
    b.metric("Test ROC-AUC", f"{m['roc_auc']:.3f}")
    c.metric(
        "Test cost / applicant",
        f"{m['expected_cost']:.3f}",
        help="Lower is better. "
        "'Decline everyone' costs 0.700 per applicant under the same cost matrix.",
    )
    e.metric("Approval rate", f"{m['approval_rate']:.0%}")
    st.markdown(
        f"**Cost matrix:** false negative (approve a defaulter) = "
        f"{info['cost_matrix']['false_negative']:g}, false positive (decline a good customer) = "
        f"{info['cost_matrix']['false_positive']:g}.  \n"
        f"Test set: only 200 applicants, so estimates are noisy (ROC-AUC 95% CI ~0.68-0.83)."
    )
    cols = st.columns(2)
    for i, (fn, cap) in enumerate(
        [
            ("01_roc_pr.png", "ROC / precision-recall"),
            ("02_cost_vs_threshold.png", "Cost vs. decision threshold"),
            ("03_calibration.png", "Calibration"),
            ("05_model_comparison.png", "Model comparison (CV)"),
            ("07_global_importance.png", "Global feature importance"),
            ("06_shap_summary.png", "SHAP summary"),
        ]
    ):
        if (FIGURES / fn).exists():
            cols[i % 2].image(str(FIGURES / fn), caption=cap)

# -------------------------------------------------------------- tab 3: fairness
with tab_fair:
    st.markdown(
        "The model is **trained without `sex` and `age`**. A Fairlearn audit compared four variants "
        "(with / without each attribute) on out-of-fold predictions:"
    )
    audit = ROOT / "reports" / "fairness_audit.csv"
    if audit.exists():
        df = pd.read_csv(audit).drop(columns=["variant"])
        st.dataframe(df.round(3), hide_index=True, width="stretch")
    if (FIGURES / "09_fairness_groups.png").exists():
        st.image(
            str(FIGURES / "09_fairness_groups.png"),
            caption="Decline rate and false-positive rate by group (served model)",
        )
    st.markdown(
        "**Take-aways**\n"
        "* Dropping `sex` costs almost nothing in accuracy (AUC -0.003) and roughly halves the "
        "sex disparity.\n"
        "* Dropping `age` costs ~0.007 AUC but substantially narrows the age-group gap.\n"
        "* **Disparity does not disappear**: applicants <= 25 are still declined more often, "
        "because other features (loan duration, savings, housing) act as proxies and because their "
        "observed default rate really is higher in this data. Removing a column is not the same as "
        "removing bias.\n\n"
        "**Limitations:** 1,000 applicants from a 1990s German dataset; label noise; a cost matrix "
        "that ignores interest income; no reject-inference (we only see outcomes for approved "
        "loans). This is a demonstration, not a lending decision system."
    )
