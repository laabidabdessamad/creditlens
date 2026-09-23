"""Global and local explanations with SHAP (offline analysis; the API uses native contributions).

python -m creditlens.explain
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from creditlens.config import FIGURES_DIR, MODEL_PATH, REPORTS_DIR
from creditlens.data import train_test
from creditlens.model import CreditRiskModel

LABELS = {
    "checking_account": "Checking account",
    "saving_accounts": "Savings account",
    "duration": "Loan duration",
    "log_credit_amount": "Credit amount (log)",
    "monthly_payment": "Monthly payment",
    "purpose": "Loan purpose",
    "housing": "Housing",
    "job": "Job level",
    "age": "Age",
    "sex": "Sex",
}


def shap_values(model: CreditRiskModel, Xt: np.ndarray) -> np.ndarray:
    est = model.estimator
    if hasattr(est, "coef_"):
        expl = shap.LinearExplainer(est, model.background_mean.reshape(1, -1))
        sv = expl.shap_values(Xt)
    else:
        sv = shap.TreeExplainer(est).shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if getattr(sv, "ndim", 2) == 3 else sv)
    return np.asarray(sv)


def local_bar(ax, names, vals, title):
    order = np.argsort(np.abs(vals))[::-1][:7][::-1]
    colors = ["#c0392b" if vals[i] > 0 else "#2e8b57" for i in order]
    ax.barh([LABELS.get(names[i], names[i]) for i in order], vals[order], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Contribution to risk (log-odds)  <- safer | riskier ->")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = CreditRiskModel.load(MODEL_PATH)
    _, X_test, _, y_test = train_test()
    Xt = model.transform(X_test)
    sv = shap_values(model, Xt)

    # --- consistency check: the API's native contributions must equal SHAP
    groups, grouped, _ = model.contributions(X_test)
    sv_grouped = np.zeros_like(grouped)
    for j, name in enumerate(model.feature_names):
        sv_grouped[:, groups.index(name.split("=")[0])] += sv[:, j]
    max_diff = float(np.abs(sv_grouped - grouped).max())

    # 1. beeswarm on encoded features
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, Xt, feature_names=model.feature_names, show=False, max_display=15)
    plt.title("SHAP summary: what drives predicted default risk")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 2. grouped global importance
    imp = pd.Series(np.abs(grouped).mean(axis=0), index=groups).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([LABELS.get(i, i) for i in imp.index], imp.values, color="#1f5fa8")
    ax.set(
        xlabel="Mean |SHAP| (log-odds)",
        title="Global feature importance (grouped by original feature)",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_global_importance.png", dpi=140)
    plt.close(fig)

    # 3. local explanations: riskiest declined applicant vs safest approved applicant
    p = model.predict_proba(X_test)
    i_hi, i_lo = int(np.argmax(p)), int(np.argmin(p))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    local_bar(
        axes[0],
        groups,
        grouped[i_hi],
        f"Highest-risk applicant  (P(bad) = {p[i_hi]:.0%})  -> DECLINE",
    )
    local_bar(
        axes[1],
        groups,
        grouped[i_lo],
        f"Lowest-risk applicant  (P(bad) = {p[i_lo]:.0%})  -> APPROVE",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_local_explanations.png", dpi=140)
    plt.close(fig)

    out = {
        "native_vs_shap_max_abs_diff": max_diff,
        "global_importance": {k: float(v) for k, v in imp.sort_values(ascending=False).items()},
    }
    (REPORTS_DIR / "explainability.json").write_text(json.dumps(out, indent=2))
    print(f"native-vs-SHAP max diff: {max_diff:.2e}")
    print("top features:", list(out["global_importance"])[:5])


if __name__ == "__main__":
    main()
