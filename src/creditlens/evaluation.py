"""Metrics, bootstrap confidence intervals and evaluation figures."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from creditlens import cost as C
from creditlens.config import FIGURES_DIR, REPORTS_DIR, load_params

PALETTE = {"main": "#1f5fa8", "bad": "#c0392b", "good": "#2e8b57", "grey": "#7f8c8d"}


def bootstrap_ci(fn, y, p, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of ``fn(y, p)`` (stratification not needed at n=200)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holdout_report(y, p_cal, p_raw, threshold: float, fn_cost=5.0, fp_cost=1.0) -> dict:
    """All headline numbers for the held-out test set (computed exactly once)."""
    y = np.asarray(y)
    pred = (np.asarray(p_cal) >= threshold).astype(int)
    cost_fn = lambda yy, pp: C.expected_cost(yy, (pp >= threshold).astype(int), fn_cost, fp_cost)  # noqa: E731
    base = C.baseline_costs(y, fn_cost, fp_cost)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    cost = cost_fn(y, p_cal)
    saving_fn = lambda yy, pp: (  # noqa: E731
        C.baseline_costs(yy, fn_cost, fp_cost)["decline_all"] - cost_fn(yy, pp)
    )
    return {
        "n_test": int(len(y)),
        "bad_rate": float(y.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p_cal)),
        "roc_auc_ci95": bootstrap_ci(roc_auc_score, y, p_cal),
        "pr_auc": float(average_precision_score(y, p_cal)),
        "brier_raw": float(brier_score_loss(y, p_raw)),
        "brier_calibrated": float(brier_score_loss(y, p_cal)),
        "expected_cost": float(cost),
        "expected_cost_ci95": bootstrap_ci(cost_fn, y, p_cal),
        "cost_approve_all": base["approve_all"],
        "cost_decline_all": base["decline_all"],
        "cost_at_threshold_0.5": C.expected_cost(
            y, (np.asarray(p_cal) >= 0.5).astype(int), fn_cost, fp_cost
        ),
        "saving_vs_decline_all": float(base["decline_all"] - cost),
        "saving_vs_decline_all_ci95": bootstrap_ci(saving_fn, y, p_cal),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "recall_bad": float(tp / max(tp + fn, 1)),
        "precision_bad": float(tp / max(tp + fp, 1)),
        "approval_rate": float((pred == 0).mean()),
    }


# ------------------------------------------------------------------------- figures
def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_figures() -> None:
    """Build evaluation figures from the artifacts written by train.py."""
    fn_c, fp_c = (load_params()["cost"][k] for k in ("false_negative", "false_positive"))
    oof = pd.read_csv(REPORTS_DIR / "oof_train.csv")
    test = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())
    thr = metrics["test"]["threshold"]
    comp = pd.read_csv(REPORTS_DIR / "model_comparison.csv")

    # 1. ROC + PR
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    RocCurveDisplay.from_predictions(
        test["y"],
        test["p_cal"],
        ax=ax[0],
        curve_kwargs={"color": PALETTE["main"]},
        name=f"CreditLens (AUC {metrics['test']['roc_auc']:.3f})",
    )
    ax[0].plot([0, 1], [0, 1], "--", color=PALETTE["grey"])
    ax[0].set_title("ROC curve (held-out test set)")
    pr, rc, _ = precision_recall_curve(test["y"], test["p_cal"])
    ax[1].plot(rc, pr, color=PALETTE["main"])
    ax[1].axhline(test["y"].mean(), ls="--", color=PALETTE["grey"], label="No-skill (bad rate)")
    ax[1].set(xlabel="Recall (bad)", ylabel="Precision (bad)", title="Precision-Recall curve")
    ax[1].legend()
    _save(fig, "01_roc_pr.png")

    # 2. Cost vs threshold
    grid = C.DEFAULT_GRID
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(
        grid,
        C.cost_curve(oof["y"], oof["p_cal"], fn_c, fp_c),
        label="Train (out-of-fold)",
        color=PALETTE["main"],
    )
    ax.plot(
        grid, C.cost_curve(test["y"], test["p_cal"], fn_c, fp_c), label="Test", color=PALETTE["bad"]
    )
    base = C.baseline_costs(test["y"], fn_c, fp_c)
    ax.axhline(base["decline_all"], ls="--", color=PALETTE["grey"], label="Decline everyone")
    ax.axhline(base["approve_all"], ls=":", color=PALETTE["grey"], label="Approve everyone")
    ax.axvline(thr, color="black", lw=1, label=f"Chosen threshold {thr:.2f}")
    ax.set(
        xlabel="Decision threshold on P(bad)",
        ylabel="Expected cost per applicant",
        title=f"Cost-sensitive threshold selection (FN={fn_c:g} x FP)",
    )
    ax.legend(fontsize=8)
    _save(fig, "02_cost_vs_threshold.png")

    # 3. Calibration
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for col, lab, color in (
        ("p_raw", "Raw model", PALETTE["grey"]),
        ("p_cal", "Calibrated (served)", PALETTE["main"]),
    ):
        fr, mp = calibration_curve(test["y"], test[col], n_bins=6, strategy="quantile")
        ax.plot(mp, fr, "o-", label=lab, color=color)
    ax.plot([0, 1], [0, 1], "--", color="black", lw=1)
    ax.set(
        xlabel="Predicted P(bad)", ylabel="Observed frequency of bad", title="Calibration (test)"
    )
    ax.legend()
    _save(fig, "03_calibration.png")

    # 4. Confusion matrix at chosen threshold
    pred = (test["p_cal"] >= thr).astype(int)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay.from_predictions(
        test["y"], pred, display_labels=["good", "bad"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix @ threshold {thr:.2f}\n(rows: truth, cols: decision)")
    ax.set_xlabel("Predicted (bad = decline)")
    _save(fig, "04_confusion_matrix.png")

    # 5. Model comparison (cross-validation)
    comp = comp.sort_values("cv_roc_auc")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(
        comp["model"],
        comp["cv_roc_auc"],
        xerr=comp["cv_roc_auc_std"],
        color=PALETTE["main"],
        alpha=0.85,
        capsize=3,
    )
    ax.axvline(0.5, ls="--", color=PALETTE["grey"])
    ax.set_xlim(0.45, max(0.85, comp["cv_roc_auc"].max() + 0.05))
    ax.set(
        xlabel="Cross-validated ROC-AUC (mean +/- std)",
        title="Model comparison (tuned with Optuna)",
    )
    _save(fig, "05_model_comparison.png")
    print("figures written to", FIGURES_DIR)


if __name__ == "__main__":
    make_figures()
