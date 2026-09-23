"""Fairness audit with Fairlearn.

Compares four variants of the champion model (with/without the sensitive attributes
`sex` and `age`) on out-of-fold predictions over the full dataset, and reports
performance, cost and group disparities.

    python -m creditlens.fairness
"""

from __future__ import annotations

import json
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from creditlens import cost as C
from creditlens.config import FIGURES_DIR, REPORTS_DIR, load_params
from creditlens.data import load_clean, split_xy
from creditlens.model import _logit
from creditlens.train import make_pipeline, oof_predict

AGE_BINS = [0, 25, 35, 50, 200]
AGE_LABELS = ["<=25", "26-35", "36-50", ">50"]


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BINS, labels=AGE_LABELS).astype(str)


def _variants() -> list[tuple[str, ...]]:
    sens = ("sex", "age")
    return [c for r in range(0, 3) for c in combinations(sens, r)]


def audit(family: str, best_params: dict) -> tuple[pd.DataFrame, dict]:
    params = load_params()
    seed, fn_c, fp_c = (
        params["seed"],
        params["cost"]["false_negative"],
        params["cost"]["false_positive"],
    )
    X, y = split_xy(load_clean())
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    groups = {"sex": X["sex"].astype(str).values, "age_group": age_group(X["age"]).values}

    rows, group_tables = [], {}
    for sens in _variants():
        pipe = make_pipeline(family, best_params, pos_weight, seed, sens)
        p_raw = oof_predict(pipe, X, y, seed)
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw).reshape(-1, 1), y)
        p = cal.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]
        thr, cost = C.best_threshold(y, p, fn_c, fp_c)
        pred = (p >= thr).astype(int)
        row = {
            "variant": "+".join(sens)
            if sens
            else "none (served)"
            if False
            else ("+".join(sens) or "none"),
            "sensitive_in_model": ",".join(sens) or "none",
            "roc_auc": roc_auc_score(y, p),
            "threshold": thr,
            "expected_cost": cost,
            "decline_rate": float(pred.mean()),
        }
        for gname, gvals in groups.items():
            row[f"dp_diff_{gname}"] = demographic_parity_difference(
                y, pred, sensitive_features=gvals
            )
            row[f"eo_diff_{gname}"] = equalized_odds_difference(y, pred, sensitive_features=gvals)
        rows.append(row)
        for gname, gvals in groups.items():
            mf = MetricFrame(
                metrics={
                    "decline_rate": selection_rate,
                    "tpr_bad_caught": true_positive_rate,
                    "fpr_good_declined": false_positive_rate,
                },
                y_true=y,
                y_pred=pred,
                sensitive_features=gvals,
            )
            t = mf.by_group.copy()
            t["actual_bad_rate"] = pd.Series(y.values).groupby(gvals).mean()
            t["n"] = pd.Series(gvals).value_counts()
            t["avg_cost"] = pd.Series(
                [
                    C.expected_cost(y.values[gvals == g], pred[gvals == g], fn_c, fp_c)
                    for g in t.index
                ],
                index=t.index,
            )
            group_tables[(row["sensitive_in_model"], gname)] = t
    return pd.DataFrame(rows), group_tables


def main() -> None:
    meta = json.loads((REPORTS_DIR / "metrics.json").read_text())
    from creditlens.config import MODEL_PATH
    from creditlens.model import CreditRiskModel

    m = CreditRiskModel.load(MODEL_PATH)
    family, bp = m.metadata["family"], m.metadata["best_params"]
    table, groups = audit(family, bp)
    table.to_csv(REPORTS_DIR / "fairness_audit.csv", index=False)

    served = ",".join(m.sensitive) or "none"
    frames = []
    for (variant, gname), t in groups.items():
        if variant in (served, "sex,age"):
            tt = t.reset_index().rename(columns={t.index.name or "sensitive_feature_0": "group"})
            tt.insert(0, "attribute", gname)
            tt.insert(0, "variant", variant)
            frames.append(tt)
    pd.concat(frames).to_csv(REPORTS_DIR / "fairness_groups.csv", index=False)

    # figure: decline rate vs actual bad rate per group, served model
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, gname in zip(axes, ("sex", "age_group"), strict=False):
        t = groups[(served, gname)].reindex(AGE_LABELS if gname == "age_group" else None)
        x = np.arange(len(t))
        ax.bar(x - 0.27, t["actual_bad_rate"], 0.27, label="Actual bad rate", color="#7f8c8d")
        ax.bar(x, t["decline_rate"], 0.27, label="Decline rate", color="#1f5fa8")
        ax.bar(
            x + 0.27,
            t["fpr_good_declined"],
            0.27,
            label="Good applicants declined (FPR)",
            color="#c0392b",
        )
        ax.set_xticks(x, t.index)
        ax.set_title(f"By {gname.replace('_', ' ')} (model uses: {served})")
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_fairness_groups.png", dpi=140)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    base = table.set_index("sensitive_in_model")
    print(
        "\nAUC loss from dropping ALL sensitive features:",
        round(base.loc["sex,age", "roc_auc"] - base.loc["none", "roc_auc"], 4),
    )
    print(
        "AUC loss from dropping sex only:",
        round(base.loc["sex,age", "roc_auc"] - base.loc["age", "roc_auc"], 4),
    )
    _ = meta


if __name__ == "__main__":
    main()
