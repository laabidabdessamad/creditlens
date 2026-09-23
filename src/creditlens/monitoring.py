"""Data / prediction drift monitoring with Evidently.

Simulates two production batches against the training data (the reference):

* ``stable``  - the held-out test split (same population)
* ``shifted`` - a stress scenario ("downturn"): larger and longer loans, more business loans,
                more applicants without a checking account

Outputs interactive HTML reports plus a compact JSON summary in reports/monitoring/.

    python -m creditlens.monitoring
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from creditlens.config import MODEL_PATH, NO_ACCOUNT, REPORTS_DIR
from creditlens.data import train_test
from creditlens.model import CreditRiskModel

OUT = REPORTS_DIR / "monitoring"
NUMERIC = ["job", "credit_amount", "duration", "predicted_risk"]
CATEGORICAL = ["housing", "saving_accounts", "checking_account", "purpose"]


def simulate_shift(X: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A plausible downturn: bigger/longer loans, more business loans, thinner bank records."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    X["credit_amount"] = (X["credit_amount"] * 1.6).round().astype(int)
    X["duration"] = (X["duration"] + 10).clip(upper=72)
    X.loc[rng.random(len(X)) < 0.30, "purpose"] = "business"
    X.loc[rng.random(len(X)) < 0.40, "checking_account"] = np.nan
    return X


def _frame(model: CreditRiskModel, X: pd.DataFrame) -> pd.DataFrame:
    df = X[
        [
            "job",
            "housing",
            "saving_accounts",
            "checking_account",
            "credit_amount",
            "duration",
            "purpose",
        ]
    ].copy()
    df["predicted_risk"] = model.predict_proba(X)
    for c in ("saving_accounts", "checking_account"):
        df[c] = df[c].astype(object).where(df[c].notna(), NO_ACCOUNT)
    return df


def _summarise(snapshot) -> dict:
    d = snapshot.dict()
    drifted, columns = 0, []
    for m in d["metrics"]:
        name = m["metric_name"]
        if not name.startswith("ValueDrift"):
            continue
        col = re.search(r"column=([^,]+)", name).group(1)
        thr = m["config"]["threshold"]
        is_p = "p_value" in name
        flagged = m["value"] < thr if is_p else m["value"] > thr
        drifted += int(flagged)
        columns.append(
            {
                "column": col,
                "score": round(float(m["value"]), 4),
                "drift": bool(flagged),
                "test": re.search(r"method=([^,]+)", name).group(1),
            }
        )
    return {
        "n_columns": len(columns),
        "n_drifted": drifted,
        "share_drifted": round(drifted / max(len(columns), 1), 3),
        "columns": columns,
    }


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    model = CreditRiskModel.load(MODEL_PATH)
    X_train, X_test, _, _ = train_test()
    ref = _frame(model, X_train)
    definition = DataDefinition(numerical_columns=NUMERIC, categorical_columns=CATEGORICAL)
    ref_ds = Dataset.from_pandas(ref, data_definition=definition)

    summary = {}
    for name, X in (("stable", X_test), ("shifted", simulate_shift(X_test))):
        cur = _frame(model, X)
        snap = Report([DataDriftPreset()]).run(
            current_data=Dataset.from_pandas(cur, data_definition=definition), reference_data=ref_ds
        )
        snap.save_html(str(OUT / f"drift_{name}.html"))
        s = _summarise(snap)
        s["mean_predicted_risk"] = round(float(cur["predicted_risk"].mean()), 4)
        s["decline_rate"] = round(float((cur["predicted_risk"] >= model.threshold).mean()), 4)
        summary[name] = s
    summary["reference"] = {
        "mean_predicted_risk": round(float(ref["predicted_risk"].mean()), 4),
        "decline_rate": round(float((ref["predicted_risk"] >= model.threshold).mean()), 4),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    s = run()
    for k in ("stable", "shifted"):
        print(
            f"{k:8s} drifted columns {s[k]['n_drifted']}/{s[k]['n_columns']}  "
            f"mean risk {s[k]['mean_predicted_risk']:.3f}  decline rate {s[k]['decline_rate']:.1%}"
        )
        print("         ", [c["column"] for c in s[k]["columns"] if c["drift"]])
    print("reference: ", s["reference"])
