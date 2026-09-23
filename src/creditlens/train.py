"""Training: Optuna tuning per model family, MLflow tracking, calibration, cost threshold.

    python -m creditlens.train [--sensitive age sex] [--trials-scale 0.5]

Model selection uses cross-validation on the TRAIN split only. The held-out test split is
evaluated exactly once, for the champion model, at the very end.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from xgboost import XGBClassifier

from creditlens import __version__
from creditlens import cost as C
from creditlens.config import (
    FIGURES_DIR,
    MODEL_PATH,
    REPORTS_DIR,
    ROOT,
    SENSITIVE_FEATURES,
    load_params,
)
from creditlens.data import train_test
from creditlens.evaluation import holdout_report
from creditlens.model import CreditRiskModel, _logit
from creditlens.pipeline import build_pipeline, encoded_feature_names, serving_preprocessor

log = logging.getLogger("creditlens.train")
FAMILIES = ("logistic_regression", "random_forest", "xgboost")
IMBALANCE = ("none", "smote", "class_weight")


# --------------------------------------------------------------------- search spaces
def suggest_params(trial: optuna.Trial, family: str) -> dict:
    p = {"imbalance": trial.suggest_categorical("imbalance", IMBALANCE)}
    if family == "logistic_regression":
        p["C"] = trial.suggest_float("C", 1e-3, 10.0, log=True)
    elif family == "random_forest":
        p["n_estimators"] = trial.suggest_int("n_estimators", 100, 400, step=50)
        p["max_depth"] = trial.suggest_int("max_depth", 3, 12)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 12)
        p["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5])
    elif family == "xgboost":
        p["n_estimators"] = trial.suggest_int("n_estimators", 100, 500, step=50)
        p["max_depth"] = trial.suggest_int("max_depth", 2, 6)
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        p["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True)
    return p


def make_estimator(family: str, p: dict, pos_weight: float, seed: int):
    weighted = p.get("imbalance") == "class_weight"
    if family == "logistic_regression":
        return LogisticRegression(
            C=p["C"], max_iter=3000, class_weight="balanced" if weighted else None
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"],
            class_weight="balanced" if weighted else None,
            n_jobs=1,
            random_state=seed,
        )
    if family == "xgboost":
        return XGBClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            min_child_weight=p["min_child_weight"],
            reg_lambda=p["reg_lambda"],
            scale_pos_weight=pos_weight if weighted else 1.0,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )
    raise ValueError(family)


def make_pipeline(family: str, p: dict, pos_weight: float, seed: int, sensitive=()):
    return build_pipeline(
        make_estimator(family, p, pos_weight, seed),
        use_smote=p.get("imbalance") == "smote",
        sensitive=tuple(sensitive),
        seed=seed,
    )


# ------------------------------------------------------------------------ tuning
def tune_family(family, X, y, params, sensitive, n_trials) -> optuna.Study:
    seed = params["seed"]
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    cv = RepeatedStratifiedKFold(
        n_splits=params["cv_folds"], n_repeats=params["cv_repeats"], random_state=seed
    )

    def objective(trial: optuna.Trial) -> float:
        p = suggest_params(trial, family)
        pipe = make_pipeline(family, p, pos_weight, seed, sensitive)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        trial.set_user_attr("std", float(scores.std()))
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"{family}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def oof_predict(pipe, X, y, seed, folds=5) -> np.ndarray:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def plot_optuna_history(studies: dict[str, optuna.Study]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for fam, st in studies.items():
        vals = [t.value for t in st.trials]
        ax.plot(np.maximum.accumulate(vals), label=fam)
    ax.set(xlabel="Optuna trial", ylabel="Best CV ROC-AUC so far", title="Hyperparameter search")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "00_optuna_history.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------- main
def run(sensitive: tuple[str, ...], trials_scale: float = 1.0, save: bool = True) -> dict:
    params = load_params()
    seed, fn_c, fp_c = (
        params["seed"],
        params["cost"]["false_negative"],
        params["cost"]["false_positive"],
    )
    X_train, X_test, y_train, y_test = train_test()
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    REPORTS_DIR.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment("creditlens")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rows, studies, best_params = [], {}, {}
    with mlflow.start_run(run_name="training") as parent:
        mlflow.log_params(
            {
                "seed": seed,
                "sensitive_in_model": ",".join(sensitive) or "none",
                "cost_fn": fn_c,
                "cost_fp": fp_c,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )
        # --- reference baseline: the prior (no skill)
        rows.append(
            {
                "model": "dummy_prior",
                "cv_roc_auc": 0.5,
                "cv_roc_auc_std": 0.0,
                "oof_pr_auc": float(y_train.mean()),
                "oof_brier": float(
                    brier_score_loss(y_train, np.full(len(y_train), y_train.mean()))
                ),
                "oof_min_cost": min(C.baseline_costs(y_train, fn_c, fp_c).values()),
                "imbalance": "-",
                "best_params": "{}",
            }
        )

        for family in FAMILIES:
            n_trials = max(3, int(params["tuning"][family] * trials_scale))
            log.info("Tuning %s (%d trials)...", family, n_trials)
            with mlflow.start_run(run_name=family, nested=True):
                study = tune_family(family, X_train, y_train, params, sensitive, n_trials)
                studies[family] = study
                bp = study.best_trial.params
                best_params[family] = bp
                pipe = make_pipeline(family, bp, pos_weight, seed, sensitive)
                oof = oof_predict(pipe, X_train, y_train, seed)
                thr, min_cost = C.best_threshold(y_train, oof, fn_c, fp_c)
                row = {
                    "model": family,
                    "cv_roc_auc": float(study.best_value),
                    "cv_roc_auc_std": float(study.best_trial.user_attrs["std"]),
                    "oof_pr_auc": float(average_precision_score(y_train, oof)),
                    "oof_brier": float(brier_score_loss(y_train, oof)),
                    "oof_min_cost": float(min_cost),
                    "imbalance": bp["imbalance"],
                    "best_params": json.dumps(bp),
                }
                rows.append(row)
                mlflow.log_params({f"best_{k}": v for k, v in bp.items()})
                mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, float)})
                mlflow.log_metric("n_trials", n_trials)

        comp = pd.DataFrame(rows)
        comp.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
        plot_optuna_history(studies)

        # --- champion = best cross-validated ROC-AUC (test set NOT involved)
        champion = comp[comp.model != "dummy_prior"].sort_values("cv_roc_auc").iloc[-1]["model"]
        bp = best_params[champion]
        log.info("Champion: %s  params=%s", champion, bp)
        pipe = make_pipeline(champion, bp, pos_weight, seed, sensitive)

        # --- calibration (Platt on out-of-fold logits) + cost-optimal threshold
        oof_raw = oof_predict(pipe, X_train, y_train, seed)
        calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(
            _logit(oof_raw).reshape(-1, 1), y_train
        )
        oof_cal = calibrator.predict_proba(_logit(oof_raw).reshape(-1, 1))[:, 1]
        threshold, oof_cost = C.best_threshold(y_train, oof_cal, fn_c, fp_c)
        pd.DataFrame({"y": y_train.values, "p_raw": oof_raw, "p_cal": oof_cal}).to_csv(
            REPORTS_DIR / "oof_train.csv", index=False
        )

        # --- final fit on the full training split, assemble the deployable artifact
        pipe.fit(X_train, y_train)
        preprocessor, clf = serving_preprocessor(pipe)
        background = preprocessor.transform(X_train).mean(axis=0)
        raw_X = X_train
        metadata = {
            "model_version": __version__,
            "family": champion,
            "best_params": bp,
            "trained_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "cost_matrix": {"false_negative": fn_c, "false_positive": fp_c},
            "theoretical_threshold": C.theoretical_threshold(fn_c, fp_c),
            "sensitive_features_in_model": list(sensitive),
            "input_ranges": {
                c: [float(raw_X[c].min()), float(raw_X[c].max())]
                for c in ("age", "credit_amount", "duration")
            },
            "versions": {
                "python": platform.python_version(),
                "sklearn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "training": {
                "n_train": int(len(X_train)),
                "bad_rate": float(y_train.mean()),
                "cv_roc_auc": float(comp[comp.model == champion].cv_roc_auc.iloc[0]),
                "oof_expected_cost": float(oof_cost),
            },
        }
        model = CreditRiskModel(
            preprocessor=preprocessor,
            estimator=clf,
            calibrator=calibrator,
            threshold=float(threshold),
            sensitive=tuple(sensitive),
            feature_names=encoded_feature_names(sensitive),
            background_mean=np.asarray(background),
            metadata=metadata,
        )

        # --- ONE evaluation on the held-out test split
        p_cal, p_raw = model.predict_proba(X_test), model.raw_proba(X_test)
        test_report = holdout_report(y_test, p_cal, p_raw, threshold, fn_c, fp_c)
        pd.DataFrame(
            {
                "y": y_test.values,
                "p_raw": p_raw,
                "p_cal": p_cal,
                "sex": X_test["sex"].values,
                "age": X_test["age"].values,
            }
        ).to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
        metrics = {
            "champion": champion,
            "test": test_report,
            "train_oof_cost": oof_cost,
            "cv_roc_auc": metadata["training"]["cv_roc_auc"],
        }
        metadata["test"] = {
            k: test_report[k] for k in ("roc_auc", "pr_auc", "expected_cost", "approval_rate")
        }
        (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

        mlflow.log_params({"champion": champion, "threshold": threshold})
        mlflow.log_metrics(
            {
                "test_roc_auc": test_report["roc_auc"],
                "test_pr_auc": test_report["pr_auc"],
                "test_expected_cost": test_report["expected_cost"],
                "test_saving_vs_decline_all": test_report["saving_vs_decline_all"],
                "test_brier_calibrated": test_report["brier_calibrated"],
            }
        )
        if save:
            model.save(MODEL_PATH)
            mlflow.log_artifact(str(MODEL_PATH))
            for f in ("metrics.json", "model_comparison.csv"):
                mlflow.log_artifact(str(REPORTS_DIR / f))
        log.info(
            "run id %s | test AUC %.3f | test cost %.3f | threshold %.2f",
            parent.info.run_id,
            test_report["roc_auc"],
            test_report["expected_cost"],
            threshold,
        )
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sensitive",
        nargs="*",
        choices=SENSITIVE_FEATURES,
        default=None,
        help="sensitive features allowed in the model (default: params.yaml)",
    )
    ap.add_argument("--trials-scale", type=float, default=1.0, help="scale tuning budget")
    a = ap.parse_args()
    sens = tuple(
        load_params()["sensitive_features_in_model"] if a.sensitive is None else a.sensitive
    )
    run(sens, a.trials_scale)


if __name__ == "__main__":
    main()
