"""Shared fixtures. Tests use synthetic data that follows the schema, so CI does not need
the real dataset (which is DVC-tracked)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from creditlens.config import CATEGORIES


def _choice_with_nan(rng, options, n, p_missing):
    """rng.choice on a mixed list would turn NaN into the string 'nan' - mask it explicitly."""
    values = pd.Series(rng.choice(options, n), dtype=object)
    return values.where(rng.random(n) >= p_missing, np.nan)


def make_synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    duration = rng.integers(4, 60, n)
    amount = rng.integers(300, 12000, n)
    checking = _choice_with_nan(rng, ["little", "moderate", "rich"], n, 0.4)
    # risk depends on duration, amount and missing/low checking -> learnable signal
    z = 0.04 * duration + 0.0002 * amount + 0.8 * pd.isna(checking) - 2.0
    risk = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "bad", "good")
    return pd.DataFrame(
        {
            "age": rng.integers(19, 75, n),
            "sex": rng.choice(CATEGORIES["sex"], n),
            "job": rng.integers(0, 4, n),
            "housing": rng.choice(CATEGORIES["housing"], n),
            "saving_accounts": _choice_with_nan(rng, ["little", "moderate", "rich"], n, 0.2),
            "checking_account": checking,
            "credit_amount": amount,
            "duration": duration,
            "purpose": rng.choice(CATEGORIES["purpose"], n),
            "risk": risk,
        }
    )


@pytest.fixture(scope="session")
def synthetic() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture(scope="session")
def xy(synthetic):
    from creditlens.data import clean, split_xy

    return split_xy(clean(synthetic))


def _ensure_model_for_api_tests() -> None:
    """test_api.py / test_app.py exercise the real API, which loads a CreditRiskModel from
    MODEL_PATH at import time. On a fresh checkout (before `dvc repro` has been run) there is
    no trained model yet, so train a small one here on synthetic data and point MODEL_PATH at
    it - this must run before `api.main` is first imported, i.e. at conftest module load time,
    not inside a fixture. An explicit MODEL_PATH or an existing models/credit_model.joblib
    (e.g. after running the real pipeline) is left untouched.
    """
    from creditlens.config import MODEL_PATH as DEFAULT_MODEL_PATH

    if os.getenv("MODEL_PATH") or DEFAULT_MODEL_PATH.exists():
        return

    from sklearn.linear_model import LogisticRegression

    from creditlens import cost as C
    from creditlens.data import clean, split_xy
    from creditlens.model import CreditRiskModel, _logit
    from creditlens.pipeline import build_pipeline, encoded_feature_names, serving_preprocessor

    X, y = split_xy(clean(make_synthetic(n=400, seed=1)))
    pipe = build_pipeline(LogisticRegression(max_iter=500, class_weight="balanced")).fit(X, y)
    preprocessor, estimator = serving_preprocessor(pipe)
    Xt = preprocessor.transform(X)
    p_raw = estimator.predict_proba(Xt)[:, 1]
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw).reshape(-1, 1), y)
    p_cal = calibrator.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]
    threshold, _ = C.best_threshold(y, p_cal, fn_cost=5.0, fp_cost=1.0)

    model = CreditRiskModel(
        preprocessor=preprocessor,
        estimator=estimator,
        calibrator=calibrator,
        threshold=threshold,
        sensitive=(),
        feature_names=encoded_feature_names(()),
        background_mean=Xt.mean(axis=0),
        metadata={
            "model_version": "test-fixture",
            "family": "logistic_regression",
            "trained_at": "n/a (synthetic test fixture, not the real pipeline)",
            "cost_matrix": {"false_negative": 5.0, "false_positive": 1.0},
            "sensitive_features_in_model": [],
            "input_ranges": {"age": [18, 100], "credit_amount": [100, 20000], "duration": [1, 72]},
            "test": {"roc_auc": 0.0, "pr_auc": 0.0, "expected_cost": 0.0, "approval_rate": 0.0},
        },
    )
    tmp_path = Path(tempfile.mkdtemp(prefix="creditlens_test_model_")) / "credit_model.joblib"
    model.save(tmp_path)
    os.environ["MODEL_PATH"] = str(tmp_path)


_ensure_model_for_api_tests()
