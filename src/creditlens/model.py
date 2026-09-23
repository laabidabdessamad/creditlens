"""The deployable model artifact.

`CreditRiskModel` bundles everything needed at serving time into ONE joblib file:
preprocessing, classifier, probability calibrator and the cost-optimal decision threshold.
Training, the API and the tests all go through this single class, so there is no
train/serve skew.

Explanations are computed natively (XGBoost TreeSHAP via ``pred_contribs`` / linear
contributions) so the API image does not need the heavy `shap` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class CreditRiskModel:
    preprocessor: Pipeline
    estimator: object
    calibrator: object | None
    threshold: float
    sensitive: tuple[str, ...]
    feature_names: list[str]
    background_mean: np.ndarray
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ prediction
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(X), dtype=float)

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated P(bad) straight from the classifier."""
        return self.estimator.predict_proba(self.transform(X))[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default P(bad)."""
        p = self.raw_proba(X)
        if self.calibrator is None:
            return p
        return self.calibrator.predict_proba(_logit(p).reshape(-1, 1))[:, 1]

    def decline(self, X: pd.DataFrame) -> np.ndarray:
        """1 = decline (predicted bad), 0 = approve, using the cost-optimal threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ---------------------------------------------------------------- explanations
    def _margin_contributions(self, Xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-encoded-feature contributions to the raw model margin (log-odds)."""
        est = self.estimator
        if hasattr(est, "get_booster"):  # XGBoost: exact TreeSHAP, no extra dependency
            import xgboost as xgb

            contribs = est.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
            return contribs[:, :-1], contribs[:, -1]
        if hasattr(est, "coef_"):  # logistic regression: coef * (x - background mean)
            coef = est.coef_.ravel()
            contribs = (Xt - self.background_mean) * coef
            base = np.full(len(Xt), float(est.intercept_[0] + coef @ self.background_mean))
            return contribs, base
        import shap  # random forest etc. (offline use only)

        sv = shap.TreeExplainer(est).shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        base = shap.TreeExplainer(est).expected_value
        base = np.atleast_1d(base)[-1]
        return sv, np.full(len(Xt), float(base))

    def contributions(self, X: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Feature-level contributions (one-hot columns summed back to their feature).

        Returns (feature_names, contributions[n, n_features], base_value[n]) in *log-odds of
        the raw classifier*. Positive values push toward "bad credit risk".
        """
        contribs, base = self._margin_contributions(self.transform(X))
        groups: list[str] = []
        for name in self.feature_names:
            g = name.split("=")[0]
            if g not in groups:
                groups.append(g)
        grouped = np.zeros((len(contribs), len(groups)))
        for j, name in enumerate(self.feature_names):
            grouped[:, groups.index(name.split("=")[0])] += contribs[:, j]
        return groups, grouped, base

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: str | Path) -> CreditRiskModel:
        return joblib.load(path)
