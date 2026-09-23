"""Business cost logic: asymmetric misclassification costs and threshold selection.

Convention: y = 1 means *bad* credit risk (default). Prediction 1 = decline the loan.
    false negative = approve a loan that defaults       (expensive)
    false positive = decline a loan that would be repaid (cheap: lost margin)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix

DEFAULT_GRID = np.round(np.arange(0.02, 0.96, 0.01), 2)


def expected_cost(y_true, y_pred, fn_cost: float = 5.0, fp_cost: float = 1.0) -> float:
    """Average cost per applicant."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float((fn_cost * fn + fp_cost * fp) / len(y_true))


def cost_curve(y_true, p, fn_cost=5.0, fp_cost=1.0, grid=DEFAULT_GRID) -> np.ndarray:
    y_true, p = np.asarray(y_true), np.asarray(p)
    return np.array([expected_cost(y_true, (p >= t).astype(int), fn_cost, fp_cost) for t in grid])


def theoretical_threshold(fn_cost: float = 5.0, fp_cost: float = 1.0) -> float:
    """Optimal threshold for *perfectly calibrated* probabilities: fp / (fp + fn)."""
    return fp_cost / (fp_cost + fn_cost)


def best_threshold(
    y_true, p, fn_cost=5.0, fp_cost=1.0, grid=DEFAULT_GRID, smooth: int = 5
) -> tuple[float, float]:
    """Cost-minimising decision threshold on out-of-fold probabilities.

    The empirical cost curve is noisy on ~1000 samples, so it is smoothed with a moving
    average before taking the argmin. Returns (threshold, raw expected cost at threshold).
    """
    curve = cost_curve(y_true, p, fn_cost, fp_cost, grid)
    k = max(1, smooth)
    smoothed = np.convolve(np.pad(curve, (k // 2, k // 2), mode="edge"), np.ones(k) / k, "valid")
    i = int(np.argmin(smoothed))
    return float(grid[i]), float(curve[i])


def baseline_costs(y_true, fn_cost=5.0, fp_cost=1.0) -> dict[str, float]:
    """Reference policies every model must beat."""
    y_true = np.asarray(y_true)
    return {
        "approve_all": expected_cost(y_true, np.zeros_like(y_true), fn_cost, fp_cost),
        "decline_all": expected_cost(y_true, np.ones_like(y_true), fn_cost, fp_cost),
    }
