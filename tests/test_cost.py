import numpy as np
import pytest

from creditlens import cost as C


def test_expected_cost_by_hand():
    y = np.array([1, 1, 0, 0])
    pred = np.array([0, 1, 1, 0])  # 1 FN, 1 FP
    assert C.expected_cost(y, pred, fn_cost=5, fp_cost=1) == pytest.approx((5 + 1) / 4)


def test_baselines():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 10% bad
    b = C.baseline_costs(y, 5, 1)
    assert b["approve_all"] == pytest.approx(0.5)  # 1 FN * 5 / 10
    assert b["decline_all"] == pytest.approx(0.9)  # 9 FP / 10


def test_theoretical_threshold():
    assert C.theoretical_threshold(5, 1) == pytest.approx(1 / 6)
    assert C.theoretical_threshold(1, 1) == pytest.approx(0.5)


def test_best_threshold_tracks_cost_ratio():
    rng = np.random.default_rng(0)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(int)  # perfectly calibrated
    t_cheap_fn, _ = C.best_threshold(y, p, fn_cost=1, fp_cost=1)
    t_costly_fn, _ = C.best_threshold(y, p, fn_cost=5, fp_cost=1)
    assert t_costly_fn < t_cheap_fn  # expensive FN -> decline more readily
    assert t_costly_fn == pytest.approx(1 / 6, abs=0.06)  # matches theory on calibrated scores
    assert t_cheap_fn == pytest.approx(0.5, abs=0.06)
