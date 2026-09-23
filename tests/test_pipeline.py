import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from creditlens.model import CreditRiskModel
from creditlens.pipeline import (
    build_pipeline,
    encoded_feature_names,
    feature_lists,
    serving_preprocessor,
)


def _fit(xy, est=None, smote=False, sensitive=()):
    X, y = xy
    est = est or LogisticRegression(max_iter=500)
    return build_pipeline(est, use_smote=smote, sensitive=sensitive).fit(X, y), X, y


@pytest.mark.parametrize("smote", [False, True])
def test_pipeline_fits_and_predicts(xy, smote):
    pipe, X, _ = _fit(xy, smote=smote)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (len(X),) and ((p >= 0) & (p <= 1)).all()


def test_encoded_feature_names_match_matrix_width(xy):
    for sens in [(), ("sex",), ("age",), ("sex", "age")]:
        pipe, X, _ = _fit(xy, sensitive=sens)
        Xt = serving_preprocessor(pipe)[0].transform(X)
        assert Xt.shape[1] == len(encoded_feature_names(sens))


def test_sensitive_features_excluded_by_default():
    cat, num = feature_lists(())
    assert "sex" not in cat and "age" not in num
    cat, num = feature_lists(("sex", "age"))
    assert "sex" in cat and "age" in num


def test_smote_only_changes_training_not_prediction(xy):
    X, y = xy
    a = build_pipeline(LogisticRegression(max_iter=500), use_smote=True).fit(X, y)
    Xt = serving_preprocessor(a)[0].transform(X)
    assert len(Xt) == len(X)  # the sampler is not applied at transform time


def test_unseen_category_does_not_crash(xy):
    pipe, X, _ = _fit(xy)
    weird = X.head(3).copy()
    weird["purpose"] = "spaceship"
    weird["housing"] = "castle"
    p = pipe.predict_proba(weird)[:, 1]
    assert np.isfinite(p).all()


def test_missing_accounts_handled(xy):
    pipe, X, _ = _fit(xy)
    row = X.head(1).copy()
    row["saving_accounts"] = None
    row["checking_account"] = None
    assert np.isfinite(pipe.predict_proba(row)).all()


def test_serving_split_reproduces_pipeline_predictions(xy):
    pipe, X, _ = _fit(xy, est=XGBClassifier(n_estimators=30, max_depth=2, verbosity=0))
    pre, clf = serving_preprocessor(pipe)
    np.testing.assert_allclose(
        clf.predict_proba(pre.transform(X)), pipe.predict_proba(X), rtol=1e-6
    )


def test_native_xgboost_contributions_are_additive(xy):
    """base + sum(contributions) must equal the raw model margin (TreeSHAP additivity)."""
    pipe, X, _ = _fit(xy, est=XGBClassifier(n_estimators=30, max_depth=3, verbosity=0))
    pre, clf = serving_preprocessor(pipe)
    model = CreditRiskModel(
        preprocessor=pre,
        estimator=clf,
        calibrator=None,
        threshold=0.2,
        sensitive=(),
        feature_names=encoded_feature_names(()),
        background_mean=pre.transform(X).mean(axis=0),
    )
    groups, contrib, base = model.contributions(X.head(20))
    p = model.raw_proba(X.head(20))
    margin = np.log(p / (1 - p))
    np.testing.assert_allclose(base + contrib.sum(axis=1), margin, atol=1e-4)
    assert "sex" not in groups and "age" not in groups


def test_logistic_contributions_are_additive(xy):
    pipe, X, _ = _fit(xy)
    pre, clf = serving_preprocessor(pipe)
    model = CreditRiskModel(
        preprocessor=pre,
        estimator=clf,
        calibrator=None,
        threshold=0.2,
        sensitive=(),
        feature_names=encoded_feature_names(()),
        background_mean=pre.transform(X).mean(axis=0),
    )
    _, contrib, base = model.contributions(X.head(15))
    p = model.raw_proba(X.head(15))
    np.testing.assert_allclose(base + contrib.sum(axis=1), np.log(p / (1 - p)), atol=1e-6)


def test_model_roundtrip(tmp_path, xy):
    pipe, X, _ = _fit(xy)
    pre, clf = serving_preprocessor(pipe)
    m = CreditRiskModel(
        pre,
        clf,
        None,
        0.2,
        (),
        encoded_feature_names(()),
        pre.transform(X).mean(axis=0),
        {"model_version": "t"},
    )
    path = m.save(tmp_path / "m.joblib")
    m2 = CreditRiskModel.load(path)
    pd.testing.assert_series_equal(pd.Series(m.predict_proba(X)), pd.Series(m2.predict_proba(X)))
