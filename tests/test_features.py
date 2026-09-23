import numpy as np
import pandas as pd

from creditlens.features import FeatureEngineer


def _row(**kw):
    base = dict(saving_accounts="little", checking_account=None, credit_amount=2400, duration=12)
    base.update(kw)
    return pd.DataFrame([base])


def test_missing_accounts_become_explicit_category():
    out = FeatureEngineer().transform(_row(saving_accounts=None, checking_account=np.nan))
    assert out.loc[0, "saving_accounts"] == "no_account"
    assert out.loc[0, "checking_account"] == "no_account"


def test_existing_accounts_untouched():
    out = FeatureEngineer().transform(_row(saving_accounts="rich", checking_account="little"))
    assert out.loc[0, "saving_accounts"] == "rich"
    assert out.loc[0, "checking_account"] == "little"


def test_engineered_columns():
    out = FeatureEngineer().transform(_row(credit_amount=2400, duration=12))
    assert out.loc[0, "monthly_payment"] == 200
    assert np.isclose(out.loc[0, "log_credit_amount"], np.log1p(2400))


def test_transform_does_not_mutate_input():
    df = _row()
    FeatureEngineer().transform(df)
    assert "monthly_payment" not in df.columns
