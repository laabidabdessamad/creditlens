import numpy as np
import pandas as pd
import pandera.errors as pa_errors
import pytest

from creditlens.data import clean, split_xy
from creditlens.schema import validate


def test_clean_keeps_valid_data(synthetic):
    df = clean(synthetic)
    assert len(df) <= len(synthetic)
    assert df["saving_accounts"].isna().any()  # missing == no account is preserved as NaN


def test_target_encoding(xy):
    _, y = xy
    assert set(y.unique()) <= {0, 1}
    assert y.mean() > 0  # 'bad' is the positive class


@pytest.mark.parametrize(
    "column,bad_value",
    [
        ("age", 5),
        ("job", 9),
        ("housing", "castle"),
        ("credit_amount", -10),
        ("duration", 0),
        ("purpose", "yacht"),
        ("risk", "maybe"),
    ],
)
def test_schema_rejects_invalid_values(synthetic, column, bad_value):
    df = synthetic.copy()
    df.loc[0, column] = bad_value
    with pytest.raises(pa_errors.SchemaErrors):
        validate(df)


def test_schema_rejects_unexpected_column(synthetic):
    with pytest.raises(pa_errors.SchemaErrors):
        validate(synthetic.assign(surprise=1))


def test_clean_drops_duplicates(synthetic):
    doubled = pd.concat([synthetic, synthetic.iloc[:10]], ignore_index=True)
    assert len(clean(doubled)) == len(clean(synthetic))


def test_split_xy_drops_target(synthetic):
    X, _ = split_xy(clean(synthetic))
    assert "risk" not in X.columns
    assert np.isfinite(X["credit_amount"]).all()
