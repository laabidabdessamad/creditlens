"""Pandera schemas: the data contract for training data and model inputs."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from creditlens.config import CATEGORIES, JOB_LEVELS

_acc = {k: CATEGORIES[k][1:] for k in ("saving_accounts", "checking_account")}

DATA_SCHEMA = pa.DataFrameSchema(
    {
        "age": pa.Column(int, pa.Check.in_range(18, 100)),
        "sex": pa.Column(str, pa.Check.isin(CATEGORIES["sex"]), required=False),
        "job": pa.Column(int, pa.Check.isin(JOB_LEVELS)),
        "housing": pa.Column(str, pa.Check.isin(CATEGORIES["housing"])),
        # NaN = no account / unknown, so these two are nullable
        "saving_accounts": pa.Column(str, pa.Check.isin(_acc["saving_accounts"]), nullable=True),
        "checking_account": pa.Column(str, pa.Check.isin(_acc["checking_account"]), nullable=True),
        "credit_amount": pa.Column(int, pa.Check.gt(0)),
        "duration": pa.Column(int, pa.Check.in_range(1, 120)),
        "purpose": pa.Column(str, pa.Check.isin(CATEGORIES["purpose"])),
        "risk": pa.Column(str, pa.Check.isin(["good", "bad"]), required=False),
    },
    strict=True,
    coerce=True,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate (and coerce) a dataframe against the data contract."""
    return DATA_SCHEMA.validate(df, lazy=True)
