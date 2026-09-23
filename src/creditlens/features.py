"""Feature engineering as a scikit-learn transformer (shared by training and the API)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from creditlens.config import NO_ACCOUNT


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Turns raw applicant fields into model-ready engineered columns.

    * missing savings / checking account  -> explicit ``no_account`` category
    * ``monthly_payment``      = credit_amount / duration   (repayment burden proxy)
    * ``log_credit_amount``    = log1p(credit_amount)       (tames the right skew)

    The transformer is stateless, so it can never leak information from the test set.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in ("saving_accounts", "checking_account"):
            X[col] = X[col].astype(object).where(X[col].notna(), NO_ACCOUNT)
        X["monthly_payment"] = X["credit_amount"] / X["duration"]
        X["log_credit_amount"] = np.log1p(X["credit_amount"])
        return X
