"""Model pipeline factory.

Order of steps (leakage-safe, everything is fitted on training folds only):

    engineer -> ordinal(+scale) -> [SMOTENC] -> one-hot -> classifier

SMOTENC (not plain SMOTE) is used because the data mixes categorical and numeric
features; plain SMOTE would invent meaningless fractional one-hot values.
The sampler lives inside the imblearn Pipeline, so it is only ever applied to the
training part of each CV fold and is skipped at prediction time.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from creditlens.config import CATEGORIES
from creditlens.features import FeatureEngineer

BASE_CATEGORICAL = ["housing", "saving_accounts", "checking_account", "purpose"]
BASE_NUMERIC = ["job", "duration", "log_credit_amount", "monthly_payment"]


def feature_lists(sensitive: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """Categorical and numeric feature names used by the model."""
    cat = BASE_CATEGORICAL + (["sex"] if "sex" in sensitive else [])
    num = BASE_NUMERIC + (["age"] if "age" in sensitive else [])
    return cat, num


def encoded_feature_names(sensitive: tuple[str, ...] = ()) -> list[str]:
    """Names of the columns produced by the final encoder (matches model input order)."""
    cat, num = feature_lists(sensitive)
    names = [f"{c}={v}" for c in cat for v in CATEGORIES[c]]
    return names + num


def build_pipeline(
    estimator,
    use_smote: bool = False,
    sensitive: tuple[str, ...] = (),
    seed: int = 42,
    smote_k: int = 5,
) -> ImbPipeline:
    cat, num = feature_lists(sensitive)
    n_cat, n_num = len(cat), len(num)

    ordinal = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    categories=[CATEGORIES[c] for c in cat],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                cat,
            ),
            ("num", StandardScaler(), num),
        ],
        verbose_feature_names_out=False,
    )
    encode = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    categories=[np.arange(len(CATEGORIES[c]), dtype=float) for c in cat],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                list(range(n_cat)),
            ),
            ("num", "passthrough", list(range(n_cat, n_cat + n_num))),
        ]
    )
    sampler = (
        SMOTENC(categorical_features=list(range(n_cat)), k_neighbors=smote_k, random_state=seed)
        if use_smote
        else "passthrough"
    )
    return ImbPipeline(
        [
            ("engineer", FeatureEngineer()),
            ("ordinal", ordinal),
            ("smote", sampler),
            ("encode", encode),
            ("clf", estimator),
        ]
    )


def serving_preprocessor(fitted: ImbPipeline) -> tuple[Pipeline, object]:
    """Split a fitted training pipeline into (plain-sklearn preprocessor, classifier).

    Dropping the sampler means the API does not need imbalanced-learn at all.
    """
    steps = [(n, s) for n, s in fitted.steps if n in ("engineer", "ordinal", "encode")]
    return Pipeline(steps), fitted.named_steps["clf"]
