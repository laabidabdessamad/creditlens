"""Loading, cleaning, validating and splitting the German Credit data."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from creditlens.config import (
    CLEAN_DATA_PATH,
    COLUMN_MAP,
    POSITIVE_LABEL,
    RAW_DATA_PATH,
    TARGET,
    load_params,
)
from creditlens.schema import validate

log = logging.getLogger(__name__)


def load_raw(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, drop the pandas index column and normalise column names."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.rename(columns=COLUMN_MAP)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning: strip whitespace, drop exact duplicates, validate the contract.

    Missing ``saving_accounts`` / ``checking_account`` values are *kept as NaN*: they mean
    "no account / unknown" and are turned into an explicit category inside the model
    pipeline (so the API and the training code share exactly one implementation).
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) != n_before:
        log.warning("Dropped %d duplicate rows", n_before - len(df))
    return validate(df)


def load_clean(path=RAW_DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features / binary target (1 = bad credit risk = default)."""
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def train_test(df: pd.DataFrame | None = None):
    """Deterministic stratified train/test split driven by params.yaml."""
    p = load_params()
    df = load_clean() if df is None else df
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=p["test_size"], stratify=y, random_state=p["seed"])


def main() -> None:
    """DVC stage `prepare`: raw CSV -> validated clean CSV."""
    logging.basicConfig(level=logging.INFO)
    df = load_clean()
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    log.info("Wrote %s (%d rows, %d columns)", CLEAN_DATA_PATH, *df.shape)


if __name__ == "__main__":
    main()
