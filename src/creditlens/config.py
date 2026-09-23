"""Project-wide constants and parameter loading."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml


def _find_root() -> Path:
    """Repository root: $CREDITLENS_ROOT, else the nearest parent folder containing params.yaml
    (so notebooks and scripts work from any sub-directory), else the current directory."""
    if env := os.getenv("CREDITLENS_ROOT"):
        return Path(env)
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "params.yaml").exists():
            return p
    return here


ROOT = _find_root()

RAW_DATA_PATH = ROOT / "data" / "raw" / "german_credit_data.csv"
CLEAN_DATA_PATH = ROOT / "data" / "processed" / "clean.csv"
MODEL_PATH = ROOT / "models" / "credit_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PARAMS_PATH = ROOT / "params.yaml"

TARGET = "risk"
POSITIVE_LABEL = "bad"  # the event we predict: default / bad credit risk  -> y = 1

# Canonical column names (snake_case) for the raw CSV headers
COLUMN_MAP = {
    "Age": "age",
    "Sex": "sex",
    "Job": "job",
    "Housing": "housing",
    "Saving accounts": "saving_accounts",
    "Checking account": "checking_account",
    "Credit amount": "credit_amount",
    "Duration": "duration",
    "Purpose": "purpose",
    "Risk": "risk",
}

NO_ACCOUNT = "no_account"  # missing account information == no account / unknown

# Fixed category lists => stable encodings between training and serving, and unseen
# categories are handled gracefully instead of crashing the API.
CATEGORIES: dict[str, list[str]] = {
    "sex": ["female", "male"],
    "housing": ["own", "rent", "free"],
    "saving_accounts": [NO_ACCOUNT, "little", "moderate", "quite rich", "rich"],
    "checking_account": [NO_ACCOUNT, "little", "moderate", "rich"],
    "purpose": [
        "car",
        "radio/TV",
        "furniture/equipment",
        "business",
        "education",
        "repairs",
        "domestic appliances",
        "vacation/others",
    ],
}

# Job is an ordinal skill level: 0 unskilled non-resident, 1 unskilled resident,
# 2 skilled, 3 highly skilled/management -> treated as a numeric feature.
JOB_LEVELS = [0, 1, 2, 3]

SENSITIVE_FEATURES = ("sex", "age")


@lru_cache(maxsize=1)
def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)
