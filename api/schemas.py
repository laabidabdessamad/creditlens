"""Pydantic request / response contracts for the CreditLens API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Housing = Literal["own", "rent", "free"]
Savings = Literal["little", "moderate", "quite rich", "rich"]
Checking = Literal["little", "moderate", "rich"]
Purpose = Literal[
    "car",
    "radio/TV",
    "furniture/equipment",
    "business",
    "education",
    "repairs",
    "domestic appliances",
    "vacation/others",
]


class Applicant(BaseModel):
    """A loan applicant. Omit (null) the account fields when the applicant has no such account
    or the balance is unknown - the model treats that as its own informative category.

    Note: `sex` and `age` are deliberately NOT accepted - they are protected attributes and
    the model does not use them (see the fairness audit in the repository).
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "job": 2,
                "housing": "own",
                "saving_accounts": "little",
                "checking_account": "moderate",
                "credit_amount": 4500,
                "duration": 24,
                "purpose": "car",
            }
        },
    )

    job: int = Field(
        ge=0,
        le=3,
        description="0 unskilled non-resident, 1 unskilled resident, 2 skilled, 3 highly skilled",
    )
    housing: Housing
    saving_accounts: Savings | None = Field(default=None, description="null = no account / unknown")
    checking_account: Checking | None = Field(
        default=None, description="null = no account / unknown"
    )
    credit_amount: int = Field(
        gt=0, le=1_000_000, description="Requested amount (Deutsche Mark, as in the source data)"
    )
    duration: int = Field(ge=1, le=120, description="Loan duration in months")
    purpose: Purpose


class Prediction(BaseModel):
    probability_of_default: float = Field(description="Calibrated P(bad credit risk), 0-1")
    decision: Literal["approve", "decline"]
    decision_threshold: float = Field(description="Decline if probability_of_default >= this value")
    risk_band: Literal["low", "medium", "high"]
    model_version: str
    warnings: list[str] = []


class Contribution(BaseModel):
    feature: str
    label: str
    value: str = Field(description="The applicant's value for this feature, human readable")
    contribution: float = Field(description="Effect on risk in log-odds; > 0 increases risk")
    direction: Literal["increases risk", "decreases risk"]


class Explanation(Prediction):
    base_log_odds: float = Field(description="Average model output (log-odds) before any feature")
    contributions: list[Contribution]
    note: str = (
        "Contributions are exact TreeSHAP values of the raw model in log-odds; the served "
        "probability additionally passes through a monotonic calibration step."
    )


class Health(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str


class ModelInfo(BaseModel):
    model_version: str
    algorithm: str
    trained_at: str
    decision_threshold: float
    cost_matrix: dict[str, float]
    sensitive_features_in_model: list[str]
    input_ranges: dict[str, list[float]]
    test_metrics: dict[str, float]
    disclaimer: str
