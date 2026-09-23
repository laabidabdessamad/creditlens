"""CreditLens FastAPI service.

Run locally:   uvicorn api.main:app --reload
Docs:          http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    Applicant,
    Contribution,
    Explanation,
    Health,
    ModelInfo,
    Prediction,
)
from creditlens.model import CreditRiskModel

log = logging.getLogger("creditlens.api")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/credit_model.joblib"))
DISCLAIMER = (
    "Portfolio/demo project trained on the 1,000-row German Credit dataset. "
    "Not for use in real lending decisions."
)

LABELS = {
    "checking_account": "Checking account",
    "saving_accounts": "Savings account",
    "duration": "Loan duration",
    "log_credit_amount": "Credit amount",
    "monthly_payment": "Monthly payment",
    "purpose": "Loan purpose",
    "housing": "Housing",
    "job": "Job level",
}
JOB_NAMES = {
    0: "unskilled, non-resident",
    1: "unskilled, resident",
    2: "skilled",
    3: "highly skilled / management",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = CreditRiskModel.load(MODEL_PATH)
    if model.sensitive:
        raise RuntimeError(
            f"Model uses sensitive features {model.sensitive}; this API schema intentionally "
            "does not accept them."
        )
    app.state.model = model
    log.info(
        "Loaded model %s (%s) from %s",
        model.metadata["model_version"],
        model.metadata["family"],
        MODEL_PATH,
    )
    yield


app = FastAPI(
    title="CreditLens API",
    version="1.0.0",
    description=(
        "Cost-sensitive credit risk scoring with per-applicant explanations.\n\n" + DISCLAIMER
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _model(request: Request) -> CreditRiskModel:
    return request.app.state.model


def _to_frame(a: Applicant) -> pd.DataFrame:
    return pd.DataFrame([a.model_dump()])


def _warnings(m: CreditRiskModel, a: Applicant) -> list[str]:
    out = []
    for col in ("credit_amount", "duration"):
        lo, hi = m.metadata["input_ranges"][col]
        v = getattr(a, col)
        if not lo <= v <= hi:
            out.append(
                f"{col}={v} is outside the training range [{lo:g}, {hi:g}]; "
                "the score is an extrapolation - treat with caution."
            )
    return out


def _band(p: float, thr: float) -> str:
    return "low" if p < 0.5 * thr else "medium" if p < thr else "high"


def _prediction(m: CreditRiskModel, a: Applicant) -> tuple[Prediction, pd.DataFrame]:
    X = _to_frame(a)
    p = float(m.predict_proba(X)[0])
    pred = Prediction(
        probability_of_default=round(p, 4),
        decision="decline" if p >= m.threshold else "approve",
        decision_threshold=round(m.threshold, 4),
        risk_band=_band(p, m.threshold),
        model_version=m.metadata["model_version"],
        warnings=_warnings(m, a),
    )
    return pred, X


@app.get("/", include_in_schema=False)
def root():
    return {"service": "CreditLens API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=Health, tags=["ops"])
def health(request: Request):
    m = _model(request)
    return Health(status="ok", model_loaded=True, model_version=m.metadata["model_version"])


@app.get("/model-info", response_model=ModelInfo, tags=["model"])
def model_info(request: Request):
    m = _model(request)
    md = m.metadata
    return ModelInfo(
        model_version=md["model_version"],
        algorithm=md["family"],
        trained_at=md["trained_at"],
        decision_threshold=round(m.threshold, 4),
        cost_matrix={k: float(v) for k, v in md["cost_matrix"].items()},
        sensitive_features_in_model=list(m.sensitive),
        input_ranges=md["input_ranges"],
        test_metrics={k: float(v) for k, v in md["test"].items()},
        disclaimer=DISCLAIMER,
    )


@app.post("/predict", response_model=Prediction, tags=["scoring"])
def predict(applicant: Applicant, request: Request):
    pred, _ = _prediction(_model(request), applicant)
    return pred


@app.post("/explain", response_model=Explanation, tags=["scoring"])
def explain(applicant: Applicant, request: Request):
    m = _model(request)
    pred, X = _prediction(m, applicant)
    try:
        groups, contrib, base = m.contributions(X)
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("explanation failed")
        raise HTTPException(status_code=500, detail="Explanation unavailable") from exc

    a = applicant
    values = {
        "checking_account": a.checking_account or "no account / unknown",
        "saving_accounts": a.saving_accounts or "no account / unknown",
        "duration": f"{a.duration} months",
        "log_credit_amount": f"DM {a.credit_amount:,}",
        "monthly_payment": f"DM {a.credit_amount / a.duration:,.0f} / month",
        "purpose": a.purpose,
        "housing": a.housing,
        "job": JOB_NAMES[a.job],
    }
    items = [
        Contribution(
            feature=g,
            label=LABELS.get(g, g),
            value=values.get(g, ""),
            contribution=round(float(c), 4),
            direction="increases risk" if c > 0 else "decreases risk",
        )
        for g, c in zip(groups, contrib[0], strict=True)
    ]
    items.sort(key=lambda c: abs(c.contribution), reverse=True)
    return Explanation(
        **pred.model_dump(), base_log_odds=round(float(np.ravel(base)[0]), 4), contributions=items
    )


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):  # pragma: no cover
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
