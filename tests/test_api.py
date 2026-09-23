"""API tests run against the committed, trained model artifact (models/credit_model.joblib)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

BODY = {
    "job": 2,
    "housing": "own",
    "saving_accounts": "little",
    "checking_account": "moderate",
    "credit_amount": 4500,
    "duration": 24,
    "purpose": "car",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_model_info_is_consistent(client):
    info = client.get("/model-info").json()
    assert info["sensitive_features_in_model"] == []  # no protected attributes in the model
    assert info["cost_matrix"] == {"false_negative": 5.0, "false_positive": 1.0}
    assert 0 < info["decision_threshold"] < 1


def test_predict_contract_and_decision_rule(client):
    r = client.post("/predict", json=BODY)
    assert r.status_code == 200
    out = r.json()
    assert 0 <= out["probability_of_default"] <= 1
    expected = (
        "decline" if out["probability_of_default"] >= out["decision_threshold"] else "approve"
    )
    assert out["decision"] == expected


def test_optional_account_fields(client):
    r = client.post("/predict", json={**BODY, "saving_accounts": None, "checking_account": None})
    assert r.status_code == 200


def test_risk_direction_is_sensible(client):
    safe = {**BODY, "credit_amount": 700, "duration": 6, "checking_account": "rich"}
    risky = {**BODY, "credit_amount": 12000, "duration": 48, "checking_account": "little"}
    p_safe = client.post("/predict", json=safe).json()["probability_of_default"]
    p_risky = client.post("/predict", json=risky).json()["probability_of_default"]
    assert p_risky > p_safe


@pytest.mark.parametrize(
    "patch",
    [
        {"job": 7},
        {"housing": "castle"},
        {"credit_amount": -5},
        {"duration": 0},
        {"purpose": "yacht"},
        {"checking_account": "billionaire"},
    ],
)
def test_validation_errors(client, patch):
    assert client.post("/predict", json={**BODY, **patch}).status_code == 422


def test_missing_required_field(client):
    body = {k: v for k, v in BODY.items() if k != "credit_amount"}
    assert client.post("/predict", json=body).status_code == 422


@pytest.mark.parametrize("field,value", [("sex", "male"), ("age", 30)])
def test_protected_attributes_are_not_accepted(client, field, value):
    assert client.post("/predict", json={**BODY, field: value}).status_code == 422


def test_out_of_range_warning(client):
    out = client.post("/predict", json={**BODY, "credit_amount": 900_000}).json()
    assert any("credit_amount" in w for w in out["warnings"])


def test_explain_matches_predict_and_is_sorted(client):
    pred = client.post("/predict", json=BODY).json()
    exp = client.post("/explain", json=BODY).json()
    assert exp["probability_of_default"] == pred["probability_of_default"]
    mags = [abs(c["contribution"]) for c in exp["contributions"]]
    assert mags == sorted(mags, reverse=True)
    assert {c["feature"] for c in exp["contributions"]} >= {"duration", "checking_account"}
    for c in exp["contributions"]:
        assert (c["contribution"] > 0) == (c["direction"] == "increases risk")
