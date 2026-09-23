"""Smoke-test a running CreditLens API (local or deployed).

python scripts/smoke_test.py https://creditlens-api.onrender.com
"""

import sys
import time

import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
BODY = {
    "job": 2,
    "housing": "own",
    "saving_accounts": "little",
    "checking_account": "moderate",
    "credit_amount": 4500,
    "duration": 24,
    "purpose": "car",
}

t0 = time.time()
for _ in range(30):  # free hosting tiers may need a while to wake up
    try:
        if requests.get(f"{BASE}/health", timeout=10).ok:
            break
    except requests.RequestException:
        pass
    time.sleep(5)
else:
    sys.exit(f"API at {BASE} did not become healthy")
print(f"health ok after {time.time() - t0:.0f}s")

pred = requests.post(f"{BASE}/predict", json=BODY, timeout=20).json()
print("predict:", pred)
assert 0 <= pred["probability_of_default"] <= 1
exp = requests.post(f"{BASE}/explain", json=BODY, timeout=20).json()
print("top drivers:", [(c["label"], c["contribution"]) for c in exp["contributions"][:3]])
print("model-info:", requests.get(f"{BASE}/model-info", timeout=20).json()["test_metrics"])
print("ALL GOOD")
