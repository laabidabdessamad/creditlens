# Deployment guide

```
git push ──► GitHub Actions CI (ruff + pytest + docker build)
                    │ success on main
                    ▼
            Deploy workflow: build image ──► Docker Hub ──► Render deploy hook (pinned image tag)
                                                                  │
   Streamlit Community Cloud (UI) ──── HTTPS ────► Render (FastAPI container) ──► /health smoke test
```

Proposed names (claim them early; Render service names are globally unique):

| Thing | Value |
|---|---|
| GitHub repo | `creditlens` |
| Docker Hub image | `<dockerhub-user>/creditlens-api` |
| Render service | `creditlens-api` → `https://creditlens-api.onrender.com` |
| Streamlit app | `creditlens` → `https://creditlens.streamlit.app` |

If a name is taken, use another and update `CREDITLENS_API_URL` (Streamlit secret) and the
`RENDER_API_URL` repo variable.

## 1. Train once, then push to GitHub
This is a fresh copy: there is no `.dvc/` (not yet initialized), no `models/credit_model.joblib`, and
no DVC remote by default (step 5 sets one up). Set up DVC and train first:
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e . --no-deps

git init
dvc init
dvc add data/raw/german_credit_data.csv   # starts DVC-tracking the raw dataset
dvc repro                                 # writes models/credit_model.joblib and reports/*
```
Then commit everything, including the model artifact (25 KB) - it is committed on purpose so the
Docker build and CI don't need a DVC remote:
```bash
git add . && git commit -m "CreditLens v1.0"
git branch -M main
git remote add origin https://github.com/<you>/creditlens.git
git push -u origin main
```

## 2. Docker Hub
1. Create an account and a **public** repository `creditlens-api` (public = Render needs no credentials).
2. Account settings → Security → **New access token** (Read & Write).
3. GitHub repo → Settings → Secrets and variables → Actions → add secrets
   `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.

## 3. First image + Render service
1. Actions tab → **Deploy API** → *Run workflow*. This pushes `latest` to Docker Hub. (The Render steps are
   skipped automatically because the hook secret does not exist yet.)
2. Render → **New → Web Service → Existing image** → `docker.io/<dockerhub-user>/creditlens-api:latest`.
   * Name: `creditlens-api`, Instance type: **Free**.
   * Settings → **Health Check Path**: `/health`.
3. Settings → **Deploy Hook**: copy the URL.
4. GitHub → add secret `RENDER_DEPLOY_HOOK_URL` (that URL) and repository *variable*
   `RENDER_API_URL` = `https://creditlens-api.onrender.com`.

From now on every push to `main` that passes CI is built, pushed and deployed automatically, and the
workflow finishes with a live `/health` + `/predict` smoke test.

Verify by hand: `python scripts/smoke_test.py https://creditlens-api.onrender.com`

> **Free tier:** Render spins the service down after ~15 minutes idle; the next request takes 30-60 s.
> The Streamlit app shows a "waking up the API" status while it retries `/health`.

## 4. Streamlit Community Cloud
1. share.streamlit.io → **New app** → repo `creditlens`, branch `main`, main file `app/streamlit_app.py`.
2. Advanced settings → Python 3.12; Secrets:
   ```toml
   CREDITLENS_API_URL = "https://creditlens-api.onrender.com"
   ```
3. Dependencies come from `app/requirements.txt` (thin client: streamlit, requests, pandas, matplotlib).
   The app also displays figures from `reports/figures/` (committed to the repo).

## 5. DVC remote + MLflow on DagsHub (optional but recommended)
```bash
dvc remote add -d origin https://dagshub.com/<you>/creditlens.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <you>
dvc remote modify origin --local password <dagshub-token>
dvc push                                   # uploads data/raw/german_credit_data.csv
```
To log runs to DagsHub instead of the local `mlflow.db`:
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<you>/creditlens.mlflow
export MLFLOW_TRACKING_USERNAME=<you>
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>
```
Locally: `make mlflow-ui`.

## 6. Retraining workflow
```bash
# edit params.yaml or code
dvc repro                       # data check -> train -> figures -> audit -> drift
pytest && ruff check .
git add -A && git commit -m "Retrain: <why>" && git push
```
Keep `requirements-api.txt` in sync with the environment that trained the model; the pickle must be loaded
with the same scikit-learn / xgboost / numpy / pandas versions (`model.metadata["versions"]` records them).

## Troubleshooting
| Symptom | Likely cause |
|---|---|
| Render deploy hook returns 4xx | Wrong/rotated hook URL; the hook must be for the same service |
| API crashes at start with a pickle/attribute error | Library versions differ from training - rebuild from `requirements-api.txt` |
| Streamlit shows "API unreachable" | Free instance still waking (wait ~1 min) or wrong `CREDITLENS_API_URL` |
| Deploy workflow does nothing after CI | It only runs when CI **succeeded on `main`**; check the branch name |
