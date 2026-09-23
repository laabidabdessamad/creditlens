.PHONY: setup data train evaluate explain fairness monitor pipeline test lint api app docker

setup:            ## create venv and install everything
	python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt && pip install -e . --no-deps

pipeline:         ## reproduce every artifact (data validation -> train -> figures -> audit -> drift)
	dvc repro

train:
	python -m creditlens.train

test:
	pytest

lint:
	ruff check . && ruff format --check .

api:              ## http://localhost:8000/docs
	uvicorn api.main:app --reload

app:              ## http://localhost:8501  (needs the API running)
	CREDITLENS_API_URL=http://localhost:8000 streamlit run app/streamlit_app.py

docker:
	docker build -t creditlens-api . && docker run --rm -p 10000:10000 creditlens-api

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
