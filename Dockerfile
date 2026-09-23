# ---- CreditLens API image -------------------------------------------------------------
# Small runtime image: only what is needed to load the pickled model and serve it.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# xgboost needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) dependencies first (cached layer unless requirements change)
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 2) application code + the trained model artifact (committed in git, see docs/DEPLOYMENT.md)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
COPY api ./api
COPY models ./models

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen(f'http://localhost:{os.getenv(\"PORT\",\"10000\")}/health', timeout=4)"

# Render injects $PORT (default 10000); fall back to 10000 locally.
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
