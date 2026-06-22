# Image de service de l'API de scoring.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Runtime OpenMP requis par LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dépendances (couche cachée tant que pyproject/uv.lock ne changent pas)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) Code + paquet + modèle entraîné (models/model.joblib doit exister :
#    lancer `readmission-calibrate` avant le build)
COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

# Sert l'API FastAPI
CMD ["uvicorn", "readmission_risk.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
