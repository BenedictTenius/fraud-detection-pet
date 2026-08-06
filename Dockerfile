FROM ghcr.io/astral-sh/uv:0.11.18 AS uv

FROM python:3.10.20-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home app

COPY --chown=app:app src ./src
COPY --chown=app:app main.py README.md ./

USER app

FROM nvcr.io/nvidia/tritonserver:26.06-py3 AS triton

RUN python3 -m pip install --no-cache-dir "catboost==1.2.10"

CMD ["tritonserver", "--model-repository=/models"]

FROM base AS trainer-gpu

CMD ["python", "-m", "src.model.baseline"]

FROM base AS runtime

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "src.web.app:app", "--host=0.0.0.0", "--port=8000", "--workers=1"]
