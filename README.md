# Fraud Detection

Fraud scoring with CatBoost, LightGBM, FastAPI, NVIDIA Triton, Prometheus and
Grafana.

## Serving architecture

```text
HTTP JSON -> FastAPI -> FP32 feature tensor -> Triton gRPC
                                            ├─ LightGBM / FIL
                                            └─ CatBoost / Python backend

FastAPI /metrics -> Prometheus -> PSI recording rules -> Grafana
```

FastAPI owns validation, feature preparation, thresholds and the public JSON
contract. Triton owns model loading, batching and inference. Both models accept
the same tensor shape: `[batch, 8]`.

## Prepare and train

Place the PaySim CSV at
`data/raw/PS_20174392719_1491204439457_log.csv`, then run:

```bash
uv sync --locked --all-groups
uv run python -m src.data.process
uv run python -m src.model.baseline
```

Training performs a chronological train/validation/test split. The decision
threshold is selected on validation data and applied unchanged to test data.
Each model produces metrics, learning curves, feature importance and two
TreeSHAP artifacts:

```text
artifacts/baseline/
├── catboost_shap_importance.csv
├── lightgbm_shap_importance.csv
└── plots/
    ├── catboost_shap_summary.png
    └── lightgbm_shap_summary.png
```

CatBoost `ShapValues` and LightGBM `pred_contrib` calculate exact TreeSHAP
contributions. Values are reported in raw-score space and are calculated from a
deterministic validation sample. Explainability stays in the training pipeline,
so the API does not load a second copy of either model.

Optional: copy the default local settings before starting:

```bash
cp .env.example .env
```

## Start

Compose starts `api`, `triton`, `prometheus` and `grafana`. The default setup
runs both LightGBM/FIL and CatBoost inference on CPU, so an NVIDIA GPU is not
needed:

```bash
docker compose up --build --detach
docker compose ps
curl http://localhost:8000/health
```

Ports:

- FastAPI HTTP: `8000`;
- Triton HTTP diagnostics: `8100`;
- Triton gRPC: `8101`;
- Triton Prometheus metrics: `8102`;
- Prometheus: `9090`;
- Grafana: `3000`.

All published ports bind to `127.0.0.1`. The stack is local-only until an
authenticated reverse proxy is placed in front of it.

OpenAPI is available at <http://localhost:8000/docs>.

## Predictions

The default model comes from `FRAUD_MODEL_NAME`. A model can also be selected
explicitly in the URL:

```bash
curl -X POST http://localhost:8000/predict/lightgbm \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "TRANSFER",
    "amount": 150000,
    "oldbalanceOrg": 150000,
    "oldbalanceDest": 2000
  }'

curl -X POST http://localhost:8000/predict/catboost \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "TRANSFER",
    "amount": 150000,
    "oldbalanceOrg": 150000,
    "oldbalanceDest": 2000
  }'
```

Available endpoints:

- `POST /predict` — default model;
- `POST /predict/batch` — default model, batch request;
- `POST /predict/{model}` — `catboost` or `lightgbm`;
- `POST /predict/{model}/batch` — selected model, batch request.

Thresholds stay in FastAPI and are read from
`artifacts/baseline/metrics.json`. Triton returns only `fraud_score`.

Every HTTP response contains `X-Request-ID`. Application and access logs are
single-line JSON records containing correlation ID, model, batch size, status
and latency. Transaction payloads and balances are never logged.

The optional GPU override is kept for deployment on another machine:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build --detach
```

## Online drift monitoring

Grafana is provisioned automatically with the **Fraud Detection — Online
Drift** dashboard. Open <http://localhost:3000> and sign in with
`FRAUD_GRAFANA_ADMIN_USER` and `FRAUD_GRAFANA_ADMIN_PASSWORD`. Development
defaults are `admin` / `admin`; replace the password outside a local machine.

The API exposes bounded-cardinality counters at
<http://localhost:8000/metrics/>. Prometheus aggregates them into PSI over one-
and 24-hour windows. The dashboard contains:

- PSI for `amount`, source and destination balances, and transaction type;
- model-specific score PSI for CatBoost and LightGBM;
- traffic, fraud alert rate and Triton inference latency;
- warning at PSI above `0.10` and critical state above `0.25`, after at least
  5,000 observations in the one-hour window.

Fixed bin edges and reference ratios live in
`artifacts/baseline/drift_reference.json`. They are built from validation data
and validation scores whenever training runs. The content hash becomes the
metric `version`, preventing distributions from different model versions from
being mixed.

Monitoring code and infrastructure configuration are separated by
responsibility:

```text
src/metrics/
└── drift.py

observability/
├── prometheus/
│   ├── prometheus.yml
│   └── rules.yml
└── grafana/
    ├── datasource.yml
    ├── dashboard-provider.yml
    └── fraud-drift-dashboard.json
```

PSI detects distribution shift, not model quality. Precision and recall require
delayed ground-truth labels joined back to stored prediction events.

## Model repository

Triton configuration is stored in `triton_models`:

```text
triton_models/
├── fraud_lightgbm/
│   ├── config.pbtxt
│   └── 1/lightgbm.txt
└── fraud_catboost/
    ├── config.pbtxt
    ├── configs/gpu.pbtxt
    └── 1/model.py
```

The custom Triton image is pinned to `26.06` and adds the CatBoost Python
package. The Python backend reads `catboost.cbm` from the read-only artifacts
mount.

## File inference

`main.py` uses the same Triton gateway as FastAPI. Start Triton first, then run:

```bash
uv run python main.py --model lightgbm \
  --input data/processed/fraud_test.csv \
  --output artifacts/baseline/predictions.csv
```

For a non-default gRPC address, set `FRAUD_TRITON_GRPC_URL`.

Training is run from the project environment and uses CPU by default
(`FRAUD_TRAIN_DEVICE=cpu`).

## Quality checks

Tests are grouped by domain:

```text
tests/
├── data/
├── metrics/
├── model/
└── web/
```

Run the same checks as GitHub Actions:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src main.py
uv run pytest --cov --cov-report=term-missing
```

CI runs linting, formatting, static type checking, 70% minimum branch-aware
coverage and a runtime Docker image build. API tests use an in-process ASGI
transport and do not require Triton, Docker or the full dataset.

The API rejects unknown transaction types, extra JSON fields, non-finite
numbers, invalid one-hot values, out-of-range balances and malformed model
scores. Internal Triton errors are logged but are not returned to clients. The
API container is non-root, read-only, capability-free and has
`no-new-privileges` enabled.

Stop all services with:

```bash
docker compose down
```
