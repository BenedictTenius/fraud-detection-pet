import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from src.config import CONFIG
from src.metrics.drift import PrometheusDriftMonitor
from src.model.training import FraudPredictionService, TritonPredictionGateway
from src.model.types import ModelName, PredictionUnavailableError
from src.web.logging import RequestLoggingMiddleware, configure_logging
from src.web.types import (
    BatchPredictionAnswer,
    BatchPredictionData,
    PredictionAnswer,
    PredictionData,
)

configure_logging(CONFIG.logging)
logger = logging.getLogger(__name__)


def create_prediction_service() -> FraudPredictionService:
    return FraudPredictionService(
        gateway=TritonPredictionGateway(CONFIG.triton),
        default_model=cast(ModelName, CONFIG.prediction.default_model),
        paths=CONFIG.paths,
        prediction_config=CONFIG.prediction,
        target_column=CONFIG.training.target_column,
        categorical_column=CONFIG.data.categorical_column,
        observer=PrometheusDriftMonitor(CONFIG.paths.drift_reference_file),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.prediction_service = create_prediction_service()
    logger.info(
        "fraud_api_started",
        extra={"default_model": app.state.prediction_service.model_name},
    )
    try:
        yield
    finally:
        logger.info("fraud_api_stopping")
        await app.state.prediction_service.close()
        del app.state.prediction_service


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(RequestLoggingMiddleware)
app.mount("/metrics", make_asgi_app())


@app.exception_handler(PredictionUnavailableError)
async def prediction_unavailable(
    request: Request,
    error: PredictionUnavailableError,
) -> JSONResponse:
    logger.warning(
        "prediction_unavailable",
        extra={"http_path": request.url.path, "error": str(error)[:1_000]},
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Prediction service is temporarily unavailable"},
    )


async def get_prediction_service(request: Request) -> FraudPredictionService:
    return cast(
        FraudPredictionService,
        request.app.state.prediction_service,
    )


PredictionService = Annotated[
    FraudPredictionService,
    Depends(get_prediction_service),
]


@app.get("/health")
async def health(service: PredictionService) -> dict[str, object]:
    if not await service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton or one of its models is not ready",
        )
    return {
        "status": "ok",
        "serving": "triton",
        "default_model": service.model_name,
        "thresholds": service.thresholds,
    }


@app.post("/predict", response_model=PredictionAnswer)
async def predict(
    data: PredictionData,
    service: PredictionService,
) -> PredictionAnswer:
    prediction = await service.predict_one(data.model_dump())
    return PredictionAnswer.model_validate(prediction)


@app.post("/predict/batch", response_model=BatchPredictionAnswer)
async def predict_batch(
    data: BatchPredictionData,
    service: PredictionService,
) -> BatchPredictionAnswer:
    transactions = pd.DataFrame(
        transaction.model_dump() for transaction in data.transactions
    )
    predictions = await service.predict_batch(transactions)
    return BatchPredictionAnswer(
        predictions=[
            PredictionAnswer.model_validate(record)
            for record in predictions.to_dict(orient="records")
        ]
    )


@app.post("/predict/{model_name}", response_model=PredictionAnswer)
async def predict_with_model(
    model_name: ModelName,
    data: PredictionData,
    service: PredictionService,
) -> PredictionAnswer:
    prediction = await service.predict_one(data.model_dump(), model_name=model_name)
    return PredictionAnswer.model_validate(prediction)


@app.post(
    "/predict/{model_name}/batch",
    response_model=BatchPredictionAnswer,
)
async def predict_batch_with_model(
    model_name: ModelName,
    data: BatchPredictionData,
    service: PredictionService,
) -> BatchPredictionAnswer:
    transactions = pd.DataFrame(
        transaction.model_dump() for transaction in data.transactions
    )
    predictions = await service.predict_batch(transactions, model_name=model_name)
    return BatchPredictionAnswer(
        predictions=[
            PredictionAnswer.model_validate(record)
            for record in predictions.to_dict(orient="records")
        ]
    )
