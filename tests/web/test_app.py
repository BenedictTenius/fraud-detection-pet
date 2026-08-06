import asyncio
from collections.abc import Mapping
from typing import Any, ClassVar

import httpx2
import pandas as pd

from src.model.types import (
    ModelName,
    PredictionResult,
    PredictionUnavailableError,
)
from src.web import app as web_app

TRANSACTION = {
    "type": "TRANSFER",
    "amount": 100.0,
    "oldbalanceOrg": 100.0,
    "oldbalanceDest": 0.0,
}


class FakePredictionService:
    model_name: ModelName = "lightgbm"
    thresholds: ClassVar[dict[ModelName, float]] = {
        "catboost": 0.7,
        "lightgbm": 0.2,
    }

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.closed = False

    async def is_ready(self) -> bool:
        return self.available

    async def close(self) -> None:
        self.closed = True

    async def predict_one(
        self,
        transaction: Mapping[str, Any],
        model_name: ModelName | None = None,
        threshold: float | None = None,
    ) -> PredictionResult:
        del transaction, threshold
        if not self.available:
            raise PredictionUnavailableError("Triton is unavailable")
        selected = model_name or self.model_name
        selected_threshold = self.thresholds[selected]
        return PredictionResult(
            model=selected,
            fraud_score=0.9,
            threshold=selected_threshold,
            is_fraud=True,
        )

    async def predict_batch(
        self,
        data: pd.DataFrame,
        model_name: ModelName | None = None,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        del threshold
        selected = model_name or self.model_name
        return pd.DataFrame(
            {
                "model": [selected] * len(data),
                "fraud_score": [0.9] * len(data),
                "threshold": [self.thresholds[selected]] * len(data),
                "is_fraud": [True] * len(data),
            }
        )


async def request(method: str, path: str, **kwargs: Any) -> httpx2.Response:
    transport = httpx2.ASGITransport(app=web_app.app)
    async with web_app.lifespan(web_app.app):
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, path, **kwargs)


def test_health_and_request_id(monkeypatch: Any) -> None:
    service = FakePredictionService()
    monkeypatch.setattr(web_app, "create_prediction_service", lambda: service)

    response = asyncio.run(
        request("GET", "/health", headers={"X-Request-ID": "test-123"})
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-123"
    assert response.json()["default_model"] == "lightgbm"
    assert service.closed is True


def test_single_and_batch_predictions(monkeypatch: Any) -> None:
    monkeypatch.setattr(web_app, "create_prediction_service", FakePredictionService)

    async def perform_requests() -> tuple[httpx2.Response, httpx2.Response]:
        transport = httpx2.ASGITransport(app=web_app.app)
        async with web_app.lifespan(web_app.app):
            async with httpx2.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                single = await client.post("/predict/catboost", json=TRANSACTION)
                batch = await client.post(
                    "/predict/lightgbm/batch",
                    json={"transactions": [TRANSACTION, TRANSACTION]},
                )
                return single, batch

    single, batch = asyncio.run(perform_requests())

    assert single.status_code == 200
    assert single.json()["model"] == "catboost"
    assert single.json()["threshold"] == 0.7
    assert len(batch.json()["predictions"]) == 2


def test_invalid_payload_is_rejected(monkeypatch: Any) -> None:
    monkeypatch.setattr(web_app, "create_prediction_service", FakePredictionService)

    response = asyncio.run(
        request("POST", "/predict", json={**TRANSACTION, "amount": -1})
    )

    assert response.status_code == 422
    assert "X-Request-ID" in response.headers


def test_unsafe_request_id_and_extreme_amount_are_rejected(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(web_app, "create_prediction_service", FakePredictionService)

    response = asyncio.run(
        request(
            "POST",
            "/predict",
            headers={"X-Request-ID": "invalid request id"},
            json={**TRANSACTION, "amount": 1e20},
        )
    )

    assert response.status_code == 422
    assert response.headers["X-Request-ID"] != "invalid request id"
    assert len(response.headers["X-Request-ID"]) == 32


def test_prediction_unavailable_returns_503(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        web_app,
        "create_prediction_service",
        lambda: FakePredictionService(available=False),
    )

    response = asyncio.run(request("POST", "/predict", json=TRANSACTION))

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Prediction service is temporarily unavailable"
    }
