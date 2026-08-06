import asyncio

import numpy as np
import pandas as pd
import pytest

from src.config import PathConfig, PredictionConfig
from src.model.training import FraudPredictionService
from src.model.types import ModelName, PredictionUnavailableError


class FakeGateway:
    def __init__(self, scores: np.ndarray) -> None:
        self.scores = scores
        self.features: np.ndarray | None = None
        self.closed = False

    async def predict(self, model_name: ModelName, features: np.ndarray) -> np.ndarray:
        del model_name
        self.features = features.copy()
        return self.scores[: len(features)]

    async def is_ready(self, model_names: tuple[ModelName, ...]) -> bool:
        return model_names == ("catboost", "lightgbm")

    async def close(self) -> None:
        self.closed = True


def service(gateway: FakeGateway, paths: PathConfig) -> FraudPredictionService:
    return FraudPredictionService(
        gateway=gateway,
        default_model="lightgbm",
        paths=paths,
        prediction_config=PredictionConfig(),
        target_column="isFraud",
        categorical_column="type",
    )


def test_single_prediction_prepares_features_in_contract_order(
    model_paths: PathConfig,
) -> None:
    gateway = FakeGateway(np.array([0.8], dtype="float32"))
    prediction = asyncio.run(
        service(gateway, model_paths).predict_one(
            {
                "type": "TRANSFER",
                "amount": 100.0,
                "oldbalanceOrg": 100.0,
                "oldbalanceDest": 0.0,
            }
        )
    )

    assert prediction.model == "lightgbm"
    assert prediction.fraud_score == pytest.approx(0.8)
    assert prediction.threshold == pytest.approx(0.2)
    assert prediction.is_fraud is True
    assert gateway.features is not None
    np.testing.assert_array_equal(
        gateway.features[0],
        np.array([100, 100, 0, 0, 0, 0, 0, 1], dtype="float32"),
    )


def test_batch_prediction_preserves_index(model_paths: PathConfig) -> None:
    gateway = FakeGateway(np.array([0.1, 0.9], dtype="float32"))
    data = pd.DataFrame(
        [
            {
                "type": "PAYMENT",
                "amount": 10,
                "oldbalanceOrg": 20,
                "oldbalanceDest": 30,
            },
            {
                "type": "CASH_OUT",
                "amount": 40,
                "oldbalanceOrg": 50,
                "oldbalanceDest": 60,
            },
        ],
        index=[10, 20],
    )

    result = asyncio.run(service(gateway, model_paths).predict_batch(data))

    assert result.index.tolist() == [10, 20]
    assert result["is_fraud"].tolist() == [False, True]


def test_prediction_rejects_unknown_category(model_paths: PathConfig) -> None:
    gateway = FakeGateway(np.array([0.1], dtype="float32"))

    with pytest.raises(ValueError, match="Unknown transaction types"):
        asyncio.run(
            service(gateway, model_paths).predict_one(
                {
                    "type": "CRYPTO",
                    "amount": 10,
                    "oldbalanceOrg": 20,
                    "oldbalanceDest": 30,
                }
            )
        )


def test_prediction_rejects_invalid_threshold(model_paths: PathConfig) -> None:
    gateway = FakeGateway(np.array([0.1], dtype="float32"))

    with pytest.raises(ValueError, match="between 0 and 1"):
        asyncio.run(
            service(gateway, model_paths).predict_batch(
                pd.DataFrame(
                    {
                        "type": ["PAYMENT"],
                        "amount": [10],
                        "oldbalanceOrg": [20],
                        "oldbalanceDest": [30],
                    }
                ),
                threshold=1.0,
            )
        )


def test_prediction_rejects_out_of_range_values(model_paths: PathConfig) -> None:
    gateway = FakeGateway(np.array([0.1], dtype="float32"))

    with pytest.raises(ValueError, match="outside the allowed range"):
        asyncio.run(
            service(gateway, model_paths).predict_batch(
                pd.DataFrame(
                    {
                        "type": ["PAYMENT"],
                        "amount": [1e13],
                        "oldbalanceOrg": [20],
                        "oldbalanceDest": [30],
                    }
                )
            )
        )


def test_prediction_rejects_invalid_one_hot_values(
    model_paths: PathConfig,
) -> None:
    gateway = FakeGateway(np.array([0.1], dtype="float32"))
    data = pd.DataFrame(
        {
            "amount": [10],
            "oldbalanceOrg": [20],
            "oldbalanceDest": [30],
            "type_CASH_IN": [0],
            "type_CASH_OUT": [0],
            "type_DEBIT": [0],
            "type_PAYMENT": [1],
            "type_TRANSFER": [1],
        }
    )

    with pytest.raises(ValueError, match="valid one-hot encoding"):
        asyncio.run(service(gateway, model_paths).predict_batch(data))


def test_prediction_rejects_non_finite_features(model_paths: PathConfig) -> None:
    gateway = FakeGateway(np.array([0.1], dtype="float32"))

    with pytest.raises(ValueError, match="only finite values"):
        asyncio.run(
            service(gateway, model_paths).predict_batch(
                pd.DataFrame(
                    {
                        "type": ["PAYMENT"],
                        "amount": [np.inf],
                        "oldbalanceOrg": [20],
                        "oldbalanceDest": [30],
                    }
                )
            )
        )


@pytest.mark.parametrize(
    "scores",
    [np.array([np.nan]), np.array([1.1]), np.array([[0.1]])],
)
def test_prediction_rejects_invalid_gateway_scores(
    model_paths: PathConfig,
    scores: np.ndarray,
) -> None:
    gateway = FakeGateway(scores)

    with pytest.raises(PredictionUnavailableError):
        asyncio.run(
            service(gateway, model_paths).predict_batch(
                pd.DataFrame(
                    {
                        "type": ["PAYMENT"],
                        "amount": [10],
                        "oldbalanceOrg": [20],
                        "oldbalanceDest": [30],
                    }
                )
            )
        )


def test_readiness_and_close_are_delegated(model_paths: PathConfig) -> None:
    gateway = FakeGateway(np.array([0.1], dtype="float32"))
    prediction_service = service(gateway, model_paths)

    assert asyncio.run(prediction_service.is_ready()) is True
    asyncio.run(prediction_service.close())
    assert gateway.closed is True
