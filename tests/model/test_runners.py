import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import (
    CatBoostConfig,
    LightGBMConfig,
    TrainingConfig,
    TritonConfig,
)
from src.model.training import (
    CatBoostRunner,
    LightGBMRunner,
    TritonPredictionGateway,
)
from src.model.types import DataSplit, ModelName, PredictionUnavailableError


def dataset() -> tuple[DataSplit, DataSplit]:
    random = np.random.default_rng(42)
    features = pd.DataFrame(
        random.normal(size=(240, 4)),
        columns=["amount", "balance", "payment", "transfer"],
    )
    target = (features["amount"] + features["balance"] * 0.5 > 0).astype("int8")
    return (
        DataSplit(features.iloc[:180], target.iloc[:180]),
        DataSplit(features.iloc[180:], target.iloc[180:]),
    )


@pytest.mark.parametrize("model_name", ["catboost", "lightgbm"])
def test_tree_runner_predictions_and_shap_are_consistent(
    model_name: ModelName,
    tmp_path: Path,
) -> None:
    training = TrainingConfig(log_period=0)
    if model_name == "catboost":
        runner = CatBoostRunner(
            CatBoostConfig(iterations=20, early_stopping_rounds=5), training
        )
    else:
        runner = LightGBMRunner(
            LightGBMConfig(
                n_estimators=20,
                min_child_samples=5,
                early_stopping_rounds=5,
            ),
            training,
        )
    train, valid = dataset()

    runner.fit(train, valid, class_weight=1.0)
    probability = runner.predict_proba(valid.features)
    explanation = runner.explain(valid.features)

    assert probability.shape == (len(valid.target),)
    assert explanation.values.shape == valid.features.shape
    assert runner.feature_importance(valid.features.columns.tolist()).shape == (4,)
    assert runner.training_history()
    assert runner.best_iteration >= 0

    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    raw_score = np.log(clipped / (1 - clipped))
    np.testing.assert_allclose(
        explanation.base_values + explanation.values.sum(axis=1),
        raw_score,
        rtol=1e-4,
        atol=1e-4,
    )

    output = tmp_path / f"model{runner.file_extension}"
    runner.save(output)
    assert output.is_file()


@pytest.mark.parametrize("model_name", ["catboost", "lightgbm"])
def test_unfitted_runner_fails_explicitly(model_name: ModelName) -> None:
    if model_name == "catboost":
        runner = CatBoostRunner(CatBoostConfig(), TrainingConfig())
    else:
        runner = LightGBMRunner(LightGBMConfig(), TrainingConfig())

    with pytest.raises(RuntimeError, match="not fitted"):
        runner.predict_proba(pd.DataFrame({"amount": [1.0]}))


class FakeResponse:
    def __init__(self, output: np.ndarray | None) -> None:
        self._output = output

    def as_numpy(self, name: str) -> np.ndarray | None:
        assert name == "output__0"
        return self._output


class FakeTritonClient:
    def __init__(self, outputs: list[np.ndarray | None]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.closed = False

    async def infer(self, **kwargs: Any) -> FakeResponse:
        assert kwargs["model_name"] == "fraud_lightgbm"
        output = self.outputs[self.calls]
        self.calls += 1
        return FakeResponse(output)

    async def is_server_live(self, **kwargs: Any) -> bool:
        return True

    async def is_server_ready(self, **kwargs: Any) -> bool:
        return True

    async def is_model_ready(self, name: str, **kwargs: Any) -> bool:
        return name in {"fraud_catboost", "fraud_lightgbm"}

    async def close(self) -> None:
        self.closed = True


def gateway_with(
    client: FakeTritonClient, max_batch_size: int = 2
) -> TritonPredictionGateway:
    return TritonPredictionGateway(
        TritonConfig(max_batch_size=max_batch_size), client=client
    )


def test_triton_gateway_chunks_and_validates_scores() -> None:
    client = FakeTritonClient(
        [
            np.array([[0.1], [0.2]], dtype="float32"),
            np.array([[0.3]], dtype="float32"),
        ]
    )
    gateway = gateway_with(client)

    scores = asyncio.run(gateway.predict("lightgbm", np.ones((3, 8), dtype="float32")))

    np.testing.assert_allclose(scores, [0.1, 0.2, 0.3])
    assert client.calls == 2
    assert asyncio.run(gateway.is_ready(("catboost", "lightgbm"))) is True
    asyncio.run(gateway.close())
    assert client.closed is True


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (None, "has no score output"),
        (np.array([[0.1], [0.2]]), "scores for 1 rows"),
        (np.array([[np.nan]]), "invalid probability scores"),
        (np.array([[1.1]]), "invalid probability scores"),
    ],
)
def test_triton_gateway_rejects_invalid_responses(
    output: np.ndarray | None,
    message: str,
) -> None:
    gateway = gateway_with(FakeTritonClient([output]))

    with pytest.raises(PredictionUnavailableError, match=message):
        asyncio.run(gateway.predict("lightgbm", np.ones((1, 8), dtype="float32")))


def test_triton_gateway_rejects_invalid_input_shape() -> None:
    gateway = gateway_with(FakeTritonClient([]))

    with pytest.raises(ValueError, match="two-dimensional"):
        asyncio.run(gateway.predict("lightgbm", np.ones(8)))
    with pytest.raises(ValueError, match="must not be empty"):
        asyncio.run(gateway.predict("lightgbm", np.ones((0, 8))))
