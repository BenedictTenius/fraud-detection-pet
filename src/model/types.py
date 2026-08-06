from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
import pandas as pd

MetricValue: TypeAlias = float | int
Metrics: TypeAlias = dict[str, MetricValue]
ExperimentReport: TypeAlias = dict[str, Any]
TrainingHistory: TypeAlias = dict[str, dict[str, list[float]]]
ModelName: TypeAlias = Literal["catboost", "lightgbm"]


class PredictionUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class DataSplit:
    features: pd.DataFrame
    target: pd.Series


@dataclass(frozen=True)
class DatasetBundle:
    train: DataSplit
    valid: DataSplit
    test: DataSplit


@dataclass(frozen=True)
class ShapExplanation:
    values: np.ndarray
    base_values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("SHAP values must be a two-dimensional matrix")
        if self.base_values.shape != (len(self.values),):
            raise ValueError("SHAP base values must contain one value per row")
        if not np.isfinite(self.values).all():
            raise ValueError("SHAP values must be finite")
        if not np.isfinite(self.base_values).all():
            raise ValueError("SHAP base values must be finite")


@dataclass(frozen=True)
class PredictionSummary:
    model: ModelName
    rows: int
    fraud_alerts: int
    threshold: float
    output_path: Path


@dataclass(frozen=True)
class PredictionResult:
    model: ModelName
    fraud_score: float
    threshold: float
    is_fraud: bool


class ModelRunner(Protocol):
    name: ModelName
    file_extension: str

    @property
    def class_weight_power(self) -> float: ...

    @property
    def best_iteration(self) -> int: ...

    def fit(
        self,
        train: DataSplit,
        valid: DataSplit,
        class_weight: float,
    ) -> None: ...

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...

    def feature_importance(self, feature_names: list[str]) -> pd.Series: ...

    def training_history(self) -> TrainingHistory: ...

    def explain(self, features: pd.DataFrame) -> ShapExplanation: ...

    def save(self, path: Path) -> None: ...


class PredictionGateway(Protocol):
    async def predict(
        self, model_name: ModelName, features: np.ndarray
    ) -> np.ndarray: ...

    async def is_ready(self, model_names: tuple[ModelName, ...]) -> bool: ...

    async def close(self) -> None: ...


class PredictionObserver(Protocol):
    def observe(
        self,
        data: pd.DataFrame,
        model: ModelName,
        scores: np.ndarray,
        threshold: float,
        duration_seconds: float,
    ) -> None: ...
