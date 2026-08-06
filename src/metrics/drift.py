import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram

from src.model.types import ModelName

DRIFT_OBSERVATIONS = Counter(
    "fraud_drift_bin_observations",
    "Transactions observed in a fixed drift bin",
    ("kind", "name", "model", "version", "bin"),
)
REFERENCE_RATIO = Gauge(
    "fraud_drift_reference_ratio",
    "Reference share of transactions in a fixed drift bin",
    ("kind", "name", "model", "version", "bin"),
)
PREDICTIONS = Counter(
    "fraud_predictions",
    "Predictions returned by the fraud service",
    ("model", "version", "decision"),
)
SCORE_SUM = Counter(
    "fraud_prediction_score_sum",
    "Sum of fraud scores returned by the fraud service",
    ("model", "version"),
)
INFERENCE_DURATION = Histogram(
    "fraud_inference_duration_seconds",
    "Triton inference duration including all chunks",
    ("model",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
MONITORING_ERRORS = Counter(
    "fraud_drift_monitoring_errors",
    "Errors caught while recording drift metrics",
    ("reason",),
)


class DriftReferenceBuilder:
    def __init__(self, output_path: Path, bin_count: int = 10) -> None:
        if bin_count < 2:
            raise ValueError("PSI bin_count must be at least 2")
        self._output_path = output_path
        self._bin_count = bin_count
        self._distributions: list[dict[str, Any]] = []

    def fit_features(
        self,
        features: pd.DataFrame,
        categorical_column: str,
    ) -> None:
        prefix = f"{categorical_column}_"
        numeric_names = [
            name for name in features.columns if not name.startswith(prefix)
        ]
        for name in numeric_names:
            self._distributions.append(
                self._numeric_distribution(
                    kind="feature",
                    name=name,
                    model="all",
                    values=features[name].to_numpy(dtype="float64"),
                )
            )

        category_columns = [
            name for name in features.columns if name.startswith(prefix)
        ]
        if not category_columns:
            raise ValueError(f"No encoded categories found for '{categorical_column}'")
        categories = [name.removeprefix(prefix) for name in category_columns]
        ratios = features[category_columns].mean().to_numpy(dtype="float64")
        ratios = ratios / ratios.sum()
        self._distributions.append(
            {
                "kind": "feature",
                "name": categorical_column,
                "model": "all",
                "binning": "categorical",
                "categories": [*categories, "__other__"],
                "bins": [*categories, "__other__"],
                "ratios": [*ratios.tolist(), 0.0],
            }
        )

    def add_scores(
        self,
        model: ModelName,
        scores: np.ndarray,
        threshold: float,
    ) -> None:
        self._distributions.append(
            self._numeric_distribution(
                kind="score",
                name="fraud_score",
                model=model,
                values=np.asarray(scores, dtype="float64"),
                extra_edges=np.asarray([threshold], dtype="float64"),
            )
        )

    def save(self) -> Path:
        if not self._distributions:
            raise RuntimeError("No drift distributions were fitted")
        body = {
            "schema_version": 1,
            "distributions": self._distributions,
        }
        canonical = json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        payload = {
            "version": hashlib.sha256(canonical).hexdigest()[:12],
            **body,
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return self._output_path

    def _numeric_distribution(
        self,
        kind: str,
        name: str,
        model: str,
        values: np.ndarray,
        extra_edges: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if values.ndim != 1 or len(values) == 0:
            raise ValueError(f"Reference values for {name} must be one-dimensional")
        if not np.isfinite(values).all():
            raise ValueError(f"Reference values for {name} must be finite")

        quantiles = np.linspace(0, 1, self._bin_count + 1)[1:-1]
        candidates = np.quantile(values, quantiles)
        if extra_edges is not None:
            candidates = np.concatenate((candidates, extra_edges))
        edges = np.unique(candidates[candidates < values.max()])
        indices = np.searchsorted(edges, values, side="left")
        counts = np.bincount(indices, minlength=len(edges) + 1)
        ratios = counts / counts.sum()
        return {
            "kind": kind,
            "name": name,
            "model": model,
            "binning": "numeric",
            "edges": edges.tolist(),
            "bins": [f"{index:02d}" for index in range(len(counts))],
            "ratios": ratios.tolist(),
        }


class PrometheusDriftMonitor:
    def __init__(self, reference_path: Path) -> None:
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported drift reference schema")
        self._version = str(payload["version"])
        self._distributions = {
            (item["kind"], item["name"], item["model"]): item
            for item in payload["distributions"]
        }
        self._initialize_metrics()

    @property
    def version(self) -> str:
        return self._version

    def observe(
        self,
        data: pd.DataFrame,
        model: ModelName,
        scores: np.ndarray,
        threshold: float,
        duration_seconds: float,
    ) -> None:
        try:
            for key, distribution in self._distributions.items():
                kind, _, distribution_model = key
                if kind == "feature" and distribution_model == "all":
                    self._observe_feature(data, distribution)

            score_distribution = self._distributions[("score", "fraud_score", model)]
            self._observe_numeric(scores, score_distribution)

            predictions = np.asarray(scores) >= threshold
            fraud = int(predictions.sum())
            PREDICTIONS.labels(model, self._version, "fraud").inc(fraud)
            PREDICTIONS.labels(model, self._version, "legitimate").inc(
                len(predictions) - fraud
            )
            SCORE_SUM.labels(model, self._version).inc(float(np.sum(scores)))
            INFERENCE_DURATION.labels(model).observe(duration_seconds)
        except Exception as error:
            MONITORING_ERRORS.labels(type(error).__name__).inc()

    def _initialize_metrics(self) -> None:
        for distribution in self._distributions.values():
            labels = (
                distribution["kind"],
                distribution["name"],
                distribution["model"],
                self._version,
            )
            for bin_name, ratio in zip(
                distribution["bins"], distribution["ratios"], strict=True
            ):
                DRIFT_OBSERVATIONS.labels(*labels, bin_name)
                REFERENCE_RATIO.labels(*labels, bin_name).set(float(ratio))

        for model in ("catboost", "lightgbm"):
            PREDICTIONS.labels(model, self._version, "fraud")
            PREDICTIONS.labels(model, self._version, "legitimate")
            SCORE_SUM.labels(model, self._version)
            INFERENCE_DURATION.labels(model)

    def _observe_feature(
        self, data: pd.DataFrame, distribution: dict[str, Any]
    ) -> None:
        name = distribution["name"]
        if distribution["binning"] == "numeric":
            if name not in data:
                raise KeyError(f"Missing monitored feature: {name}")
            self._observe_numeric(data[name].to_numpy(dtype="float64"), distribution)
            return

        categories = distribution["categories"]
        values = self._category_values(data, name, categories)
        category_index = {value: index for index, value in enumerate(categories)}
        other = category_index["__other__"]
        indices = np.fromiter(
            (category_index.get(str(value), other) for value in values),
            dtype="int64",
            count=len(values),
        )
        self._increment(distribution, indices)

    def _observe_numeric(
        self, values: np.ndarray, distribution: dict[str, Any]
    ) -> None:
        values = np.asarray(values, dtype="float64")
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Observed drift values must be finite and one-dimensional")
        edges = np.asarray(distribution["edges"], dtype="float64")
        self._increment(distribution, np.searchsorted(edges, values, side="left"))

    @staticmethod
    def _category_values(
        data: pd.DataFrame,
        name: str,
        categories: list[str],
    ) -> np.ndarray:
        if name in data:
            return data[name].astype(str).to_numpy()

        values = np.full(len(data), "__other__", dtype=object)
        for category in categories:
            column = f"{name}_{category}"
            if category != "__other__" and column in data:
                values[data[column].to_numpy() == 1] = category
        return values

    def _increment(self, distribution: dict[str, Any], indices: np.ndarray) -> None:
        counts = np.bincount(indices, minlength=len(distribution["bins"]))
        labels = (
            distribution["kind"],
            distribution["name"],
            distribution["model"],
            self._version,
        )
        for bin_name, count in zip(distribution["bins"], counts, strict=True):
            if count:
                DRIFT_OBSERVATIONS.labels(*labels, bin_name).inc(int(count))
