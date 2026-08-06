import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from tritonclient.grpc import InferInput
from tritonclient.grpc.aio import InferenceServerClient
from tritonclient.utils import InferenceServerException

from src.config import (
    CatBoostConfig,
    LightGBMConfig,
    PathConfig,
    PredictionConfig,
    TrainingConfig,
    TritonConfig,
)
from src.model.types import (
    DatasetBundle,
    DataSplit,
    ModelName,
    PredictionGateway,
    PredictionObserver,
    PredictionResult,
    PredictionSummary,
    PredictionUnavailableError,
    ShapExplanation,
    TrainingHistory,
)

logger = logging.getLogger(__name__)


class DatasetLoader:
    def __init__(self, paths: PathConfig, target_column: str) -> None:
        self._paths = paths
        self._target_column = target_column

    def load(self, path: Path) -> DataSplit:
        data = pd.read_csv(path, dtype="float32")
        if self._target_column not in data:
            raise ValueError(
                f"Target column '{self._target_column}' is missing in {path}"
            )
        if not np.isfinite(data.to_numpy(dtype="float64")).all():
            raise ValueError(f"Non-finite values found in {path}")

        target = data.pop(self._target_column).astype("int8")
        if not set(target.unique()).issubset({0, 1}):
            raise ValueError(f"Target in {path} must contain only 0 and 1")
        return DataSplit(features=data, target=target)

    def load_all(self) -> DatasetBundle:
        bundle = DatasetBundle(
            train=self.load(self._paths.train_data),
            valid=self.load(self._paths.valid_data),
            test=self.load(self._paths.test_data),
        )
        expected = bundle.train.features.columns.tolist()
        for name, split in (("valid", bundle.valid), ("test", bundle.test)):
            if split.features.columns.tolist() != expected:
                raise ValueError(f"Feature schema of {name} does not match train")
        return bundle


class CatBoostRunner:
    name: ModelName = "catboost"
    file_extension = ".cbm"

    def __init__(
        self,
        config: CatBoostConfig,
        training_config: TrainingConfig,
    ) -> None:
        self._config = config
        self._training_config = training_config
        self._model: CatBoostClassifier | None = None

    @property
    def best_iteration(self) -> int:
        return int(self._fitted_model().get_best_iteration())

    @property
    def class_weight_power(self) -> float:
        return self._config.class_weight_power

    def fit(
        self,
        train: DataSplit,
        valid: DataSplit,
        class_weight: float,
    ) -> None:
        device_parameters = {}
        if self._training_config.device == "gpu":
            device_parameters = {
                "task_type": "GPU",
                "devices": self._training_config.gpu_devices,
            }

        self._model = CatBoostClassifier(
            iterations=self._config.iterations,
            learning_rate=self._config.learning_rate,
            depth=self._config.depth,
            l2_leaf_reg=self._config.l2_leaf_reg,
            loss_function="Logloss",
            eval_metric="PRAUC",
            scale_pos_weight=class_weight,
            random_seed=self._training_config.random_seed,
            thread_count=-1,
            allow_writing_files=False,
            verbose=self._training_config.log_period,
            **device_parameters,
        )
        self._model.fit(
            train.features,
            train.target,
            eval_set=(valid.features, valid.target),
            early_stopping_rounds=self._config.early_stopping_rounds,
            use_best_model=True,
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self._fitted_model().predict_proba(features)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        return pd.Series(self._fitted_model().feature_importances_, index=feature_names)

    def training_history(self) -> TrainingHistory:
        return self._fitted_model().get_evals_result()

    def explain(self, features: pd.DataFrame) -> ShapExplanation:
        contributions = np.asarray(
            self._fitted_model().get_feature_importance(
                Pool(features), type="ShapValues"
            ),
            dtype="float64",
        )
        return ShapExplanation(
            values=contributions[:, :-1],
            base_values=contributions[:, -1],
        )

    def save(self, path: Path) -> None:
        self._fitted_model().save_model(str(path))

    def _fitted_model(self) -> CatBoostClassifier:
        if self._model is None:
            raise RuntimeError("CatBoost model is not fitted")
        return self._model


class LightGBMRunner:
    name: ModelName = "lightgbm"
    file_extension = ".txt"

    def __init__(
        self,
        config: LightGBMConfig,
        training_config: TrainingConfig,
    ) -> None:
        self._config = config
        self._training_config = training_config
        self._model: lgb.LGBMClassifier | None = None

    @property
    def best_iteration(self) -> int:
        return int(self._fitted_model().best_iteration_)

    @property
    def class_weight_power(self) -> float:
        return self._config.class_weight_power

    def fit(
        self,
        train: DataSplit,
        valid: DataSplit,
        class_weight: float,
    ) -> None:
        self._model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self._config.n_estimators,
            learning_rate=self._config.learning_rate,
            num_leaves=self._config.num_leaves,
            max_depth=self._config.max_depth,
            min_child_samples=self._config.min_child_samples,
            subsample=self._config.subsample,
            subsample_freq=1,
            colsample_bytree=self._config.colsample_bytree,
            reg_lambda=self._config.reg_lambda,
            scale_pos_weight=class_weight,
            random_state=self._training_config.random_seed,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        self._model.fit(
            train.features,
            train.target,
            eval_X=(train.features, valid.features),
            eval_y=cast(Any, (train.target, valid.target)),
            eval_names=["train", "valid"],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(
                    self._config.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=True,
                ),
                lgb.log_evaluation(self._training_config.log_period),
            ],
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        model = self._fitted_model()
        probability = model.predict_proba(features, num_iteration=model.best_iteration_)
        return np.asarray(probability)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        return pd.Series(self._fitted_model().feature_importances_, index=feature_names)

    def training_history(self) -> TrainingHistory:
        return self._fitted_model().evals_result_

    def explain(self, features: pd.DataFrame) -> ShapExplanation:
        model = self._fitted_model()
        contributions = np.asarray(
            model.booster_.predict(
                features,
                pred_contrib=True,
                num_iteration=model.best_iteration_,
            ),
            dtype="float64",
        )
        return ShapExplanation(
            values=contributions[:, :-1],
            base_values=contributions[:, -1],
        )

    def save(self, path: Path) -> None:
        model = self._fitted_model()
        model.booster_.save_model(str(path), num_iteration=model.best_iteration_)

    def _fitted_model(self) -> lgb.LGBMClassifier:
        if self._model is None:
            raise RuntimeError("LightGBM model is not fitted")
        return self._model


class TritonPredictionGateway:
    def __init__(
        self,
        config: TritonConfig,
        client: InferenceServerClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or InferenceServerClient(url=config.grpc_url)

    async def close(self) -> None:
        await self._client.close()

    async def is_ready(self, model_names: tuple[ModelName, ...]) -> bool:
        try:
            if not await self._client.is_server_live(
                client_timeout=self._config.request_timeout
            ):
                return False
            if not await self._client.is_server_ready(
                client_timeout=self._config.request_timeout
            ):
                return False
            return all(
                [
                    await self._client.is_model_ready(
                        self._config.model_name(model_name),
                        client_timeout=self._config.request_timeout,
                    )
                    for model_name in model_names
                ]
            )
        except (InferenceServerException, OSError):
            return False

    async def predict(self, model_name: ModelName, features: np.ndarray) -> np.ndarray:
        matrix = np.ascontiguousarray(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Triton input must be a two-dimensional matrix")
        if len(matrix) == 0:
            raise ValueError("Triton input must not be empty")

        chunks = []
        for start in range(0, len(matrix), self._config.max_batch_size):
            chunks.append(
                await self._predict_chunk(
                    model_name,
                    matrix[start : start + self._config.max_batch_size],
                )
            )
        return np.concatenate(chunks)

    async def _predict_chunk(
        self, model_name: ModelName, features: np.ndarray
    ) -> np.ndarray:
        request_input = InferInput(
            self._config.input_name,
            features.shape,
            "FP32",
        )
        request_input.set_data_from_numpy(features)

        try:
            response = await self._client.infer(
                model_name=self._config.model_name(model_name),
                inputs=[request_input],
                client_timeout=self._config.request_timeout,
            )
        except (InferenceServerException, OSError) as error:
            raise PredictionUnavailableError(
                f"Triton inference failed for {model_name}: {error}"
            ) from error

        output = response.as_numpy(self._config.output_name)
        if output is None:
            raise PredictionUnavailableError(
                f"Triton response for {model_name} has no score output"
            )
        scores = np.asarray(output, dtype=np.float32).reshape(-1)
        if len(scores) != len(features):
            raise PredictionUnavailableError(
                f"Triton returned {len(scores)} scores for {len(features)} rows"
            )
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise PredictionUnavailableError(
                f"Triton returned invalid probability scores for {model_name}"
            )
        return scores


class FraudPredictionService:
    _models: tuple[ModelName, ...] = ("catboost", "lightgbm")

    def __init__(
        self,
        gateway: PredictionGateway,
        default_model: ModelName,
        paths: PathConfig,
        prediction_config: PredictionConfig,
        target_column: str,
        categorical_column: str,
        observer: PredictionObserver | None = None,
    ) -> None:
        self._gateway = gateway
        self._default_model = default_model
        self._paths = paths
        self._config = prediction_config
        self._target_column = target_column
        self._categorical_column = categorical_column
        self._observer = observer
        self._feature_names = list(prediction_config.feature_names)
        self._thresholds = self._load_thresholds()

    @property
    def model_name(self) -> ModelName:
        return self._default_model

    @property
    def thresholds(self) -> dict[ModelName, float]:
        return self._thresholds.copy()

    async def close(self) -> None:
        await self._gateway.close()

    async def is_ready(self) -> bool:
        return await self._gateway.is_ready(self._models)

    async def predict_one(
        self,
        transaction: Mapping[str, Any],
        model_name: ModelName | None = None,
        threshold: float | None = None,
    ) -> PredictionResult:
        selected_model = model_name or self._default_model
        result = (
            await self.predict_batch(
                pd.DataFrame([transaction]),
                model_name=selected_model,
                threshold=threshold,
            )
        ).iloc[0]
        return PredictionResult(
            model=selected_model,
            fraud_score=float(result[self._config.score_column]),
            threshold=float(result["threshold"]),
            is_fraud=bool(result[self._config.prediction_column]),
        )

    async def predict_batch(
        self,
        data: pd.DataFrame,
        model_name: ModelName | None = None,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        if data.empty:
            raise ValueError("Prediction batch must not be empty")

        selected_model = model_name or self._default_model
        selected_threshold = self._select_threshold(selected_model, threshold)
        features = self._prepare_features(data)
        started_at = time.perf_counter()
        scores = self._validate_scores(
            await self._gateway.predict(selected_model, features.to_numpy(copy=False)),
            len(features),
        )
        duration_seconds = time.perf_counter() - started_at
        if self._observer is not None:
            self._observer.observe(
                data=data,
                model=selected_model,
                scores=scores,
                threshold=selected_threshold,
                duration_seconds=duration_seconds,
            )

        fraud_alerts = int((scores >= selected_threshold).sum())
        logger.info(
            "prediction_batch_completed",
            extra={
                "model": selected_model,
                "batch_size": len(features),
                "fraud_alerts": fraud_alerts,
                "threshold": selected_threshold,
                "duration_ms": round(duration_seconds * 1_000, 3),
            },
        )

        return pd.DataFrame(
            {
                "model": selected_model,
                self._config.score_column: scores,
                "threshold": selected_threshold,
                self._config.prediction_column: scores >= selected_threshold,
            },
            index=data.index,
        )

    async def predict_file(
        self,
        input_path: Path,
        output_path: Path,
        model_name: ModelName | None = None,
        threshold: float | None = None,
    ) -> PredictionSummary:
        selected_model = model_name or self._default_model
        data = pd.read_csv(input_path)
        actual = data.pop(self._target_column) if self._target_column in data else None
        result = (
            await self.predict_batch(
                data,
                model_name=selected_model,
                threshold=threshold,
            )
        ).reset_index(drop=True)
        result.insert(0, "row_id", np.arange(len(result)))
        if actual is not None:
            result[self._target_column] = actual.to_numpy(dtype="int8")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        return PredictionSummary(
            model=selected_model,
            rows=len(data),
            fraud_alerts=int(result[self._config.prediction_column].sum()),
            threshold=float(result["threshold"].iloc[0]),
            output_path=output_path,
        )

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        dummy_prefix = f"{self._categorical_column}_"
        if self._categorical_column in data:
            known_categories = {
                name.removeprefix(dummy_prefix)
                for name in self._feature_names
                if name.startswith(dummy_prefix)
            }
            unknown = set(data[self._categorical_column].unique()) - known_categories
            if unknown:
                raise ValueError(f"Unknown transaction types: {sorted(unknown)}")
            data = pd.get_dummies(
                data, columns=[self._categorical_column], dtype="int8"
            )

        missing = set(self._feature_names) - set(data.columns)
        missing_numeric = {
            name for name in missing if not name.startswith(dummy_prefix)
        }
        if missing_numeric:
            raise ValueError(f"Missing model features: {sorted(missing_numeric)}")
        for name in missing - missing_numeric:
            data[name] = 0

        features = data.loc[:, self._feature_names].astype("float32")
        if not np.isfinite(features.to_numpy(copy=False)).all():
            raise ValueError("Prediction data must contain only finite values")

        numeric_names = [
            name for name in self._feature_names if not name.startswith(dummy_prefix)
        ]
        numeric = features.loc[:, numeric_names]
        if (numeric < 0).any().any() or (
            numeric > self._config.max_transaction_value
        ).any().any():
            raise ValueError("Prediction values are outside the allowed range")

        category_names = [
            name for name in self._feature_names if name.startswith(dummy_prefix)
        ]
        categories = features.loc[:, category_names]
        is_binary = ((categories == 0) | (categories == 1)).all().all()
        if not is_binary or not (categories.sum(axis=1) == 1).all():
            raise ValueError("Transaction type must be valid one-hot encoding")
        return features

    @staticmethod
    def _validate_scores(scores: np.ndarray, expected_rows: int) -> np.ndarray:
        values = np.asarray(scores, dtype="float32")
        if values.shape != (expected_rows,):
            raise PredictionUnavailableError(
                f"Prediction gateway returned shape {values.shape}, "
                f"expected ({expected_rows},)"
            )
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise PredictionUnavailableError(
                "Prediction gateway returned invalid probability scores"
            )
        return values

    def _select_threshold(
        self, model_name: ModelName, threshold: float | None
    ) -> float:
        selected = self._thresholds[model_name] if threshold is None else threshold
        if not 0 < selected < 1:
            raise ValueError("Prediction threshold must be between 0 and 1")
        return selected

    def _load_thresholds(self) -> dict[ModelName, float]:
        report = json.loads(self._paths.metrics_file.read_text(encoding="utf-8"))
        thresholds: dict[ModelName, float] = {}
        for model_name in self._models:
            try:
                thresholds[model_name] = float(
                    report["models"][model_name]["validation"]["threshold"]
                )
            except KeyError as error:
                raise ValueError(f"Threshold for {model_name} is missing") from error
        return thresholds
