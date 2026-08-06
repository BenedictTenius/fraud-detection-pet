import argparse
import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import CONFIG, AppConfig
from src.model.training import CatBoostRunner, LightGBMRunner
from src.model.types import DataSplit, ModelName, ModelRunner

TuningValue: TypeAlias = int | float
ParameterSet: TypeAlias = dict[str, TuningValue]
RunnerBuilder: TypeAlias = Callable[[ModelName, ParameterSet], ModelRunner]


@dataclass(frozen=True)
class TemporalDataset:
    features: pd.DataFrame
    target: pd.Series
    steps: np.ndarray

    def __post_init__(self) -> None:
        row_count = len(self.features)
        if row_count == 0 or len(self.target) != row_count:
            raise ValueError("Tuning dataset must contain aligned non-empty rows")
        if self.steps.shape != (row_count,):
            raise ValueError("Tuning steps must contain one value per row")
        if np.any(self.steps[1:] < self.steps[:-1]):
            raise ValueError("Tuning data must be sorted by step")
        if set(self.target.unique()) != {0, 1}:
            raise ValueError("Tuning target must contain both classes")
        if not np.isfinite(self.features.to_numpy(dtype="float64")).all():
            raise ValueError("Tuning features must be finite")


@dataclass(frozen=True)
class TemporalFoldPlan:
    number: int
    train_stop: int
    valid_start: int
    valid_stop: int
    train_end_step: int
    valid_start_step: int
    valid_end_step: int


@dataclass(frozen=True)
class TemporalFold:
    plan: TemporalFoldPlan
    train: DataSplit
    valid: DataSplit


class TemporalDatasetBuilder:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def load(self) -> TemporalDataset:
        data_config = self._config.data
        prediction_config = self._config.prediction
        category = data_config.categorical_column
        target = data_config.target_column
        step = data_config.time_column
        numeric_features = [
            name
            for name in prediction_config.feature_names
            if not name.startswith(f"{category}_")
        ]
        columns = [step, category, *numeric_features, target]
        dtype = {
            step: "int32",
            category: "category",
            target: "int8",
            **dict.fromkeys(numeric_features, "float32"),
        }
        data = pd.read_csv(
            self._config.paths.raw_data,
            usecols=columns,
            dtype=cast(Any, dtype),
        ).sort_values(step, kind="stable")

        final_test_start = data[step].quantile(
            data_config.train_share + data_config.valid_share,
            interpolation="lower",
        )
        data = data[data[step] <= final_test_start].reset_index(drop=True)
        categories = [
            name.removeprefix(f"{category}_")
            for name in prediction_config.feature_names
            if name.startswith(f"{category}_")
        ]
        unknown_categories = set(data[category].astype(str).unique()) - set(categories)
        if unknown_categories:
            raise ValueError(f"Unknown tuning categories: {sorted(unknown_categories)}")

        encoded = data[[*numeric_features, category]].copy()
        encoded[category] = pd.Categorical(encoded[category], categories=categories)
        features = pd.get_dummies(
            encoded,
            columns=[category],
            dtype="int8",
        ).reindex(columns=prediction_config.feature_names, fill_value=0)
        return TemporalDataset(
            features=features.astype("float32"),
            target=data[target].astype("int8"),
            steps=data[step].to_numpy(dtype="int64"),
        )


class ExpandingWindowSplitter:
    def __init__(self, fold_count: int, gap_steps: int) -> None:
        if fold_count < 2:
            raise ValueError("Temporal fold_count must be at least two")
        if gap_steps < 0:
            raise ValueError("Temporal gap_steps must not be negative")
        self._fold_count = fold_count
        self._gap_steps = gap_steps

    def plan(self, steps: np.ndarray) -> tuple[TemporalFoldPlan, ...]:
        unique_steps = np.unique(steps)
        splitter = TimeSeriesSplit(
            n_splits=self._fold_count,
            gap=self._gap_steps,
        )
        try:
            ranges = list(splitter.split(unique_steps))
        except ValueError as error:
            raise ValueError(f"Cannot build temporal folds: {error}") from error

        plans = []
        for number, (train_indices, valid_indices) in enumerate(ranges, start=1):
            train_end_step = int(unique_steps[train_indices[-1]])
            valid_start_step = int(unique_steps[valid_indices[0]])
            valid_end_step = int(unique_steps[valid_indices[-1]])
            plans.append(
                TemporalFoldPlan(
                    number=number,
                    train_stop=int(
                        np.searchsorted(steps, train_end_step, side="right")
                    ),
                    valid_start=int(
                        np.searchsorted(steps, valid_start_step, side="left")
                    ),
                    valid_stop=int(
                        np.searchsorted(steps, valid_end_step, side="right")
                    ),
                    train_end_step=train_end_step,
                    valid_start_step=valid_start_step,
                    valid_end_step=valid_end_step,
                )
            )
        return tuple(plans)

    @staticmethod
    def materialize(
        dataset: TemporalDataset,
        plan: TemporalFoldPlan,
    ) -> TemporalFold:
        train = DataSplit(
            features=dataset.features.iloc[: plan.train_stop],
            target=dataset.target.iloc[: plan.train_stop],
        )
        valid = DataSplit(
            features=dataset.features.iloc[plan.valid_start : plan.valid_stop],
            target=dataset.target.iloc[plan.valid_start : plan.valid_stop],
        )
        for name, target in (("train", train.target), ("validation", valid.target)):
            if set(target.unique()) != {0, 1}:
                raise ValueError(f"Temporal fold {plan.number} {name} has one class")
        return TemporalFold(plan=plan, train=train, valid=valid)


class ModelSearchSpace:
    @staticmethod
    def suggest(model_name: ModelName, trial: optuna.Trial) -> ParameterSet:
        if model_name == "catboost":
            return {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.20, log=True
                ),
                "depth": trial.suggest_int("depth", 5, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                "random_strength": trial.suggest_float(
                    "random_strength", 1e-3, 10.0, log=True
                ),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", 0.0, 5.0
                ),
                "class_weight_power": trial.suggest_float(
                    "class_weight_power", 0.0, 1.0, step=0.25
                ),
            }
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 20, 500, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "class_weight_power": trial.suggest_float(
                "class_weight_power", 0.0, 1.0, step=0.25
            ),
        }


class ConfiguredRunnerBuilder:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._training = replace(config.training, log_period=0)

    def __call__(
        self,
        model_name: ModelName,
        parameters: ParameterSet,
    ) -> ModelRunner:
        if model_name == "catboost":
            catboost_config = replace(
                self._config.catboost,
                **cast(Any, parameters),
            )
            return CatBoostRunner(catboost_config, self._training)
        lightgbm_config = replace(
            self._config.lightgbm,
            **cast(Any, parameters),
        )
        return LightGBMRunner(lightgbm_config, self._training)


class TemporalObjective:
    def __init__(
        self,
        model_name: ModelName,
        dataset: TemporalDataset,
        plans: Sequence[TemporalFoldPlan],
        runner_builder: RunnerBuilder,
    ) -> None:
        self._model_name = model_name
        self._dataset = dataset
        self._plans = plans
        self._runner_builder = runner_builder

    def __call__(self, trial: optuna.Trial) -> float:
        parameters = ModelSearchSpace.suggest(self._model_name, trial)
        scores = []
        best_iterations = []
        for plan in self._plans:
            fold = ExpandingWindowSplitter.materialize(self._dataset, plan)
            runner = self._runner_builder(self._model_name, parameters)
            class_weight = self._class_ratio(fold.train.target) ** float(
                parameters["class_weight_power"]
            )
            runner.fit(fold.train, fold.valid, class_weight)
            probability = runner.predict_proba(fold.valid.features)
            score = float(average_precision_score(fold.valid.target, probability))
            scores.append(score)
            best_iterations.append(runner.best_iteration)
            trial.report(float(np.mean(scores)), step=plan.number - 1)
            if trial.should_prune():
                trial.set_user_attr("fold_scores", scores)
                raise optuna.TrialPruned()

        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("fold_score_std", float(np.std(scores)))
        trial.set_user_attr("best_iterations", best_iterations)
        return float(np.mean(scores))

    @staticmethod
    def _class_ratio(target: pd.Series) -> float:
        positive = int(target.sum())
        negative = len(target) - positive
        if positive == 0 or negative == 0:
            raise ValueError("Temporal training target must contain both classes")
        return negative / positive


class HyperparameterTuner:
    def __init__(
        self,
        config: AppConfig,
        output_dir: Path,
        fold_count: int,
        gap_steps: int,
        startup_trials: int,
        runner_builder: RunnerBuilder | None = None,
    ) -> None:
        self._config = config
        self._output_dir = output_dir
        self._fold_count = fold_count
        self._gap_steps = gap_steps
        self._splitter = ExpandingWindowSplitter(fold_count, gap_steps)
        self._startup_trials = startup_trials
        self._runner_builder = runner_builder or ConfiguredRunnerBuilder(config)

    def run(
        self,
        dataset: TemporalDataset,
        model_names: Sequence[ModelName],
        trials: int,
        timeout_seconds: int,
    ) -> Path:
        if trials < 1 or timeout_seconds < 1:
            raise ValueError("Tuning trials and timeout must be positive")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        plans = self._splitter.plan(dataset.steps)
        data_version = self._data_version()
        study_signature = self._study_signature(data_version)
        storage_path = (self._output_dir / "optuna.db").resolve()
        storage = f"sqlite:///{storage_path}"
        model_results = {}

        for model_name in model_names:
            study = optuna.create_study(
                study_name=(
                    f"fraud-{model_name}-temporal-v1-"
                    f"f{self._fold_count}-g{self._gap_steps}"
                ),
                storage=storage,
                direction="maximize",
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(
                    seed=self._config.training.random_seed,
                    n_startup_trials=self._startup_trials,
                ),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=self._startup_trials,
                    n_warmup_steps=1,
                ),
            )
            existing_signature = study.user_attrs.get("study_signature")
            if existing_signature is None:
                study.set_user_attr("study_signature", study_signature)
            elif existing_signature != study_signature:
                raise ValueError(
                    f"Stored Optuna study for {model_name} uses different data "
                    "or features; choose another output directory"
                )
            objective = TemporalObjective(
                model_name,
                dataset,
                plans,
                self._runner_builder,
            )
            study.optimize(
                objective,
                n_trials=trials,
                timeout=timeout_seconds,
                n_jobs=1,
                gc_after_trial=True,
            )
            completed = [
                trial
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            ]
            if not completed:
                raise RuntimeError(f"No completed Optuna trials for {model_name}")
            study.trials_dataframe().to_csv(
                self._output_dir / f"{model_name}_trials.csv",
                index=False,
            )
            model_results[model_name] = {
                "study_name": study.study_name,
                "best_trial": study.best_trial.number,
                "best_average_precision": float(study.best_value),
                "best_params": dict(study.best_params),
                "completed_trials": len(completed),
                "total_trials": len(study.trials),
            }
            print(
                f"{model_name}: best temporal AP={study.best_value:.6f}, "
                f"trial={study.best_trial.number}"
            )

        return self._save_report(
            model_results,
            plans,
            trials,
            timeout_seconds,
            data_version,
        )

    def _save_report(
        self,
        model_results: dict[ModelName, dict[str, object]],
        plans: Sequence[TemporalFoldPlan],
        trials: int,
        timeout_seconds: int,
        data_version: str,
    ) -> Path:
        output_path = self._output_dir / "best_params.json"
        previous_models = {}
        if output_path.exists():
            previous = json.loads(output_path.read_text(encoding="utf-8"))
            if previous.get("schema_version") != 1:
                raise ValueError("Unsupported tuning report schema")
            if previous.get("data_version") != data_version:
                raise ValueError(
                    "Existing tuning report uses another dataset version; "
                    "choose another output directory"
                )
            if previous.get("folds") != self._fold_report(plans):
                raise ValueError(
                    "Existing tuning report uses another temporal split; "
                    "choose another output directory"
                )
            previous_models = previous.get("models", {})
        previous_models.update(model_results)
        payload = {
            "schema_version": 1,
            "objective": "mean_temporal_average_precision",
            "validation": "expanding_window",
            "data_version": data_version,
            "config": {
                **asdict(self._config.tuning),
                "trials": trials,
                "fold_count": self._fold_count,
                "gap_steps": self._gap_steps,
                "timeout_seconds": timeout_seconds,
                "startup_trials": self._startup_trials,
            },
            "folds": self._fold_report(plans),
            "models": previous_models,
        }
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def _data_version(self) -> str:
        raw_path = self._config.paths.raw_data
        pointer_path = raw_path.with_name(f"{raw_path.name}.dvc")
        if pointer_path.is_file():
            match = re.search(
                r"\bmd5:\s*([0-9a-f]{32})\b",
                pointer_path.read_text(encoding="utf-8"),
            )
            if match:
                return match.group(1)
        stat = raw_path.stat()
        return f"size-{stat.st_size}-mtime-{stat.st_mtime_ns}"

    def _study_signature(self, data_version: str) -> str:
        payload = {
            "data_version": data_version,
            "features": self._config.prediction.feature_names,
            "pretest_share": (
                self._config.data.train_share + self._config.data.valid_share
            ),
            "fold_count": self._fold_count,
            "gap_steps": self._gap_steps,
            "search_space_version": 1,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @staticmethod
    def _fold_report(
        plans: Sequence[TemporalFoldPlan],
    ) -> list[dict[str, int]]:
        return [
            {
                "number": plan.number,
                "train_end_step": plan.train_end_step,
                "valid_start_step": plan.valid_start_step,
                "valid_end_step": plan.valid_end_step,
            }
            for plan in plans
        ]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def parse_args() -> argparse.Namespace:
    tuning = CONFIG.tuning
    parser = argparse.ArgumentParser(
        description="Tune fraud models with temporal cross-validation"
    )
    parser.add_argument(
        "--model",
        choices=("catboost", "lightgbm", "all"),
        default="all",
    )
    parser.add_argument("--trials", type=positive_int, default=tuning.trials)
    parser.add_argument("--folds", type=positive_int, default=tuning.fold_count)
    parser.add_argument(
        "--gap-steps",
        type=non_negative_int,
        default=tuning.gap_steps,
    )
    parser.add_argument(
        "--timeout",
        type=positive_int,
        default=tuning.timeout_seconds,
        help="Maximum seconds per model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONFIG.paths.artifact_dir.parent / "tuning",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = ("catboost", "lightgbm") if args.model == "all" else (args.model,)
    model_names = cast(tuple[ModelName, ...], selected)
    print("Loading pre-test data for temporal tuning...")
    dataset = TemporalDatasetBuilder(CONFIG).load()
    print(
        f"Loaded {len(dataset.target):,} rows across "
        f"{len(np.unique(dataset.steps))} time steps"
    )
    output_path = HyperparameterTuner(
        config=CONFIG,
        output_dir=args.output_dir,
        fold_count=args.folds,
        gap_steps=args.gap_steps,
        startup_trials=CONFIG.tuning.startup_trials,
    ).run(
        dataset=dataset,
        model_names=model_names,
        trials=args.trials,
        timeout_seconds=args.timeout,
    )
    print(f"Best parameters saved to {output_path}")


if __name__ == "__main__":
    main()
