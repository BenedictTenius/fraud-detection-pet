import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG, TuningConfig
from src.model.baseline import load_tuned_parameters
from src.model.tuning import (
    ExpandingWindowSplitter,
    HyperparameterTuner,
    ParameterSet,
    TemporalDatasetBuilder,
)
from src.model.types import DataSplit, ModelName


def write_raw_dataset(path: Path) -> None:
    rows = []
    transaction_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    for step in range(1, 25):
        for index in range(5):
            fraud = int(index == 0)
            rows.append(
                {
                    "step": step,
                    "type": transaction_types[index],
                    "amount": float(1_000 * fraud + step + index),
                    "oldbalanceOrg": float(2_000 - step * 10),
                    "oldbalanceDest": float(step * 20),
                    "isFraud": fraud,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def tuning_dataset(tmp_path: Path):
    raw_path = tmp_path / "paysim.csv"
    write_raw_dataset(raw_path)
    config = replace(CONFIG, paths=replace(CONFIG.paths, raw_data=raw_path))
    return config, TemporalDatasetBuilder(config).load()


def test_temporal_builder_excludes_final_test_and_preserves_schema(
    tmp_path: Path,
) -> None:
    config, dataset = tuning_dataset(tmp_path)

    assert dataset.steps.max() < 24
    assert dataset.features.columns.tolist() == list(config.prediction.feature_names)
    assert dataset.features.dtypes.eq("float32").all()


def test_expanding_window_split_has_step_gap_and_both_classes(
    tmp_path: Path,
) -> None:
    _, dataset = tuning_dataset(tmp_path)
    splitter = ExpandingWindowSplitter(fold_count=3, gap_steps=1)
    plans = splitter.plan(dataset.steps)

    assert len(plans) == 3
    for plan in plans:
        fold = splitter.materialize(dataset, plan)
        assert plan.train_end_step < plan.valid_start_step
        assert plan.valid_start_step - plan.train_end_step >= 2
        assert set(fold.train.target.unique()) == {0, 1}
        assert set(fold.valid.target.unique()) == {0, 1}


class FakeRunner:
    file_extension = ".fake"

    def __init__(self, model_name: ModelName) -> None:
        self.name = model_name

    @property
    def class_weight_power(self) -> float:
        return 0.0

    @property
    def best_iteration(self) -> int:
        return 1

    def fit(
        self,
        train: DataSplit,
        valid: DataSplit,
        class_weight: float,
    ) -> None:
        assert len(train.target) > 0
        assert len(valid.target) > 0
        assert class_weight > 0

    @staticmethod
    def predict_proba(features: pd.DataFrame) -> np.ndarray:
        return (features["amount"].to_numpy() > 500).astype("float64") * 0.8 + 0.1

    @staticmethod
    def feature_importance(feature_names: list[str]) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def training_history() -> dict[str, dict[str, list[float]]]:
        raise NotImplementedError

    @staticmethod
    def explain(features: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def save(path: Path) -> None:
        raise NotImplementedError


def fake_runner_builder(
    model_name: ModelName,
    parameters: ParameterSet,
) -> FakeRunner:
    assert parameters
    return FakeRunner(model_name)


def test_tuner_persists_study_trials_and_best_parameters(tmp_path: Path) -> None:
    config, dataset = tuning_dataset(tmp_path)
    output_dir = tmp_path / "tuning"
    report_path = HyperparameterTuner(
        config=config,
        output_dir=output_dir,
        fold_count=3,
        gap_steps=1,
        startup_trials=1,
        runner_builder=fake_runner_builder,
    ).run(
        dataset=dataset,
        model_names=("catboost", "lightgbm"),
        trials=1,
        timeout_seconds=30,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["validation"] == "expanding_window"
    assert report["models"]["catboost"]["temporal_validation"]["fold_f1"] == [
        1.0,
        1.0,
        1.0,
    ]
    assert report["models"]["lightgbm"]["completed_trials"] == 1
    assert report["models"]["lightgbm"]["best_params"]
    assert (output_dir / "optuna.db").is_file()
    assert (output_dir / "catboost_trials.csv").is_file()
    assert (output_dir / "lightgbm_trials.csv").is_file()
    assert (output_dir / "temporal_cv_f1.png").is_file()

    metrics = pd.read_csv(output_dir / "temporal_cv_metrics.csv")
    assert metrics.groupby("model").size().to_dict() == {
        "catboost": 3,
        "lightgbm": 3,
    }
    assert metrics["f1"].eq(1.0).all()


def test_baseline_loads_known_tuned_parameters(tmp_path: Path) -> None:
    path = tmp_path / "best_params.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "catboost": {"best_params": {"depth": 8, "learning_rate": 0.03}}
                },
            }
        ),
        encoding="utf-8",
    )

    parameters = load_tuned_parameters(path)

    assert parameters == {"catboost": {"depth": 8, "learning_rate": 0.03}}


def test_baseline_rejects_unknown_tuned_parameter(tmp_path: Path) -> None:
    path = tmp_path / "best_params.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "lightgbm": {"best_params": {"unknown": 1.0}},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown tuned parameters"):
        load_tuned_parameters(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trials": 0},
        {"fold_count": 1},
        {"gap_steps": -1},
        {"timeout_seconds": 0},
        {"startup_trials": -1},
    ],
)
def test_tuning_config_rejects_invalid_values(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        TuningConfig(**kwargs)
