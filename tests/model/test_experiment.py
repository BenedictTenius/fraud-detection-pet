from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model.experiment import (
    ArtifactStore,
    BinaryEvaluator,
    ReportPlotter,
    ThresholdSelector,
)
from src.model.types import DatasetBundle, DataSplit, ShapExplanation


def test_threshold_selector_honours_minimum_recall() -> None:
    target = pd.Series([0, 0, 0, 1, 1], dtype="int8")
    probability = np.array([0.05, 0.10, 0.40, 0.60, 0.90])

    threshold = ThresholdSelector(minimum_recall=1.0).select(target, probability)

    assert threshold == pytest.approx(0.6)


def test_binary_evaluator_uses_the_selected_threshold() -> None:
    target = pd.Series([0, 0, 1, 1], dtype="int8")
    probability = np.array([0.1, 0.8, 0.7, 0.9])

    metrics = BinaryEvaluator.evaluate(target, probability, threshold=0.75)

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["threshold"] == 0.75


def test_shap_artifacts_are_saved(tmp_path: Path) -> None:
    features = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0],
            "oldbalanceOrg": [5.0, 25.0, 20.0],
        }
    )
    explanation = ShapExplanation(
        values=np.array([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.1]]),
        base_values=np.zeros(3),
    )

    csv_path = ArtifactStore(tmp_path).save_shap_importance(
        "lightgbm", features.columns.tolist(), explanation
    )
    plot_path = ReportPlotter(tmp_path).save_shap_summary(
        "lightgbm", features, explanation, max_display=2
    )

    importance = pd.read_csv(csv_path, index_col="feature")
    assert importance.index[0] == "amount"
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0


def test_shap_explanation_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        ShapExplanation(values=np.array([1.0]), base_values=np.array([0.0]))


class FakeHistoryModel:
    name = "lightgbm"

    @staticmethod
    def training_history() -> dict[str, dict[str, list[float]]]:
        return {
            "train": {"average_precision": [0.5, 0.7]},
            "valid": {"average_precision": [0.4, 0.6]},
        }


def test_evaluation_plots_are_saved(tmp_path: Path) -> None:
    plotter = ReportPlotter(tmp_path)
    target = pd.Series([0, 0, 0, 1, 1, 1], dtype="int8")
    probability = np.array([0.05, 0.1, 0.4, 0.6, 0.8, 0.95])

    learning = plotter.save_training_curves(FakeHistoryModel())
    evaluation = plotter.save_evaluation_dashboard(
        "lightgbm", target, probability, threshold=0.5
    )
    importance = plotter.save_feature_importance(
        "lightgbm", pd.Series({"amount": 2.0, "balance": 1.0})
    )

    assert all(path.is_file() for path in (learning, evaluation, importance))


def test_feature_distribution_plot_is_saved(tmp_path: Path) -> None:
    train = DataSplit(
        features=pd.DataFrame(
            {
                "amount": np.linspace(1, 100, 100),
                "type_TRANSFER": [0, 1] * 50,
            }
        ),
        target=pd.Series([0] * 100, dtype="int8"),
    )
    valid = DataSplit(
        features=pd.DataFrame(
            {
                "amount": np.linspace(5, 105, 40),
                "type_TRANSFER": [0, 1] * 20,
            }
        ),
        target=pd.Series([0] * 40, dtype="int8"),
    )
    test = DataSplit(
        features=pd.DataFrame(
            {
                "amount": np.linspace(10, 110, 40),
                "type_TRANSFER": [1, 0] * 20,
            }
        ),
        target=pd.Series([0] * 40, dtype="int8"),
    )

    output_path = ReportPlotter(tmp_path).save_feature_distributions(
        DatasetBundle(train=train, valid=valid, test=test)
    )

    assert output_path.name == "feature_distributions.png"
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
