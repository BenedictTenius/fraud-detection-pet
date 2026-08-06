import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fraud-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import AppConfig
from src.metrics.drift import DriftReferenceBuilder
from src.model.training import DatasetLoader
from src.model.types import (
    DatasetBundle,
    ExperimentReport,
    Metrics,
    ModelRunner,
    ShapExplanation,
)


class ThresholdSelector:
    def __init__(self, minimum_recall: float) -> None:
        if not 0 < minimum_recall <= 1:
            raise ValueError("minimum_recall must be between 0 and 1")
        self._minimum_recall = minimum_recall

    def select(self, target: pd.Series, probability: np.ndarray) -> float:
        precision, recall, thresholds = precision_recall_curve(target, probability)
        eligible = np.flatnonzero(recall[:-1] >= self._minimum_recall)
        best_precision = precision[:-1][eligible].max()
        best = eligible[np.isclose(precision[:-1][eligible], best_precision)][-1]
        return float(thresholds[best])


class BinaryEvaluator:
    @staticmethod
    def evaluate(
        target: pd.Series,
        probability: np.ndarray,
        threshold: float,
    ) -> Metrics:
        prediction = (probability >= threshold).astype("int8")
        precision_curve, recall_curve, _ = precision_recall_curve(target, probability)
        fpr_curve, tpr_curve, roc_thresholds = roc_curve(target, probability)
        ks_values = tpr_curve - fpr_curve
        finite_indices = np.flatnonzero(np.isfinite(roc_thresholds))
        ks_index = int(finite_indices[np.argmax(ks_values[finite_indices])])
        tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
        prevalence = float(target.mean())
        precision = float(precision_score(target, prediction, zero_division=0))
        return {
            "pr_auc": float(auc(recall_curve, precision_curve)),
            "average_precision": float(average_precision_score(target, probability)),
            "roc_auc": float(roc_auc_score(target, probability)),
            "ks_statistic": float(ks_values[ks_index]),
            "ks_threshold": float(roc_thresholds[ks_index]),
            "precision": precision,
            "recall": float(recall_score(target, prediction, zero_division=0)),
            "f1": float(f1_score(target, prediction, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
            "matthews_corrcoef": float(matthews_corrcoef(target, prediction)),
            "specificity": float(tn / (tn + fp)),
            "false_positive_rate": float(fp / (fp + tn)),
            "false_negative_rate": float(fn / (fn + tp)),
            "brier_score": float(brier_score_loss(target, probability)),
            "log_loss": float(log_loss(target, probability)),
            "alert_rate": float(prediction.mean()),
            "lift": float(precision / prevalence),
            "threshold": threshold,
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }


class ArtifactStore:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def save_model(self, model: ModelRunner, feature_names: list[str]) -> pd.Series:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        model.save(self._output_dir / f"{model.name}{model.file_extension}")
        importance = (
            model.feature_importance(feature_names)
            .rename("importance")
            .rename_axis("feature")
            .sort_values(ascending=False)
        )
        importance.to_csv(self._output_dir / f"{model.name}_importance.csv")
        return importance

    def save_report(self, report: ExperimentReport) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / "metrics.json"
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return output_path

    def save_shap_importance(
        self,
        model_name: str,
        feature_names: list[str],
        explanation: ShapExplanation,
    ) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        importance = pd.DataFrame(
            {
                "mean_absolute_shap": np.abs(explanation.values).mean(axis=0),
                "mean_shap": explanation.values.mean(axis=0),
            },
            index=feature_names,
        ).sort_values("mean_absolute_shap", ascending=False)
        importance.index.name = "feature"
        output_path = self._output_dir / f"{model_name}_shap_importance.csv"
        importance.to_csv(output_path)
        return output_path


class ReportPlotter:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def save_training_curves(self, model: ModelRunner) -> Path:
        history = model.training_history()
        metrics = sorted(
            {
                metric
                for dataset_metrics in history.values()
                for metric in dataset_metrics
            }
        )
        if not metrics:
            raise ValueError(f"Training history for {model.name} is empty")

        figure, axes = plt.subplots(
            len(metrics), 1, figsize=(11, 4 * len(metrics)), squeeze=False
        )
        for axis, metric in zip(axes.flat, metrics, strict=True):
            for dataset, dataset_metrics in history.items():
                values = dataset_metrics.get(metric)
                if values:
                    axis.plot(range(1, len(values) + 1), values, label=dataset)
            axis.set(title=metric, xlabel="Iteration", ylabel=metric)
            axis.grid(alpha=0.25)
            axis.legend()

        figure.suptitle(f"{model.name} training history", fontsize=14)
        figure.tight_layout()
        return self._save_figure(figure, f"{model.name}_learning_curves.png")

    def save_evaluation_dashboard(
        self,
        model_name: str,
        target: pd.Series,
        probability: np.ndarray,
        threshold: float,
    ) -> Path:
        metrics = BinaryEvaluator.evaluate(target, probability, threshold)
        prediction = (probability >= threshold).astype("int8")
        precision, recall, pr_thresholds = precision_recall_curve(target, probability)
        fpr, tpr, roc_thresholds = roc_curve(target, probability)
        matrix = confusion_matrix(target, prediction, labels=[0, 1])

        figure, axes = plt.subplots(2, 3, figsize=(18, 10))
        self._plot_pr_curve(axes[0, 0], recall, precision, metrics)
        self._plot_roc_curve(axes[0, 1], fpr, tpr, metrics)
        self._plot_ks_curve(axes[0, 2], fpr, tpr, roc_thresholds, metrics)
        self._plot_confusion_matrix(axes[1, 0], matrix)
        self._plot_score_distribution(axes[1, 1], target, probability, threshold)
        self._plot_threshold_metrics(
            axes[1, 2], precision, recall, pr_thresholds, threshold
        )

        figure.suptitle(f"{model_name} — test evaluation", fontsize=16)
        figure.tight_layout()
        return self._save_figure(figure, f"{model_name}_evaluation.png")

    def save_feature_importance(self, model_name: str, importance: pd.Series) -> Path:
        top = importance.nlargest(20).sort_values()
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.barh(
            top.index,
            top.to_numpy(dtype="float64"),
            color="#4472C4",
        )
        axis.set(
            title=f"{model_name} feature importance",
            xlabel="Importance",
            ylabel="Feature",
        )
        axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        return self._save_figure(figure, f"{model_name}_feature_importance.png")

    def save_shap_summary(
        self,
        model_name: str,
        features: pd.DataFrame,
        explanation: ShapExplanation,
        max_display: int,
    ) -> Path:
        importance = np.abs(explanation.values).mean(axis=0)
        display_count = min(max_display, features.shape[1])
        order = np.argsort(importance)[-display_count:]
        random = np.random.default_rng(42)

        figure, axis = plt.subplots(figsize=(12, max(5, display_count * 0.55)))
        color_map = plt.get_cmap("coolwarm")
        for row, feature_index in enumerate(order):
            feature_values = features.iloc[:, feature_index].to_numpy(dtype="float64")
            lower, upper = np.quantile(feature_values, (0.05, 0.95))
            if upper > lower:
                colors = np.clip((feature_values - lower) / (upper - lower), 0, 1)
            else:
                colors = np.full(len(feature_values), 0.5)
            jitter = np.clip(random.normal(0, 0.09, len(features)), -0.3, 0.3)
            axis.scatter(
                explanation.values[:, feature_index],
                row + jitter,
                c=colors,
                cmap=color_map,
                vmin=0,
                vmax=1,
                s=9,
                alpha=0.5,
                edgecolors="none",
                rasterized=True,
            )

        axis.axvline(0, color="grey", linewidth=1)
        axis.set(
            title=f"{model_name} — TreeSHAP summary",
            xlabel="SHAP contribution to raw fraud score",
            ylabel="Feature",
            yticks=np.arange(display_count),
            yticklabels=features.columns[order],
        )
        axis.grid(axis="x", alpha=0.2)
        color_bar = figure.colorbar(
            plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0, 1)),
            ax=axis,
            pad=0.02,
        )
        color_bar.set_label("Feature value")
        color_bar.set_ticks((0, 1), labels=("Low", "High"))
        figure.tight_layout()
        return self._save_figure(figure, f"{model_name}_shap_summary.png")

    @staticmethod
    def _plot_pr_curve(
        axis: plt.Axes,
        recall: np.ndarray,
        precision: np.ndarray,
        metrics: Metrics,
    ) -> None:
        indices = np.linspace(0, len(recall) - 1, min(10_000, len(recall)), dtype=int)
        axis.plot(recall[indices], precision[indices], color="#4472C4")
        axis.set(
            title=(
                f"Precision-Recall\nPR-AUC={metrics['pr_auc']:.4f}, "
                f"AP={metrics['average_precision']:.4f}"
            ),
            xlabel="Recall",
            ylabel="Precision",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        axis.grid(alpha=0.25)

    @staticmethod
    def _plot_roc_curve(
        axis: plt.Axes,
        fpr: np.ndarray,
        tpr: np.ndarray,
        metrics: Metrics,
    ) -> None:
        indices = np.linspace(0, len(fpr) - 1, min(10_000, len(fpr)), dtype=int)
        axis.plot(fpr[indices], tpr[indices], color="#70AD47", label="Model")
        axis.plot([0, 1], [0, 1], "--", color="grey", label="Random")
        axis.set(
            title=f"ROC curve\nAUC={metrics['roc_auc']:.4f}",
            xlabel="False positive rate",
            ylabel="True positive rate",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        axis.grid(alpha=0.25)
        axis.legend()

    @staticmethod
    def _plot_ks_curve(
        axis: plt.Axes,
        fpr: np.ndarray,
        tpr: np.ndarray,
        thresholds: np.ndarray,
        metrics: Metrics,
    ) -> None:
        finite = np.isfinite(thresholds)
        axis.plot(thresholds[finite], tpr[finite], label="TPR")
        axis.plot(thresholds[finite], fpr[finite], label="FPR")
        axis.axvline(metrics["ks_threshold"], color="#C00000", linestyle="--")
        axis.set(
            title=(
                f"KS statistic={metrics['ks_statistic']:.4f}\n"
                f"KS threshold={metrics['ks_threshold']:.4f}"
            ),
            xlabel="Decision threshold",
            ylabel="Rate",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        axis.grid(alpha=0.25)
        axis.legend()

    @staticmethod
    def _plot_confusion_matrix(axis: plt.Axes, matrix: np.ndarray) -> None:
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                text_color = (
                    "white" if matrix[row, column] > matrix.max() / 2 else "black"
                )
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    color=text_color,
                )
        axis.set(
            title="Confusion matrix",
            xlabel="Predicted class",
            ylabel="Actual class",
            xticks=(0, 1),
            yticks=(0, 1),
            xticklabels=("Legitimate", "Fraud"),
            yticklabels=("Legitimate", "Fraud"),
        )
        axis.figure.colorbar(image, ax=axis, fraction=0.046)

    @staticmethod
    def _plot_score_distribution(
        axis: plt.Axes,
        target: pd.Series,
        probability: np.ndarray,
        threshold: float,
    ) -> None:
        axis.hist(
            probability[target.to_numpy() == 0],
            bins=60,
            range=(0, 1),
            alpha=0.65,
            label="Legitimate",
            density=True,
        )
        axis.hist(
            probability[target.to_numpy() == 1],
            bins=60,
            range=(0, 1),
            alpha=0.65,
            label="Fraud",
            density=True,
        )
        axis.axvline(threshold, color="#C00000", linestyle="--")
        axis.set(
            title="Score distribution",
            xlabel="Fraud probability",
            ylabel="Density",
            yscale="log",
            xlim=(0, 1),
        )
        axis.legend()

    @staticmethod
    def _plot_threshold_metrics(
        axis: plt.Axes,
        precision: np.ndarray,
        recall: np.ndarray,
        thresholds: np.ndarray,
        selected_threshold: float,
    ) -> None:
        f1 = np.divide(
            2 * precision[:-1] * recall[:-1],
            precision[:-1] + recall[:-1],
            out=np.zeros_like(thresholds),
            where=(precision[:-1] + recall[:-1]) > 0,
        )
        indices = np.linspace(
            0, len(thresholds) - 1, min(2_000, len(thresholds)), dtype=int
        )
        axis.plot(thresholds[indices], precision[:-1][indices], label="Precision")
        axis.plot(thresholds[indices], recall[:-1][indices], label="Recall")
        axis.plot(thresholds[indices], f1[indices], label="F1")
        axis.axvline(selected_threshold, color="#C00000", linestyle="--")
        axis.set(
            title="Metrics by threshold",
            xlabel="Decision threshold",
            ylabel="Metric value",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        axis.grid(alpha=0.25)
        axis.legend()

    def _save_figure(self, figure: plt.Figure, filename: str) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / filename
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return output_path


class BaselineExperiment:
    def __init__(
        self,
        config: AppConfig,
        models: list[ModelRunner],
        artifact_store: ArtifactStore,
        plotter: ReportPlotter,
        drift_reference: DriftReferenceBuilder,
    ) -> None:
        self._config = config
        self._models = models
        self._artifact_store = artifact_store
        self._plotter = plotter
        self._drift_reference = drift_reference
        self._selector = ThresholdSelector(config.training.minimum_recall)

    def run(self) -> ExperimentReport:
        print("Loading train, validation and test datasets...")
        data = DatasetLoader(
            self._config.paths, self._config.training.target_column
        ).load_all()
        class_ratio = self._class_ratio(data.train.target)
        report = self._create_report(data)
        self._drift_reference.fit_features(
            data.valid.features, self._config.data.categorical_column
        )

        for model in self._models:
            class_weight = class_ratio**model.class_weight_power
            print(f"\nTraining {model.name} (scale_pos_weight={class_weight:.2f})...")
            model.fit(data.train, data.valid, class_weight)
            result, valid_probability, test_probability = self._evaluate_model(
                model, data, class_weight
            )
            report["models"][model.name] = result
            self._drift_reference.add_scores(
                model.name,
                valid_probability,
                result["validation"]["threshold"],
            )

            importance = self._artifact_store.save_model(
                model, data.train.features.columns.tolist()
            )
            shap_features = self._shap_sample(data.valid.features)
            explanation = model.explain(shap_features)
            shap_csv = self._artifact_store.save_shap_importance(
                model.name,
                shap_features.columns.tolist(),
                explanation,
            )
            shap_plot = self._plotter.save_shap_summary(
                model.name,
                shap_features,
                explanation,
                self._config.explainability.max_display,
            )
            result["explainability"] = {
                "method": "TreeSHAP",
                "output_space": "raw_score",
                "dataset": "validation",
                "sample_rows": len(shap_features),
                "importance_file": str(
                    shap_csv.relative_to(self._config.paths.artifact_dir)
                ),
                "summary_plot": str(
                    shap_plot.relative_to(self._config.paths.artifact_dir)
                ),
            }
            self._plotter.save_training_curves(model)
            self._plotter.save_evaluation_dashboard(
                model.name,
                data.test.target,
                test_probability,
                result["test"]["threshold"],
            )
            self._plotter.save_feature_importance(model.name, importance)
            self._print_result(model.name, result)

        metrics_path = self._artifact_store.save_report(report)
        drift_path = self._drift_reference.save()
        print(f"\nMetrics saved to {metrics_path}")
        print(f"Drift reference saved to {drift_path}")
        return report

    def _evaluate_model(
        self,
        model: ModelRunner,
        data: DatasetBundle,
        class_weight: float,
    ) -> tuple[ExperimentReport, np.ndarray, np.ndarray]:
        valid_probability = model.predict_proba(data.valid.features)
        threshold = self._selector.select(data.valid.target, valid_probability)
        test_probability = model.predict_proba(data.test.features)
        result = {
            "best_iteration": model.best_iteration,
            "scale_pos_weight": class_weight,
            "validation": BinaryEvaluator.evaluate(
                data.valid.target, valid_probability, threshold
            ),
            "test": BinaryEvaluator.evaluate(
                data.test.target, test_probability, threshold
            ),
        }
        return result, valid_probability, test_probability

    def _create_report(self, data: DatasetBundle) -> ExperimentReport:
        return {
            "config": {
                "training": asdict(self._config.training),
                "explainability": asdict(self._config.explainability),
                "prediction": asdict(self._config.prediction),
                "catboost": asdict(self._config.catboost),
                "lightgbm": asdict(self._config.lightgbm),
            },
            "datasets": self._dataset_summary(data),
            "models": {},
        }

    @staticmethod
    def _class_ratio(target: pd.Series) -> float:
        positive = int(target.sum())
        negative = len(target) - positive
        if positive == 0 or negative == 0:
            raise ValueError("Training target must contain both classes")
        return negative / positive

    @staticmethod
    def _dataset_summary(data: DatasetBundle) -> ExperimentReport:
        summary = {}
        for name, split in (
            ("train", data.train),
            ("valid", data.valid),
            ("test", data.test),
        ):
            positives = int(split.target.sum())
            summary[name] = {
                "rows": len(split.target),
                "fraud": positives,
                "fraud_rate": positives / len(split.target),
            }
        return summary

    def _shap_sample(self, features: pd.DataFrame) -> pd.DataFrame:
        sample_size = min(self._config.explainability.sample_size, len(features))
        if sample_size == len(features):
            return features.reset_index(drop=True)
        return features.sample(
            n=sample_size,
            random_state=self._config.training.random_seed,
        ).reset_index(drop=True)

    @staticmethod
    def _print_result(name: str, result: ExperimentReport) -> None:
        test = result["test"]
        print(
            f"{name}: PR-AUC={test['pr_auc']:.4f}, "
            f"KS={test['ks_statistic']:.4f}, "
            f"precision={test['precision']:.4f}, "
            f"recall={test['recall']:.4f}, "
            f"threshold={test['threshold']:.4f}"
        )
