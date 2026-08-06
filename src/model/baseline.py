import argparse
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast

from src.config import CONFIG, CatBoostConfig, LightGBMConfig
from src.metrics.drift import DriftReferenceBuilder
from src.model.experiment import ArtifactStore, BaselineExperiment, ReportPlotter
from src.model.training import CatBoostRunner, LightGBMRunner
from src.model.types import ModelName


def load_tuned_parameters(path: Path) -> dict[ModelName, dict[str, int | float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported tuning report schema")
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("Tuning report models must be an object")

    allowed = {
        "catboost": {field.name for field in fields(CatBoostConfig)},
        "lightgbm": {field.name for field in fields(LightGBMConfig)},
    }
    result: dict[ModelName, dict[str, int | float]] = {}
    for model_name in ("catboost", "lightgbm"):
        model = models.get(model_name)
        if model is None:
            continue
        if not isinstance(model, dict):
            raise ValueError(f"Tuning result for {model_name} must be an object")
        parameters = model.get("best_params")
        if not isinstance(parameters, dict):
            raise ValueError(f"Best parameters for {model_name} are missing")
        unknown = set(parameters) - allowed[model_name]
        if unknown:
            raise ValueError(
                f"Unknown tuned parameters for {model_name}: {sorted(unknown)}"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in parameters.values()
        ):
            raise ValueError(f"Tuned parameters for {model_name} must be numeric")
        result[model_name] = cast(dict[str, int | float], parameters)
    if not result:
        raise ValueError("Tuning report contains no supported models")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection baselines")
    parser.add_argument(
        "--tuned-params",
        type=Path,
        help="Optuna best_params.json produced by src.model.tuning",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tuned = load_tuned_parameters(args.tuned_params) if args.tuned_params else {}
    catboost_config = replace(
        CONFIG.catboost,
        **cast(Any, tuned.get("catboost", {})),
    )
    lightgbm_config = replace(
        CONFIG.lightgbm,
        **cast(Any, tuned.get("lightgbm", {})),
    )
    config = replace(
        CONFIG,
        catboost=catboost_config,
        lightgbm=lightgbm_config,
    )
    experiment = BaselineExperiment(
        config=config,
        models=[
            CatBoostRunner(catboost_config, config.training),
            LightGBMRunner(lightgbm_config, config.training),
        ],
        artifact_store=ArtifactStore(config.paths.artifact_dir),
        plotter=ReportPlotter(config.paths.plots_dir),
        drift_reference=DriftReferenceBuilder(config.paths.drift_reference_file),
    )
    experiment.run()


if __name__ == "__main__":
    main()
