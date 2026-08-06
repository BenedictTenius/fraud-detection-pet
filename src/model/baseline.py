from src.config import CONFIG
from src.metrics.drift import DriftReferenceBuilder
from src.model.experiment import ArtifactStore, BaselineExperiment, ReportPlotter
from src.model.training import CatBoostRunner, LightGBMRunner


def main() -> None:
    experiment = BaselineExperiment(
        config=CONFIG,
        models=[
            CatBoostRunner(CONFIG.catboost, CONFIG.training),
            LightGBMRunner(CONFIG.lightgbm, CONFIG.training),
        ],
        artifact_store=ArtifactStore(CONFIG.paths.artifact_dir),
        plotter=ReportPlotter(CONFIG.paths.plots_dir),
        drift_reference=DriftReferenceBuilder(CONFIG.paths.drift_reference_file),
    )
    experiment.run()


if __name__ == "__main__":
    main()
