from src.config import CONFIG
from src.data.pipeline import (
    CsvDataReader,
    CsvDatasetWriter,
    DataProcessingPipeline,
    FraudDataCleaner,
    FraudFeatureTransformer,
    QuantileTimeSplitter,
)


def main() -> None:
    pipeline = DataProcessingPipeline(
        reader=CsvDataReader(CONFIG.paths.raw_data),
        cleaner=FraudDataCleaner(CONFIG.data),
        splitter=QuantileTimeSplitter(CONFIG.data),
        feature_transformer=FraudFeatureTransformer(CONFIG.data),
        writer=CsvDatasetWriter(CONFIG.paths.processed_dir),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
