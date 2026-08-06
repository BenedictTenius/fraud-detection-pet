from pathlib import Path

import pandas as pd

from src.config import DataConfig
from src.data.types import (
    DataReader,
    DatasetWriter,
    DataSplitter,
    DataTransformer,
    FeatureTransformer,
    SplitName,
)


class CsvDataReader:
    def __init__(self, path: Path) -> None:
        self._path = path

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self._path)


class FraudDataCleaner:
    def __init__(self, config: DataConfig) -> None:
        self._config = config

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return (
            data.sort_values(self._config.time_column, kind="stable")
            .drop(columns=list(self._config.drop_columns))
            .reset_index(drop=True)
        )


class QuantileTimeSplitter:
    def __init__(self, config: DataConfig) -> None:
        self._config = config

    def split(self, data: pd.DataFrame) -> dict[SplitName, pd.DataFrame]:
        time = data[self._config.time_column]
        train_end = time.quantile(
            self._config.train_share, interpolation="lower"
        )
        valid_end = time.quantile(
            self._config.train_share + self._config.valid_share,
            interpolation="lower",
        )
        return {
            "train": data[time <= train_end],
            "valid": data[(time > train_end) & (time <= valid_end)],
            "test": data[time > valid_end],
        }


class FraudFeatureTransformer:
    def __init__(self, config: DataConfig) -> None:
        self._config = config
        self._categories: list[str] | None = None

    def fit(self, data: pd.DataFrame) -> None:
        self._categories = sorted(
            data[self._config.categorical_column].unique().tolist()
        )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._categories is None:
            raise RuntimeError("Feature transformer must be fitted first")

        features = data.drop(
            columns=[self._config.time_column, self._config.target_column]
        ).copy()
        category = self._config.categorical_column
        features[category] = pd.Categorical(
            features[category], categories=self._categories
        )
        features = pd.get_dummies(features, columns=[category], dtype="int8")
        features[self._config.target_column] = data[
            self._config.target_column
        ].to_numpy(dtype="int8")
        return features


class CsvDatasetWriter:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def write(self, name: SplitName, data: pd.DataFrame) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / f"fraud_{name}.csv"
        data.to_csv(output_path, index=False)
        return output_path


class DataProcessingPipeline:
    def __init__(
        self,
        reader: DataReader,
        cleaner: DataTransformer,
        splitter: DataSplitter,
        feature_transformer: FeatureTransformer,
        writer: DatasetWriter,
    ) -> None:
        self._reader = reader
        self._cleaner = cleaner
        self._splitter = splitter
        self._feature_transformer = feature_transformer
        self._writer = writer

    def run(self) -> None:
        data = self._cleaner.transform(self._reader.read())
        splits = self._splitter.split(data)
        self._feature_transformer.fit(splits["train"])

        for name, split in splits.items():
            prepared = self._feature_transformer.transform(split)
            output_path = self._writer.write(name, prepared)
            print(f"Saved {len(prepared):,} rows to {output_path}")
