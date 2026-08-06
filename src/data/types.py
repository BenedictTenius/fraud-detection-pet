from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import pandas as pd

SplitName: TypeAlias = Literal["train", "valid", "test"]
DataSplits: TypeAlias = dict[SplitName, pd.DataFrame]


class DataReader(Protocol):
    def read(self) -> pd.DataFrame: ...


class DataTransformer(Protocol):
    def transform(self, data: pd.DataFrame) -> pd.DataFrame: ...


class DataSplitter(Protocol):
    def split(self, data: pd.DataFrame) -> DataSplits: ...


class FeatureTransformer(Protocol):
    def fit(self, data: pd.DataFrame) -> None: ...

    def transform(self, data: pd.DataFrame) -> pd.DataFrame: ...


class DatasetWriter(Protocol):
    def write(self, name: SplitName, data: pd.DataFrame) -> Path: ...
