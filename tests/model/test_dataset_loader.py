from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import PathConfig
from src.model.training import DatasetLoader


def write_split(path: Path, columns: tuple[str, ...] = ("amount", "balance")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame({name: [1.0, 2.0] for name in columns})
    data["isFraud"] = [0, 1]
    data.to_csv(path, index=False)


def test_dataset_loader_requires_matching_schemas(model_paths: PathConfig) -> None:
    write_split(model_paths.train_data)
    write_split(model_paths.valid_data)
    write_split(model_paths.test_data, columns=("balance", "amount"))

    with pytest.raises(ValueError, match="schema of test"):
        DatasetLoader(model_paths, "isFraud").load_all()


def test_dataset_loader_rejects_non_finite_values(
    model_paths: PathConfig,
) -> None:
    path = model_paths.processed_dir / "invalid.csv"
    write_split(path)
    data = pd.read_csv(path)
    data.loc[0, "amount"] = np.inf
    data.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Non-finite values"):
        DatasetLoader(model_paths, "isFraud").load(path)


def test_dataset_loader_returns_binary_target(model_paths: PathConfig) -> None:
    path = model_paths.processed_dir / "valid.csv"
    write_split(path)

    split = DatasetLoader(model_paths, "isFraud").load(path)

    assert split.features.columns.tolist() == ["amount", "balance"]
    assert split.target.tolist() == [0, 1]
    assert split.target.dtype == "int8"
