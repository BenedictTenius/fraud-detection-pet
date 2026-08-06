from pathlib import Path

import pandas as pd

from src.config import DataConfig
from src.data.pipeline import (
    CsvDatasetWriter,
    FraudDataCleaner,
    FraudFeatureTransformer,
    QuantileTimeSplitter,
)


def raw_transactions() -> pd.DataFrame:
    transaction_types = [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    ]
    return pd.DataFrame(
        {
            "step": list(range(20, 0, -1)),
            "type": transaction_types * 4,
            "amount": [float(value * 10) for value in range(20)],
            "nameOrig": [f"C{value}" for value in range(20)],
            "oldbalanceOrg": [1_000.0] * 20,
            "newbalanceOrig": [900.0] * 20,
            "nameDest": [f"M{value}" for value in range(20)],
            "oldbalanceDest": [500.0] * 20,
            "newbalanceDest": [600.0] * 20,
            "isFraud": [0] * 18 + [1, 1],
            "isFlaggedFraud": [0] * 20,
        }
    )


def test_clean_split_and_transform_preserve_time_order() -> None:
    config = DataConfig()
    cleaned = FraudDataCleaner(config).transform(raw_transactions())

    assert cleaned["step"].is_monotonic_increasing
    assert set(config.drop_columns).isdisjoint(cleaned.columns)

    splits = QuantileTimeSplitter(config).split(cleaned)
    assert max(splits["train"]["step"]) < min(splits["valid"]["step"])
    assert max(splits["valid"]["step"]) < min(splits["test"]["step"])
    assert sum(map(len, splits.values())) == len(cleaned)

    transformer = FraudFeatureTransformer(config)
    transformer.fit(splits["train"])
    prepared = transformer.transform(splits["valid"])

    assert "step" not in prepared
    assert "type" not in prepared
    assert prepared["isFraud"].dtype == "int8"
    assert {
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    }.issubset(prepared.columns)


def test_writer_creates_a_reproducible_csv(tmp_path: Path) -> None:
    data = pd.DataFrame({"amount": [10.0], "isFraud": [0]})
    output = CsvDatasetWriter(tmp_path / "processed").write("train", data)

    assert output == tmp_path / "processed/fraud_train.csv"
    pd.testing.assert_frame_equal(pd.read_csv(output), data)
