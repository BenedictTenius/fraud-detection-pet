import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metrics.drift import DriftReferenceBuilder, PrometheusDriftMonitor


def reference_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": np.arange(1, 101, dtype="float64"),
            "oldbalanceOrg": np.arange(101, 201, dtype="float64"),
            "type_PAYMENT": [1, 0] * 50,
            "type_TRANSFER": [0, 1] * 50,
        }
    )


def test_reference_is_versioned_and_can_observe_raw_categories(
    tmp_path: Path,
) -> None:
    output = tmp_path / "drift_reference.json"
    builder = DriftReferenceBuilder(output, bin_count=5)
    builder.fit_features(reference_features(), "type")
    builder.add_scores(
        "lightgbm", np.linspace(0, 1, 100, endpoint=False), threshold=0.4
    )
    builder.add_scores(
        "catboost", np.linspace(0, 1, 100, endpoint=False), threshold=0.6
    )
    builder.save()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["version"]) == 12
    assert all(
        sum(distribution["ratios"]) == pytest.approx(1.0)
        for distribution in payload["distributions"]
    )

    monitor = PrometheusDriftMonitor(output)
    monitor.observe(
        data=pd.DataFrame(
            {
                "amount": [10.0, 90.0],
                "oldbalanceOrg": [110.0, 190.0],
                "type": ["PAYMENT", "TRANSFER"],
            }
        ),
        model="lightgbm",
        scores=np.array([0.1, 0.9]),
        threshold=0.4,
        duration_seconds=0.01,
    )
    assert monitor.version == payload["version"]


def test_reference_rejects_non_finite_values(tmp_path: Path) -> None:
    builder = DriftReferenceBuilder(tmp_path / "reference.json")
    features = reference_features()
    features.loc[0, "amount"] = np.nan

    with pytest.raises(ValueError, match="must be finite"):
        builder.fit_features(features, "type")
