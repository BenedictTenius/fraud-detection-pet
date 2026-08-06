import json
from pathlib import Path

import pytest

from src.config import PathConfig


@pytest.fixture
def model_paths(tmp_path: Path) -> PathConfig:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "metrics.json").write_text(
        json.dumps(
            {
                "models": {
                    "catboost": {"validation": {"threshold": 0.7}},
                    "lightgbm": {"validation": {"threshold": 0.2}},
                }
            }
        ),
        encoding="utf-8",
    )
    return PathConfig(
        raw_data=tmp_path / "raw.csv",
        processed_dir=tmp_path / "processed",
        artifact_dir=artifact_dir,
    )
