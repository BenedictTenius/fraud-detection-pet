import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
MAX_TRANSACTION_VALUE = 1_000_000_000_000
load_dotenv(PROJECT_DIR / ".env")


def env_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else PROJECT_DIR / path


def env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).lower()
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {expected}")
    return value


def env_positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def env_positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class PathConfig:
    raw_data: Path = field(
        default_factory=lambda: env_path(
            "FRAUD_RAW_DATA",
            "data/raw/PS_20174392719_1491204439457_log.csv",
        )
    )
    processed_dir: Path = field(
        default_factory=lambda: env_path("FRAUD_PROCESSED_DIR", "data/processed")
    )
    artifact_dir: Path = field(
        default_factory=lambda: env_path("FRAUD_ARTIFACT_DIR", "artifacts/baseline")
    )

    @property
    def train_data(self) -> Path:
        return self.processed_dir / "fraud_train.csv"

    @property
    def valid_data(self) -> Path:
        return self.processed_dir / "fraud_valid.csv"

    @property
    def test_data(self) -> Path:
        return self.processed_dir / "fraud_test.csv"

    @property
    def metrics_file(self) -> Path:
        return self.artifact_dir / "metrics.json"

    @property
    def drift_reference_file(self) -> Path:
        return self.artifact_dir / "drift_reference.json"

    @property
    def plots_dir(self) -> Path:
        return self.artifact_dir / "plots"

    @property
    def prediction_file(self) -> Path:
        return self.artifact_dir / "predictions.csv"


@dataclass(frozen=True)
class DataConfig:
    time_column: str = "step"
    target_column: str = "isFraud"
    categorical_column: str = "type"
    train_share: float = 0.70
    valid_share: float = 0.15
    drop_columns: tuple[str, ...] = (
        "nameOrig",
        "nameDest",
        "newbalanceOrig",
        "newbalanceDest",
        "isFlaggedFraud",
    )

    def __post_init__(self) -> None:
        if not 0 < self.train_share < 1:
            raise ValueError("train_share must be between 0 and 1")
        if not 0 < self.valid_share < 1 - self.train_share:
            raise ValueError("valid_share must leave data for the test split")


@dataclass(frozen=True)
class TrainingConfig:
    target_column: str = "isFraud"
    random_seed: int = 42
    minimum_recall: float = 0.80
    log_period: int = 50
    device: str = field(
        default_factory=lambda: env_choice("FRAUD_TRAIN_DEVICE", "cpu", {"cpu", "gpu"})
    )
    gpu_devices: str = field(
        default_factory=lambda: os.getenv("FRAUD_GPU_DEVICES", "0")
    )


@dataclass(frozen=True)
class ExplainabilityConfig:
    sample_size: int = 10_000
    max_display: int = 15


@dataclass(frozen=True)
class LoggingConfig:
    level: str = field(
        default_factory=lambda: env_choice(
            "FRAUD_LOG_LEVEL",
            "info",
            {"debug", "info", "warning", "error", "critical"},
        ).upper()
    )
    service: str = "fraud-api"
    environment: str = field(
        default_factory=lambda: os.getenv("FRAUD_ENVIRONMENT", "development")
    )


@dataclass(frozen=True)
class PredictionConfig:
    default_model: str = field(
        default_factory=lambda: env_choice(
            "FRAUD_MODEL_NAME",
            "lightgbm",
            {"catboost", "lightgbm"},
        )
    )
    score_column: str = "fraud_score"
    prediction_column: str = "is_fraud"
    max_transaction_value: float = MAX_TRANSACTION_VALUE
    feature_names: tuple[str, ...] = (
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    )


@dataclass(frozen=True)
class TritonConfig:
    grpc_url: str = field(
        default_factory=lambda: os.getenv("FRAUD_TRITON_GRPC_URL", "localhost:8101")
    )
    request_timeout: float = field(
        default_factory=lambda: env_positive_float("FRAUD_TRITON_REQUEST_TIMEOUT", 5.0)
    )
    max_batch_size: int = field(
        default_factory=lambda: env_positive_int("FRAUD_TRITON_MAX_BATCH_SIZE", 4096)
    )
    input_name: str = "input__0"
    output_name: str = "output__0"
    catboost_model: str = "fraud_catboost"
    lightgbm_model: str = "fraud_lightgbm"

    def model_name(self, model: str) -> str:
        if model == "catboost":
            return self.catboost_model
        if model == "lightgbm":
            return self.lightgbm_model
        raise ValueError(f"Unsupported model: {model}")


@dataclass(frozen=True)
class CatBoostConfig:
    iterations: int = 500
    learning_rate: float = 0.05
    depth: int = 7
    l2_leaf_reg: float = 5.0
    early_stopping_rounds: int = 50
    class_weight_power: float = 1.0


@dataclass(frozen=True)
class LightGBMConfig:
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 100
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    class_weight_power: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    triton: TritonConfig = field(default_factory=TritonConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)


CONFIG = AppConfig()
