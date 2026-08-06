from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import MAX_TRANSACTION_VALUE
from src.model.types import ModelName

TransactionType = Literal[
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


class PredictionData(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: TransactionType
    amount: float = Field(ge=0, le=MAX_TRANSACTION_VALUE)
    oldbalanceOrg: float = Field(ge=0, le=MAX_TRANSACTION_VALUE)
    oldbalanceDest: float = Field(ge=0, le=MAX_TRANSACTION_VALUE)


class PredictionAnswer(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model: ModelName
    fraud_score: float
    threshold: float
    is_fraud: bool


class BatchPredictionData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[PredictionData] = Field(
        min_length=1,
        max_length=10_000,
    )


class BatchPredictionAnswer(BaseModel):
    predictions: list[PredictionAnswer]
