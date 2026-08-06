import argparse
import asyncio
from pathlib import Path
from typing import cast

from src.config import CONFIG
from src.model.training import FraudPredictionService, TritonPredictionGateway
from src.model.types import ModelName


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fraudulent transactions")
    parser.add_argument(
        "--model",
        choices=("catboost", "lightgbm"),
        default=CONFIG.prediction.default_model,
    )
    parser.add_argument("--input", type=Path, default=CONFIG.paths.test_data)
    parser.add_argument("--output", type=Path, default=CONFIG.paths.prediction_file)
    parser.add_argument("--threshold", type=float)
    return parser.parse_args()


async def run() -> None:
    args = parse_args()
    service = FraudPredictionService(
        gateway=TritonPredictionGateway(CONFIG.triton),
        default_model=cast(ModelName, args.model),
        paths=CONFIG.paths,
        prediction_config=CONFIG.prediction,
        target_column=CONFIG.training.target_column,
        categorical_column=CONFIG.data.categorical_column,
    )
    try:
        summary = await service.predict_file(
            input_path=args.input,
            output_path=args.output,
            model_name=cast(ModelName, args.model),
            threshold=args.threshold,
        )
    finally:
        await service.close()
    print(
        f"{summary.model}: {summary.fraud_alerts:,} alerts from "
        f"{summary.rows:,} rows at threshold={summary.threshold:.6f}"
    )
    print(f"Predictions saved to {summary.output_path}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
