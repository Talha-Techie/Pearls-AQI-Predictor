"""Model registry helpers."""

from __future__ import annotations

import os
from pathlib import Path
import argparse
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature

from app.feature_pipeline.engineer import MODEL_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_benchmarks"
    / "final_test_metrics.csv"
)



MODELS = {
    "24h": MODEL_DIR / "aqi_ridge_24h.joblib",
    "48h": MODEL_DIR / "aqi_ridge_48h.joblib",
    "72h": MODEL_DIR / "aqi_ridge_72h.joblib",
}


def configure_mlflow() -> None:
    load_dotenv()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not configured."
        )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        os.getenv(
            "MLFLOW_EXPERIMENT_NAME",
            "aqi-forecasting",
        )
    )


def register_models(
    dataset_path: Path,
) -> None:
    configure_mlflow()

    metrics = pd.read_csv(METRICS_PATH)

    training_df = pd.read_parquet(
    dataset_path,
    columns=MODEL_FEATURE_COLUMNS,
)

    input_example = training_df.iloc[[0]].copy()

    for horizon, model_path in MODELS.items():

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        row = metrics.loc[
            metrics["horizon"].astype(str) == horizon
        ]

        if row.empty:
            raise RuntimeError(
                f"Metrics missing for {horizon}"
            )

        row = row.iloc[0]

        model = joblib.load(model_path)

        sample_prediction = model.predict(
            input_example
        )

        signature = infer_signature(
            input_example,
            sample_prediction,
        )

        registered_name = (
            f"aqi-ridge-{horizon}"
        )

        with mlflow.start_run(
            run_name=f"ridge-{horizon}-final"
        ):
            mlflow.log_params(
                {
                    "model_type": "Ridge Regression",
                    "ridge_alpha": 1.0,
                    "forecast_horizon": horizon,
                    "feature_count": len(
                        MODEL_FEATURE_COLUMNS
                    ),
                }
            )

            mlflow.log_metrics(
                {
                    "test_mae": float(row["mae"]),
                    "test_rmse": float(row["rmse"]),
                    "test_r2": float(row["r2"]),
                    "baseline_mae": float(
                        row["baseline_mae"]
                    ),
                    "baseline_rmse": float(
                        row["baseline_rmse"]
                    ),
                    "baseline_r2": float(
                        row["baseline_r2"]
                    ),
                }
            )

            mlflow.log_dict(
                {
                    "forecast_horizon": horizon,
                    "features": MODEL_FEATURE_COLUMNS,
                },
                "feature_metadata.json",
            )

            result = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_name,
            )

            print(
                f"\nRegistered: {registered_name}"
            )
            print(
                f"Model URI: {result.model_uri}"
            )

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Register AQI champion models "
            "in MLflow Model Registry."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    register_models(
        dataset_path=args.input
    )


if __name__ == "__main__":
    main()