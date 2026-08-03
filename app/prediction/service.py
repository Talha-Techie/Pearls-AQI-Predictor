"""Prediction service for AQI forecasts."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from dotenv import load_dotenv
from feast import FeatureStore

from app.feature_pipeline.engineer import MODEL_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEAST_REPO = PROJECT_ROOT / "feature_repo"

FEATURE_SERVICE_NAME = "aqi_prediction_features_v1"


class PredictionServiceError(RuntimeError):
    """Raised when AQI prediction cannot be generated."""


def configure_mlflow() -> None:
    load_dotenv()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI"
    )

    if not tracking_uri:
        raise PredictionServiceError(
            "MLFLOW_TRACKING_URI is not configured."
        )

    mlflow.set_tracking_uri(
        tracking_uri
    )


@lru_cache(maxsize=1)
def get_feature_store() -> FeatureStore:
    return FeatureStore(
        repo_path=str(FEAST_REPO)
    )


@lru_cache(maxsize=3)
def load_registered_model(
    horizon: str,
):
    configure_mlflow()

    model_uri = (
        f"models:/aqi-ridge-{horizon}/1"
    )

    return mlflow.sklearn.load_model(
        model_uri
    )


def get_online_feature_vector(
    city: str,
) -> pd.DataFrame:
    store = get_feature_store()

    feature_service = (
        store.get_feature_service(
            FEATURE_SERVICE_NAME
        )
    )

    response = store.get_online_features(
        features=feature_service,
        entity_rows=[
            {
                "city": city,
            }
        ],
    ).to_dict()

    feature_data = {}

    for feature in MODEL_FEATURE_COLUMNS:
        values = response.get(feature)

        if not values:
            raise PredictionServiceError(
                f"Feature missing from Feast: {feature}"
            )

        value = values[0]

        if value is None:
            raise PredictionServiceError(
                f"Feature contains null value: {feature}"
            )

        feature_data[feature] = [value]

    dataframe = pd.DataFrame(
        feature_data,
        columns=MODEL_FEATURE_COLUMNS,
    )

    if dataframe.isna().any().any():
        raise PredictionServiceError(
            "Online feature vector contains missing values."
        )

    return dataframe


def predict_aqi(
    city: str = "Lahore",
) -> dict:
    features = get_online_feature_vector(
        city
    )

    current_aqi = float(
        features.iloc[0]["us_aqi"]
    )

    forecasts = {}

    for horizon in (
        "24h",
        "48h",
        "72h",
    ):
        model = load_registered_model(
            horizon
        )

        prediction = float(
            model.predict(features)[0]
        )

        forecasts[horizon] = round(
            max(0.0, prediction),
            1,
        )

    return {
        "city": city,
        "current_aqi": round(
            current_aqi,
            1,
        ),
        "forecast": forecasts,
        "feature_count": len(
            MODEL_FEATURE_COLUMNS
        ),
        "feature_source": (
            "Feast online store"
        ),
        "model_source": (
            "DagsHub MLflow Model Registry"
        ),
    }


def main() -> None:
    result = predict_aqi()

    print(
        "\nAQI Prediction Service"
    )
    print(
        "----------------------"
    )

    print(
        f"City: {result['city']}"
    )

    print(
        f"Current AQI: "
        f"{result['current_aqi']}"
    )

    print(
        "\nForecast:"
    )

    for horizon, prediction in (
        result["forecast"].items()
    ):
        print(
            f"  +{horizon}: "
            f"{prediction}"
        )

    print(
        "\nFeature source:",
        result["feature_source"],
    )

    print(
        "Model source:",
        result["model_source"],
    )


if __name__ == "__main__":
    main()