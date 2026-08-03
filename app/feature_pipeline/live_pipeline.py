from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from feast import FeatureStore
from feast.data_source import PushMode

from app.config.settings import get_settings
from app.feature_pipeline.engineer import (
    MODEL_FEATURE_COLUMNS,
    engineer_inference_features,
)
from app.feature_pipeline.validator import validate_hourly_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEAST_REPO = PROJECT_ROOT / "feature_repo"

PAST_HOURS = 48


class LiveFeaturePipelineError(RuntimeError):
    """Raised when live feature generation fails."""


def _fetch_json(
    url: str,
    params: dict,
    timeout: float,
) -> dict:
    with httpx.Client(
        timeout=timeout
    ) as client:
        response = client.get(
            url,
            params=params,
        )
        response.raise_for_status()
        return response.json()


def _hourly_to_dataframe(
    payload: dict,
    columns: list[str],
) -> pd.DataFrame:
    hourly = payload.get("hourly")

    if not hourly:
        raise LiveFeaturePipelineError(
            "API response does not contain hourly data."
        )

    if "time" not in hourly:
        raise LiveFeaturePipelineError(
            "Hourly API response does not contain time."
        )

    data = {
        "timestamp": pd.to_datetime(
            hourly["time"],
            utc=True,
        )
    }

    for column in columns:
        values = hourly.get(column)

        if values is None:
            raise LiveFeaturePipelineError(
                f"Missing hourly variable: {column}"
            )

        data[column] = values

    return pd.DataFrame(data)


def fetch_recent_weather() -> pd.DataFrame:
    settings = get_settings()

    variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "pressure_msl",
        "wind_speed_10m",
        "wind_direction_10m",
    ]

    payload = _fetch_json(
        settings.weather_api_url,
        {
            "latitude": settings.latitude,
            "longitude": settings.longitude,
            "hourly": ",".join(variables),
            "past_hours": PAST_HOURS,
            "forecast_hours": 1,
            "timezone": "UTC",
        },
        settings.http_timeout_seconds,
    )

    return _hourly_to_dataframe(
        payload,
        variables,
    )


def fetch_recent_air_quality() -> pd.DataFrame:
    settings = get_settings()

    variables = [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "us_aqi",
    ]

    payload = _fetch_json(
        settings.air_quality_api_url,
        {
            "latitude": settings.latitude,
            "longitude": settings.longitude,
            "hourly": ",".join(variables),
            "past_hours": PAST_HOURS,
            "forecast_hours": 1,
            "timezone": "UTC",
        },
        settings.http_timeout_seconds,
    )

    return _hourly_to_dataframe(
        payload,
        variables,
    )


def collect_recent_hourly_data() -> pd.DataFrame:
    settings = get_settings()

    weather = fetch_recent_weather()
    air_quality = fetch_recent_air_quality()

    merged = weather.merge(
        air_quality,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise LiveFeaturePipelineError(
            "Weather and AQI datasets have no matching timestamps."
        )

    merged.insert(
        1,
        "city",
        settings.city,
    )

    merged.insert(
        2,
        "latitude",
        settings.latitude,
    )

    merged.insert(
        3,
        "longitude",
        settings.longitude,
    )

    return merged.sort_values(
        "timestamp"
    ).reset_index(drop=True)


def build_latest_feature_vector() -> pd.DataFrame:
    raw = collect_recent_hourly_data()

    validate_hourly_dataset(raw)

    engineered = engineer_inference_features(
        raw
    )

    latest = engineered.iloc[[-1]].copy()

    columns = [
        "timestamp",
        "city",
        *MODEL_FEATURE_COLUMNS,
    ]

    latest = latest[columns]

    if latest[MODEL_FEATURE_COLUMNS].isna().any().any():
        raise LiveFeaturePipelineError(
            "Latest feature vector contains missing values."
        )

    return latest


def push_latest_features(
    features: pd.DataFrame,
) -> None:
    store = FeatureStore(
        repo_path=str(FEAST_REPO)
    )

    store.push(
        "aqi_features_push_source",
        features,
        to=PushMode.ONLINE,
    )


def run_live_feature_pipeline() -> pd.DataFrame:
    latest = build_latest_feature_vector()

    print("\nLive AQI Feature Pipeline")
    print("-------------------------")

    print(
        f"City: {latest.iloc[0]['city']}"
    )

    print(
        "Timestamp:",
        latest.iloc[0]["timestamp"],
    )

    print(
        f"Model features: {len(MODEL_FEATURE_COLUMNS)}"
    )

    print(
        "Missing values:",
        int(
            latest[
                MODEL_FEATURE_COLUMNS
            ].isna().sum().sum()
        ),
    )

    print(
        "Current AQI:",
        latest.iloc[0]["us_aqi"],
    )

    print(
        "PM2.5:",
        latest.iloc[0]["pm2_5"],
    )

    print(
        "PM10:",
        latest.iloc[0]["pm10"],
    )

    print(
        "\nPushing latest vector to Feast..."
    )

    push_latest_features(latest)

    print(
        "Feast online update: PASS"
    )

    return latest


if __name__ == "__main__":
    run_live_feature_pipeline()