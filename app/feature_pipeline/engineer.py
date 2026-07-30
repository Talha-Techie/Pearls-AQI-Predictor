from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.feature_pipeline.validator import (
    validate_hourly_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


TARGET_COLUMNS = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
]


MODEL_FEATURE_COLUMNS = [
    # Current weather
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "wind_speed_10m",

    # Circular wind representation
    "wind_direction_sin",
    "wind_direction_cos",

    # Current pollution
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",

    # Current AQI
    "us_aqi",

    # Calendar features
    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    # Cyclical calendar features
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",

    # AQI lags
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",

    # Pollution lags
    "pm2_5_lag_1h",
    "pm2_5_lag_6h",
    "pm2_5_lag_24h",
    "pm10_lag_24h",

    # Historical rolling features
    "aqi_rolling_mean_6h",
    "aqi_rolling_mean_12h",
    "aqi_rolling_mean_24h",
    "aqi_rolling_std_24h",

    "pm2_5_rolling_mean_6h",
    "pm2_5_rolling_mean_24h",
    "pm10_rolling_mean_24h",

    # AQI trend
    "aqi_change_1h",
    "aqi_change_rate_1h",
]


class FeatureEngineeringError(RuntimeError):
    """Raised when feature generation fails."""


def _add_time_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    df = dataframe.copy()

    timestamp = df["timestamp"]

    df["hour"] = timestamp.dt.hour
    df["day_of_week"] = timestamp.dt.dayofweek
    df["month"] = timestamp.dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )

    wind_radians = np.deg2rad(
        df["wind_direction_10m"]
    )

    df["wind_direction_sin"] = np.sin(
        wind_radians
    )

    df["wind_direction_cos"] = np.cos(
        wind_radians
    )

    return df


def _add_lag_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    df = dataframe.copy()

    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = (
            df["us_aqi"].shift(lag)
        )

    for lag in [1, 6, 24]:
        df[f"pm2_5_lag_{lag}h"] = (
            df["pm2_5"].shift(lag)
        )

    df["pm10_lag_24h"] = (
        df["pm10"].shift(24)
    )

    return df


def _add_rolling_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    df = dataframe.copy()

    # Shift first so rolling statistics contain
    # only observations before the current timestamp.
    previous_aqi = df["us_aqi"].shift(1)
    previous_pm25 = df["pm2_5"].shift(1)
    previous_pm10 = df["pm10"].shift(1)

    df["aqi_rolling_mean_6h"] = (
        previous_aqi
        .rolling(
            window=6,
            min_periods=6,
        )
        .mean()
    )

    df["aqi_rolling_mean_12h"] = (
        previous_aqi
        .rolling(
            window=12,
            min_periods=12,
        )
        .mean()
    )

    df["aqi_rolling_mean_24h"] = (
        previous_aqi
        .rolling(
            window=24,
            min_periods=24,
        )
        .mean()
    )

    df["aqi_rolling_std_24h"] = (
        previous_aqi
        .rolling(
            window=24,
            min_periods=24,
        )
        .std()
    )

    df["pm2_5_rolling_mean_6h"] = (
        previous_pm25
        .rolling(
            window=6,
            min_periods=6,
        )
        .mean()
    )

    df["pm2_5_rolling_mean_24h"] = (
        previous_pm25
        .rolling(
            window=24,
            min_periods=24,
        )
        .mean()
    )

    df["pm10_rolling_mean_24h"] = (
        previous_pm10
        .rolling(
            window=24,
            min_periods=24,
        )
        .mean()
    )

    return df


def _add_change_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    df = dataframe.copy()

    previous_aqi = df["us_aqi"].shift(1)

    df["aqi_change_1h"] = (
        df["us_aqi"] - previous_aqi
    )

    safe_previous_aqi = (
        previous_aqi.replace(0, np.nan)
    )

    df["aqi_change_rate_1h"] = (
        df["aqi_change_1h"]
        / safe_previous_aqi
    )

    return df


def _add_targets(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    df = dataframe.copy()

    df["target_aqi_24h"] = (
        df["us_aqi"].shift(-24)
    )

    df["target_aqi_48h"] = (
        df["us_aqi"].shift(-48)
    )

    df["target_aqi_72h"] = (
        df["us_aqi"].shift(-72)
    )

    return df


def engineer_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    validation = validate_hourly_dataset(
        dataframe
    )

    print("\nRaw dataset validation passed.")
    print(
        f"Rows: {validation['rows']:,}"
    )
    print(
        f"Range: "
        f"{validation['start_timestamp']} -> "
        f"{validation['end_timestamp']}"
    )

    df = dataframe.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df.sort_values(
            ["city", "timestamp"]
        )
        .reset_index(drop=True)
    )

    # For now the project operates on one city.
    # Group-aware feature generation will be used
    # when multi-city support is introduced.
    city_count = df["city"].nunique()

    if city_count != 1:
        raise FeatureEngineeringError(
            "Current feature pipeline expects exactly "
            f"one city, but received {city_count}."
        )

    df = _add_time_features(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_change_features(df)
    df = _add_targets(df)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    required_training_columns = (
        MODEL_FEATURE_COLUMNS
        + TARGET_COLUMNS
    )

    training_df = df.dropna(
        subset=required_training_columns
    ).reset_index(drop=True)

    if training_df.empty:
        raise FeatureEngineeringError(
            "No usable training rows remain after "
            "feature engineering."
        )

    return training_df


def process_file(
    input_path: Path,
    output_path: Path | None = None,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    dataframe = pd.read_parquet(
        input_path
    )

    original_rows = len(dataframe)

    engineered = engineer_features(
        dataframe
    )

    if output_path is None:
        output_path = (
            PROCESSED_DATA_DIR
            / f"features_{input_path.stem}.parquet"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    engineered.to_parquet(
        output_path,
        index=False,
    )

    print("\nFeature engineering completed.")
    print(
        f"Raw rows:      {original_rows:,}"
    )
    print(
        f"Training rows: {len(engineered):,}"
    )
    print(
        f"Model features: "
        f"{len(MODEL_FEATURE_COLUMNS)}"
    )
    print(
        f"Targets:       "
        f"{len(TARGET_COLUMNS)}"
    )

    print("\nTargets:")
    for target in TARGET_COLUMNS:
        print(f"  - {target}")

    print(
        f"\nProcessed dataset saved to:\n"
        f"{output_path.resolve()}"
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate leakage-safe AQI "
            "forecasting features."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Historical parquet dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional processed output path.",
    )

    args = parser.parse_args()

    process_file(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()