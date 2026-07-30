from __future__ import annotations

import numpy as np
import pandas as pd

from app.feature_pipeline.engineer import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    engineer_features,
)


def make_dataset(
    hours: int = 200,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=hours,
        freq="h",
        tz="UTC",
    )

    sequence = np.arange(
        1,
        hours + 1,
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "city": ["Lahore"] * hours,
            "latitude": [31.5204] * hours,
            "longitude": [74.3587] * hours,

            "temperature_2m": 20 + sequence * 0.01,
            "relative_humidity_2m": [60] * hours,
            "precipitation": [0.0] * hours,
            "pressure_msl": [1010.0] * hours,
            "wind_speed_10m": [5.0] * hours,
            "wind_direction_10m": [180] * hours,

            "pm10": 50 + sequence,
            "pm2_5": 20 + sequence,
            "carbon_monoxide": 200 + sequence,
            "nitrogen_dioxide": 10 + sequence * 0.01,
            "sulphur_dioxide": 5 + sequence * 0.01,
            "ozone": 40 + sequence * 0.01,

            # Predictable sequence makes lag/target
            # correctness easy to verify.
            "us_aqi": sequence,
        }
    )


def test_feature_engineering_returns_rows() -> None:
    df = make_dataset()

    processed = engineer_features(df)

    assert not processed.empty


def test_all_model_features_exist() -> None:
    df = make_dataset()

    processed = engineer_features(df)

    for column in MODEL_FEATURE_COLUMNS:
        assert column in processed.columns


def test_all_targets_exist() -> None:
    df = make_dataset()

    processed = engineer_features(df)

    for column in TARGET_COLUMNS:
        assert column in processed.columns


def test_no_missing_training_values() -> None:
    df = make_dataset()

    processed = engineer_features(df)

    required_columns = (
        MODEL_FEATURE_COLUMNS
        + TARGET_COLUMNS
    )

    assert (
        processed[required_columns]
        .isna()
        .sum()
        .sum()
        == 0
    )


def test_24_hour_lag_is_correct() -> None:
    df = make_dataset()

    processed = engineer_features(df)

    first_row = processed.iloc[0]

    timestamp = first_row["timestamp"]

    source_row = df.loc[
        df["timestamp"]
        == timestamp - pd.Timedelta(hours=24)
    ].iloc[0]

    assert (
        first_row["aqi_lag_24h"]
        == source_row["us_aqi"]
    )


def test_1_hour_lag_is_correct() -> None:
    df = make_dataset()

    processed = engineer_features(df)

    first_row = processed.iloc[0]

    timestamp = first_row["timestamp"]

    source_row = df.loc[
        df["timestamp"]
        == timestamp - pd.Timedelta(hours=1)
    ].iloc[0]

    assert (
        first_row["aqi_lag_1h"]
        == source_row["us_aqi"]
    )


def test_24_hour_target_is_correct() -> None:
    df = make_dataset()

    processed = engineer_features(df)

    first_row = processed.iloc[0]

    timestamp = first_row["timestamp"]

    target_row = df.loc[
        df["timestamp"]
        == timestamp + pd.Timedelta(hours=24)
    ].iloc[0]

    assert (
        first_row["target_aqi_24h"]
        == target_row["us_aqi"]
    )


def test_48_hour_target_is_correct() -> None:
    df = make_dataset()

    processed = engineer_features(df)

    first_row = processed.iloc[0]

    timestamp = first_row["timestamp"]

    target_row = df.loc[
        df["timestamp"]
        == timestamp + pd.Timedelta(hours=48)
    ].iloc[0]

    assert (
        first_row["target_aqi_48h"]
        == target_row["us_aqi"]
    )


def test_72_hour_target_is_correct() -> None:
    df = make_dataset()

    processed = engineer_features(df)

    first_row = processed.iloc[0]

    timestamp = first_row["timestamp"]

    target_row = df.loc[
        df["timestamp"]
        == timestamp + pd.Timedelta(hours=72)
    ].iloc[0]

    assert (
        first_row["target_aqi_72h"]
        == target_row["us_aqi"]
    )


def test_output_row_count_is_correct() -> None:
    df = make_dataset(
        hours=200
    )

    processed = engineer_features(df)

    # First 24 rows are lost due to the longest lag.
    # Last 72 rows are lost due to the longest target.
    expected_rows = 200 - 24 - 72

    assert len(processed) == expected_rows