from __future__ import annotations

import pandas as pd
import pytest

from app.feature_pipeline.validator import (
    DataValidationError,
    validate_hourly_dataset,
)


def make_valid_dataset(hours: int = 100) -> pd.DataFrame:
    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=hours,
        freq="h",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "city": ["Lahore"] * hours,
            "latitude": [31.5204] * hours,
            "longitude": [74.3587] * hours,
            "temperature_2m": [25.0] * hours,
            "relative_humidity_2m": [60] * hours,
            "precipitation": [0.0] * hours,
            "pressure_msl": [1010.0] * hours,
            "wind_speed_10m": [5.0] * hours,
            "wind_direction_10m": [180] * hours,
            "pm10": [80.0] * hours,
            "pm2_5": [40.0] * hours,
            "carbon_monoxide": [300.0] * hours,
            "nitrogen_dioxide": [10.0] * hours,
            "sulphur_dioxide": [5.0] * hours,
            "ozone": [70.0] * hours,
            "us_aqi": [100] * hours,
        }
    )


def test_valid_dataset_passes() -> None:
    df = make_valid_dataset()

    result = validate_hourly_dataset(df)

    assert result["valid"] is True
    assert result["rows"] == 100
    assert result["missing_values"] == 0
    assert result["duplicate_records"] == 0
    assert result["irregular_intervals"] == 0


def test_missing_required_column_fails() -> None:
    df = make_valid_dataset()

    df = df.drop(
        columns=["pm2_5"]
    )

    with pytest.raises(
        DataValidationError,
        match="Missing required columns",
    ):
        validate_hourly_dataset(df)


def test_missing_value_fails() -> None:
    df = make_valid_dataset()

    df.loc[10, "us_aqi"] = None

    with pytest.raises(
        DataValidationError,
        match="Missing values detected",
    ):
        validate_hourly_dataset(df)


def test_duplicate_timestamp_fails() -> None:
    df = make_valid_dataset()

    duplicated = pd.concat(
        [
            df,
            df.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        DataValidationError,
        match="duplicate",
    ):
        validate_hourly_dataset(duplicated)


def test_invalid_humidity_fails() -> None:
    df = make_valid_dataset()

    df.loc[0, "relative_humidity_2m"] = 120

    with pytest.raises(
        DataValidationError,
        match="humidity",
    ):
        validate_hourly_dataset(df)


def test_negative_pollution_fails() -> None:
    df = make_valid_dataset()

    df.loc[0, "pm2_5"] = -10

    with pytest.raises(
        DataValidationError,
        match="negative",
    ):
        validate_hourly_dataset(df)


def test_non_hourly_interval_fails() -> None:
    df = make_valid_dataset()

    df = df.drop(
        index=20
    ).reset_index(drop=True)

    with pytest.raises(
        DataValidationError,
        match="non-hourly",
    ):
        validate_hourly_dataset(df)