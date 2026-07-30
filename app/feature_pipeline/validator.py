"""Validation utilities for collected and engineered AQI features."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "city",
    "latitude",
    "longitude",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]


NON_NEGATIVE_COLUMNS = [
    "precipitation",
    "wind_speed_10m",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]


class DataValidationError(ValueError):
    """Raised when incoming AQI data fails validation."""


def validate_hourly_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """
    Validate raw historical AQI/weather data before
    feature engineering.

    Returns a validation summary when successful.
    """

    if dataframe.empty:
        raise DataValidationError(
            "Dataset is empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise DataValidationError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    df = dataframe.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamps = int(
        df["timestamp"].isna().sum()
    )

    if invalid_timestamps:
        raise DataValidationError(
            f"Found {invalid_timestamps} invalid timestamps."
        )

    duplicate_count = int(
        df.duplicated(
            subset=["city", "timestamp"]
        ).sum()
    )

    if duplicate_count:
        raise DataValidationError(
            f"Found {duplicate_count} duplicate "
            "city/timestamp records."
        )

    df = df.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    missing_counts = (
        df[REQUIRED_COLUMNS]
        .isna()
        .sum()
    )

    columns_with_missing = {
        column: int(count)
        for column, count in missing_counts.items()
        if count > 0
    }

    if columns_with_missing:
        raise DataValidationError(
            "Missing values detected: "
            f"{columns_with_missing}"
        )

    humidity_invalid = (
        (df["relative_humidity_2m"] < 0)
        | (df["relative_humidity_2m"] > 100)
    )

    if humidity_invalid.any():
        raise DataValidationError(
            "Relative humidity contains values "
            "outside the valid 0-100 range."
        )

    wind_direction_invalid = (
        (df["wind_direction_10m"] < 0)
        | (df["wind_direction_10m"] > 360)
    )

    if wind_direction_invalid.any():
        raise DataValidationError(
            "Wind direction contains values "
            "outside the valid 0-360 range."
        )

    for column in NON_NEGATIVE_COLUMNS:
        negative_count = int(
            (df[column] < 0).sum()
        )

        if negative_count:
            raise DataValidationError(
                f"Column '{column}' contains "
                f"{negative_count} negative values."
            )

    irregular_intervals = 0

    for _, city_df in df.groupby(
        "city",
        sort=False,
    ):
        differences = (
            city_df["timestamp"]
            .sort_values()
            .diff()
            .dropna()
        )

        irregular_intervals += int(
            (
                differences
                != pd.Timedelta(hours=1)
            ).sum()
        )

    if irregular_intervals:
        raise DataValidationError(
            f"Found {irregular_intervals} "
            "non-hourly timestamp intervals."
        )

    return {
        "valid": True,
        "rows": len(df),
        "columns": len(df.columns),
        "cities": sorted(
            df["city"].unique().tolist()
        ),
        "start_timestamp": df["timestamp"].min(),
        "end_timestamp": df["timestamp"].max(),
        "duplicate_records": duplicate_count,
        "missing_values": 0,
        "irregular_intervals": irregular_intervals,
    }