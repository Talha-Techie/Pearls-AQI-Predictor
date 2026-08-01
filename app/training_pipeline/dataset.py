"""Dataset loading and preparation for model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.feature_pipeline.engineer import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)


MAX_FORECAST_HORIZON_HOURS = max(
    int(
        target
        .removeprefix("target_aqi_")
        .removesuffix("h")
    )
    for target in TARGET_COLUMNS
)


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_training_dataset(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {path}"
        )

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    required = (
        MODEL_FEATURE_COLUMNS
        + TARGET_COLUMNS
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing training columns: {missing}"
        )

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


def temporal_split(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    purge_hours: int = MAX_FORECAST_HORIZON_HOURS,
) -> DatasetSplit:
    """
    Create leakage-safe temporal train/validation/test splits.

    Rows immediately before validation and test boundaries
    are purged so future forecast targets cannot cross into
    the next evaluation period.
    """

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "Train + validation ratios must be less than 1."
        )

    if purge_hours < 0:
        raise ValueError(
            "purge_hours cannot be negative."
        )

    df = (
        dataframe
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    row_count = len(df)

    train_boundary = int(
        row_count * train_ratio
    )

    validation_boundary = (
        train_boundary
        + int(
            row_count * validation_ratio
        )
    )

    train_end = int(
        train_boundary - purge_hours
    )

    validation_end = (
        validation_boundary - purge_hours
    )

    if train_end <= 0:
        raise ValueError(
            "Dataset is too small for the requested "
            "training split and purge horizon."
        )

    if validation_end <= train_boundary:
        raise ValueError(
            "Dataset is too small for the requested "
            "validation split and purge horizon."
        )

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_boundary:validation_end
    ].copy()

    test = df.iloc[
        validation_boundary:
    ].copy()

    if (
        train.empty
        or validation.empty
        or test.empty
    ):
        raise ValueError(
            "One or more temporal splits are empty."
        )

    max_horizon = pd.Timedelta(
        hours=purge_hours
    )

    if (
        train["timestamp"].max()
        + max_horizon
        >= validation["timestamp"].min()
    ):
        raise ValueError(
            "Training targets overlap the "
            "validation period."
        )

    if (
        validation["timestamp"].max()
        + max_horizon
        >= test["timestamp"].min()
    ):
        raise ValueError(
            "Validation targets overlap the test period."
        )

    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
    )
