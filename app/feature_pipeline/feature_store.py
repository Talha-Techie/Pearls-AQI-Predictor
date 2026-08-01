from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.feature_pipeline.engineer import MODEL_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_aqi_history_2023-01-01_2026-07-29.parquet"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "feature_repo"
    / "data"
    / "aqi_features.parquet"
)


class FeatureStorePreparationError(RuntimeError):
    """Raised when Feast feature data cannot be prepared."""


def prepare_feature_store_data(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed feature dataset not found: {input_path}"
        )

    dataframe = pd.read_parquet(input_path)

    if dataframe.empty:
        raise FeatureStorePreparationError(
            "Processed feature dataset is empty."
        )

    required_columns = [
        "timestamp",
        "city",
        *MODEL_FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise FeatureStorePreparationError(
            f"Missing Feast columns: {missing_columns}"
        )

    feast_df = dataframe[
        required_columns
    ].copy()

    feast_df["timestamp"] = pd.to_datetime(
        feast_df["timestamp"],
        utc=True,
    )

    feast_df = (
        feast_df
        .sort_values(
            ["city", "timestamp"]
        )
        .drop_duplicates(
            subset=["city", "timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    missing_values = int(
        feast_df.isna().sum().sum()
    )

    if missing_values:
        raise FeatureStorePreparationError(
            f"Feature Store dataset contains "
            f"{missing_values} missing values."
        )

    target_columns = {
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h",
    }

    leaked_targets = (
        target_columns
        & set(feast_df.columns)
    )

    if leaked_targets:
        raise FeatureStorePreparationError(
            "Forecast targets must not be included "
            f"in the online Feature Store: {leaked_targets}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feast_df.to_parquet(
        output_path,
        index=False,
    )

    print("\nFeast feature dataset prepared.")
    print("-------------------------------")

    print(
        f"Rows:          {len(feast_df):,}"
    )

    print(
        f"Model features:{len(MODEL_FEATURE_COLUMNS):>4}"
    )

    print(
        f"Total columns: {len(feast_df.columns)}"
    )

    print(
        f"Missing values:{missing_values:>4}"
    )

    print(
        f"\nRange:"
        f"\n{feast_df['timestamp'].min()}"
        f"\n→ {feast_df['timestamp'].max()}"
    )

    print(
        f"\nSaved to:\n{output_path.resolve()}"
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare inference-safe AQI features "
            "for the Feast Feature Store."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    prepare_feature_store_data(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()