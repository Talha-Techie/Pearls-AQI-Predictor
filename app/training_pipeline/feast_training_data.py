from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from feast import FeatureStore

from app.feature_pipeline.engineer import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = PROJECT_ROOT / "feature_repo"


class FeastTrainingDataError(RuntimeError):
    """Raised when historical Feast training data cannot be built."""


def build_training_dataset(
    labels_path: Path,
    output_path: Path,
) -> Path:
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels dataset not found: {labels_path}"
        )

    labels_df = pd.read_parquet(
        labels_path
    )

    required_columns = [
        "timestamp",
        "city",
        *TARGET_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in labels_df.columns
    ]

    if missing_columns:
        raise FeastTrainingDataError(
            "Labels dataset is missing columns: "
            f"{missing_columns}"
        )

    entity_df = labels_df[
        [
            "city",
            "timestamp",
            *TARGET_COLUMNS,
        ]
    ].copy()

    entity_df = entity_df.rename(
        columns={
            "timestamp": "event_timestamp",
        }
    )

    entity_df["event_timestamp"] = (
        pd.to_datetime(
            entity_df["event_timestamp"],
            utc=True,
        )
    )

    store = FeatureStore(
        repo_path=str(
            FEATURE_REPO
        )
    )

    feature_service = (
        store.get_feature_service(
            "aqi_prediction_features_v1"
        )
    )

    retrieval_job = (
        store.get_historical_features(
            features=feature_service,
            entity_df=entity_df,
        )
    )

    historical_df = (
        retrieval_job.to_df()
    )

    historical_df = (
        historical_df.rename(
            columns={
                "event_timestamp": "timestamp",
            }
        )
    )

    required_output = [
        "timestamp",
        "city",
        *MODEL_FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    ]

    missing_output = [
        column
        for column in required_output
        if column not in historical_df.columns
    ]

    if missing_output:
        raise FeastTrainingDataError(
            "Feast historical retrieval is missing: "
            f"{missing_output}"
        )

    historical_df = historical_df[
        required_output
    ].copy()

    historical_df = (
        historical_df
        .dropna()
        .sort_values(
            [
                "city",
                "timestamp",
            ]
        )
        .drop_duplicates(
            subset=[
                "city",
                "timestamp",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if historical_df.empty:
        raise FeastTrainingDataError(
            "Feast returned no usable training rows."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    historical_df.to_parquet(
        output_path,
        index=False,
    )

    print(
        "\nFeast Historical Training Dataset"
    )
    print(
        "--------------------------------"
    )
    print(
        f"Rows:           "
        f"{len(historical_df):,}"
    )
    print(
        f"Model features: "
        f"{len(MODEL_FEATURE_COLUMNS)}"
    )
    print(
        f"Targets:        "
        f"{len(TARGET_COLUMNS)}"
    )
    print(
        f"Missing values: "
        f"{historical_df.isna().sum().sum()}"
    )
    print(
        f"\nRange:"
        f"\n{historical_df['timestamp'].min()}"
        f"\n→ {historical_df['timestamp'].max()}"
    )
    print(
        f"\nSaved to:\n"
        f"{output_path.resolve()}"
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve historical AQI training "
            "features through Feast."
        )
    )

    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    build_training_dataset(
        labels_path=args.labels,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()