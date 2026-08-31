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

# Prevent Feast/Dask from creating a huge
# city-level point-in-time join in memory.
RETRIEVAL_BATCH_SIZE = 128


class FeastTrainingDataError(RuntimeError):
    """Raised when Feast training data cannot be prepared."""


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

    labels_df = labels_df[
        required_columns
    ].copy()

    labels_df["timestamp"] = pd.to_datetime(
        labels_df["timestamp"],
        utc=True,
    )

    labels_df = (
        labels_df
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

    total_rows = len(
        labels_df
    )

    total_batches = (
        total_rows
        + RETRIEVAL_BATCH_SIZE
        - 1
    ) // RETRIEVAL_BATCH_SIZE

    print(
        "\nFeast Historical Feature Retrieval"
    )
    print(
        "----------------------------------"
    )
    print(
        f"Rows requested:  {total_rows:,}"
    )
    print(
        f"Batch size:      "
        f"{RETRIEVAL_BATCH_SIZE}"
    )
    print(
        f"Total batches:   "
        f"{total_batches}"
    )

    retrieved_batches: list[
        pd.DataFrame
    ] = []

    for batch_number, start_index in enumerate(
        range(
            0,
            total_rows,
            RETRIEVAL_BATCH_SIZE,
        ),
        start=1,
    ):
        end_index = min(
            start_index
            + RETRIEVAL_BATCH_SIZE,
            total_rows,
        )

        print(
            f"Retrieving batch "
            f"{batch_number}/{total_batches} "
            f"(rows {start_index + 1:,}"
            f"-{end_index:,})...",
            flush=True,
        )

        # Keep only entity + event time in the
        # Feast query. Targets are joined later.
        entity_batch = labels_df.iloc[
            start_index:end_index
        ][
            [
                "city",
                "timestamp",
            ]
        ].copy()

        entity_batch = entity_batch.rename(
            columns={
                "timestamp":
                    "event_timestamp",
            }
        )

        retrieval_job = (
            store.get_historical_features(
                features=feature_service,
                entity_df=entity_batch,
            )
        )

        batch_df = retrieval_job.to_df()

        batch_df = batch_df.rename(
            columns={
                "event_timestamp":
                    "timestamp",
            }
        )

        batch_df["timestamp"] = (
            pd.to_datetime(
                batch_df["timestamp"],
                utc=True,
            )
        )

        retrieved_batches.append(
            batch_df
        )

    if not retrieved_batches:
        raise FeastTrainingDataError(
            "Feast returned no historical batches."
        )

    features_df = pd.concat(
        retrieved_batches,
        ignore_index=True,
    )

    features_df = (
        features_df
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

    #
    # Join forecast targets AFTER Feast retrieval.
    #
    historical_df = features_df.merge(
        labels_df,
        on=[
            "city",
            "timestamp",
        ],
        how="inner",
        validate="one_to_one",
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
            "Historical training data is "
            "missing columns: "
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
        .reset_index(drop=True)
    )

    if historical_df.empty:
        raise FeastTrainingDataError(
            "Feast returned no usable "
            "training rows."
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
        "\nRange:"
    )
    print(
        historical_df[
            "timestamp"
        ].min()
    )
    print(
        "→",
        historical_df[
            "timestamp"
        ].max()
    )

    print(
        f"\nSaved to:\n"
        f"{output_path.resolve()}"
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve historical AQI "
            "training features through Feast."
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