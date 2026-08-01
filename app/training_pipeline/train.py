"""Model training entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import Ridge

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.feature_pipeline.engineer import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)

from app.training_pipeline.dataset import (
    load_training_dataset,
    temporal_split,
)

from app.training_pipeline.evaluate import (
    regression_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_benchmarks"
)


def build_models() -> dict:
    return {
        "ridge": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(
                        alpha=1.0,
                    ),
                ),
            ]
        ),

        "random_forest": (
            RandomForestRegressor(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        ),

        "gradient_boosting": (
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
        ),
    }


def persistence_predictions(
    dataframe: pd.DataFrame,
) -> pd.Series:
    return dataframe["us_aqi"]


def run_benchmark(
    input_path: Path,
) -> pd.DataFrame:

    df = load_training_dataset(
        input_path
    )

    splits = temporal_split(df)

    print("\nTemporal Dataset Split")
    print("----------------------")

    print(
        f"Train:      {len(splits.train):,}"
    )

    print(
        f"Validation: {len(splits.validation):,}"
    )

    print(
        f"Test:       {len(splits.test):,}"
    )

    print(
        "\nTrain range:"
        f"\n{splits.train.timestamp.min()}"
        f"\n→ {splits.train.timestamp.max()}"
    )

    print(
        "\nValidation range:"
        f"\n{splits.validation.timestamp.min()}"
        f"\n→ {splits.validation.timestamp.max()}"
    )

    print(
        "\nTest range:"
        f"\n{splits.test.timestamp.min()}"
        f"\n→ {splits.test.timestamp.max()}"
    )

    X_train = splits.train[
        MODEL_FEATURE_COLUMNS
    ]

    X_validation = splits.validation[
        MODEL_FEATURE_COLUMNS
    ]

    results: list[dict] = []

    for target in TARGET_COLUMNS:
        horizon = target.replace(
            "target_aqi_",
            "",
        )

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Forecast horizon: {horizon}"
        )

        print(
            f"{'=' * 60}"
        )

        y_train = splits.train[target]
        y_validation = (
            splits.validation[target]
        )

        baseline_predictions = (
            persistence_predictions(
                splits.validation
            )
        )

        baseline_metrics = (
            regression_metrics(
                y_validation,
                baseline_predictions,
            )
        )

        results.append(
            {
                "horizon": horizon,
                "model": "persistence",
                **baseline_metrics,
            }
        )

        print(
            "\nPersistence baseline:"
            f"\n  MAE:  "
            f"{baseline_metrics['mae']:.3f}"
            f"\n  RMSE: "
            f"{baseline_metrics['rmse']:.3f}"
            f"\n  R²:   "
            f"{baseline_metrics['r2']:.3f}"
        )

        models = build_models()

        for model_name, model in models.items():

            print(
                f"\nTraining {model_name}..."
            )

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_validation
            )

            metrics = regression_metrics(
                y_validation,
                predictions,
            )

            results.append(
                {
                    "horizon": horizon,
                    "model": model_name,
                    **metrics,
                }
            )

            print(
                f"  MAE:  "
                f"{metrics['mae']:.3f}"
            )

            print(
                f"  RMSE: "
                f"{metrics['rmse']:.3f}"
            )

            print(
                f"  R²:   "
                f"{metrics['r2']:.3f}"
            )

    results_df = pd.DataFrame(
        results
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        REPORT_DIR
        / "validation_benchmark.csv"
    )

    json_path = (
        REPORT_DIR
        / "validation_benchmark.json"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        "\n\nValidation Benchmark"
    )

    print(
        results_df
        .sort_values(
            [
                "horizon",
                "rmse",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to:\n"
        f"{csv_path.resolve()}"
    )

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark AQI forecasting models."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    run_benchmark(
        args.input
    )


if __name__ == "__main__":
    main()