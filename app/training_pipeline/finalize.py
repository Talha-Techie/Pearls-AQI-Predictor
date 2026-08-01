from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

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

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_benchmarks"
)


def build_ridge_model() -> Pipeline:
    """
    Build the champion Ridge regression pipeline.

    Scaling remains inside the pipeline so training
    and inference always use identical preprocessing.
    """

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(alpha=1.0),
            ),
        ]
    )


def horizon_from_target(
    target: str,
) -> str:
    return (
        target
        .removeprefix("target_aqi_")
    )


def save_model_metadata(
    metadata: dict[str, Any],
    path: Path,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            default=str,
        )


def finalize_champions(
    input_path: Path,
) -> pd.DataFrame:

    dataframe = load_training_dataset(
        input_path
    )

    splits = temporal_split(
        dataframe
    )

    # Validation has already served its purpose:
    # model selection.
    #
    # We can now combine train + validation for the
    # final model while preserving the purge gap
    # that already exists between the datasets.
    development_data = pd.concat(
        [
            splits.train,
            splits.validation,
        ],
        ignore_index=True,
    ).sort_values(
        "timestamp"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nFinal Model Training")
    print("--------------------")

    print(
        f"Development rows: "
        f"{len(development_data):,}"
    )

    print(
        f"Test rows:        "
        f"{len(splits.test):,}"
    )

    print(
        "\nDevelopment range:"
        f"\n{development_data['timestamp'].min()}"
        f"\n→ {development_data['timestamp'].max()}"
    )

    print(
        "\nTest range:"
        f"\n{splits.test['timestamp'].min()}"
        f"\n→ {splits.test['timestamp'].max()}"
    )

    X_development = development_data[
        MODEL_FEATURE_COLUMNS
    ]

    X_test = splits.test[
        MODEL_FEATURE_COLUMNS
    ]

    results: list[dict[str, Any]] = []

    for target in TARGET_COLUMNS:

        horizon = horizon_from_target(
            target
        )

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Final champion: Ridge ({horizon})"
        )

        print(
            f"{'=' * 60}"
        )

        y_development = (
            development_data[target]
        )

        y_test = splits.test[target]

        model = build_ridge_model()

        model.fit(
            X_development,
            y_development,
        )

        predictions = model.predict(
            X_test
        )

        test_metrics = regression_metrics(
            y_test,
            predictions,
        )

        # Persistence remains our final reference
        # baseline on the untouched test period.
        baseline_predictions = (
            splits.test["us_aqi"]
        )

        baseline_metrics = regression_metrics(
            y_test,
            baseline_predictions,
        )

        model_path = (
            MODEL_DIR
            / f"aqi_ridge_{horizon}.joblib"
        )

        metadata_path = (
            REPORT_DIR
            / f"aqi_ridge_{horizon}_metadata.json"
        )

        joblib.dump(
            model,
            model_path,
        )

        metadata = {
            "model_name": f"aqi_ridge_{horizon}",
            "model_type": "Ridge Regression",
            "forecast_horizon": horizon,
            "target": target,
            "alpha": 1.0,
            "feature_count": len(
                MODEL_FEATURE_COLUMNS
            ),
            "features": MODEL_FEATURE_COLUMNS,
            "development_rows": len(
                development_data
            ),
            "test_rows": len(
                splits.test
            ),
            "development_start": (
                development_data[
                    "timestamp"
                ].min()
            ),
            "development_end": (
                development_data[
                    "timestamp"
                ].max()
            ),
            "test_start": (
                splits.test[
                    "timestamp"
                ].min()
            ),
            "test_end": (
                splits.test[
                    "timestamp"
                ].max()
            ),
            "test_metrics": test_metrics,
            "persistence_baseline": (
                baseline_metrics
            ),
        }

        save_model_metadata(
            metadata,
            metadata_path,
        )

        results.append(
            {
                "horizon": horizon,
                "model": "ridge",
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "r2": test_metrics["r2"],
                "baseline_mae": (
                    baseline_metrics["mae"]
                ),
                "baseline_rmse": (
                    baseline_metrics["rmse"]
                ),
                "baseline_r2": (
                    baseline_metrics["r2"]
                ),
            }
        )

        print(
            "\nTest metrics:"
        )

        print(
            f"  MAE:  "
            f"{test_metrics['mae']:.3f}"
        )

        print(
            f"  RMSE: "
            f"{test_metrics['rmse']:.3f}"
        )

        print(
            f"  R²:   "
            f"{test_metrics['r2']:.3f}"
        )

        print(
            "\nPersistence baseline:"
        )

        print(
            f"  MAE:  "
            f"{baseline_metrics['mae']:.3f}"
        )

        print(
            f"  RMSE: "
            f"{baseline_metrics['rmse']:.3f}"
        )

        print(
            f"  R²:   "
            f"{baseline_metrics['r2']:.3f}"
        )

        print(
            f"\nModel saved:\n"
            f"{model_path.resolve()}"
        )

    results_df = pd.DataFrame(
        results
    )

    csv_path = (
        REPORT_DIR
        / "final_test_metrics.csv"
    )

    json_path = (
        REPORT_DIR
        / "final_test_metrics.json"
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
        "\n\nFinal Test Results"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nFinal metrics saved to:\n"
        f"{csv_path.resolve()}"
    )

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain selected AQI champion models "
            "and perform final test evaluation."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    finalize_champions(
        input_path=args.input
    )


if __name__ == "__main__":
    main()