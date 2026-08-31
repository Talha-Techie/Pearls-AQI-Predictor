from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_REPO = PROJECT_ROOT / "feature_repo"

START_DATE = "2023-01-01"


def run_command(
    command: list[str],
    cwd: Path | None = None,
) -> None:
    print(
        "\n> " + " ".join(command),
        flush=True,
    )

    subprocess.run(
        command,
        cwd=cwd or PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    end_date = (
        datetime.now(
            timezone.utc
        ).date()
        - timedelta(days=1)
    ).isoformat()

    suffix = (
        f"{START_DATE}_{end_date}"
    )

    historical_path = (
        PROJECT_ROOT
        / "data"
        / "historical"
        / f"aqi_history_{suffix}.parquet"
    )

    processed_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"features_aqi_history_{suffix}.parquet"
    )

    feast_source_path = (
        FEATURE_REPO
        / "data"
        / "aqi_features.parquet"
    )

    feast_training_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"feast_training_{suffix}.parquet"
    )

    print(
        "\nDaily AQI Training Pipeline"
    )
    print(
        "==========================="
    )
    print(
        f"Historical range: "
        f"{START_DATE} -> {end_date}"
    )

    # 1. Historical backfill
    run_command(
        [
            sys.executable,
            "-m",
            "app.feature_pipeline.backfill",
            "--start",
            START_DATE,
            "--end",
            end_date,
        ]
    )

    # 2. Leakage-safe feature engineering
    run_command(
        [
            sys.executable,
            "-m",
            "app.feature_pipeline.engineer",
            "--input",
            str(historical_path),
            "--output",
            str(processed_path),
        ]
    )

    # 3. Prepare Feast offline feature source
    run_command(
        [
            sys.executable,
            "-m",
            "app.feature_pipeline.feature_store",
            "--input",
            str(processed_path),
            "--output",
            str(feast_source_path),
        ]
    )

    # 4. Apply Feast definitions
    run_command(
        [
            "feast",
            "apply",
        ],
        cwd=FEATURE_REPO,
    )

    # 5. Retrieve historical features through Feast
    run_command(
        [
            sys.executable,
            "-m",
            "app.training_pipeline.feast_training_data",
            "--labels",
            str(processed_path),
            "--output",
            str(feast_training_path),
        ]
    )

    # 6. Benchmark candidate models
    run_command(
        [
            sys.executable,
            "-m",
            "app.training_pipeline.train",
            "--input",
            str(feast_training_path),
        ]
    )

    # 7. Retrain/finalize Ridge champions
    run_command(
        [
            sys.executable,
            "-m",
            "app.training_pipeline.finalize",
            "--input",
            str(feast_training_path),
        ]
    )

    # 8. Register models in DagsHub MLflow
    run_command(
        [
            sys.executable,
            "-m",
            "app.training_pipeline.registry",
            "--input",
            str(feast_training_path),
        ]
    )

    print(
        "\nDaily AQI Training Pipeline: PASS"
    )


if __name__ == "__main__":
    main()