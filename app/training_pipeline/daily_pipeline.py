from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_START_DATE = "2023-01-01"


def run_command(
    command: list[str],
) -> None:
    print(
        "\nRunning:",
        " ".join(command),
    )

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    today = datetime.now(
        timezone.utc
    ).date()

    # Avoid incomplete current-day observations.
    end_date = (
        today
        - timedelta(days=1)
    ).isoformat()

    print(
        "\nDaily AQI Training Pipeline"
    )
    print(
        "==========================="
    )

    print(
        f"Historical range: "
        f"{HISTORICAL_START_DATE} "
        f"→ {end_date}"
    )

    # 1. Historical data
    run_command(
        [
            sys.executable,
            "-m",
            "app.feature_pipeline.backfill",
            "--start-date",
            HISTORICAL_START_DATE,
            "--end-date",
            end_date,
        ]
    )

    # 2. Produce the model-ready feature dataset.
    run_command(
        [
            sys.executable,
            "-m",
            "app.feature_pipeline.engineer",
        ]
    )

    # 3. Prepare Feast offline source.
    run_command(
        [
            sys.executable,
            "-m",
            "app.feature_pipeline.feature_store",
        ]
    )

    # 4. Apply Feast definitions.
    run_command(
        [
            "feast",
            "apply",
        ]
    )

    # 5. Benchmark candidate models.
    run_command(
        [
            sys.executable,
            "-m",
            "app.training_pipeline.train",
        ]
    )

    # 6. Train/finalize champion models.
    run_command(
        [
            sys.executable,
            "-m",
            "app.training_pipeline.finalize",
        ]
    )

    print(
        "\nDaily training pipeline: PASS"
    )


if __name__ == "__main__":
    main()