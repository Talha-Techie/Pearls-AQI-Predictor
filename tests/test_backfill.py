from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.feature_pipeline import backfill


def test_invalid_date_range_fails() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be after",
    ):
        backfill._validate_date_range(
            date(2024, 2, 1),
            date(2024, 1, 1),
        )


def test_month_chunks_are_correct() -> None:
    chunks = list(
        backfill._month_chunks(
            date(2024, 1, 15),
            date(2024, 3, 10),
        )
    )

    assert chunks == [
        (
            date(2024, 1, 15),
            date(2024, 1, 31),
        ),
        (
            date(2024, 2, 1),
            date(2024, 2, 29),
        ),
        (
            date(2024, 3, 1),
            date(2024, 3, 10),
        ),
    ]


def test_hourly_payload_conversion() -> None:
    payload = {
        "hourly": {
            "time": [
                "2024-01-01T00:00",
                "2024-01-01T01:00",
            ],
            "temperature_2m": [
                20.0,
                21.0,
            ],
        }
    }

    df = backfill._hourly_payload_to_dataframe(
        payload,
        source_name="Test API",
    )

    assert len(df) == 2

    assert "timestamp" in df.columns

    assert str(
        df["timestamp"].dt.tz
    ) == "UTC"

    assert (
        df.iloc[0]["temperature_2m"]
        == 20.0
    )


def test_missing_hourly_payload_fails() -> None:
    with pytest.raises(
        backfill.BackfillError,
        match="returned no hourly data",
    ):
        backfill._hourly_payload_to_dataframe(
            {},
            source_name="Test API",
        )


def test_merge_weather_and_air_quality() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=3,
        freq="h",
        tz="UTC",
    )

    weather = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_2m": [
                20.0,
                21.0,
                22.0,
            ],
        }
    )

    air_quality = pd.DataFrame(
        {
            "timestamp": timestamps,
            "us_aqi": [
                100,
                110,
                120,
            ],
        }
    )

    merged = (
        backfill._merge_weather_and_air_quality(
            weather,
            air_quality,
        )
    )

    assert len(merged) == 3

    assert "city" in merged.columns
    assert "latitude" in merged.columns
    assert "longitude" in merged.columns

    assert "temperature_2m" in merged.columns
    assert "us_aqi" in merged.columns

    assert merged["timestamp"].is_unique


def test_merge_without_matching_timestamps_fails() -> None:
    weather = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01T00:00:00Z"]
            ),
            "temperature_2m": [20.0],
        }
    )

    air_quality = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02T00:00:00Z"]
            ),
            "us_aqi": [100],
        }
    )

    with pytest.raises(
        backfill.BackfillError,
        match="no matching timestamps",
    ):
        backfill._merge_weather_and_air_quality(
            weather,
            air_quality,
        )


def test_backfill_pipeline_saves_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:

    raw_dir = tmp_path / "raw"
    historical_dir = tmp_path / "historical"

    monkeypatch.setattr(
        backfill,
        "RAW_DATA_DIR",
        raw_dir,
    )

    monkeypatch.setattr(
        backfill,
        "HISTORICAL_DATA_DIR",
        historical_dir,
    )

    def fake_weather(
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:

        timestamps = pd.date_range(
            start=start_date,
            end=(
                pd.Timestamp(end_date)
                + pd.Timedelta(hours=23)
            ),
            freq="h",
            tz="UTC",
        )

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "temperature_2m": [25.0]
                * len(timestamps),
            }
        )

    def fake_air_quality(
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:

        timestamps = pd.date_range(
            start=start_date,
            end=(
                pd.Timestamp(end_date)
                + pd.Timedelta(hours=23)
            ),
            freq="h",
            tz="UTC",
        )

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "us_aqi": [100]
                * len(timestamps),
            }
        )

    monkeypatch.setattr(
        backfill,
        "fetch_historical_weather",
        fake_weather,
    )

    monkeypatch.setattr(
        backfill,
        "fetch_historical_air_quality",
        fake_air_quality,
    )

    result = backfill.backfill(
        date(2024, 1, 1),
        date(2024, 1, 2),
    )

    assert len(result) == 48

    assert (
        raw_dir
        / "weather_2024-01-01_2024-01-02.parquet"
    ).exists()

    assert (
        raw_dir
        / "air_quality_2024-01-01_2024-01-02.parquet"
    ).exists()

    assert (
        historical_dir
        / "aqi_history_2024-01-01_2024-01-02.parquet"
    ).exists()