"""Backfill jobs for historical AQI feature generation."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config.settings import get_settings


settings = get_settings()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
HISTORICAL_DATA_DIR = PROJECT_ROOT / "data" / "historical"


WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
]


AIR_QUALITY_VARIABLES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]


class BackfillError(RuntimeError):
    """Raised when historical data cannot be backfilled."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _request_json(
    url: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    with httpx.Client(timeout=settings.http_timeout_seconds) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def _validate_date_range(
    start_date: date,
    end_date: date,
) -> None:
    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) cannot be after "
            f"end_date ({end_date})."
        )

    if end_date >= datetime.now(timezone.utc).date():
        raise ValueError(
            "Historical backfill should end before today's date."
        )


def _month_chunks(
    start_date: date,
    end_date: date,
) -> Iterator[tuple[date, date]]:
    """
    Split a large date range into monthly chunks.

    Smaller API requests are more reliable and make retries cheaper.
    """

    current = start_date

    while current <= end_date:
        next_month = (
            current.replace(day=28) + timedelta(days=4)
        ).replace(day=1)

        chunk_end = min(
            next_month - timedelta(days=1),
            end_date,
        )

        yield current, chunk_end

        current = chunk_end + timedelta(days=1)


def _hourly_payload_to_dataframe(
    payload: dict[str, Any],
    source_name: str,
) -> pd.DataFrame:
    hourly = payload.get("hourly")

    if not hourly:
        raise BackfillError(
            f"{source_name} returned no hourly data."
        )

    if "time" not in hourly:
        raise BackfillError(
            f"{source_name} response does not contain timestamps."
        )

    dataframe = pd.DataFrame(hourly)

    dataframe["timestamp"] = pd.to_datetime(
        dataframe.pop("time"),
        utc=True,
        errors="raise",
    )

    dataframe = dataframe.sort_values("timestamp")

    dataframe = dataframe.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    return dataframe


def fetch_historical_weather(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    params = {
        "latitude": settings.latitude,
        "longitude": settings.longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
    }

    payload = _request_json(
        settings.historical_weather_api_url,
        params,
    )

    return _hourly_payload_to_dataframe(
        payload,
        source_name="Historical Weather API",
    )


def fetch_historical_air_quality(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    params = {
        "latitude": settings.latitude,
        "longitude": settings.longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(AIR_QUALITY_VARIABLES),
        "timezone": "UTC",
    }

    payload = _request_json(
        settings.air_quality_api_url,
        params,
    )

    return _hourly_payload_to_dataframe(
        payload,
        source_name="Historical Air Quality API",
    )


def _merge_weather_and_air_quality(
    weather: pd.DataFrame,
    air_quality: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.merge(
        weather,
        air_quality,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise BackfillError(
            "Weather and air-quality data produced no matching timestamps."
        )

    merged.insert(1, "city", settings.city)
    merged.insert(2, "latitude", settings.latitude)
    merged.insert(3, "longitude", settings.longitude)

    merged = merged.sort_values("timestamp")

    merged = merged.drop_duplicates(
        subset=["timestamp", "city"],
        keep="last",
    )

    return merged.reset_index(drop=True)


def _save_dataframe(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        path,
        index=False,
    )


def backfill(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    _validate_date_range(
        start_date,
        end_date,
    )

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORICAL_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    weather_frames: list[pd.DataFrame] = []
    air_quality_frames: list[pd.DataFrame] = []

    print(
        f"\nStarting historical backfill for "
        f"{settings.city}"
    )

    print(
        f"Range: {start_date} -> {end_date}\n"
    )

    for chunk_start, chunk_end in _month_chunks(
        start_date,
        end_date,
    ):
        print(
            f"Fetching {chunk_start} -> {chunk_end}..."
        )

        weather = fetch_historical_weather(
            chunk_start,
            chunk_end,
        )

        air_quality = fetch_historical_air_quality(
            chunk_start,
            chunk_end,
        )

        weather_frames.append(weather)
        air_quality_frames.append(air_quality)

        print(
            f"  Weather rows:     {len(weather)}"
        )

        print(
            f"  Air-quality rows: {len(air_quality)}"
        )

    weather_data = pd.concat(
        weather_frames,
        ignore_index=True,
    )

    air_quality_data = pd.concat(
        air_quality_frames,
        ignore_index=True,
    )

    weather_data = weather_data.drop_duplicates(
        subset=["timestamp"],
    ).sort_values("timestamp")

    air_quality_data = air_quality_data.drop_duplicates(
        subset=["timestamp"],
    ).sort_values("timestamp")

    merged = _merge_weather_and_air_quality(
        weather_data,
        air_quality_data,
    )

    file_suffix = (
        f"{start_date.isoformat()}_"
        f"{end_date.isoformat()}"
    )

    weather_path = (
        RAW_DATA_DIR
        / f"weather_{file_suffix}.parquet"
    )

    air_quality_path = (
        RAW_DATA_DIR
        / f"air_quality_{file_suffix}.parquet"
    )

    merged_path = (
        HISTORICAL_DATA_DIR
        / f"aqi_history_{file_suffix}.parquet"
    )

    _save_dataframe(
        weather_data,
        weather_path,
    )

    _save_dataframe(
        air_quality_data,
        air_quality_path,
    )

    _save_dataframe(
        merged,
        merged_path,
    )

    missing_aqi = int(
        merged["us_aqi"].isna().sum()
    )

    print("\nBackfill completed successfully.")
    print(f"Historical rows: {len(merged):,}")
    print(f"Missing AQI rows: {missing_aqi:,}")
    print(f"Columns: {len(merged.columns)}")

    print(
        f"\nMerged dataset saved to:\n"
        f"{merged_path}"
    )

    print(
        "\nTimestamp range:"
        f"\n  {merged['timestamp'].min()}"
        f"\n  {merged['timestamp'].max()}"
    )

    return merged


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)

    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. "
            "Expected YYYY-MM-DD."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill historical weather and "
            "air-quality data."
        )
    )

    parser.add_argument(
        "--start",
        required=True,
        type=_parse_date,
        help="Start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end",
        required=True,
        type=_parse_date,
        help="End date in YYYY-MM-DD format.",
    )

    args = parser.parse_args()

    backfill(
        start_date=args.start,
        end_date=args.end,
    )


if __name__ == "__main__":
    main()