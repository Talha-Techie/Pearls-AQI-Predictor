from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config.settings import get_settings


settings = get_settings()


class DataCollectionError(RuntimeError):
    """Raised when external AQI/weather data cannot be collected."""


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


def fetch_weather() -> dict[str, Any]:
    params = {
        "latitude": settings.latitude,
        "longitude": settings.longitude,
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "pressure_msl",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
        "timezone": "UTC",
    }

    return _request_json(
        settings.weather_api_url,
        params,
    )


def fetch_air_quality() -> dict[str, Any]:
    params = {
        "latitude": settings.latitude,
        "longitude": settings.longitude,
        "current": ",".join(
            [
                "us_aqi",
                "pm10",
                "pm2_5",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
            ]
        ),
        "timezone": "UTC",
    }

    return _request_json(
        settings.air_quality_api_url,
        params,
    )


def collect_current_data() -> dict[str, Any]:
    try:
        weather = fetch_weather()
        air_quality = fetch_air_quality()

        weather_current = weather.get("current", {})
        air_current = air_quality.get("current", {})

        if not weather_current:
            raise DataCollectionError("Weather API returned no current data.")

        if not air_current:
            raise DataCollectionError("Air-quality API returned no current data.")

        return {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "city": settings.city,
            "latitude": settings.latitude,
            "longitude": settings.longitude,

            "temperature_2m": weather_current.get("temperature_2m"),
            "relative_humidity_2m": weather_current.get(
                "relative_humidity_2m"
            ),
            "precipitation": weather_current.get("precipitation"),
            "pressure_msl": weather_current.get("pressure_msl"),
            "wind_speed_10m": weather_current.get("wind_speed_10m"),
            "wind_direction_10m": weather_current.get(
                "wind_direction_10m"
            ),

            "pm10": air_current.get("pm10"),
            "pm2_5": air_current.get("pm2_5"),
            "carbon_monoxide": air_current.get("carbon_monoxide"),
            "nitrogen_dioxide": air_current.get("nitrogen_dioxide"),
            "sulphur_dioxide": air_current.get("sulphur_dioxide"),
            "ozone": air_current.get("ozone"),
            "us_aqi": air_current.get("us_aqi"),
        }

    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise DataCollectionError(
            f"Failed to collect AQI data: {exc}"
        ) from exc


if __name__ == "__main__":
    from pprint import pprint

    pprint(collect_current_data())