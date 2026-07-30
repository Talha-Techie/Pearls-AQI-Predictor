from __future__ import annotations

import pytest

from app.feature_pipeline import collector
from app.feature_pipeline.collector import (
    DataCollectionError,
    collect_current_data,
)


def test_collect_current_data_combines_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    weather_payload = {
        "current": {
            "temperature_2m": 35.5,
            "relative_humidity_2m": 55,
            "precipitation": 0.0,
            "pressure_msl": 1001.2,
            "wind_speed_10m": 8.5,
            "wind_direction_10m": 180,
        }
    }

    air_quality_payload = {
        "current": {
            "pm10": 120.0,
            "pm2_5": 65.0,
            "carbon_monoxide": 350.0,
            "nitrogen_dioxide": 12.0,
            "sulphur_dioxide": 8.0,
            "ozone": 90.0,
            "us_aqi": 155,
        }
    }

    monkeypatch.setattr(
        collector,
        "fetch_weather",
        lambda: weather_payload,
    )

    monkeypatch.setattr(
        collector,
        "fetch_air_quality",
        lambda: air_quality_payload,
    )

    result = collect_current_data()

    assert result["city"] == collector.settings.city
    assert result["latitude"] == collector.settings.latitude
    assert result["longitude"] == collector.settings.longitude

    assert result["temperature_2m"] == 35.5
    assert result["relative_humidity_2m"] == 55
    assert result["pressure_msl"] == 1001.2

    assert result["pm2_5"] == 65.0
    assert result["pm10"] == 120.0
    assert result["us_aqi"] == 155

    assert "collected_at" in result


def test_missing_weather_data_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(
        collector,
        "fetch_weather",
        lambda: {"current": {}},
    )

    monkeypatch.setattr(
        collector,
        "fetch_air_quality",
        lambda: {
            "current": {
                "us_aqi": 100,
            }
        },
    )

    with pytest.raises(
        DataCollectionError,
        match="Weather API returned no current data",
    ):
        collect_current_data()


def test_missing_air_quality_data_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(
        collector,
        "fetch_weather",
        lambda: {
            "current": {
                "temperature_2m": 30,
            }
        },
    )

    monkeypatch.setattr(
        collector,
        "fetch_air_quality",
        lambda: {"current": {}},
    )

    with pytest.raises(
        DataCollectionError,
        match="Air-quality API returned no current data",
    ):
        collect_current_data()


def test_fetch_weather_uses_expected_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    captured: dict = {}

    def fake_request(
        url: str,
        params: dict,
    ) -> dict:
        captured["url"] = url
        captured["params"] = params

        return {
            "current": {
                "temperature_2m": 30,
            }
        }

    monkeypatch.setattr(
        collector,
        "_request_json",
        fake_request,
    )

    collector.fetch_weather()

    assert (
        captured["url"]
        == collector.settings.weather_api_url
    )

    current_variables = captured["params"]["current"]

    assert "temperature_2m" in current_variables
    assert "relative_humidity_2m" in current_variables
    assert "pressure_msl" in current_variables
    assert "wind_speed_10m" in current_variables


def test_fetch_air_quality_uses_expected_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    captured: dict = {}

    def fake_request(
        url: str,
        params: dict,
    ) -> dict:
        captured["url"] = url
        captured["params"] = params

        return {
            "current": {
                "us_aqi": 100,
            }
        }

    monkeypatch.setattr(
        collector,
        "_request_json",
        fake_request,
    )

    collector.fetch_air_quality()

    assert (
        captured["url"]
        == collector.settings.air_quality_api_url
    )

    current_variables = captured["params"]["current"]

    assert "us_aqi" in current_variables
    assert "pm2_5" in current_variables
    assert "pm10" in current_variables
    assert "ozone" in current_variables