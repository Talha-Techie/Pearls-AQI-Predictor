"""Alerting logic for AQI prediction thresholds."""

from __future__ import annotations

from typing import Any


def get_aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


def get_health_guidance(aqi: float) -> str:
    category = get_aqi_category(aqi)

    guidance = {
        "Good": (
            "Air quality is satisfactory. "
            "Normal outdoor activities are suitable."
        ),
        "Moderate": (
            "Air quality is acceptable. "
            "Sensitive individuals should monitor symptoms."
        ),
        "Unhealthy for Sensitive Groups": (
            "Sensitive groups should reduce prolonged "
            "or heavy outdoor activity."
        ),
        "Unhealthy": (
            "Everyone may begin to experience health effects. "
            "Reduce prolonged outdoor activity."
        ),
        "Very Unhealthy": (
            "Health alert conditions. "
            "Avoid unnecessary outdoor exposure."
        ),
        "Hazardous": (
            "Health emergency conditions. "
            "Avoid outdoor activity where possible."
        ),
    }

    return guidance[category]


def get_alert_level(aqi: float) -> str:
    if aqi <= 100:
        return "none"

    if aqi <= 150:
        return "advisory"

    if aqi <= 200:
        return "warning"

    if aqi <= 300:
        return "high"

    return "critical"


def build_aqi_status(
    aqi: float,
) -> dict[str, Any]:
    return {
        "aqi": round(float(aqi), 1),
        "category": get_aqi_category(aqi),
        "alert_level": get_alert_level(aqi),
        "health_guidance": get_health_guidance(aqi),
        "alert": bool(aqi > 100),
    }


def enrich_prediction(
    prediction: dict,
) -> dict:
    current_aqi = float(
        prediction["current_aqi"]
    )

    forecast = prediction["forecast"]

    enriched_forecast = {}

    for horizon, value in forecast.items():
        enriched_forecast[horizon] = (
            build_aqi_status(
                float(value)
            )
        )

    return {
        **prediction,
        "current_status": build_aqi_status(
            current_aqi
        ),
        "forecast_status": enriched_forecast,
        "hazard_alert": any(
            float(value) > 200
            for value in forecast.values()
        ),
    }