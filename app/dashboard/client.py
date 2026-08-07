from __future__ import annotations

import requests


API_BASE_URL = "http://127.0.0.1:8000/api/v1"


class DashboardAPIError(RuntimeError):
    """Raised when the dashboard cannot reach the AQI API."""


def refresh_features() -> dict:
    response = requests.post(
        f"{API_BASE_URL}/features/refresh",
        timeout=60,
    )

    if not response.ok:
        raise DashboardAPIError(
            f"Feature refresh failed: {response.status_code}"
        )

    return response.json()


def get_forecast(
    city: str = "Lahore",
) -> dict:
    response = requests.get(
        f"{API_BASE_URL}/forecast",
        params={"city": city},
        timeout=60,
    )

    if not response.ok:
        raise DashboardAPIError(
            f"Forecast API failed: {response.status_code}"
        )

    return response.json()


def get_live_forecast(
    city: str = "Lahore",
) -> dict:
    refresh = refresh_features()
    forecast = get_forecast(
        city
    )

    return {
        **forecast,
        **refresh,
    }


def get_explanation(
    horizon: str,
    city: str = "Lahore",
    top_n: int = 5,
) -> dict:
    response = requests.get(
        f"{API_BASE_URL}/explain/{horizon}",
        params={
            "city": city,
            "top_n": top_n,
        },
        timeout=60,
    )

    if not response.ok:
        raise DashboardAPIError(
            f"Explanation API failed: {response.status_code}"
        )

    return response.json()
