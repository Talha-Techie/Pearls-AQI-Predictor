import pandas as pd

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


SAMPLE_FORECAST = {
    "city": "Lahore",
    "current_aqi": 161.0,
    "forecast": {
        "24h": 147.7,
        "48h": 139.8,
        "72h": 135.3,
    },
    "feature_count": 42,
    "feature_source": (
        "Feast online store"
    ),
    "model_source": (
        "DagsHub MLflow Model Registry"
    ),
    "current_status": {
        "aqi": 161.0,
        "category": "Unhealthy",
        "alert_level": "warning",
        "health_guidance": "Test guidance",
        "alert": True,
    },
    "forecast_status": {
        "24h": {
            "aqi": 147.7,
            "category": (
                "Unhealthy for Sensitive Groups"
            ),
            "alert_level": "advisory",
            "health_guidance": "Test guidance",
            "alert": True,
        },
        "48h": {
            "aqi": 139.8,
            "category": (
                "Unhealthy for Sensitive Groups"
            ),
            "alert_level": "advisory",
            "health_guidance": "Test guidance",
            "alert": True,
        },
        "72h": {
            "aqi": 135.3,
            "category": (
                "Unhealthy for Sensitive Groups"
            ),
            "alert_level": "advisory",
            "health_guidance": "Test guidance",
            "alert": True,
        },
    },
    "hazard_alert": False,
}


def test_health_endpoint() -> None:
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert (
        response.json()["status"]
        == "healthy"
    )


def test_forecast_endpoint(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.api.routes.predict_aqi",
        lambda city: SAMPLE_FORECAST,
    )

    response = client.get(
        "/api/v1/forecast?city=Lahore"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["city"] == "Lahore"
    assert data["current_aqi"] == 161.0
    assert data["forecast"]["24h"] == 147.7


def test_refresh_endpoint(
    monkeypatch,
) -> None:
    latest = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(
                    "2026-08-29T17:00:00Z"
                ),
                "city": "Lahore",
                "us_aqi": 161,
                "pm2_5": 70.8,
                "pm10": 104.3,
            }
        ]
    )

    monkeypatch.setattr(
        "app.api.routes."
        "run_live_feature_pipeline",
        lambda: latest,
    )

    response = client.post(
        "/api/v1/features/refresh"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "updated"
    assert data["current_aqi"] == 161.0


def test_live_forecast_endpoint(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.api.routes."
        "run_live_feature_pipeline",
        lambda: None,
    )

    monkeypatch.setattr(
        "app.api.routes.predict_aqi",
        lambda city: SAMPLE_FORECAST,
    )

    response = client.post(
        "/api/v1/forecast/live?city=Lahore"
    )

    assert response.status_code == 200

    assert (
        response.json()["forecast"]["72h"]
        == 135.3
    )


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"