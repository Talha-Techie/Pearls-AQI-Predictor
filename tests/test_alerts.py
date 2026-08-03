from app.prediction.alerts import (
    build_aqi_status,
    enrich_prediction,
    get_aqi_category,
)


def test_aqi_categories() -> None:
    assert get_aqi_category(25) == "Good"
    assert get_aqi_category(75) == "Moderate"

    assert (
        get_aqi_category(125)
        == "Unhealthy for Sensitive Groups"
    )

    assert get_aqi_category(175) == "Unhealthy"

    assert (
        get_aqi_category(250)
        == "Very Unhealthy"
    )

    assert get_aqi_category(350) == "Hazardous"


def test_boundary_values() -> None:
    assert get_aqi_category(50) == "Good"
    assert get_aqi_category(100) == "Moderate"

    assert (
        get_aqi_category(150)
        == "Unhealthy for Sensitive Groups"
    )

    assert get_aqi_category(200) == "Unhealthy"

    assert (
        get_aqi_category(300)
        == "Very Unhealthy"
    )

    assert get_aqi_category(301) == "Hazardous"


def test_high_aqi_generates_alert() -> None:
    status = build_aqi_status(220)

    assert status["alert"] is True
    assert status["alert_level"] == "high"


def test_prediction_is_enriched() -> None:
    prediction = {
        "city": "Lahore",
        "current_aqi": 161.0,
        "forecast": {
            "24h": 147.7,
            "48h": 139.8,
            "72h": 135.3,
        },
    }

    result = enrich_prediction(
        prediction
    )

    assert (
        result["current_status"]["category"]
        == "Unhealthy"
    )

    assert (
        result["forecast_status"]["24h"]["category"]
        == "Unhealthy for Sensitive Groups"
    )

    assert result["hazard_alert"] is False