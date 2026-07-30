from app.config.settings import Settings


def test_settings_can_be_created() -> None:
    settings = Settings(
        city="Lahore",
        latitude=31.5204,
        longitude=74.3587,
        timezone="Asia/Karachi",
    )

    assert settings.app_name == "Pearls AQI Predictor"
    assert settings.app_env == "development"

    assert settings.city == "Lahore"
    assert settings.latitude == 31.5204
    assert settings.longitude == 74.3587
    assert settings.timezone == "Asia/Karachi"

    assert settings.weather_api_url.startswith(
        "https://"
    )

    assert settings.air_quality_api_url.startswith(
        "https://"
    )

    assert settings.historical_weather_api_url.startswith(
        "https://"
    )

    assert settings.http_timeout_seconds > 0