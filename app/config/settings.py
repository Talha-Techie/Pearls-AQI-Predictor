from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Pearls AQI Predictor"
    app_env: str = "development"

    city: str
    latitude: float
    longitude: float
    timezone: str = "UTC"

    weather_api_url: str = "https://api.open-meteo.com/v1/forecast"
    air_quality_api_url: str = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    http_timeout_seconds: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()