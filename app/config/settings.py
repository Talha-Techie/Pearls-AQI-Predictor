from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    """

    OPENWEATHER_API_KEY: str = Field(..., description="OpenWeather API key")

    DEFAULT_CITY: str = Field(default="Lahore")

    LATITUDE: float = Field(default=31.5204)

    LONGITUDE: float = Field(default=74.3587)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
