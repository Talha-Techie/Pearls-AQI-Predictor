from __future__ import annotations

from pydantic import BaseModel


class AQIStatus(BaseModel):
    aqi: float
    category: str
    alert_level: str
    health_guidance: str
    alert: bool


class ForecastResponse(BaseModel):
    city: str
    current_aqi: float
    forecast: dict[str, float]

    feature_count: int
    feature_source: str
    model_source: str

    current_status: AQIStatus
    forecast_status: dict[str, AQIStatus]

    hazard_alert: bool


class FeatureRefreshResponse(BaseModel):
    status: str
    city: str
    timestamp: str
    current_aqi: float
    pm2_5: float
    pm10: float


class HealthResponse(BaseModel):
    status: str
    service: str
