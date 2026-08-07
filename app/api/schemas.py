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
    ozone: float | None = None
    nitrogen_dioxide: float | None = None
    carbon_monoxide: float | None = None
    sulphur_dioxide: float | None = None


class HealthResponse(BaseModel):
    status: str
    service: str


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str


class ExplanationResponse(BaseModel):
    city: str
    horizon: str
    prediction: float
    base_value: float
    top_features: list[
        FeatureContribution
    ]
    feature_count: int
    method: str
