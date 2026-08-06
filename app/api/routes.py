from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    ExplanationResponse,
    FeatureRefreshResponse,
    ForecastResponse,
)
from app.feature_pipeline.live_pipeline import (
    run_live_feature_pipeline,
)
from app.prediction.explain import (
    ExplanationServiceError,
    explain_horizon,
)
from app.prediction.service import (
    PredictionServiceError,
    predict_aqi,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["AQI Prediction"],
)


@router.get(
    "/forecast",
    response_model=ForecastResponse,
)
def get_forecast(
    city: str = Query(
        default="Lahore",
        min_length=1,
    ),
) -> ForecastResponse:
    """
    Return AQI forecasts using the latest
    feature vector already stored in Feast.
    """

    try:
        result = predict_aqi(city)

        return ForecastResponse(
            **result
        )

    except PredictionServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "AQI prediction service "
                "is temporarily unavailable."
            ),
        ) from exc


@router.post(
    "/features/refresh",
    response_model=FeatureRefreshResponse,
)
def refresh_features() -> FeatureRefreshResponse:
    """
    Fetch fresh weather/AQI data,
    engineer the latest 42 features,
    and push them to Feast.
    """

    try:
        latest = (
            run_live_feature_pipeline()
        )

        row = latest.iloc[0]

        return FeatureRefreshResponse(
            status="updated",
            city=str(row["city"]),
            timestamp=(
                row["timestamp"].isoformat()
            ),
            current_aqi=float(
                row["us_aqi"]
            ),
            pm2_5=float(
                row["pm2_5"]
            ),
            pm10=float(
                row["pm10"]
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to refresh live "
                "AQI features."
            ),
        ) from exc


@router.post(
    "/forecast/live",
    response_model=ForecastResponse,
)
def get_live_forecast(
    city: str = Query(
        default="Lahore",
        min_length=1,
    ),
) -> ForecastResponse:
    """
    Refresh live features first,
    then generate 24h/48h/72h forecasts.
    """

    try:
        run_live_feature_pipeline()

        result = predict_aqi(city)

        return ForecastResponse(
            **result
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to generate "
                "live AQI forecast."
            ),
        ) from exc


@router.get(
    "/explain/{horizon}",
    response_model=ExplanationResponse,
)
def explain_forecast(
    horizon: str,
    city: str = Query(
        default="Lahore",
        min_length=1,
    ),
    top_n: int = Query(
        default=8,
        ge=1,
        le=20,
    ),
) -> ExplanationResponse:

    try:
        result = explain_horizon(
            city=city,
            horizon=horizon,
            top_n=top_n,
        )

        return ExplanationResponse(
            **result
        )

    except ExplanationServiceError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to generate "
                "forecast explanation."
            ),
        ) from exc
