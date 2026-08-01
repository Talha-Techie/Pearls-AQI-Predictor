"""
Feast feature definitions for the Pearls AQI Predictor.

This repository exposes inference-safe AQI, pollutant,
weather, temporal, lag, rolling, and trend features.

Forecast labels are intentionally excluded from the
online feature view.
"""
from datetime import timedelta

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
)
from feast.value_type import ValueType

from feast.types import (
    Float64,
    Int64,
)


# ------------------------------------------------------------------
# Entity
# ------------------------------------------------------------------

city = Entity(
    name="city",
    join_keys=["city"],
    value_type=ValueType.STRING,
    description=(
        "City for which AQI features are generated."
    ),
)


# ------------------------------------------------------------------
# Offline batch source
# ------------------------------------------------------------------

aqi_features_source = FileSource(
    name="aqi_features_source",
    path="data/aqi_features.parquet",
    timestamp_field="timestamp",
)


# ------------------------------------------------------------------
# Feature View
# ------------------------------------------------------------------

aqi_hourly_features = FeatureView(
    name="aqi_hourly_features",
    entities=[city],
    ttl=timedelta(days=3650),
    schema=[
        # Weather
        Field(
            name="temperature_2m",
            dtype=Float64,
        ),
        Field(
            name="relative_humidity_2m",
            dtype=Int64,
        ),
        Field(
            name="precipitation",
            dtype=Float64,
        ),
        Field(
            name="pressure_msl",
            dtype=Float64,
        ),
        Field(
            name="wind_speed_10m",
            dtype=Float64,
        ),

        # Circular wind direction
        Field(
            name="wind_direction_sin",
            dtype=Float64,
        ),
        Field(
            name="wind_direction_cos",
            dtype=Float64,
        ),

        # Pollutants
        Field(
            name="pm10",
            dtype=Float64,
        ),
        Field(
            name="pm2_5",
            dtype=Float64,
        ),
        Field(
            name="carbon_monoxide",
            dtype=Float64,
        ),
        Field(
            name="nitrogen_dioxide",
            dtype=Float64,
        ),
        Field(
            name="sulphur_dioxide",
            dtype=Float64,
        ),
        Field(
            name="ozone",
            dtype=Float64,
        ),

        # Current AQI
        Field(
            name="us_aqi",
            dtype=Int64,
        ),

        # Calendar
        Field(
            name="hour",
            dtype=Int64,
        ),
        Field(
            name="day_of_week",
            dtype=Int64,
        ),
        Field(
            name="month",
            dtype=Int64,
        ),
        Field(
            name="is_weekend",
            dtype=Int64,
        ),

        # Cyclical time
        Field(
            name="hour_sin",
            dtype=Float64,
        ),
        Field(
            name="hour_cos",
            dtype=Float64,
        ),
        Field(
            name="day_of_week_sin",
            dtype=Float64,
        ),
        Field(
            name="day_of_week_cos",
            dtype=Float64,
        ),
        Field(
            name="month_sin",
            dtype=Float64,
        ),
        Field(
            name="month_cos",
            dtype=Float64,
        ),

        # AQI lags
        Field(
            name="aqi_lag_1h",
            dtype=Float64,
        ),
        Field(
            name="aqi_lag_3h",
            dtype=Float64,
        ),
        Field(
            name="aqi_lag_6h",
            dtype=Float64,
        ),
        Field(
            name="aqi_lag_12h",
            dtype=Float64,
        ),
        Field(
            name="aqi_lag_24h",
            dtype=Float64,
        ),

        # Pollution lags
        Field(
            name="pm2_5_lag_1h",
            dtype=Float64,
        ),
        Field(
            name="pm2_5_lag_6h",
            dtype=Float64,
        ),
        Field(
            name="pm2_5_lag_24h",
            dtype=Float64,
        ),
        Field(
            name="pm10_lag_24h",
            dtype=Float64,
        ),

        # AQI rolling statistics
        Field(
            name="aqi_rolling_mean_6h",
            dtype=Float64,
        ),
        Field(
            name="aqi_rolling_mean_12h",
            dtype=Float64,
        ),
        Field(
            name="aqi_rolling_mean_24h",
            dtype=Float64,
        ),
        Field(
            name="aqi_rolling_std_24h",
            dtype=Float64,
        ),

        # Pollutant rolling statistics
        Field(
            name="pm2_5_rolling_mean_6h",
            dtype=Float64,
        ),
        Field(
            name="pm2_5_rolling_mean_24h",
            dtype=Float64,
        ),
        Field(
            name="pm10_rolling_mean_24h",
            dtype=Float64,
        ),

        # AQI trend
        Field(
            name="aqi_change_1h",
            dtype=Float64,
        ),
        Field(
            name="aqi_change_rate_1h",
            dtype=Float64,
        ),
    ],
    source=aqi_features_source,
    online=True,
    description=(
        "Inference-safe hourly weather, pollution, "
        "temporal, lag and rolling AQI features."
    ),
)


# ------------------------------------------------------------------
# Feature Service
# ------------------------------------------------------------------

aqi_prediction_features_v1 = FeatureService(
    name="aqi_prediction_features_v1",
    features=[
        aqi_hourly_features,
    ],
)
