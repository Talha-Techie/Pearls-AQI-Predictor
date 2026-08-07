from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import pandas as pd
import streamlit as st

from app.dashboard.client import (
    DashboardAPIError,
    get_explanation,
    get_live_forecast,
)


METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_benchmarks"
    / "final_test_metrics.csv"
)


st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
)


def format_value(
    value: object,
    suffix: str = "",
) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.1f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def load_model_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        METRICS_PATH
    )


def show_status(
    title: str,
    status: dict,
) -> None:
    st.subheader(title)

    st.metric(
        "AQI",
        status["aqi"],
    )

    st.write(
        f"**Category:** {status['category']}"
    )

    st.write(
        f"**Alert Level:** "
        f"{status['alert_level'].title()}"
    )

    st.info(
        status["health_guidance"]
    )


def show_explanation(
    horizon: str,
    city: str,
) -> None:
    explanation = get_explanation(
        horizon=horizon,
        city=city,
        top_n=5,
    )

    df = pd.DataFrame(
        explanation["top_features"]
    )

    st.subheader(
        f"{horizon} Forecast Explanation"
    )

    st.metric(
        "Forecast AQI",
        explanation["prediction"],
    )

    chart_df = (
        df[
            [
                "feature",
                "contribution",
            ]
        ]
        .set_index("feature")
    )

    st.bar_chart(chart_df)

    df["direction"] = df["direction"].str.title()

    st.dataframe(
        df[
            [
                "feature",
                "value",
                "contribution",
                "direction",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.title(
        "🌍 Pearls AQI Predictor"
    )

    st.caption(
        "Real-time air quality monitoring "
        "with 24h, 48h and 72h ML forecasts."
    )

    city = st.sidebar.selectbox(
        "City",
        options=["Lahore"],
    )

    refresh = st.sidebar.button(
        "Refresh Live Forecast",
        type="primary",
    )

    if (
        "forecast_data"
        not in st.session_state
        or refresh
    ):
        try:
            with st.spinner(
                "Fetching live AQI data and "
                "generating forecasts..."
            ):
                st.session_state[
                    "forecast_data"
                ] = get_live_forecast(
                    city
                )

        except DashboardAPIError as exc:
            st.error(str(exc))
            st.stop()

        except Exception:
            st.error(
                "Unable to load AQI forecast. "
                "Make sure FastAPI is running."
            )
            st.stop()

    data = st.session_state[
        "forecast_data"
    ]

    st.subheader(
        f"Current Air Quality — {data['city']}"
    )

    current = data["current_status"]

    if "timestamp" in data:
        st.caption(
            f"Last updated: {data['timestamp']}"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current AQI",
            data["current_aqi"],
        )

    with col2:
        st.metric(
            "Category",
            current["category"],
        )

    with col3:
        st.metric(
            "Alert Level",
            current["alert_level"].title(),
        )

    if current["alert"]:
        st.warning(
            current["health_guidance"]
        )
    else:
        st.success(
            current["health_guidance"]
        )

    st.subheader(
        "Current Pollutants"
    )

    pollutant_columns = st.columns(6)

    pollutants = [
        (
            "PM2.5",
            "pm2_5",
            " µg/m³",
        ),
        (
            "PM10",
            "pm10",
            " µg/m³",
        ),
        (
            "O3",
            "ozone",
            " µg/m³",
        ),
        (
            "NO2",
            "nitrogen_dioxide",
            " µg/m³",
        ),
        (
            "CO",
            "carbon_monoxide",
            " µg/m³",
        ),
        (
            "SO2",
            "sulphur_dioxide",
            " µg/m³",
        ),
    ]

    for column, (
        label,
        key,
        suffix,
    ) in zip(
        pollutant_columns,
        pollutants,
    ):
        with column:
            st.metric(
                label,
                format_value(
                    data.get(key),
                    suffix,
                ),
            )

    st.divider()

    st.header(
        "3-Day AQI Forecast"
    )

    forecast_columns = st.columns(3)

    horizons = [
        "24h",
        "48h",
        "72h",
    ]

    for column, horizon in zip(
        forecast_columns,
        horizons,
    ):
        status = data[
            "forecast_status"
        ][horizon]

        with column:
            st.metric(
                f"+{horizon}",
                data["forecast"][
                    horizon
                ],
            )

            st.write(
                f"**{status['category']}**"
            )

            st.caption(
                status[
                    "health_guidance"
                ]
            )

    forecast_chart = pd.DataFrame(
        {
            "Period": [
                "Current",
                "+24h",
                "+48h",
                "+72h",
            ],
            "AQI": [
                data["current_aqi"],
                data["forecast"]["24h"],
                data["forecast"]["48h"],
                data["forecast"]["72h"],
            ],
        }
    )

    st.subheader(
        "AQI Forecast Trend"
    )

    st.line_chart(
        forecast_chart.set_index(
            "Period"
        )
    )

    if data["hazard_alert"]:
        st.error(
            "⚠️ Hazardous or very unhealthy "
            "air quality is forecast."
        )

    st.divider()

    st.header(
        "Forecast Explainability"
    )

    selected_horizon = st.selectbox(
        "Select forecast horizon",
        horizons,
    )

    try:
        show_explanation(
            selected_horizon,
            city,
        )

    except Exception as exc:
        st.warning(
            f"Explanation unavailable: {exc}"
        )

    st.divider()

    with st.expander(
        "Model & Pipeline Information"
    ):
        st.write(
            "**Model:**",
            "Ridge Regression",
        )

        st.write(
            "**Features:**",
            data["feature_count"],
        )

        st.write(
            "**Feature Store:**",
            data["feature_source"],
        )

        st.write(
            "**Model Registry:**",
            data["model_source"],
        )

    st.subheader(
        "Model Performance"
    )

    metrics = load_model_metrics()

    if metrics.empty:
        st.info(
            "Model performance metrics are not available."
        )
    else:
        display_metrics = metrics[
            [
                "horizon",
                "mae",
                "rmse",
                "r2",
            ]
        ].copy()

        display_metrics = display_metrics.rename(
            columns={
                "horizon": "Horizon",
                "mae": "MAE",
                "rmse": "RMSE",
                "r2": "R²",
            }
        )

        st.dataframe(
            display_metrics,
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
