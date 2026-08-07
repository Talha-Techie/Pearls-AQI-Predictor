from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

LOCAL_TIMEZONE = ZoneInfo("Asia/Karachi")


st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(46, 160, 130, 0.08),
                transparent 30%
            ),
            #0e1117;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .hero {
        padding: 24px 28px;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        background:
            linear-gradient(
                135deg,
                rgba(25, 35, 50, 0.95),
                rgba(13, 21, 30, 0.95)
            );
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        color: #9da8b6;
        font-size: 15px;
    }

    .metric-card {
        background: rgba(20, 27, 38, 0.95);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px;
        min-height: 155px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.18);
    }

    .metric-label {
        color: #8d98a8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-value {
        font-size: 36px;
        font-weight: 750;
        margin-top: 7px;
    }

    .metric-category {
        margin-top: 8px;
        font-size: 14px;
        font-weight: 600;
    }

    .live-pill {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        background: rgba(34,197,94,0.12);
        color: #72e49a;
        border: 1px solid rgba(34,197,94,0.30);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .04em;
    }

    .info-card {
        padding: 18px 22px;
        border-radius: 15px;
        background: rgba(30, 39, 52, 0.82);
        border: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 44px;
    }

    button[data-baseweb="tab"] {
        font-weight: 650;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
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


def format_updated(
    updated: datetime | None,
) -> str:
    if updated is None:
        return "Not refreshed yet"

    return updated.strftime(
        "%d %b %Y • %I:%M %p"
    )


def load_model_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        return pd.DataFrame()

    metrics = pd.read_csv(
        METRICS_PATH
    )

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

    display_metrics["MAE"] = display_metrics[
        "MAE"
    ].round(2)
    display_metrics["RMSE"] = display_metrics[
        "RMSE"
    ].round(2)
    display_metrics["R²"] = display_metrics[
        "R²"
    ].round(3)

    return display_metrics


def metric_card(
    label: str,
    value: float,
    category: str,
) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {label}
            </div>
            <div class="metric-value">
                {value:.1f}
            </div>
            <div class="metric-category">
                {category}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pollutant_card(
    label: str,
    value: object,
    suffix: str,
) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="metric-label">
                {label}
            </div>
            <div style="font-size:24px;font-weight:700;margin-top:6px;">
                {format_value(value, suffix)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_aqi_gauge(
    current_aqi: float,
) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current_aqi,
            number={
                "font": {
                    "size": 44,
                }
            },
            title={
                "text": "Current AQI",
            },
            gauge={
                "axis": {
                    "range": [0, 500],
                    "tickwidth": 1,
                },
                "bar": {
                    "thickness": 0.25,
                },
                "steps": [
                    {
                        "range": [0, 50],
                        "color": "#2b8750",
                    },
                    {
                        "range": [50, 100],
                        "color": "#a08a2e",
                    },
                    {
                        "range": [100, 150],
                        "color": "#a5682e",
                    },
                    {
                        "range": [150, 200],
                        "color": "#993d42",
                    },
                    {
                        "range": [200, 300],
                        "color": "#71418c",
                    },
                    {
                        "range": [300, 500],
                        "color": "#662f3c",
                    },
                ],
            },
        )
    )

    figure.update_layout(
        height=310,
        margin=dict(
            l=30,
            r=30,
            t=55,
            b=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "white",
        },
    )

    return figure


def forecast_chart(
    data: dict,
) -> go.Figure:
    forecast_df = pd.DataFrame(
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

    figure = px.line(
        forecast_df,
        x="Period",
        y="AQI",
        markers=True,
    )

    figure.update_traces(
        line={
            "width": 4,
        },
        marker={
            "size": 10,
        },
    )

    figure.update_layout(
        title="AQI Forecast Trend",
        height=310,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "white",
        },
        yaxis_title="AQI",
        xaxis_title=None,
    )

    return figure


def shap_chart(
    explanation: dict,
) -> go.Figure:
    dataframe = pd.DataFrame(
        explanation["top_features"]
    )

    dataframe = dataframe.sort_values(
        "contribution"
    )

    figure = px.bar(
        dataframe,
        x="contribution",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={
            "increase": "#ef6262",
            "decrease": "#52b788",
        },
    )

    figure.update_layout(
        title=(
            f"{explanation['horizon']} "
            "Forecast Feature Contributions"
        ),
        xaxis_title="SHAP contribution",
        yaxis_title=None,
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "white",
        },
        legend_title=None,
    )

    return figure


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

    st.plotly_chart(
        shap_chart(explanation),
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    display_df = df.copy()
    display_df["direction"] = display_df[
        "direction"
    ].str.title()

    st.dataframe(
        display_df[
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


def render_sidebar() -> tuple[str, bool]:
    with st.sidebar:
        st.markdown(
            "## 🌍 AQI Predictor"
        )

        st.caption(
            "Air-quality intelligence platform"
        )

        st.divider()

        city = st.selectbox(
            "Monitoring Location",
            ["Lahore"],
        )

        refresh = st.button(
            "↻ Refresh Live Data",
            use_container_width=True,
            type="primary",
        )

        st.divider()

        st.markdown(
            "### System Status"
        )

        st.success(
            "● API Online"
        )

        st.success(
            "● Feature Store Online"
        )

        st.success(
            "● Models Available"
        )

        st.divider()

        st.caption(
            "Forecast horizons"
        )

        st.write(
            "24h • 48h • 72h"
        )

    return city, refresh


def render_hero(
    city: str,
    updated: datetime | None,
) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:20px;
            ">
                <div>
                    <div class="hero-title">
                        🌍 Pearls AQI Predictor
                    </div>
                    <div class="hero-subtitle">
                        Real-time air quality intelligence,
                        ML forecasting and explainable predictions
                    </div>
                    <div class="hero-subtitle" style="margin-top:8px;">
                        {city} • Last updated {format_updated(updated)}
                    </div>
                </div>

                <div class="live-pill">
                    ● LIVE
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(
    data: dict,
) -> None:
    cols = st.columns(4)

    with cols[0]:
        metric_card(
            "Current AQI",
            float(data["current_aqi"]),
            data["current_status"]["category"],
        )

    with cols[1]:
        metric_card(
            "+24 Hours",
            float(data["forecast"]["24h"]),
            data["forecast_status"]["24h"]["category"],
        )

    with cols[2]:
        metric_card(
            "+48 Hours",
            float(data["forecast"]["48h"]),
            data["forecast_status"]["48h"]["category"],
        )

    with cols[3]:
        metric_card(
            "+72 Hours",
            float(data["forecast"]["72h"]),
            data["forecast_status"]["72h"]["category"],
        )


def render_pollutants(
    data: dict,
) -> None:
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
            pollutant_card(
                label,
                data.get(key),
                suffix,
            )


def render_model_performance() -> None:
    metrics = load_model_metrics()

    if metrics.empty:
        st.info(
            "Model performance metrics are not available."
        )
        return

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    city, refresh = render_sidebar()

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

                st.session_state[
                    "last_updated"
                ] = datetime.now(
                    LOCAL_TIMEZONE
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

    updated = st.session_state.get(
        "last_updated"
    )

    render_hero(
        city=data["city"],
        updated=updated,
    )

    render_kpi_cards(data)

    left, right = st.columns(
        [
            0.38,
            0.62,
        ]
    )

    with left:
        st.plotly_chart(
            build_aqi_gauge(
                float(data["current_aqi"])
            ),
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

    with right:
        st.plotly_chart(
            forecast_chart(data),
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

    overview_tab, explain_tab, model_tab, system_tab = st.tabs(
        [
            "📊 Overview",
            "🧠 Explainability",
            "📈 Model Performance",
            "⚙️ System",
        ]
    )

    horizons = [
        "24h",
        "48h",
        "72h",
    ]

    with overview_tab:
        st.subheader(
            "Health Advisory"
        )

        status = data[
            "current_status"
        ]

        if status["alert"]:
            st.warning(
                status["health_guidance"]
            )
        else:
            st.success(
                status["health_guidance"]
            )

        if data["hazard_alert"]:
            st.error(
                "Hazardous or very unhealthy "
                "air quality is forecast."
            )

        render_pollutants(data)

    with explain_tab:
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

    with model_tab:
        st.subheader(
            "Model Performance"
        )

        render_model_performance()

        st.markdown(
            """
            <div class="info-card">
                <strong>Model:</strong> Ridge Regression<br>
                <strong>Forecast horizons:</strong> 24h, 48h, 72h<br>
                <strong>Evaluation split:</strong> leakage-safe temporal test set
            </div>
            """,
            unsafe_allow_html=True,
        )

    with system_tab:
        st.subheader(
            "Prediction Pipeline"
        )

        st.markdown(
            """
            **Live APIs**
            → **Feature Engineering**
            → **Feast**
            → **MLflow Model Registry**
            → **Ridge Models**
            → **FastAPI**
            → **Streamlit**

            ---
            """
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Features",
            str(data["feature_count"]),
        )

        c2.metric(
            "Forecast Models",
            "3",
        )

        c3.metric(
            "Feature Store",
            "Feast",
        )

        c4.metric(
            "Registry",
            "MLflow",
        )

        st.markdown(
            f"""
            <div class="info-card">
                <strong>Feature source:</strong> {data["feature_source"]}<br>
                <strong>Model source:</strong> {data["model_source"]}<br>
                <strong>Serving API:</strong> FastAPI<br>
                <strong>Dashboard:</strong> Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
