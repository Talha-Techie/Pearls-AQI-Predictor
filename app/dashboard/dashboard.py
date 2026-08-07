from __future__ import annotations

import sys
import textwrap
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

# AQI category -> (color, glow color, label) — used everywhere so the
# whole UI agrees on what "unhealthy" looks like.
AQI_BANDS = [
    (0, 50, "#22c55e", "Good"),
    (50, 100, "#eab308", "Moderate"),
    (100, 150, "#f97316", "Unhealthy (Sensitive)"),
    (150, 200, "#ef4444", "Unhealthy"),
    (200, 300, "#a855f7", "Very Unhealthy"),
    (300, 500, "#7f1d1d", "Hazardous"),
]


def aqi_color(value: float) -> str:
    for lo, hi, color, _ in AQI_BANDS:
        if lo <= value < hi:
            return color
    return AQI_BANDS[-1][2]


st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
)


st.html(
"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>

:root {
    --bg-0: #05070a;
    --bg-1: #0a0e14;
    --panel: rgba(18, 24, 34, 0.72);
    --panel-solid: rgba(15, 20, 29, 0.96);
    --border: rgba(148, 163, 184, 0.14);
    --border-hover: rgba(148, 163, 184, 0.28);
    --text-primary: #eef1f6;
    --text-muted: #8891a3;
    --accent: #34d8b0;
    --accent-soft: rgba(52, 216, 176, 0.14);
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', -apple-system, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(52,216,176,0.10), transparent 38%),
        radial-gradient(circle at 92% 12%, rgba(124,92,255,0.10), transparent 42%),
        radial-gradient(circle at 50% 100%, rgba(52,216,176,0.05), transparent 55%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
    background-attachment: fixed;
}

.block-container {
    max-width: 1440px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

/* ---------- animated background grid + noise ---------- */
.grid-overlay {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(148,163,184,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148,163,184,0.035) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse 80% 60% at 50% 0%, black 20%, transparent 75%);
}

/* ---------- hero ---------- */
.hero {
    position: relative;
    padding: 30px 34px;
    border: 1px solid var(--border);
    border-radius: 22px;
    background:
        linear-gradient(135deg, rgba(30,41,56,0.92), rgba(10,14,20,0.95));
    margin-bottom: 26px;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}

.hero::before {
    content: "";
    position: absolute;
    top: -60%;
    right: -10%;
    width: 420px;
    height: 420px;
    background: radial-gradient(circle, rgba(52,216,176,0.22), transparent 70%);
    filter: blur(10px);
    animation: drift 12s ease-in-out infinite alternate;
}

@keyframes drift {
    from { transform: translate(0,0) scale(1); }
    to   { transform: translate(-30px, 40px) scale(1.15); }
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid rgba(52,216,176,0.25);
    padding: 4px 10px;
    border-radius: 999px;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
    background: linear-gradient(90deg, #ffffff, #b9c4d4);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: var(--text-muted);
    font-size: 15px;
    max-width: 560px;
    line-height: 1.5;
}

.hero-meta {
    color: #6b7688;
    font-size: 13px;
    margin-top: 14px;
    font-family: 'JetBrains Mono', monospace;
}

.live-pill {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(34,197,94,0.10);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.28);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .06em;
    font-family: 'JetBrains Mono', monospace;
}

.live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 0 0 rgba(74,222,128,0.6);
    animation: pulse-dot 1.8s infinite;
}

@keyframes pulse-dot {
    0%   { box-shadow: 0 0 0 0 rgba(74,222,128,0.55); }
    70%  { box-shadow: 0 0 0 9px rgba(74,222,128,0); }
    100% { box-shadow: 0 0 0 0 rgba(74,222,128,0); }
}

/* ---------- KPI cards ---------- */
.metric-card {
    position: relative;
    background: var(--panel);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px 22px 20px;
    min-height: 158px;
    box-shadow: 0 12px 34px rgba(0,0,0,0.22);
    transition: transform .18s ease, border-color .18s ease;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--border-hover);
}

.metric-card::after {
    content: "";
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 3px;
    background: var(--bar-color, var(--accent));
    opacity: 0.85;
}

.metric-label {
    color: var(--text-muted);
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}

.metric-value {
    font-size: 40px;
    font-weight: 700;
    margin-top: 8px;
    letter-spacing: -0.02em;
    color: var(--text-primary);
}

.metric-category {
    margin-top: 10px;
    display: inline-block;
    font-size: 12.5px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
}

/* ---------- pollutant chips ---------- */
.info-card {
    padding: 18px 18px;
    border-radius: 16px;
    background: var(--panel);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    transition: border-color .18s ease, transform .18s ease;
}

.info-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-2px);
}

/* ---------- generic surfaces ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,14,20,0.98), rgba(6,9,13,0.98));
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    font-family: 'Space Grotesk', sans-serif;
}

.stButton > button {
    border-radius: 11px;
    font-weight: 650;
    min-height: 44px;
    border: 1px solid var(--border);
    transition: all .15s ease;
}

.stButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
}

button[data-baseweb="tab"] {
    font-weight: 650;
    font-family: 'Space Grotesk', sans-serif;
}

div[data-testid="stVerticalBlock"] { gap: 0.8rem; }

div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
}

/* subtle divider */
hr, .stDivider {
    border-color: var(--border) !important;
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,0.18);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.32); }

</style>
<div class="grid-overlay"></div>
    """
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
    color = aqi_color(value)
    st.html(
        textwrap.dedent(
            f"""
            <div class="metric-card" style="--bar-color:{color};">
                <div class="metric-label">
                    {label}
                </div>
                <div class="metric-value">
                    {value:.1f}
                </div>
                <div class="metric-category" style="background:{color}22; color:{color}; border:1px solid {color}55;">
                    {category}
                </div>
            </div>
            """
        )
    )


def pollutant_card(
    label: str,
    value: object,
    suffix: str,
) -> None:
    st.html(
        textwrap.dedent(
            f"""
            <div class="info-card">
                <div class="metric-label">
                    {label}
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:23px;font-weight:600;margin-top:8px;color:#eef1f6;">
                    {format_value(value, suffix)}
                </div>
            </div>
            """
        )
    )


def build_aqi_gauge(
    current_aqi: float,
) -> go.Figure:
    needle_color = aqi_color(current_aqi)

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current_aqi,
            number={
                "font": {
                    "size": 46,
                    "family": "JetBrains Mono, monospace",
                    "color": needle_color,
                },
            },
            title={
                "text": "CURRENT AQI",
                "font": {
                    "size": 13,
                    "family": "JetBrains Mono, monospace",
                    "color": "#8891a3",
                },
            },
            gauge={
                "axis": {
                    "range": [0, 500],
                    "tickwidth": 1,
                    "tickcolor": "rgba(255,255,255,0.25)",
                    "tickfont": {"color": "#6b7688", "size": 10},
                },
                "bar": {
                    "color": needle_color,
                    "thickness": 0.28,
                },
                "bgcolor": "rgba(255,255,255,0.02)",
                "borderwidth": 0,
                "steps": [
                    {"range": [lo, hi], "color": f"{color}33"}
                    for lo, hi, color, _ in AQI_BANDS
                ],
            },
        )
    )

    figure.update_layout(
        height=320,
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "white",
            "family": "Space Grotesk, sans-serif",
        },
    )

    return figure


def forecast_chart(
    data: dict,
) -> go.Figure:
    periods = ["Current", "+24h", "+48h", "+72h"]
    values = [
        data["current_aqi"],
        data["forecast"]["24h"],
        data["forecast"]["48h"],
        data["forecast"]["72h"],
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=periods,
            y=values,
            mode="lines+markers",
            line=dict(
                width=3,
                color="#34d8b0",
                shape="spline",
            ),
            marker=dict(
                size=12,
                color=[aqi_color(v) for v in values],
                line=dict(width=2, color="#05070a"),
            ),
            fill="tozeroy",
            fillcolor="rgba(52,216,176,0.08)",
            hovertemplate="%{x}: <b>%{y:.0f}</b> AQI<extra></extra>",
        )
    )

    figure.update_layout(
        title={
            "text": "AQI FORECAST TREND",
            "font": {"size": 13, "family": "JetBrains Mono, monospace", "color": "#8891a3"},
        },
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Space Grotesk, sans-serif"},
        yaxis_title="AQI",
        xaxis_title=None,
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
        showlegend=False,
        hovermode="x unified",
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
            "increase": "#ef4444",
            "decrease": "#34d8b0",
        },
    )

    figure.update_traces(
        marker_line_width=0,
        opacity=0.9,
    )

    figure.update_layout(
        title=(
            f"{explanation['horizon']} "
            "FORECAST FEATURE CONTRIBUTIONS"
        ).upper(),
        title_font={"size": 13, "family": "JetBrains Mono, monospace", "color": "#8891a3"},
        xaxis_title="SHAP contribution",
        yaxis_title=None,
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "white",
            "family": "Space Grotesk, sans-serif",
        },
        legend_title=None,
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
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
        width='stretch',
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
        width='stretch',
        hide_index=True,
    )


def render_sidebar() -> tuple[str, bool]:
    with st.sidebar:
        st.html(
            textwrap.dedent(
                """
                <div style="padding: 4px 0 14px;">
                    <div style="font-size:22px;font-weight:700;letter-spacing:-0.01em;">
                        🌍 AQI Predictor
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                                color:#6b7688;letter-spacing:0.06em;margin-top:2px;">
                        v2.0 · PEARLS
                    </div>
                </div>
                """
            )
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
            width='stretch',
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
    st.html(
        textwrap.dedent(
            f"""
            <div class="hero">
                <div style="
                    position:relative; z-index:1;
                    display:flex;
                    justify-content:space-between;
                    align-items:flex-start;
                    gap:20px;
                    flex-wrap: wrap;
                ">
                    <div>
                        <div class="eyebrow">● machine learning · live inference</div>
                        <div class="hero-title">
                            Pearls AQI Predictor
                        </div>
                        <div class="hero-subtitle">
                            Real-time air quality intelligence with ML forecasting
                            and explainable, feature-level predictions.
                        </div>
                        <div class="hero-meta">
                            {city.upper()} &nbsp;/&nbsp; LAST UPDATED {format_updated(updated).upper()}
                        </div>
                    </div>

                    <div class="live-pill">
                        <span class="live-dot"></span> LIVE
                    </div>
                </div>
            </div>
            """
        )
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
        width='stretch',
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

    st.write("")

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
            width='stretch',
            config={
                "displayModeBar": False,
            },
        )

    with right:
        st.plotly_chart(
            forecast_chart(data),
            width='stretch',
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

        st.html(
            textwrap.dedent(
                """
                <div class="info-card">
                    <strong>Model:</strong> Ridge Regression<br>
                    <strong>Forecast horizons:</strong> 24h, 48h, 72h<br>
                    <strong>Evaluation split:</strong> leakage-safe temporal test set
                </div>
                """
            )
        )

    with system_tab:
        st.subheader(
            "Prediction Pipeline"
        )

        st.markdown(
            textwrap.dedent(
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

        st.html(
            textwrap.dedent(
                f"""
                <div class="info-card">
                    <strong>Feature source:</strong> {data["feature_source"]}<br>
                    <strong>Model source:</strong> {data["model_source"]}<br>
                    <strong>Serving API:</strong> FastAPI<br>
                    <strong>Dashboard:</strong> Streamlit
                </div>
                """
            )
        )


if __name__ == "__main__":
    main()