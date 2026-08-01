from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.config.settings import get_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
settings = get_settings()


AQI_BINS = [
    -np.inf,
    50,
    100,
    150,
    200,
    300,
    np.inf,
]

AQI_LABELS = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]


CORRELATION_COLUMNS = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
]


TARGET_COLUMNS = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
]


def load_dataset(
    input_path: Path,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    df = pd.read_parquet(input_path)

    if df.empty:
        raise ValueError(
            "Processed dataset is empty."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


def add_analysis_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    df = dataframe.copy()

    df["aqi_category"] = pd.cut(
        df["us_aqi"],
        bins=AQI_BINS,
        labels=AQI_LABELS,
    )

    df["local_timestamp"] = (
        df["timestamp"]
        .dt.tz_convert(settings.timezone)
    )

    df["year_month"] = (
        df["local_timestamp"]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
    )

    return df


def save_aqi_timeseries(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    daily = (
        df.set_index("timestamp")["us_aqi"]
        .resample("D")
        .mean()
    )

    plt.figure(figsize=(14, 6))
    plt.plot(
        daily.index,
        daily.values,
        linewidth=0.9,
    )

    plt.title(
        "Daily Average AQI Over Time"
    )
    plt.xlabel("Date")
    plt.ylabel("US AQI")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "01_aqi_timeseries.png",
        dpi=160,
    )

    plt.close()


def save_aqi_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    plt.hist(
        df["us_aqi"],
        bins=40,
        edgecolor="black",
        alpha=0.8,
    )

    plt.title(
        "AQI Distribution"
    )
    plt.xlabel("US AQI")
    plt.ylabel("Frequency")
    plt.grid(
        axis="y",
        alpha=0.25,
    )
    plt.tight_layout()

    plt.savefig(
        output_dir / "02_aqi_distribution.png",
        dpi=160,
    )

    plt.close()


def save_monthly_aqi(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    monthly = (
        df.groupby(
            df["local_timestamp"].dt.month
        )["us_aqi"]
        .mean()
        .reindex(range(1, 13))
    )

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    plt.figure(figsize=(11, 6))

    plt.bar(
        month_names,
        monthly.values,
    )

    plt.title(
        "Average AQI by Month"
    )
    plt.xlabel("Month")
    plt.ylabel("Average US AQI")
    plt.grid(
        axis="y",
        alpha=0.25,
    )
    plt.tight_layout()

    plt.savefig(
        output_dir / "03_monthly_aqi.png",
        dpi=160,
    )

    plt.close()


def save_hourly_aqi(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    hourly = (
        df.groupby(
            df["local_timestamp"].dt.hour
        )["us_aqi"]
        .mean()
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        hourly.index,
        hourly.values,
        marker="o",
    )

    plt.xticks(range(24))

    plt.title(
        "Average AQI by Hour of Day"
    )
    plt.xlabel(f"Hour ({settings.timezone})")
    plt.ylabel("Average US AQI")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "04_hourly_aqi.png",
        dpi=160,
    )

    plt.close()


def save_aqi_categories(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    counts = (
        df["aqi_category"]
        .value_counts()
        .reindex(AQI_LABELS)
        .fillna(0)
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        counts.index,
        counts.values,
    )

    plt.title(
        "AQI Category Distribution"
    )
    plt.xlabel("AQI Category")
    plt.ylabel("Number of Observations")

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "05_aqi_categories.png",
        dpi=160,
    )

    plt.close()


def save_correlation_matrix(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    correlation = (
        df[CORRELATION_COLUMNS]
        .corr()
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    image = ax.imshow(
        correlation.values,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(
        np.arange(
            len(CORRELATION_COLUMNS)
        )
    )

    ax.set_yticks(
        np.arange(
            len(CORRELATION_COLUMNS)
        )
    )

    ax.set_xticklabels(
        CORRELATION_COLUMNS,
        rotation=60,
        ha="right",
    )

    ax.set_yticklabels(
        CORRELATION_COLUMNS,
    )

    for row in range(
        len(CORRELATION_COLUMNS)
    ):
        for column in range(
            len(CORRELATION_COLUMNS)
        ):
            ax.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_title(
        "Feature Correlation Matrix"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation",
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "06_correlation_matrix.png",
        dpi=160,
    )

    plt.close(fig)


def save_pm25_relationship(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    sample_size = min(
        5000,
        len(df),
    )

    sample = df.sample(
        n=sample_size,
        random_state=42,
    )

    plt.figure(figsize=(9, 6))

    plt.scatter(
        sample["pm2_5"],
        sample["us_aqi"],
        alpha=0.25,
        s=12,
    )

    plt.title(
        "PM2.5 vs AQI"
    )
    plt.xlabel("PM2.5")
    plt.ylabel("US AQI")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        output_dir
        / "07_pm25_vs_aqi.png",
        dpi=160,
    )

    plt.close()


def save_pm10_relationship(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    sample_size = min(
        5000,
        len(df),
    )

    sample = df.sample(
        n=sample_size,
        random_state=42,
    )

    plt.figure(figsize=(9, 6))

    plt.scatter(
        sample["pm10"],
        sample["us_aqi"],
        alpha=0.25,
        s=12,
    )

    plt.title(
        "PM10 vs AQI"
    )
    plt.xlabel("PM10")
    plt.ylabel("US AQI")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        output_dir
        / "08_pm10_vs_aqi.png",
        dpi=160,
    )

    plt.close()


def save_weather_relationships(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "pressure_msl",
    ]

    correlations = (
        df[variables + ["us_aqi"]]
        .corr()["us_aqi"]
        .drop("us_aqi")
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        correlations.index,
        correlations.values,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title(
        "Weather Feature Correlation with AQI"
    )

    plt.ylabel(
        "Pearson Correlation"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "09_weather_relationships.png",
        dpi=160,
    )

    plt.close()


def save_target_distributions(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(11, 6))

    for column in TARGET_COLUMNS:
        plt.hist(
            df[column],
            bins=40,
            alpha=0.4,
            label=column,
        )

    plt.title(
        "Forecast Target Distributions"
    )
    plt.xlabel("Future AQI")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.2,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "10_forecast_target_distributions.png",
        dpi=160,
    )

    plt.close()


def save_summary_tables(
    df: pd.DataFrame,
    report_dir: Path,
) -> None:
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptive = df[
        [
            "us_aqi",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
        ]
    ].describe().T

    descriptive.to_csv(
        report_dir
        / "eda_descriptive_statistics.csv"
    )

    category_distribution = (
        df["aqi_category"]
        .value_counts()
        .reindex(AQI_LABELS)
        .fillna(0)
        .to_frame(name="count")
    )

    category_distribution[
        "percentage"
    ] = (
        category_distribution["count"]
        / len(df)
        * 100
    )

    category_distribution.to_csv(
        report_dir
        / "eda_aqi_category_distribution.csv"
    )

    correlations = (
        df[CORRELATION_COLUMNS]
        .corr()["us_aqi"]
        .sort_values(
            ascending=False
        )
        .to_frame(
            name="correlation_with_aqi"
        )
    )

    correlations.to_csv(
        report_dir
        / "eda_aqi_correlations.csv"
    )


def run_eda(
    input_path: Path,
    output_dir: Path,
) -> None:
    print(
        f"\nLoading dataset:\n"
        f"{input_path.resolve()}"
    )

    df = load_dataset(
        input_path
    )

    df = add_analysis_columns(
        df
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Range: "
        f"{df['timestamp'].min()} -> "
        f"{df['timestamp'].max()}"
    )

    print(
        "\nGenerating EDA figures..."
    )

    save_aqi_timeseries(
        df,
        output_dir,
    )

    save_aqi_distribution(
        df,
        output_dir,
    )

    save_monthly_aqi(
        df,
        output_dir,
    )

    save_hourly_aqi(
        df,
        output_dir,
    )

    save_aqi_categories(
        df,
        output_dir,
    )

    save_correlation_matrix(
        df,
        output_dir,
    )

    save_pm25_relationship(
        df,
        output_dir,
    )

    save_pm10_relationship(
        df,
        output_dir,
    )

    save_weather_relationships(
        df,
        output_dir,
    )

    save_target_distributions(
        df,
        output_dir,
    )

    save_summary_tables(
        df,
        DEFAULT_REPORT_DIR,
    )

    print(
        "\nEDA completed successfully."
    )

    print(
        f"Figures saved to:\n"
        f"{output_dir.resolve()}"
    )

    print(
        "\nStrongest AQI correlations:"
    )

    print(
        df[CORRELATION_COLUMNS]
        .corr()["us_aqi"]
        .sort_values(
            ascending=False
        )
        .round(3)
        .to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run exploratory data analysis "
            "for AQI forecasting data."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Processed parquet dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for generated figures.",
    )

    args = parser.parse_args()

    run_eda(
        input_path=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
