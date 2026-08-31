from __future__ import annotations

from functools import lru_cache
import numpy as np
import pandas as pd
import shap

from app.feature_pipeline.engineer import MODEL_FEATURE_COLUMNS
from app.prediction.service import (
    get_online_feature_vector,
    load_registered_model,
)


# PROJECT_ROOT = Path(__file__).resolve().parents[2]

# BACKGROUND_DATASET = (
#     PROJECT_ROOT
#     / "data"
#     / "processed"
#     / "features_aqi_history_2023-01-01_2026-07-29.parquet"
# )


class ExplanationServiceError(RuntimeError):
    """Raised when a model explanation cannot be generated."""


def _get_pipeline_step(
    pipeline,
    names: tuple[str, ...],
):
    for name in names:
        if name in pipeline.named_steps:
            return pipeline.named_steps[name]

    raise KeyError(names)


# @lru_cache(maxsize=1)
# def load_background_data() -> pd.DataFrame:
#     dataframe = pd.read_parquet(
#         BACKGROUND_DATASET,
#         columns=MODEL_FEATURE_COLUMNS,
#     )

    # if dataframe.empty:
    #     raise ExplanationServiceError(
    #         "SHAP background dataset is empty."
    #     )

    # # Small representative sample keeps explanation fast.
    # sample_size = min(
    #     500,
    #     len(dataframe),
    # )

    # return dataframe.sample(
    #     n=sample_size,
    #     random_state=42,
    # )


def explain_horizon(
    city: str,
    horizon: str,
    top_n: int = 8,
) -> dict:

    if horizon not in {
        "24h",
        "48h",
        "72h",
    }:
        raise ExplanationServiceError(
            f"Unsupported horizon: {horizon}"
        )

    features = get_online_feature_vector(
        city
    )

    pipeline = load_registered_model(
        horizon
    )

    if not hasattr(
        pipeline,
        "named_steps",
    ):
        raise ExplanationServiceError(
            "Registered model is not an sklearn pipeline."
        )

    try:
        scaler = _get_pipeline_step(
            pipeline,
            (
                "scaler",
                "standardscaler",
            ),
        )

        ridge = _get_pipeline_step(
            pipeline,
            (
                "model",
                "ridge",
            ),
        )

    except KeyError as exc:
        raise ExplanationServiceError(
            "Expected StandardScaler + Ridge pipeline."
        ) from exc

    background_scaled = np.zeros(
    (
        1,
        len(MODEL_FEATURE_COLUMNS),
    ),
    dtype=float,
)

    current_scaled = scaler.transform(
        features
    )

    explainer = shap.LinearExplainer(
        ridge,
        background_scaled,
    )

    shap_values = explainer(
        current_scaled
    )

    contributions = []

    values = shap_values.values[0]

    for feature, feature_value, shap_value in zip(
        MODEL_FEATURE_COLUMNS,
        features.iloc[0].tolist(),
        values,
    ):
        contributions.append(
            {
                "feature": feature,
                "value": round(
                    float(feature_value),
                    4,
                ),
                "contribution": round(
                    float(shap_value),
                    4,
                ),
                "direction": (
                    "increase"
                    if shap_value >= 0
                    else "decrease"
                ),
            }
        )

    contributions.sort(
        key=lambda item: abs(
            item["contribution"]
        ),
        reverse=True,
    )

    prediction = float(
        pipeline.predict(features)[0]
    )

    return {
        "city": city,
        "horizon": horizon,
        "prediction": round(
            max(0.0, prediction),
            1,
        ),
        "base_value": round(
            float(
                np.asarray(
                    shap_values.base_values
                ).reshape(-1)[0]
            ),
            4,
        ),
        "top_features": (
            contributions[:top_n]
        ),
        "feature_count": len(
            MODEL_FEATURE_COLUMNS
        ),
        "method": (
            "SHAP LinearExplainer"
        ),
    }


def explain_all_horizons(
    city: str = "Lahore",
    top_n: int = 8,
) -> dict:

    return {
        horizon: explain_horizon(
            city=city,
            horizon=horizon,
            top_n=top_n,
        )
        for horizon in (
            "24h",
            "48h",
            "72h",
        )
    }


def main() -> None:
    explanations = (
        explain_all_horizons()
    )

    print(
        "\nAQI Forecast Explainability"
    )
    print(
        "---------------------------"
    )

    for horizon, result in (
        explanations.items()
    ):
        print(
            f"\n{horizon} forecast: "
            f"{result['prediction']}"
        )

        print(
            "Top influences:"
        )

        for item in (
            result["top_features"][:5]
        ):
            sign = (
                "+"
                if item["contribution"] >= 0
                else ""
            )

            print(
                f"  {item['feature']:<30} "
                f"{sign}"
                f"{item['contribution']}"
            )


if __name__ == "__main__":
    main()
