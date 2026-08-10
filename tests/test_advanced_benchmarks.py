import pandas as pd
import torch

from app.feature_pipeline.engineer import (
    MODEL_FEATURE_COLUMNS,
)
from app.training_pipeline.advanced_benchmarks import (
    AQIDeepMLP,
    calculate_metrics,
    purged_split,
)


def make_dataset(
    rows: int = 1000,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01",
        periods=rows,
        freq="h",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
        }
    )


def test_purged_split_sizes() -> None:
    df = make_dataset()

    train, validation, test = (
        purged_split(df)
    )

    assert len(train) == 628
    assert len(validation) == 78
    assert len(test) == 150


def test_deep_model_output_shape() -> None:
    model = AQIDeepMLP(
        input_features=len(
            MODEL_FEATURE_COLUMNS
        )
    )

    sample = torch.zeros(
        (
            4,
            len(
                MODEL_FEATURE_COLUMNS
            ),
        )
    )

    output = model(sample)

    assert output.shape == (
        4,
        3,
    )


def test_metrics_are_computed() -> None:
    result = calculate_metrics(
        y_true=[
            100,
            120,
            140,
        ],
        y_pred=[
            105,
            118,
            135,
        ],
    )

    assert result["mae"] >= 0
    assert result["rmse"] >= 0
    assert "r2" in result