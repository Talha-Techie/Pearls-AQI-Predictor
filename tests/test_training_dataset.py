import pandas as pd

from app.training_pipeline.dataset import (
    MAX_FORECAST_HORIZON_HOURS,
    temporal_split,
)


def make_temporal_dataset(
    rows: int = 1000,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2024-01-01",
                periods=rows,
                freq="h",
                tz="UTC",
            ),
            "value": range(rows),
        }
    )


def test_temporal_split_preserves_order() -> None:
    df = make_temporal_dataset()

    split = temporal_split(df)

    assert (
        split.train["timestamp"].max()
        < split.validation["timestamp"].min()
    )

    assert (
        split.validation["timestamp"].max()
        < split.test["timestamp"].min()
    )


def test_train_targets_do_not_cross_validation() -> None:
    df = make_temporal_dataset()

    split = temporal_split(df)

    latest_train_target_time = (
        split.train["timestamp"].max()
        + pd.Timedelta(
            hours=MAX_FORECAST_HORIZON_HOURS
        )
    )

    assert (
        latest_train_target_time
        < split.validation["timestamp"].min()
    )


def test_validation_targets_do_not_cross_test() -> None:
    df = make_temporal_dataset()

    split = temporal_split(df)

    latest_validation_target_time = (
        split.validation["timestamp"].max()
        + pd.Timedelta(
            hours=MAX_FORECAST_HORIZON_HOURS
        )
    )

    assert (
        latest_validation_target_time
        < split.test["timestamp"].min()
    )


def test_expected_purged_split_sizes() -> None:
    df = make_temporal_dataset(
        rows=1000
    )

    split = temporal_split(df)

    assert len(split.train) == 628
    assert len(split.validation) == 78
    assert len(split.test) == 150
