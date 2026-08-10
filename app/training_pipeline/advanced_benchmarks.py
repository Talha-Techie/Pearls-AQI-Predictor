from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.feature_pipeline.engineer import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_aqi_history_2023-01-01_2026-07-29.parquet"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_benchmarks"
)

OUTPUT_PATH = (
    REPORT_DIR
    / "advanced_validation_metrics.csv"
)

MAX_FORECAST_HORIZON = 72

RANDOM_SEED = 42


STATISTICAL_FEATURES = [
    "us_aqi",
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_rolling_mean_24h",
    "pm2_5",
    "pm10",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "hour_sin",
    "hour_cos",
]


HORIZON_LABELS = {
    "target_aqi_24h": "24h",
    "target_aqi_48h": "48h",
    "target_aqi_72h": "72h",
}


def set_seeds() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATASET_PATH}"
        )

    df = pd.read_parquet(DATASET_PATH)

    required = [
        "timestamp",
        *MODEL_FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Dataset is missing columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


def purged_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Reproduce the existing 70/15/15 temporal split
    with a 72-hour purge between partitions.
    """

    total = len(df)

    train_boundary = int(
        total * 0.70
    )

    validation_boundary = int(
        total * 0.85
    )

    train_end = (
        train_boundary
        - MAX_FORECAST_HORIZON
    )

    validation_end = (
        validation_boundary
        - MAX_FORECAST_HORIZON
    )

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_boundary:validation_end
    ].copy()

    test = df.iloc[
        validation_boundary:
    ].copy()

    if (
        train["timestamp"].max()
        + pd.Timedelta(
            hours=MAX_FORECAST_HORIZON
        )
        >= validation["timestamp"].min()
    ):
        raise RuntimeError(
            "Train-validation leakage detected."
        )

    if (
        validation["timestamp"].max()
        + pd.Timedelta(
            hours=MAX_FORECAST_HORIZON
        )
        >= test["timestamp"].min()
    ):
        raise RuntimeError(
            "Validation-test leakage detected."
        )

    return (
        train,
        validation,
        test,
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


# ------------------------------------------------------------------
# Statistical benchmark
# ------------------------------------------------------------------

def benchmark_statistical_ols(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> list[dict]:
    """
    Direct statistical forecasting model using
    autoregressive and exogenous AQI predictors.
    """

    results = []

    x_train = train[
        STATISTICAL_FEATURES
    ].astype(float)

    x_validation = validation[
        STATISTICAL_FEATURES
    ].astype(float)

    x_train = sm.add_constant(
        x_train,
        has_constant="add",
    )

    x_validation = sm.add_constant(
        x_validation,
        has_constant="add",
    )

    for target in TARGET_COLUMNS:
        horizon = HORIZON_LABELS[
            target
        ]

        model = sm.OLS(
            train[target].astype(float),
            x_train,
        )

        fitted = model.fit()

        predictions = fitted.predict(
            x_validation
        ).to_numpy()

        metrics = calculate_metrics(
            validation[target].to_numpy(),
            predictions,
        )

        results.append(
            {
                "model": (
                    "Statistical OLS"
                ),
                "horizon": horizon,
                **metrics,
            }
        )

        print(
            f"OLS {horizon}: "
            f"MAE={metrics['mae']:.3f} "
            f"RMSE={metrics['rmse']:.3f} "
            f"R2={metrics['r2']:.3f}"
        )

    return results


# ------------------------------------------------------------------
# Deep neural-network benchmark
# ------------------------------------------------------------------

class AQIDeepMLP(nn.Module):
    def __init__(
        self,
        input_features: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_features,
                128,
            ),
            nn.ReLU(),

            nn.Dropout(
                p=0.10,
            ),

            nn.Linear(
                128,
                64,
            ),
            nn.ReLU(),

            nn.Dropout(
                p=0.10,
            ),

            nn.Linear(
                64,
                32,
            ),
            nn.ReLU(),

            nn.Linear(
                32,
                3,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


def benchmark_deep_mlp(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> list[dict]:

    set_seeds()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDeep-learning device: {device}"
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(
        train[MODEL_FEATURE_COLUMNS]
    )

    x_validation = x_scaler.transform(
        validation[MODEL_FEATURE_COLUMNS]
    )

    y_train = y_scaler.fit_transform(
        train[TARGET_COLUMNS]
    )

    y_validation = y_scaler.transform(
        validation[TARGET_COLUMNS]
    )

    x_train_tensor = torch.tensor(
        x_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    x_val_tensor = torch.tensor(
        x_validation,
        dtype=torch.float32,
        device=device,
    )

    train_loader = DataLoader(
        TensorDataset(
            x_train_tensor,
            y_train_tensor,
        ),
        batch_size=256,
        shuffle=True,
    )

    model = AQIDeepMLP(
        input_features=len(
            MODEL_FEATURE_COLUMNS
        )
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5,
    )

    criterion = nn.MSELoss()

    max_epochs = 100
    patience = 12

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        running_loss = 0.0

        for batch_x, batch_y in (
            train_loader
        ):
            batch_x = batch_x.to(
                device
            )

            batch_y = batch_y.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                batch_x
            )

            loss = criterion(
                outputs,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * batch_x.size(0)
            )

        training_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        model.eval()

        with torch.no_grad():
            validation_output = model(
                x_val_tensor
            )

            val_loss = criterion(
                validation_output,
                torch.tensor(
                    y_validation,
                    dtype=torch.float32,
                    device=device,
                ),
            ).item()

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"Epoch {epoch:03d} | "
                f"train={training_loss:.5f} | "
                f"val={val_loss:.5f}"
            )

        if (
            val_loss
            < best_loss - 1e-5
        ):
            best_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                f"Early stopping at "
                f"epoch {epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Deep model did not produce "
            "a valid checkpoint."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():
        scaled_predictions = (
            model(
                x_val_tensor
            )
            .cpu()
            .numpy()
        )

    predictions = (
        y_scaler.inverse_transform(
            scaled_predictions
        )
    )

    results = []

    for index, target in enumerate(
        TARGET_COLUMNS
    ):
        horizon = HORIZON_LABELS[
            target
        ]

        metrics = calculate_metrics(
            validation[target].to_numpy(),
            predictions[:, index],
        )

        results.append(
            {
                "model": (
                    "PyTorch Deep MLP"
                ),
                "horizon": horizon,
                **metrics,
            }
        )

        print(
            f"Deep MLP {horizon}: "
            f"MAE={metrics['mae']:.3f} "
            f"RMSE={metrics['rmse']:.3f} "
            f"R2={metrics['r2']:.3f}"
        )

    return results


def add_existing_ridge_reference(
    results: list[dict],
) -> list[dict]:
    """
    Existing leakage-safe validation results.
    These values are included only for benchmark
    comparison; no test-set values are used here.
    """

    ridge_results = [
        {
            "model": "Ridge",
            "horizon": "24h",
            "mae": 15.364,
            "rmse": 20.543,
            "r2": 0.779,
        },
        {
            "model": "Ridge",
            "horizon": "48h",
            "mae": 21.796,
            "rmse": 27.843,
            "r2": 0.596,
        },
        {
            "model": "Ridge",
            "horizon": "72h",
            "mae": 23.936,
            "rmse": 30.336,
            "r2": 0.524,
        },
    ]

    return [
        *results,
        *ridge_results,
    ]


def main() -> None:
    print(
        "\nAdvanced AQI Model Benchmark"
    )
    print(
        "============================"
    )

    dataset = load_dataset()

    train, validation, test = (
        purged_split(dataset)
    )

    print(
        f"\nTrain rows:      "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Test rows:       "
        f"{len(test):,} "
        "(NOT USED)"
    )

    print(
        "\nStatistical benchmark"
    )
    print(
        "---------------------"
    )

    results = (
        benchmark_statistical_ols(
            train,
            validation,
        )
    )

    print(
        "\nDeep-learning benchmark"
    )
    print(
        "-----------------------"
    )

    results.extend(
        benchmark_deep_mlp(
            train,
            validation,
        )
    )

    results = (
        add_existing_ridge_reference(
            results
        )
    )

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df[
        [
            "model",
            "horizon",
            "mae",
            "rmse",
            "r2",
        ]
    ]

    result_df = (
        result_df.sort_values(
            [
                "horizon",
                "rmse",
            ]
        )
        .reset_index(drop=True)
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nFinal validation comparison"
    )
    print(
        "---------------------------"
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )

    print(
        "\nIMPORTANT:"
        "\nThe final test partition was "
        "not used in this benchmark."
    )


if __name__ == "__main__":
    main()