import pandas as pd
import pytest

from app.feature_pipeline.engineer import MODEL_FEATURE_COLUMNS
from app.prediction import service


def make_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                feature: float(index + 1)
                for index, feature in enumerate(
                    MODEL_FEATURE_COLUMNS
                )
            }
        ],
        columns=MODEL_FEATURE_COLUMNS,
    )


class FakeModel:
    def __init__(self, prediction: float):
        self.prediction = prediction

    def predict(
        self,
        dataframe: pd.DataFrame,
    ):
        assert list(dataframe.columns) == (
            MODEL_FEATURE_COLUMNS
        )

        return [self.prediction]


def test_online_feature_vector_uses_model_feature_order(
    monkeypatch,
) -> None:
    feature_values = {
        feature: [float(index + 1)]
        for index, feature in enumerate(
            MODEL_FEATURE_COLUMNS
        )
    }

    feature_values["city"] = ["Lahore"]

    class FakeOnlineResult:
        def to_dict(self):
            return feature_values

    class FakeStore:
        def get_feature_service(
            self,
            name,
        ):
            assert (
                name
                == "aqi_prediction_features_v1"
            )

            return object()

        def get_online_features(
            self,
            features,
            entity_rows,
        ):
            assert entity_rows == [
                {"city": "Lahore"}
            ]

            return FakeOnlineResult()

    monkeypatch.setattr(
        service,
        "get_feature_store",
        lambda: FakeStore(),
    )

    result = (
        service.get_online_feature_vector(
            "Lahore"
        )
    )

    assert list(result.columns) == (
        MODEL_FEATURE_COLUMNS
    )

    assert len(result.columns) == 42


def test_missing_feast_feature_is_rejected(
    monkeypatch,
) -> None:
    feature_values = {
        feature: [1.0]
        for feature in MODEL_FEATURE_COLUMNS
    }

    feature_values.pop(
        MODEL_FEATURE_COLUMNS[0]
    )

    class FakeOnlineResult:
        def to_dict(self):
            return feature_values

    class FakeStore:
        def get_feature_service(
            self,
            name,
        ):
            return object()

        def get_online_features(
            self,
            features,
            entity_rows,
        ):
            return FakeOnlineResult()

    monkeypatch.setattr(
        service,
        "get_feature_store",
        lambda: FakeStore(),
    )

    with pytest.raises(
        service.PredictionServiceError
    ):
        service.get_online_feature_vector(
            "Lahore"
        )


def test_all_three_forecast_horizons_are_used(
    monkeypatch,
) -> None:
    features = make_feature_frame()

    # Make current AQI realistic.
    features.loc[
        0,
        "us_aqi",
    ] = 161.0

    monkeypatch.setattr(
        service,
        "get_online_feature_vector",
        lambda city: features,
    )

    predictions = {
        "24h": 147.7,
        "48h": 139.8,
        "72h": 135.3,
    }

    requested_horizons = []

    def fake_load_model(
        horizon: str,
    ):
        requested_horizons.append(
            horizon
        )

        return FakeModel(
            predictions[horizon]
        )

    monkeypatch.setattr(
        service,
        "load_registered_model",
        fake_load_model,
    )

    result = service.predict_aqi(
        "Lahore"
    )

    assert requested_horizons == [
        "24h",
        "48h",
        "72h",
    ]

    assert result["forecast"] == {
        "24h": 147.7,
        "48h": 139.8,
        "72h": 135.3,
    }


def test_prediction_response_structure(
    monkeypatch,
) -> None:
    features = make_feature_frame()

    features.loc[
        0,
        "us_aqi",
    ] = 161.0

    monkeypatch.setattr(
        service,
        "get_online_feature_vector",
        lambda city: features,
    )

    def fake_load_model(
        horizon: str,
    ):
        values = {
            "24h": 147.7,
            "48h": 139.8,
            "72h": 135.3,
        }

        return FakeModel(
            values[horizon]
        )

    monkeypatch.setattr(
        service,
        "load_registered_model",
        fake_load_model,
    )

    result = service.predict_aqi(
        "Lahore"
    )

    assert result["city"] == "Lahore"

    assert result["current_aqi"] == 161.0

    assert result["feature_count"] == 42

    assert (
        result["feature_source"]
        == "Feast online store"
    )

    assert (
        result["model_source"]
        == "DagsHub MLflow Model Registry"
    )

    assert "current_status" in result
    assert "forecast_status" in result

    assert set(
        result["forecast"]
    ) == {
        "24h",
        "48h",
        "72h",
    }