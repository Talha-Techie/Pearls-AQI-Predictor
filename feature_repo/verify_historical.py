import pandas as pd

from feast import FeatureStore


def main() -> None:
    store = FeatureStore(
        repo_path="."
    )

    feature_service = (
        store.get_feature_service(
            "aqi_prediction_features_v1"
        )
    )

    entity_df = pd.DataFrame(
        {
            "city": [
                "Lahore",
                "Lahore",
                "Lahore",
            ],
            "event_timestamp": pd.to_datetime(
                [
                    "2025-01-01T12:00:00Z",
                    "2025-06-01T12:00:00Z",
                    "2026-01-01T12:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    training_features = (
        store.get_historical_features(
            features=feature_service,
            entity_df=entity_df,
        )
        .to_df()
    )

    print(
        training_features[
            [
                "city",
                "event_timestamp",
                "us_aqi",
                "pm2_5",
                "aqi_lag_24h",
            ]
        ].to_string(
            index=False
        )
    )

    assert len(training_features) == 3

    assert (
        training_features[
            "us_aqi"
        ].notna().all()
    )

    print(
        "\nHistorical feature retrieval: PASS"
    )


if __name__ == "__main__":
    main()
