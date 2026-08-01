from pprint import pprint

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

    result = store.get_online_features(
        features=feature_service,
        entity_rows=[
            {
                "city": "Lahore",
            }
        ],
    ).to_dict()

    print("\nOnline AQI Feature Vector")
    print("-------------------------")

    pprint(result)

    assert result["city"][0] == "Lahore"
    assert result["us_aqi"][0] is not None
    assert result["pm2_5"][0] is not None
    assert result["aqi_lag_24h"][0] is not None

    print(
        "\nOnline feature retrieval: PASS"
    )


if __name__ == "__main__":
    main()
