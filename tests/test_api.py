import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import serve.app as api
from src.features import make_pipeline, split_xy
from src.prepare import prepare
from tests.test_prepare import ROW


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A small logistic regression trained on made-up rows, saved like the export stage does."""
    rows = [{**ROW, "age": 20 + i, "euribor3m": 1.0 + i / 10, "job": ["admin.", "student", "retired"][i % 3],
             "y": "yes" if i % 3 == 1 else "no"} for i in range(30)]
    train = prepare(pd.DataFrame(rows), drop_columns=["duration"])
    X, y = split_xy(train)
    model = make_pipeline(X, {"model": "logreg", "seed": 0, "logreg": {}}).fit(X, y)
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "meta.json").write_text(json.dumps({"version": 7, "threshold": 0.5, "features": list(X.columns)}))
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path)
    api.load_model.cache_clear()
    yield TestClient(api.app)
    api.load_model.cache_clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": 7}


def test_predict_returns_one_prediction_per_customer(client):
    customers = [api.EXAMPLE, {**api.EXAMPLE, "job": "student", "pdays": 3, "previous": 1, "poutcome": "success"}]
    response = client.post("/predict", json={"customers": customers})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == 7 and body["threshold"] == 0.5
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert 0 <= p["probability"] <= 1 and p["subscribe"] == (p["probability"] >= 0.5)


@pytest.mark.parametrize("change", [{"job": "astronaut"}, {"age": 12}, {"campaign": 0}, {"duration": 100}])
def test_invalid_input_is_rejected(client, change):
    """Unknown categories, impossible values and the leaking "duration" field are refused with 422."""
    assert client.post("/predict", json={"customers": [{**api.EXAMPLE, **change}]}).status_code == 422


def test_missing_field_and_empty_batch(client):
    incomplete = {k: v for k, v in api.EXAMPLE.items() if k != "euribor3m"}
    assert client.post("/predict", json={"customers": [incomplete]}).status_code == 422
    assert client.post("/predict", json={"customers": []}).status_code == 422


def test_features_match_training_columns_and_dtypes(client):
    _, meta = api.load_model()
    df = api.features([api.Customer(**api.EXAMPLE)], meta["features"])
    assert list(df.columns) == meta["features"]
    assert df["previously_contacted"].iloc[0] == 0 and df["pdays"].iloc[0] == -1  # 999 -> not contacted
    assert df["age"].dtype == "float64"
