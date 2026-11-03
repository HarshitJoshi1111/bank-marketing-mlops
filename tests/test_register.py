import mlflow
import mlflow.sklearn
import pytest
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier

from src.register import decide, register

CFG = {"model_name": "test-model", "alias": "champion", "min_average_precision": 0.4}


def test_decide():
    assert decide(0.30, gate=0.4, champion_score=None) == "rejected"
    assert decide(0.45, gate=0.4, champion_score=None) == "champion"
    assert decide(0.45, gate=0.4, champion_score=0.50) == "challenger"
    assert decide(0.50, gate=0.4, champion_score=0.50) == "champion"


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow stores artifacts relative to the working directory
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield MlflowClient()
    mlflow.set_tracking_uri(None)


def logged_run() -> dict:
    with mlflow.start_run() as run:
        info = mlflow.sklearn.log_model(DummyClassifier().fit([[0], [1]], [0, 1]), name="model")
    return {"run_id": run.info.run_id, "model_uri": info.model_uri, "model": "dummy"}


def test_champion_moves_only_to_better_models(tracking):
    first = register(tracking, logged_run(), {"average_precision": 0.45}, CFG)
    assert first["decision"] == "champion" and first["version"] == 1

    worse = register(tracking, logged_run(), {"average_precision": 0.42}, CFG)
    assert worse["decision"] == "challenger" and worse["version"] == 2
    assert int(tracking.get_model_version_by_alias("test-model", "champion").version) == 1

    rejected = register(tracking, logged_run(), {"average_precision": 0.2}, CFG)
    assert rejected["decision"] == "rejected" and rejected["version"] is None

    better = register(tracking, logged_run(), {"average_precision": 0.5}, CFG)
    assert better["decision"] == "champion"
    champion = tracking.get_model_version_by_alias("test-model", "champion")
    assert int(champion.version) == 3 and champion.tags["test_average_precision"] == "0.5"
