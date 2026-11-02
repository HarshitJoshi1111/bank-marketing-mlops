import numpy as np
import pandas as pd
import pytest

from src.evaluate import metrics
from src.features import make_pipeline, split_xy

PARAMS = {"model": "logreg", "seed": 0, "logreg": {"C": 1.0, "class_weight": "balanced"}, "hist_gb": {"max_iter": 20},
          "random_forest": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1, "class_weight": "balanced"}}


def toy_data(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({"campaign": rng.integers(1, 5, n), "euribor3m": x, "job": rng.choice(["admin.", "student"], n),
                         "y": (x + rng.normal(scale=0.5, size=n) > 1).astype(int)})


def test_split_xy_turns_integers_into_floats():
    X, y = split_xy(toy_data())
    assert "y" not in X.columns and len(y) == len(X)
    assert X["campaign"].dtype == "float64"


@pytest.mark.parametrize("model", ["dummy", "logreg", "random_forest", "hist_gb"])
def test_pipeline_trains_and_handles_unseen_category(model):
    X, y = split_xy(toy_data())
    pipeline = make_pipeline(X, {**PARAMS, "model": model}).fit(X, y)
    new = X.head(2).assign(job="astronaut")  # category not seen in training
    assert pipeline.predict_proba(new).shape == (2, 2)


def test_unknown_model_name():
    X, _ = split_xy(toy_data())
    with pytest.raises(ValueError):
        make_pipeline(X, {**PARAMS, "model": "xgboost"})


def test_metrics():
    y = pd.Series([0, 0, 1, 1])
    result = metrics(y, np.array([0.1, 0.6, 0.4, 0.9]), threshold=0.5)
    assert result["roc_auc"] == 0.75
    assert result["precision"] == 0.5 and result["recall"] == 0.5
    assert result["positive_rate"] == 0.5
