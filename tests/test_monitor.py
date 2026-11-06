import numpy as np
import pandas as pd

from src.monitor import decide, feature_drift, psi_categorical, psi_numeric, status

CFG = {"bins": 10, "psi_moderate": 0.1, "psi_drift": 0.25, "ignore_for_decision": ["month"]}
rng = np.random.default_rng(0)


def test_psi_numeric_same_distribution_is_small_and_shift_is_large():
    reference = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(reference, pd.Series(rng.normal(0, 1, 5000))) < 0.02
    assert psi_numeric(reference, pd.Series(rng.normal(1.5, 1, 5000))) > 0.25


def test_psi_categorical_handles_new_categories():
    reference = pd.Series(["a"] * 50 + ["b"] * 50)
    assert psi_categorical(reference, reference) == 0
    assert psi_categorical(reference, pd.Series(["a"] * 50 + ["c"] * 50)) > 0.25  # "c" never seen before


def test_status_thresholds():
    assert [status(p, 0.1, 0.25) for p in (0.05, 0.1, 0.3)] == ["stable", "moderate", "drift"]


def test_feature_drift_and_decision():
    reference = pd.DataFrame({"x": rng.normal(0, 1, 2000), "month": ["may"] * 2000})
    shifted = pd.DataFrame({"x": rng.normal(2, 1, 2000), "month": ["nov"] * 2000})
    same = pd.DataFrame({"x": rng.normal(0, 1, 2000), "month": ["nov"] * 2000})

    rows = feature_drift(reference, shifted, CFG)
    assert {r["feature"]: r["status"] for r in rows} == {"x": "drift", "month": "drift"}
    assert decide(rows, score_psi=0.0, cfg=CFG) == "retrain"
    # only the calendar feature changed: reported, but no retrain
    assert decide(feature_drift(reference, same, CFG), score_psi=0.0, cfg=CFG) == "ok"
    assert decide(feature_drift(reference, same, CFG), score_psi=0.3, cfg=CFG) == "retrain"
