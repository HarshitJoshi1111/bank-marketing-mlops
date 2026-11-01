import pandas as pd
import pytest

from src.prepare import prepare, validate

ROW = {"age": 41, "job": "admin.", "marital": "married", "education": "university.degree", "default": "no",
       "housing": "yes", "loan": "no", "contact": "cellular", "month": "may", "day_of_week": "mon",
       "duration": 120, "campaign": 1, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
       "emp.var.rate": 1.1, "cons.price.idx": 93.994, "cons.conf.idx": -36.4, "euribor3m": 4.857,
       "nr.employed": 5191.0, "y": "no"}


def frame(*changes: dict) -> pd.DataFrame:
    return pd.DataFrame([{**ROW, **c} for c in changes] or [ROW])


def test_valid_data_passes():
    validate(frame({}, {"y": "yes", "pdays": 3}))


def test_validation_reports_every_problem():
    df = frame({"job": "astronaut", "age": 140}).drop(columns=["loan"])
    with pytest.raises(ValueError) as error:
        validate(df)
    message = str(error.value)
    assert "missing columns: ['loan']" in message
    assert "astronaut" in message
    assert "age outside" in message


def test_prepare_drops_leak_and_encodes_target_and_pdays():
    df = prepare(frame({}, {"y": "yes", "pdays": 3}), drop_columns=["duration"])
    assert "duration" not in df.columns
    assert df["y"].tolist() == [0, 1]
    assert df["previously_contacted"].tolist() == [0, 1]
    assert df["pdays"].tolist() == [-1, 3]


def test_prepare_removes_duplicate_rows():
    assert len(prepare(frame({}, {}, {"age": 50}), drop_columns=["duration"])) == 2
