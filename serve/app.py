"""Prediction API for the champion model.

    uvicorn serve.app:app --port 8000        then open http://localhost:8000/docs

POST /predict takes up to 1,000 customers with the same fields as the raw data (without "duration",
which is only known after the call) and returns the probability of a subscription and the yes/no
decision at the model's threshold. Inputs are validated (types, ranges, allowed category values)
before they reach the model; feature engineering is the same function the training data went through.
"""
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.prepare import CATEGORIES, add_features

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models" / "serving"))


def one_of(column: str):
    return Literal[tuple(sorted(CATEGORIES[column]))]


EXAMPLE = {
    "age": 35, "job": "admin.", "marital": "single", "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular", "month": "may", "day_of_week": "thu",
    "campaign": 1, "pdays": 999, "previous": 0, "poutcome": "nonexistent", "emp.var.rate": -1.8,
    "cons.price.idx": 92.893, "cons.conf.idx": -46.2, "euribor3m": 1.313, "nr.employed": 5099.1,
}


class Customer(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", json_schema_extra={"examples": [EXAMPLE]})

    age: int = Field(ge=15, le=100)
    job: one_of("job")
    marital: one_of("marital")
    education: one_of("education")
    default: one_of("default")
    housing: one_of("housing")
    loan: one_of("loan")
    contact: one_of("contact")
    month: one_of("month")
    day_of_week: one_of("day_of_week")
    campaign: int = Field(ge=1, description="contacts during this campaign, including this one")
    pdays: int = Field(ge=0, description="days since the last contact of an earlier campaign; 999 = never")
    previous: int = Field(ge=0)
    poutcome: one_of("poutcome")
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    probability: float
    subscribe: bool


class PredictResponse(BaseModel):
    model_version: int
    threshold: float
    predictions: list[Prediction]


@lru_cache(maxsize=1)
def load_model():
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    return joblib.load(MODEL_DIR / "model.joblib"), meta


app = FastAPI(title="Bank marketing: term deposit prediction",
              description="Probability that a customer subscribes to a term deposit after a marketing call.")


def features(customers: list[Customer], columns: list[str]) -> pd.DataFrame:
    df = add_features(pd.DataFrame([c.model_dump(by_alias=True) for c in customers]))
    numeric = df.select_dtypes(include="number").columns
    return df.astype({c: "float64" for c in numeric})[columns]  # same dtypes and column order as in training


@app.get("/health")
def health() -> dict:
    _, meta = load_model()
    return {"status": "ok", "model_version": meta["version"]}


@app.get("/model")
def model_info() -> dict:
    _, meta = load_model()
    return meta


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model, meta = load_model()
    probabilities = model.predict_proba(features(request.customers, meta["features"]))[:, 1]
    return PredictResponse(
        model_version=meta["version"], threshold=meta["threshold"],
        predictions=[Prediction(probability=round(float(p), 4), subscribe=bool(p >= meta["threshold"]))
                     for p in probabilities],
    )
