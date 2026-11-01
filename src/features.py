"""Model pipeline: one-hot encoding for categories, scaling for numbers, then the classifier."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "y"


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET])
    # integers as floats: the model signature logged to MLflow then also accepts missing values
    ints = X.select_dtypes(include="integer").columns
    return X.astype({c: "float64" for c in ints}), df[TARGET]


def make_classifier(params: dict):
    name, seed = params["model"], params["seed"]
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=seed, **params["logreg"])
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params["random_forest"])
    raise ValueError(f"unknown model {name!r}")


def make_pipeline(X: pd.DataFrame, params: dict) -> Pipeline:
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric = X.select_dtypes(include="number").columns.tolist()
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
    ])
    return Pipeline([("preprocess", preprocess), ("model", make_classifier(params))])
