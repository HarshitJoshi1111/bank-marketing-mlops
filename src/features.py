"""Model pipeline: one-hot encoding for categories, scaling for numbers, then the classifier."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "y"
MODELS = ("dummy", "logreg", "random_forest", "hist_gb")
# MLflow saves scikit-learn models in the skops format, which only loads types marked as trusted.
# These are the tree structures of our own random forest / gradient boosting models.
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
    "sklearn._loss.link.LogitLink",
    "sklearn._loss.loss.HalfBinomialLoss",
    "sklearn._loss._loss.CyHalfBinomialLoss",
]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET])
    # integers as floats: the model signature logged to MLflow then also accepts missing values
    ints = X.select_dtypes(include="integer").columns
    return X.astype({c: "float64" for c in ints}), df[TARGET]


def make_classifier(params: dict, n_jobs: int = -1):
    """n_jobs=1 when the caller already runs several fits in parallel (cross-validation, search)."""
    name, seed = params["model"], params["seed"]
    settings = params.get(name, {})
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=seed, **settings)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **settings)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=seed, **settings)
    raise ValueError(f"unknown model {name!r}")


def make_pipeline(X: pd.DataFrame, params: dict, n_jobs: int = -1) -> Pipeline:
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric = X.select_dtypes(include="number").columns.tolist()
    preprocess = ColumnTransformer([
        # dense output: gradient boosting does not accept sparse input (the table is small anyway)
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", StandardScaler(), numeric),
    ])
    return Pipeline([("preprocess", preprocess), ("model", make_classifier(params, n_jobs))])
