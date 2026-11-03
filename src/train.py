"""Stage 4: cross-validate and train the model on the training set; everything is logged to MLflow.

Cross-validation gives out-of-fold predictions for every training row. They are used for the CV scores
(reports/cv_metrics.json, models are compared on these so the test set is only used once) and to choose
the decision threshold (src/threshold.py).

The MLflow run id, the logged model URI and the threshold are saved in models/run.json for the
evaluate and register stages.
"""
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.features import SKOPS_TRUSTED_TYPES, make_pipeline, split_xy
from src.params import ROOT, load_params
from src.threshold import choose_threshold

IN = ROOT / "data" / "split" / "train.csv"
OUT = ROOT / "models"
REPORTS = ROOT / "reports"


def flat(prefix: str, d: dict) -> dict:
    return {f"{prefix}.{k}": v for k, v in d.items()}


def cv_scores(y: pd.Series, oof: np.ndarray, cv, X: pd.DataFrame) -> dict:
    """Mean and std over the folds of ROC AUC and average precision, from the out-of-fold predictions."""
    folds = {"roc_auc": [], "average_precision": []}
    for _, val in cv.split(X, y):
        folds["roc_auc"].append(roc_auc_score(y.iloc[val], oof[val]))
        folds["average_precision"].append(average_precision_score(y.iloc[val], oof[val]))
    result = {f"cv_{m}": float(np.mean(v)) for m, v in folds.items()}
    return result | {f"cv_{m}_std": float(np.std(v)) for m, v in folds.items()}


def main() -> None:
    params = load_params()
    train_params = params["train"]
    X, y = split_xy(pd.read_csv(IN))

    cv = StratifiedKFold(n_splits=train_params["cv_folds"], shuffle=True, random_state=train_params["seed"])
    oof = cross_val_predict(make_pipeline(X, train_params, n_jobs=1), X, y, cv=cv, method="predict_proba",
                            n_jobs=-1)[:, 1]
    cv_metrics = cv_scores(y, oof, cv, X)

    t = train_params["threshold"]
    threshold = choose_threshold(y, oof, t["method"], t.get("value", 0.5), t.get("min_recall", 0.5))
    predicted = (oof >= threshold).astype(int)
    cv_metrics |= {"threshold": threshold, "cv_precision": float(precision_score(y, predicted)),
                   "cv_recall": float(recall_score(y, predicted)), "cv_f1": float(f1_score(y, predicted))}

    pipeline = make_pipeline(X, train_params).fit(X, y)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])
    model_name = train_params["model"]
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({"model": model_name, "cv_folds": train_params["cv_folds"],
                           "test_size": params["split"]["test_size"], "train_rows": len(X), "features": X.shape[1],
                           "threshold_method": t["method"], **flat(model_name, train_params.get(model_name, {}))})
        mlflow.log_metrics(cv_metrics)
        info = mlflow.sklearn.log_model(pipeline, name="model", input_example=X.head(3),
                                        skops_trusted_types=SKOPS_TRUSTED_TYPES)
        run_id = run.info.run_id

    OUT.mkdir(exist_ok=True)
    joblib.dump(pipeline, OUT / "model.joblib")
    (OUT / "run.json").write_text(json.dumps({"run_id": run_id, "model": model_name, "model_uri": info.model_uri,
                                              "threshold": round(threshold, 4)}, indent=2))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "cv_metrics.json").write_text(json.dumps({k: round(v, 4) for k, v in cv_metrics.items()}, indent=2))
    print(f"{model_name}: " + ", ".join(f"{k}={v:.3f}" for k, v in cv_metrics.items()) + f" (MLflow run {run_id})")


if __name__ == "__main__":
    main()
