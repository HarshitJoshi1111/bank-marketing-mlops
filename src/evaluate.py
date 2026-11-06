"""Stage 5: evaluate on the held-out test set.

Writes metrics for DVC (reports/metrics.json), curve data for `dvc plots`, and logs the same numbers
to the MLflow run of the trained model. With ~11 % positives, accuracy is misleading (always "no"
gives 89 %), so the main metrics are ROC AUC and average precision (area under the PR curve).
Precision, recall and F1 use the threshold chosen in the train stage (models/run.json).
"""
import json

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.features import split_xy
from src.params import ROOT, load_params

TEST = ROOT / "data" / "split" / "test.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def metrics(y_true, scores, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int)
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "positive_rate": float(y_true.mean()),
    }


def main() -> None:
    params = load_params()
    X, y = split_xy(pd.read_csv(TEST))
    model = joblib.load(MODELS / "model.joblib")
    run = json.loads((MODELS / "run.json").read_text())
    scores = model.predict_proba(X)[:, 1]
    result = {k: round(float(v), 4) for k, v in metrics(y, scores, run["threshold"]).items()}

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(result, indent=2))
    fpr, tpr, _ = roc_curve(y, scores)
    precision, recall, _ = precision_recall_curve(y, scores)
    step = max(1, len(fpr) // 200)  # keep the plot files small
    pd.DataFrame({"fpr": fpr[::step], "tpr": tpr[::step]}).to_csv(REPORTS / "roc.csv", index=False)
    step = max(1, len(recall) // 200)
    pd.DataFrame({"recall": recall[::step], "precision": precision[::step]}).to_csv(REPORTS / "pr.csv", index=False)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    with mlflow.start_run(run_id=run["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in result.items() if k != "threshold"})
    print(f"{run['model']} on test: " + ", ".join(f"{k}={v}" for k, v in result.items()))


if __name__ == "__main__":
    main()
