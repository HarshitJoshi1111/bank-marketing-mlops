"""Stage 4: cross-validate and train the model on the training set; everything is logged to MLflow.

The cross-validation scores (training set only) go to reports/cv_metrics.json; models are compared
on these, so the test set is only used once, for the final model.

The MLflow run id is saved next to the model so the evaluate stage can add the test metrics
to the same run.
"""
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.features import SKOPS_TRUSTED_TYPES, make_pipeline, split_xy
from src.params import ROOT, load_params

IN = ROOT / "data" / "split" / "train.csv"
OUT = ROOT / "models"
REPORTS = ROOT / "reports"


def flat(prefix: str, d: dict) -> dict:
    return {f"{prefix}.{k}": v for k, v in d.items()}


def main() -> None:
    params = load_params()
    train_params = params["train"]
    X, y = split_xy(pd.read_csv(IN))
    pipeline = make_pipeline(X, train_params)

    cv = StratifiedKFold(n_splits=train_params["cv_folds"], shuffle=True, random_state=train_params["seed"])
    scores = cross_validate(make_pipeline(X, train_params, n_jobs=1), X, y, cv=cv,
                            scoring=["roc_auc", "average_precision"], n_jobs=-1)
    cv_metrics = {f"cv_{m}": float(scores[f"test_{m}"].mean()) for m in ("roc_auc", "average_precision")}
    cv_metrics |= {f"cv_{m}_std": float(scores[f"test_{m}"].std()) for m in ("roc_auc", "average_precision")}

    pipeline.fit(X, y)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])
    model_name = train_params["model"]
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({"model": model_name, "cv_folds": train_params["cv_folds"],
                           "test_size": params["split"]["test_size"], "train_rows": len(X), "features": X.shape[1],
                           **flat(model_name, train_params.get(model_name, {}))})
        mlflow.log_metrics(cv_metrics)
        mlflow.sklearn.log_model(pipeline, name="model", input_example=X.head(3),
                                 skops_trusted_types=SKOPS_TRUSTED_TYPES)
        run_id = run.info.run_id

    OUT.mkdir(exist_ok=True)
    joblib.dump(pipeline, OUT / "model.joblib")
    (OUT / "run.json").write_text(json.dumps({"run_id": run_id, "model": model_name}, indent=2))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "cv_metrics.json").write_text(json.dumps({k: round(v, 4) for k, v in cv_metrics.items()}, indent=2))
    print(f"{model_name}: " + ", ".join(f"{k}={v:.3f}" for k, v in cv_metrics.items()) + f" (MLflow run {run_id})")


if __name__ == "__main__":
    main()
