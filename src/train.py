"""Stage 4: cross-validate and train the model on the training set; everything is logged to MLflow.

The MLflow run id is saved next to the model so the evaluate stage can add the test metrics
to the same run.
"""
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.features import make_pipeline, split_xy
from src.params import ROOT, load_params

IN = ROOT / "data" / "split" / "train.csv"
OUT = ROOT / "models"


def flat(prefix: str, d: dict) -> dict:
    return {f"{prefix}.{k}": v for k, v in d.items()}


def main() -> None:
    params = load_params()
    train_params = params["train"]
    X, y = split_xy(pd.read_csv(IN))
    pipeline = make_pipeline(X, train_params)

    cv = StratifiedKFold(n_splits=train_params["cv_folds"], shuffle=True, random_state=train_params["seed"])
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=-1)
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
        # MLflow stores scikit-learn models in the skops format, which only loads types marked as trusted;
        # the tree structure of the random forest is our own model, so it is trusted here.
        mlflow.sklearn.log_model(pipeline, name="model", input_example=X.head(3),
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])
        run_id = run.info.run_id

    OUT.mkdir(exist_ok=True)
    joblib.dump(pipeline, OUT / "model.joblib")
    (OUT / "run.json").write_text(json.dumps({"run_id": run_id, "model": model_name}, indent=2))
    print(f"{model_name}: " + ", ".join(f"{k}={v:.3f}" for k, v in cv_metrics.items()) + f" (MLflow run {run_id})")


if __name__ == "__main__":
    main()
