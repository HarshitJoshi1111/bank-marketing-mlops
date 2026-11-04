"""Stage 7: export the current champion from the MLflow Model Registry for serving.

The API and its Docker image do not need MLflow: they load models/serving/model.joblib and
models/serving/meta.json (version, threshold, test metrics, expected input columns). What is served is
always what the registry calls champion, not simply the last trained model.
"""
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.params import ROOT, load_params

OUT = ROOT / "models" / "serving"


def export(name: str, alias: str, out: Path = OUT) -> dict:
    version = MlflowClient().get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    tags = version.tags
    meta = {
        "name": name,
        "alias": alias,
        "version": int(version.version),
        "model_type": tags.get("model_type"),
        "run_id": tags.get("run_id"),
        "threshold": float(tags["test_threshold"]),
        "test_metrics": {k.removeprefix("test_"): float(v) for k, v in tags.items()
                         if k.startswith("test_") and k != "test_threshold"},
        "features": list(model.feature_names_in_),
    }
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    params = load_params()
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    meta = export(params["register"]["model_name"], params["register"]["alias"])
    print(f"exported {meta['name']} version {meta['version']} ({meta['model_type']}, threshold {meta['threshold']}) "
          f"to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
