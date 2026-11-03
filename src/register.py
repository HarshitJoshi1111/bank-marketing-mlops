"""Stage 6: register the evaluated model in the MLflow Model Registry (champion / challenger).

1. Quality gate: the test average precision must reach register.min_average_precision, otherwise the
   model is not registered at all.
2. A model that passes becomes a new version of the registered model, tagged with its test metrics,
   the threshold and the git-independent MLflow run id.
3. The alias (default "champion") moves to the new version only if it is at least as good as the
   current champion on the test set. Code that serves the model loads `models:/<name>@champion`, so a
   worse model is kept as a challenger version but never goes live by accident.

Writes the decision to reports/registry.json.
"""
import json

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.params import ROOT, load_params

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SCORE = "average_precision"


def decide(score: float, gate: float, champion_score: float | None) -> str:
    if score < gate:
        return "rejected"
    if champion_score is None or score >= champion_score:
        return "champion"
    return "challenger"


def current_champion(client: MlflowClient, name: str, alias: str) -> tuple[int, float] | None:
    try:
        version = client.get_model_version_by_alias(name, alias)
    except MlflowException:  # no registered model or no alias yet
        return None
    return int(version.version), float(version.tags.get(f"test_{SCORE}", "nan"))


def register(client: MlflowClient, run: dict, test_metrics: dict, cfg: dict) -> dict:
    name, alias = cfg["model_name"], cfg["alias"]
    champion = current_champion(client, name, alias)
    decision = decide(test_metrics[SCORE], cfg[f"min_{SCORE}"], champion[1] if champion else None)
    result = {"decision": decision, f"test_{SCORE}": test_metrics[SCORE], "gate": cfg[f"min_{SCORE}"],
              "previous_champion": {"version": champion[0], f"test_{SCORE}": champion[1]} if champion else None,
              "model": run["model"], "version": None}
    if decision == "rejected":
        return result
    tags = {f"test_{k}": str(v) for k, v in test_metrics.items()} | {"model_type": run["model"], "run_id": run["run_id"]}
    version = mlflow.register_model(run["model_uri"], name, tags=tags)
    if decision == "champion":
        client.set_registered_model_alias(name, alias, version.version)
    return result | {"version": int(version.version)}


def main() -> None:
    params = load_params()
    run = json.loads((MODELS / "run.json").read_text())
    test_metrics = json.loads((REPORTS / "metrics.json").read_text())
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    result = register(MlflowClient(), run, test_metrics, params["register"])
    (REPORTS / "registry.json").write_text(json.dumps(result, indent=2))
    cfg = params["register"]
    messages = {
        "rejected": f"not registered: test {SCORE} {result[f'test_{SCORE}']} is below the gate {result['gate']}",
        "champion": f"registered {cfg['model_name']} version {result['version']} as @{cfg['alias']}",
        "challenger": f"registered {cfg['model_name']} version {result['version']}; @{cfg['alias']} stays on "
                      f"version {(result['previous_champion'] or {}).get('version')} (better test {SCORE})",
    }
    print(messages[result["decision"]])


if __name__ == "__main__":
    main()
