"""Model comparison and hyperparameter search, with cross-validation on the training set only.

For every model in params.yaml (tune.models) the script measures the cross-validated score with the
default settings from `train:` and then runs a randomized search over `tune.spaces`. Everything goes to
MLflow: one parent run per model and one nested run per tried setting. The test set is not used here;
the evaluate stage uses it once, for the model that is finally chosen in params.yaml.

Selection uses the one-standard-error rule on the model types: of all models whose best score is within
one standard deviation (across the CV folds) of the overall best, the one that trains fastest is chosen,
with its best settings. Differences smaller than the fold-to-fold noise are not real, and a faster,
smaller model is cheaper to retrain and serve.

Not a DVC stage on purpose: it takes minutes and its result is a decision (which settings to put into
params.yaml), not a file the pipeline needs.

Usage:   python -m src.tune                  (search, then write the report)
         python -m src.tune --report-only    (rebuild the report from reports/tuning.json)
Results: reports/tuning.md, reports/tuning.json and the MLflow experiment "<experiment>-tuning"
"""
import json
import time
from copy import deepcopy

import mlflow
import pandas as pd
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold, cross_validate

from src.features import make_pipeline, split_xy
from src.params import ROOT, load_params

IN = ROOT / "data" / "split" / "train.csv"
REPORTS = ROOT / "reports"
SCORING = ["average_precision", "roc_auc"]


def search_space(space: dict) -> dict:
    """{'C': [...]} -> {'model__C': [...]} (parameter names inside the scikit-learn pipeline)."""
    return {f"model__{name}": values for name, values in space.items()}


def evaluate_default(X, y, train_params: dict, name: str, cv) -> dict:
    params = {**train_params, "model": name}
    scores = cross_validate(make_pipeline(X, params, n_jobs=1), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    result = {m: float(scores[f"test_{m}"].mean()) for m in SCORING}
    result |= {f"{m}_std": float(scores[f"test_{m}"].std()) for m in SCORING}
    return result | {"fit_seconds": float(scores["fit_time"].mean())}


def search(X, y, train_params: dict, name: str, space: dict, n_iter: int, scoring: str, cv) -> pd.DataFrame:
    """Randomized search; returns one row per tried setting, best first."""
    n_iter = min(n_iter, len(ParameterGrid(space)))  # small spaces are searched completely
    searcher = RandomizedSearchCV(
        make_pipeline(X, {**train_params, "model": name}, n_jobs=1), search_space(space), n_iter=n_iter,
        scoring=SCORING, refit=False, cv=cv, random_state=train_params["seed"], n_jobs=-1,
    )
    searcher.fit(X, y)
    r = searcher.cv_results_
    rows = pd.DataFrame({
        "params": [{k.removeprefix("model__"): v for k, v in p.items()} for p in r["params"]],
        **{m: r[f"mean_test_{m}"] for m in SCORING},
        **{f"{m}_std": r[f"std_test_{m}"] for m in SCORING},
        "fit_seconds": r["mean_fit_time"],
    })
    return rows.sort_values(scoring, ascending=False).reset_index(drop=True)


def log_to_mlflow(name: str, default: dict, rows: pd.DataFrame) -> None:
    with mlflow.start_run(run_name=f"tune-{name}"):
        mlflow.log_param("model", name)
        mlflow.log_metrics({f"default_{m}": v for m, v in default.items()})
        mlflow.log_metrics({f"best_{m}": float(rows.loc[0, m]) for m in SCORING})
        mlflow.log_params({f"best.{k}": v for k, v in rows.loc[0, "params"].items()})
        for i, row in rows.iterrows():
            with mlflow.start_run(run_name=f"{name}-{i}", nested=True):
                mlflow.log_params(row["params"])
                mlflow.log_metrics({k: float(row[k]) for k in rows.columns if k != "params"})


def candidates(results: list[dict], train_params: dict, scoring: str) -> list[dict]:
    """Every model's best searched setting and its default setting."""
    out = []
    for r in results:
        for source, row, settings in (("search", r["best"], r["best"]["params"]),
                                      ("default", r["default"], train_params.get(r["model"], {}))):
            out.append({"model": r["model"], "settings": settings, "source": source, "score": row[scoring],
                        "std": row[f"{scoring}_std"], "fit_seconds": row["fit_seconds"]})
    return out


def winner(results: list[dict], train_params: dict, scoring: str) -> dict:
    """One-standard-error rule: the fastest model type whose best score is within one std of the overall best.

    Speed is the fit time of the tuned setting (measured in the same search for every model), so the
    noisy timing of single settings does not decide between two settings of the same model.
    """
    per_model = []
    for r in results:
        best_setting = max(candidates([r], train_params, scoring), key=lambda c: c["score"])
        per_model.append(best_setting | {"fit_seconds": r["best"]["fit_seconds"]})
    best = max(per_model, key=lambda c: c["score"])
    close = [c for c in per_model if c["score"] >= best["score"] - best["std"]]
    return min(close, key=lambda c: c["fit_seconds"]) | {"best_score": best["score"], "best_std": best["std"],
                                                        "best_model": best["model"]}


def to_markdown(results: list[dict], positive_rate: float, cfg: dict, train_params: dict) -> str:
    w = winner(results, train_params, cfg["scoring"])
    lines = [
        "# Model comparison and tuning",
        "",
        f"{train_params['cv_folds']}-fold cross-validation on the training set, up to {cfg['n_iter']} random settings per model, "
        f"selection by {cfg['scoring']}. A model that always predicts the base rate scores "
        f"average precision {positive_rate:.3f} and ROC AUC 0.500.",
        "",
        "| model | default AP | tuned AP | ± std | tuned ROC AUC | fit s (tuned) | settings tried | search min |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        b = r["best"]
        lines.append(f"| {r['model']} | {r['default']['average_precision']:.3f} | **{b['average_precision']:.3f}** | "
                     f"{b['average_precision_std']:.3f} | {b['roc_auc']:.3f} | {b['fit_seconds']:.1f} | {r['tried']} | "
                     f"{r['seconds'] / 60:.1f} |")
    lines += [
        "",
        f"Highest score: {w['best_model']} ({cfg['scoring']} {w['best_score']:.3f} ± {w['best_std']:.3f}). "
        f"One-standard-error rule: of the models with a score of at least {w['best_score'] - w['best_std']:.3f}, "
        f"the one that trains fastest is chosen, with its best settings.",
        "",
        f"Chosen: **{w['model']}** ({w['source']} settings, {cfg['scoring']} {w['score']:.3f}, "
        f"{w['fit_seconds']:.1f} s per fit):",
        "", "```yaml", f"model: {w['model']}", f"{w['model']}:",
    ]
    lines += [f"  {k}: {json.dumps(v, default=str)}" for k, v in w["settings"].items()]
    lines += ["```", ""]
    for r in results:
        lines += [f"## {r['model']}: top 5 settings", "", "| AP | ROC AUC | settings |", "|---|---|---|"]
        for row in r["top"]:
            lines.append(f"| {row['average_precision']:.3f} | {row['roc_auc']:.3f} | `{json.dumps(row['params'], default=str)}` |")
        lines.append("")
    return "\n".join(lines)


def write_report(results: list[dict], positive_rate: float, cfg: dict, train_params: dict) -> None:
    (REPORTS / "tuning.md").write_text(to_markdown(results, positive_rate, cfg, train_params))
    w = winner(results, train_params, cfg["scoring"])
    print(f"chosen: {w['model']} ({w['source']} settings, {cfg['scoring']} {w['score']:.3f}, "
          f"{w['fit_seconds']:.1f} s per fit): {w['settings']}")
    print(f"wrote {(REPORTS / 'tuning.md').relative_to(ROOT)}")


def report_only() -> None:
    """Rebuild reports/tuning.md from reports/tuning.json without searching again."""
    params = load_params()
    y = pd.read_csv(IN)["y"]
    write_report(json.loads((REPORTS / "tuning.json").read_text()), float(y.mean()), params["tune"], params["train"])


def main() -> None:
    params = load_params()
    train_params, cfg = params["train"], params["tune"]
    X, y = split_xy(pd.read_csv(IN))
    cv = StratifiedKFold(n_splits=train_params["cv_folds"], shuffle=True, random_state=train_params["seed"])
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"] + "-tuning")

    results = []
    for name in cfg["models"]:
        start = time.perf_counter()
        default = evaluate_default(X, y, deepcopy(train_params), name, cv)
        rows = search(X, y, train_params, name, cfg["spaces"][name], cfg["n_iter"], cfg["scoring"], cv)
        seconds = time.perf_counter() - start
        log_to_mlflow(name, default, rows)
        best = rows.iloc[0].to_dict()
        results.append({"model": name, "default": default, "best": best, "tried": len(rows), "seconds": seconds,
                        "top": rows.head(5).to_dict("records")})
        print(f"{name:<14} default AP {default['average_precision']:.3f} -> tuned {best['average_precision']:.3f} "
              f"(ROC AUC {best['roc_auc']:.3f}, {len(rows)} settings, {seconds:.0f}s) best: {best['params']}", flush=True)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "tuning.json").write_text(json.dumps(results, indent=2, default=str))
    write_report(results, float(y.mean()), cfg, train_params)



if __name__ == "__main__":
    import sys

    if "--report-only" in sys.argv:
        report_only()
    else:
        main()
