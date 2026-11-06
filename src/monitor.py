"""Stage 8: data and prediction drift monitoring with the Population Stability Index (PSI).

Compares a batch of calls ("current") with the training data ("reference"):
  - PSI per feature: numbers are binned at the deciles of the reference, categories by value
  - PSI of the champion model's scores, and its predicted vs. actual positive rate
PSI < 0.1 is stable, 0.1-0.25 a moderate shift, >= 0.25 a significant shift (common rule of thumb).

The data is ordered by call date (May 2008 - Nov 2010), so the batch here is the most recent
`monitor.window` calls, a stand-in for "the calls of last month". These calls are also partly in the
training set (the split is random), so the report demonstrates the check; in production it would run
on calls the model has never seen, and a "retrain" result would trigger `dvc repro` with new data.
"""
import json

import joblib
import numpy as np
import pandas as pd

from src.features import split_xy
from src.params import ROOT, load_params

REFERENCE = ROOT / "data" / "split" / "train.csv"
PREPARED = ROOT / "data" / "prepared" / "bank.csv"
SERVING = ROOT / "models" / "serving"
REPORTS = ROOT / "reports"
EPS = 1e-4  # avoids log(0) for empty bins


def _psi(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    ref_share, cur_share = np.clip(ref_share, EPS, None), np.clip(cur_share, EPS, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    return _psi(ref, cur)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference) | set(current))
    ref = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    cur = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return _psi(ref, cur)


def status(psi: float, moderate: float, drift: float) -> str:
    return "drift" if psi >= drift else "moderate" if psi >= moderate else "stable"


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict) -> list[dict]:
    rows = []
    for col in reference.columns:
        numeric = pd.api.types.is_numeric_dtype(reference[col])
        psi = (psi_numeric(reference[col], current[col], cfg["bins"]) if numeric
               else psi_categorical(reference[col], current[col]))
        row = {"feature": col, "psi": round(psi, 4), "status": status(psi, cfg["psi_moderate"], cfg["psi_drift"])}
        if numeric:
            row |= {"reference_mean": round(float(reference[col].mean()), 3),
                    "current_mean": round(float(current[col].mean()), 3)}
        rows.append(row)
    return sorted(rows, key=lambda r: r["psi"], reverse=True)


def decide(features: list[dict], score_psi: float, cfg: dict) -> str:
    """Retrain when the scores or a non-calendar feature shifted significantly."""
    drifted = [f for f in features if f["status"] == "drift" and f["feature"] not in cfg["ignore_for_decision"]]
    return "retrain" if drifted or score_psi >= cfg["psi_drift"] else "ok"


def to_markdown(report: dict) -> str:
    s = report["scores"]
    lines = [
        "# Drift report", "",
        f"Reference: training set ({report['reference_rows']} calls). Current: the most recent "
        f"{report['current_rows']} calls. PSI < 0.1 stable, 0.1-0.25 moderate, >= 0.25 drift.", "",
        f"**Decision: {report['decision']}** ({report['drifted_features']} features with drift, "
        f"score PSI {s['psi']:.3f}; calendar features are not used for the decision)", "",
        "| | reference | current |", "|---|---|---|",
        f"| actual positive rate | {s['reference_actual_rate']:.3f} | {s['current_actual_rate']:.3f} |",
        f"| predicted positive rate (threshold {s['threshold']}) | {s['reference_predicted_rate']:.3f} | "
        f"{s['current_predicted_rate']:.3f} |",
        f"| mean score | {s['reference_mean_score']:.3f} | {s['current_mean_score']:.3f} |", "",
        "| feature | PSI | status | reference mean | current mean |", "|---|---|---|---|---|",
    ]
    for f in report["features"]:
        lines.append(f"| {f['feature']} | {f['psi']:.3f} | {f['status']} | {f.get('reference_mean', '')} | "
                     f"{f.get('current_mean', '')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    cfg = load_params()["monitor"]
    reference = pd.read_csv(REFERENCE)
    current = pd.read_csv(PREPARED).tail(cfg["window"])
    X_ref, y_ref = split_xy(reference)
    X_cur, y_cur = split_xy(current)

    model = joblib.load(SERVING / "model.joblib")
    meta = json.loads((SERVING / "meta.json").read_text())
    ref_scores = model.predict_proba(X_ref[meta["features"]])[:, 1]
    cur_scores = model.predict_proba(X_cur[meta["features"]])[:, 1]

    features = feature_drift(X_ref, X_cur, cfg)
    score_psi = psi_numeric(pd.Series(ref_scores), pd.Series(cur_scores), cfg["bins"])
    report = {
        "reference_rows": len(reference), "current_rows": len(current), "model_version": meta["version"],
        "decision": decide(features, score_psi, cfg),
        "drifted_features": sum(f["status"] == "drift" for f in features),
        "scores": {
            "psi": round(score_psi, 4), "threshold": meta["threshold"],
            "reference_actual_rate": round(float(y_ref.mean()), 4), "current_actual_rate": round(float(y_cur.mean()), 4),
            "reference_predicted_rate": round(float((ref_scores >= meta["threshold"]).mean()), 4),
            "current_predicted_rate": round(float((cur_scores >= meta["threshold"]).mean()), 4),
            "reference_mean_score": round(float(ref_scores.mean()), 4),
            "current_mean_score": round(float(cur_scores.mean()), 4),
        },
        "features": features,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "drift.json").write_text(json.dumps(report, indent=2))
    (REPORTS / "drift.md").write_text(to_markdown(report))
    top = ", ".join(f"{f['feature']} {f['psi']:.2f}" for f in features[:4])
    print(f"decision: {report['decision']} ({report['drifted_features']} drifted features; top PSI: {top}; "
          f"score PSI {score_psi:.2f}; positive rate {y_ref.mean():.3f} -> {y_cur.mean():.3f})")


if __name__ == "__main__":
    main()
