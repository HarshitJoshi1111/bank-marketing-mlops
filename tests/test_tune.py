from sklearn.model_selection import StratifiedKFold

from src.features import split_xy
from src.tune import evaluate_default, search, search_space, winner
from tests.test_features import PARAMS, toy_data


def test_search_space_prefixes_pipeline_step():
    assert search_space({"C": [0.1, 1.0]}) == {"model__C": [0.1, 1.0]}


def test_search_returns_rows_sorted_by_score_and_caps_iterations():
    X, y = split_xy(toy_data())
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    rows = search(X, y, PARAMS, "logreg", {"C": [0.01, 1.0, 100.0]}, n_iter=10, scoring="average_precision", cv=cv)
    assert len(rows) == 3  # only 3 settings exist, although 10 were requested
    assert rows["average_precision"].is_monotonic_decreasing
    assert set(rows.loc[0, "params"]) == {"C"}


def test_evaluate_default_scores_between_0_and_1():
    X, y = split_xy(toy_data())
    scores = evaluate_default(X, y, PARAMS, "logreg", StratifiedKFold(n_splits=2, shuffle=True, random_state=0))
    assert {"average_precision", "roc_auc", "average_precision_std", "fit_seconds"} <= set(scores)
    assert all(0 <= scores[m] <= 1 for m in ("average_precision", "roc_auc"))


def result(model, default_ap, best_ap, std, default_s, best_s, params):
    return {"model": model,
            "default": {"average_precision": default_ap, "average_precision_std": std, "fit_seconds": default_s},
            "best": {"params": params, "average_precision": best_ap, "average_precision_std": std, "fit_seconds": best_s}}


def test_winner_prefers_faster_model_within_one_std():
    results = [result("random_forest", 0.46, 0.471, 0.02, 10.0, 15.0, {"n_estimators": 400}),
               result("hist_gb", 0.46, 0.470, 0.02, 2.0, 1.0, {"max_iter": 100}),
               result("logreg", 0.40, 0.42, 0.02, 0.5, 0.5, {"C": 1.0})]  # fastest, but too far below
    w = winner(results, {}, "average_precision")
    assert (w["model"], w["source"], w["settings"]) == ("hist_gb", "search", {"max_iter": 100})
    assert w["best_model"] == "random_forest"


def test_winner_can_keep_default_settings():
    results = [result("logreg", 0.45, 0.44, 0.001, 0.5, 0.9, {"C": 0.1})]
    w = winner(results, {"logreg": {"C": 1.0}}, "average_precision")
    assert (w["source"], w["settings"]) == ("default", {"C": 1.0})
