# Model comparison and tuning

5-fold cross-validation on the training set, up to 20 random settings per model, selection by average_precision. A model that always predicts the base rate scores average precision 0.117 and ROC AUC 0.500.

| model | default AP | tuned AP | ± std | tuned ROC AUC | fit s (tuned) | settings tried | search min |
|---|---|---|---|---|---|---|---|
| logreg | 0.445 | **0.451** | 0.023 | 0.790 | 0.7 | 16 | 0.2 |
| random_forest | 0.469 | **0.471** | 0.019 | 0.801 | 21.9 | 20 | 5.7 |
| hist_gb | 0.466 | **0.470** | 0.021 | 0.801 | 1.9 | 20 | 0.9 |

Highest score: random_forest (average_precision 0.471 ± 0.019). One-standard-error rule: of the models with a score of at least 0.452, the one that trains fastest is chosen, with its best settings.

Chosen: **hist_gb** (search settings, average_precision 0.470, 1.9 s per fit):

```yaml
model: hist_gb
hist_gb:
  min_samples_leaf: 20
  max_leaf_nodes: 7
  max_iter: 100
  learning_rate: 0.1
  l2_regularization: 0.0
  class_weight: null
```

## logreg: top 5 settings

| AP | ROC AUC | settings |
|---|---|---|
| 0.451 | 0.790 | `{"class_weight": null, "C": 3.0}` |
| 0.450 | 0.790 | `{"class_weight": null, "C": 10.0}` |
| 0.450 | 0.790 | `{"class_weight": null, "C": 0.3}` |
| 0.450 | 0.790 | `{"class_weight": null, "C": 1.0}` |
| 0.450 | 0.791 | `{"class_weight": null, "C": 0.1}` |

## random_forest: top 5 settings

| AP | ROC AUC | settings |
|---|---|---|
| 0.471 | 0.801 | `{"n_estimators": 400, "min_samples_leaf": 20, "max_features": 0.3, "max_depth": 14, "class_weight": "balanced_subsample"}` |
| 0.470 | 0.802 | `{"n_estimators": 200, "min_samples_leaf": 20, "max_features": 0.3, "max_depth": 10, "class_weight": null}` |
| 0.469 | 0.801 | `{"n_estimators": 400, "min_samples_leaf": 10, "max_features": 0.5, "max_depth": 8, "class_weight": null}` |
| 0.468 | 0.799 | `{"n_estimators": 400, "min_samples_leaf": 20, "max_features": 0.5, "max_depth": 14, "class_weight": null}` |
| 0.468 | 0.795 | `{"n_estimators": 200, "min_samples_leaf": 10, "max_features": "sqrt", "max_depth": null, "class_weight": null}` |

## hist_gb: top 5 settings

| AP | ROC AUC | settings |
|---|---|---|
| 0.470 | 0.801 | `{"min_samples_leaf": 20, "max_leaf_nodes": 7, "max_iter": 100, "learning_rate": 0.1, "l2_regularization": 0.0, "class_weight": null}` |
| 0.470 | 0.801 | `{"min_samples_leaf": 50, "max_leaf_nodes": 31, "max_iter": 100, "learning_rate": 0.1, "l2_regularization": 1.0, "class_weight": null}` |
| 0.469 | 0.802 | `{"min_samples_leaf": 50, "max_leaf_nodes": 31, "max_iter": 200, "learning_rate": 0.02, "l2_regularization": 1.0, "class_weight": "balanced"}` |
| 0.469 | 0.803 | `{"min_samples_leaf": 50, "max_leaf_nodes": 15, "max_iter": 200, "learning_rate": 0.02, "l2_regularization": 1.0, "class_weight": null}` |
| 0.468 | 0.804 | `{"min_samples_leaf": 100, "max_leaf_nodes": 15, "max_iter": 100, "learning_rate": 0.05, "l2_regularization": 0.1, "class_weight": null}` |
