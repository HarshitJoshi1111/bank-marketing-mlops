# Bank marketing: an end-to-end MLOps pipeline

A model that predicts whether a bank customer will subscribe to a term deposit after a marketing phone call, built as a full pipeline instead of a notebook: versioned data and steps with DVC, experiment tracking and a model registry with MLflow, a FastAPI service in Docker, and drift monitoring. Every number below can be reproduced with `dvc repro`.

## The problem

The [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) contains 41,188 calls of a Portuguese bank's telemarketing campaigns between May 2008 and November 2010 (CC BY 4.0). Only 11.7 % of the customers said yes, so a model that always predicts "no" is 88 % accurate and useless. The goal is to rank customers so that the call center can call the likely ones first, and to decide who counts as "likely".

The column `duration` (length of the call) is dropped: it is only known after the call has ended and almost decides the outcome by itself. Keeping it gives impressive scores that cannot be used before a call. The dataset description says the same.

## Pipeline

```
download → prepare → split → train → evaluate → register → export → monitor
```

| stage | what it does |
|---|---|
| download | fetches the dataset from UCI |
| prepare | validates the raw data (columns, types, allowed values, missing values, plausible ages) and stops with a list of all problems if the source changes; drops `duration` and 1,784 duplicate rows; turns `pdays = 999` into a "previously contacted" flag |
| split | stratified 80/20 train/test split |
| train | 5-fold cross-validation on the training set (out-of-fold predictions), chooses the decision threshold, trains the final model, logs parameters, metrics and the model to MLflow |
| evaluate | test metrics at the chosen threshold, ROC and precision-recall curves for `dvc plots` |
| register | quality gate and champion/challenger logic in the MLflow Model Registry |
| export | copies the current champion out of the registry for serving |
| monitor | data and prediction drift report (PSI) |

All settings live in `params.yaml`. DVC only re-runs the stages whose code, data or parameters changed, and `dvc metrics diff` compares the results with any earlier commit.

## Results

Test set (7,881 calls the model never saw during training or tuning):

| step | model | threshold | average precision | ROC AUC | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| baseline | always "no" | - | 0.117 | 0.500 | - | 0.00 | - |
| 1 | logistic regression, class weights | 0.5 | 0.470 | 0.796 | 0.36 | 0.65 | 0.47 |
| 2 | tuned gradient boosting | 0.5 | 0.495 | 0.800 | 0.71 | 0.28 | 0.40 |
| 3 | tuned gradient boosting | 0.236 (from CV) | 0.495 | 0.800 | 0.47 | 0.56 | **0.51** |

Step 2 ranks better but finds far fewer customers, because the tuned model predicts calibrated probabilities around the 12 % base rate, where a cut at 0.5 is far too high. Step 3 only changes the threshold. With it, the model finds 56 % of the subscribers while calling 14 % of the customers.

## How the decisions were made

**Model selection (`python -m src.tune`).** Randomized search with 5-fold cross-validation on the training set only, up to 20 settings per model, every setting logged to MLflow as a nested run ([reports/tuning.md](reports/tuning.md)):

| model | default AP | tuned AP | ± std over folds | seconds per fit |
|---|---|---|---|---|
| logistic regression | 0.445 | 0.451 | 0.023 | 0.7 |
| random forest | 0.469 | 0.471 | 0.019 | 21.9 |
| gradient boosting | 0.466 | 0.470 | 0.021 | 1.9 |

The random forest is 0.001 better, but the fold-to-fold noise is 0.02, so the difference is not real. The selection uses the one-standard-error rule: among the models within one standard deviation of the best, take the fastest. Gradient boosting (100 trees with 7 leaves) trains 12 times faster and gives a much smaller model.

**Threshold.** Chosen on the out-of-fold predictions of the training set, never on the test set. The default rule maximizes F1; `min_recall` (find at least X % of the subscribers) and a fixed value are available in `params.yaml`, because the right trade-off depends on what a call costs compared with a missed customer.

**Registry.** A model is only registered if its test average precision is at least 0.40, and the `champion` alias only moves to a new version if it is at least as good as the current champion. A worse model stays a challenger version and never reaches the API by accident.

**Drift.** The data is ordered by call date, so the monitor compares the training set with the most recent 4,000 calls ([reports/drift.md](reports/drift.md)). Result: retrain. The economic indicators moved strongly (3-month Euribor from 3.6 to 0.8 on average, number of employees down, employment variation rate from +0.1 to -2.3) and the share of subscribers rose from 12 % to 46 %. Month and weekday always differ between periods, so they are reported but do not trigger a retrain. These recent calls are partly in the training set because the split is random, so the report shows the method; in production it would run on new calls.

## Run it

Python 3.11:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc repro                       # downloads the data and runs the whole pipeline
dvc metrics show
mlflow ui --backend-store-uri sqlite:///mlflow.db     # runs, metrics and the model registry
```

Model comparison and tuning (takes a few minutes, not part of `dvc repro`):

```bash
python -m src.tune
```

### Prediction API

```bash
uvicorn serve.app:app --port 8000          # docs at http://localhost:8000/docs
```

or in Docker, after `dvc repro` has exported the champion:

```bash
docker build -t bank-marketing-api .
docker run -p 8000:8000 bank-marketing-api
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"customers": [{
  "age": 60, "job": "retired", "marital": "married", "education": "basic.4y", "default": "no",
  "housing": "no", "loan": "no", "contact": "cellular", "month": "oct", "day_of_week": "tue",
  "campaign": 1, "pdays": 6, "previous": 2, "poutcome": "success", "emp.var.rate": -3.4,
  "cons.price.idx": 92.431, "cons.conf.idx": -26.9, "euribor3m": 0.742, "nr.employed": 5017.5}]}'
# {"model_version":1,"threshold":0.2359,"predictions":[{"probability":0.7148,"subscribe":true}]}
```

The API validates every field (types, ranges, allowed category values) and rejects `duration`. It builds its features with the same function as the training pipeline, in the same column order and data types. The image contains only the API, that feature code and the exported model, with the library versions of the training environment pinned in `requirements-serve.txt` (a saved scikit-learn model needs the same versions to load).

### Tests

```bash
python -m pytest     # 36 tests, no data or running services needed
ruff check .
```

## Project structure

```
src/
  download.py     dataset from UCI (also unpacks the nested zip)
  prepare.py      validation and feature engineering (shared with the API)
  split.py        stratified train/test split
  features.py     preprocessing + model pipeline (logistic regression, random forest, gradient boosting)
  train.py        cross-validation, threshold, final model, MLflow logging
  threshold.py    threshold rules
  evaluate.py     test metrics and curves
  register.py     quality gate and champion/challenger
  export.py       champion from the registry -> models/serving
  monitor.py      PSI drift report
  tune.py         model comparison and randomized search
serve/app.py      FastAPI service
tests/            unit tests
dvc.yaml          pipeline, params.yaml all settings, dvc.lock the exact state of the last run
```

## Limitations and next steps

- **Random split.** Calls from the same period are in both training and test data. A split by date would show how the model holds up on the future; the drift report suggests it would score worse on the last months.
- **No cost model.** The threshold maximizes F1. With real numbers for the cost of a call and the value of a deposit, it should maximize expected profit instead.
- **Monitoring is a batch report.** The API does not log its requests yet, so drift is checked on data from the pipeline, not on live traffic.
- **Local only.** DVC has no remote and MLflow runs on SQLite. For a team, both would move to shared storage (for example S3 and a tracking server), and the Docker image would run on a cloud service.
