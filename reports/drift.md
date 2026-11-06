# Drift report

Reference: training set (31523 calls). Current: the most recent 4000 calls. PSI < 0.1 stable, 0.1-0.25 moderate, >= 0.25 drift.

**Decision: retrain** (12 features with drift, score PSI 6.685; calendar features are not used for the decision)

| | reference | current |
|---|---|---|
| actual positive rate | 0.117 | 0.464 |
| predicted positive rate (threshold 0.2359) | 0.138 | 0.906 |
| mean score | 0.116 | 0.461 |

| feature | PSI | status | reference mean | current mean |
|---|---|---|---|---|
| nr.employed | 8.959 | drift | 5166.028 | 5013.453 |
| euribor3m | 7.614 | drift | 3.601 | 0.799 |
| emp.var.rate | 7.140 | drift | 0.063 | -2.341 |
| cons.conf.idx | 5.227 | drift | -40.504 | -35.149 |
| cons.price.idx | 3.928 | drift | 93.577 | 93.255 |
| month | 1.494 | drift |  |  |
| poutcome | 0.814 | drift |  |  |
| previous | 0.721 | drift | 0.177 | 0.843 |
| job | 0.533 | drift |  |  |
| default | 0.446 | drift |  |  |
| age | 0.290 | drift | 40.18 | 43.417 |
| contact | 0.266 | drift |  |  |
| education | 0.152 | moderate |  |  |
| campaign | 0.128 | moderate | 2.622 | 1.84 |
| marital | 0.034 | stable |  |  |
| housing | 0.002 | stable |  |  |
| loan | 0.002 | stable |  |  |
| day_of_week | 0.002 | stable |  |  |
| pdays | 0.000 | stable | -0.732 | 1.068 |
| previously_contacted | 0.000 | stable | 0.038 | 0.288 |
