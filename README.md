# urban-change-classification

Multi-class classification of urban development type from multi-date satellite observations. Kaggle challenge, 6 classes, **62% cross-validated accuracy** with a tuned XGBoost.

## Data

Each observation is a georeferenced polygon observed at 5 dates. Per date: a construction status label and RGB colour statistics of the satellite image. Target:

`Demolition` · `Road` · `Residential` · `Commercial` · `Industrial` · `Mega Projects`

## Feature engineering

**Dates → two distinct signals.** Absolute position (year, month, day, weekday) and elapsed days since the first observation. The latter measures the *rate* of transformation, which separates a residential development from a mega project.

**Geometry → scale and shape.** Area and perimeter, plus compactness

$$C = \frac{4\pi A}{P^2}$$

equal to 1 for a disc and tending to 0 for elongated shapes — the discriminant between roads and built parcels.

**Categorical status → indicator variables.** `pd.get_dummies` restricted to the status columns. A full `OneHotEncoder` over all categoricals produced a sparse space where most columns carried no signal.

## Dimensionality reduction

Encoding yields 309 features. A `SelectKBest` (χ²) curve under 5-fold CV rises steeply to $k \approx 60$, peaks near 80, then flattens.

![Accuracy vs number of selected features](docs/feature_selection.png)

Of the 108 highest-importance features, 81 cover 95% of cumulative importance.

## Models

| Model | CV accuracy |
|---|---:|
| k-nearest neighbours | ~40% |
| Random Forest | 50–52% |
| **XGBoost (tuned)** | **62%** |

k-NN suffers from distance concentration in high dimension. Random Forest handles the feature space but plateaus on overlapping classes. Boosting recovers the remaining 10 points by reweighting misclassified examples at each iteration.

Hyperparameters via `RandomizedSearchCV`, 50 draws over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`.

## Feature importance

![Feature importance](docs/feature_importance.png)

Construction status dominates: `change_status_date0_Prior Construction` alone accounts for 11% of total importance, about three times the next feature. Geometric features rank mid-table, date components lower, per-parcel colour statistics last.

This bounds the achievable accuracy — classes sharing a status trajectory (commercial vs industrial) are not separated by the most informative features.

## Usage

```bash
pip install -r requirements.txt
```

```python
import geopandas as gpd
from src.features import build_features, encode_target, align_columns, drop_missing_rows
from src.models import tune_xgboost, cumulative_importance_features, make_submission

train = gpd.read_file("data/train.geojson")
test = gpd.read_file("data/test.geojson")

y = encode_target(train["change_type"])
X, y = drop_missing_rows(train.drop(columns=["change_type"]), y)
X, X_test = align_columns(build_features(X), build_features(test))

search = tune_xgboost(X, y, n_iter=50)
kept = cumulative_importance_features(search.best_estimator_, list(X.columns), 0.95)
make_submission(search.best_estimator_, X_test[kept], test["index"])
```

Challenge data is not redistributed. Place `train.geojson` and `test.geojson` under `data/`.

## Layout

```
src/features.py   date decomposition, geometry, encoding, train/test alignment
src/models.py     baselines, SelectKBest curve, XGBoost search, submission
notebooks/        full run with outputs
tests/            14 tests, runnable without the challenge data
```

```bash
pytest
```

## Extensions

- Macro-F1 rather than accuracy — the global score hides per-class disparities on an imbalanced target
- Bayesian optimisation (Optuna) instead of random search
- Encode status *transitions* rather than per-date states: the sequence carries more information than the individual labels
- Spatial context features (distance to city centre, road network density)
