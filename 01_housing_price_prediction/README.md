# 01 · Housing price prediction

Regression on **California Housing** (1990 California census, 20,640 districts, 8 features). The target is the median house value per district.

## Results

| Metric | Value |
|---|---|
| Test R² | **0.850** |
| 5-fold cross-validated R² | **0.847 ± 0.007** |
| Test MAE | 0.296 (≈ $29,600) |
| Selected model | XGBoost |

Compared: linear regression (R² 0.606), Random Forest (0.804) and XGBoost (0.847).

## Methodology

A **60/20/20** split. The model is chosen on validation and only the winner is measured once on test, so the published figure is not inflated by having picked while looking at the result. R² is also reported with cross-validation, so every number comes with its standard deviation.

## Run

```bash
python 01_housing_price_prediction/train_model.py
streamlit run 01_housing_price_prediction/app.py
```

## Limitations

- The target is **capped at $500,000** in the original dataset: the most expensive districts appear truncated and the model underestimates them by design.
- This is 1990 data. It demonstrates the workflow; it is not fit for appraising anything today.
- `Latitude` and `Longitude` carry a lot of the signal, so the model generalises poorly outside California.

## Source

`sklearn.datasets.fetch_california_housing` — derived from the 1990 US census.

> **Note:** this project previously used *Boston Housing*, removed from scikit-learn in version 1.2 because it contains a feature built as a racial proxy. California Housing is the replacement recommended by the library itself.
