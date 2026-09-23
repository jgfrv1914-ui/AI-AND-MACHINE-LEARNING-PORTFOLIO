"""Regression on the California Housing dataset.

Methodology (identical across the three portfolio projects):
  - 60/20/20 split: models are fit on train, the winner is CHOSEN on
    validation, and only that winner is measured once on test. The test set
    takes part in no decision, so the published metric is not inflated.
  - R2 is also reported with 5-fold cross-validation over train+val, so every
    number comes with its standard deviation instead of a single lucky figure.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "MedInc",      # median block income (tens of thousands of USD)
    "HouseAge",    # median age of the houses
    "AveRooms",    # rooms per dwelling
    "AveBedrms",   # bedrooms per dwelling
    "Population",  # block population
    "AveOccup",    # occupants per dwelling
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"  # median value in hundreds of thousands of USD
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """Download California Housing on first run, then reuse the local CSV."""
    csv_path = DATA_DIR / "california_housing.csv"
    if not csv_path.exists():
        frame = fetch_california_housing(as_frame=True).frame
        frame.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path)


def build_models() -> dict:
    # Tree-based models are scale invariant, so only linear regression needs
    # a StandardScaler in front of it.
    return {
        "Linear regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.5,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=4,
            tree_method="hist",
            device="cpu",
        ),
    }


def score(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def main() -> None:
    data = load_dataset()
    x, y = data[FEATURES], data[TARGET]

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.40, random_state=RANDOM_STATE
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE
    )
    print(f"train={len(x_train)}  val={len(x_val)}  test={len(x_test)}\n")

    # --- 1. Model selection using ONLY the validation split --------------
    rows, fitted = [], {}
    for name, estimator in build_models().items():
        estimator.fit(x_train, y_train)
        fitted[name] = estimator
        rows.append({"model": name, **score(y_val, estimator.predict(x_val))})
        print(f"  trained: {name}")

    validation = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = validation.iloc[0]["model"]

    # --- 2. Stability: cross-validated R2 over train+val ------------------
    x_dev = pd.concat([x_train, x_val])
    y_dev = pd.concat([y_train, y_val])
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    for name, estimator in build_models().items():
        cv = cross_val_score(estimator, x_dev, y_dev, cv=folds, scoring="r2", n_jobs=1)
        cv_scores[name] = (cv.mean(), cv.std())

    # --- 3. Final measurement of the winner, once, on test ----------------
    best = fitted[best_name]
    test_score = score(y_test, best.predict(x_test))

    metrics = validation.copy()
    metrics["R2_cv_mean"] = metrics["model"].map(lambda n: cv_scores[n][0])
    metrics["R2_cv_std"] = metrics["model"].map(lambda n: cv_scores[n][1])
    metrics["split"] = "validation"
    metrics = pd.concat(
        [metrics, pd.DataFrame([{"model": best_name, **test_score, "split": "test (final)"}])],
        ignore_index=True,
    )
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)

    joblib.dump(
        {
            "pipeline": best,
            "features": FEATURES,
            "target": TARGET,
            "model_name": best_name,
            "test_r2": test_score["R2"],
            "test_mae": test_score["MAE"],
            "cv_r2_mean": cv_scores[best_name][0],
            "cv_r2_std": cv_scores[best_name][1],
        },
        MODEL_DIR / "housing_model.joblib",
    )

    print("\n=== Validation (used to pick the model) ===")
    print(validation.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== R2 with 5-fold cross-validation ===")
    for name, (mean, std) in cv_scores.items():
        print(f"  {name:18s} {mean:.3f} +/- {std:.3f}")
    print(f"\n=== Final test - {best_name} ===")
    print(f"  MAE  {test_score['MAE']:.4f}  (hundreds of thousands of USD)")
    print(f"  RMSE {test_score['RMSE']:.4f}")
    print(f"  R2   {test_score['R2']:.4f}")


if __name__ == "__main__":
    main()
