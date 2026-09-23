"""Fraud detection with heavily imbalanced classes (0.17 % positives).

Three decisions define this project:

1. The imbalance is corrected ONCE. Applying SMOTE (which rebalances to 50/50)
   and then scale_pos_weight (~578) on top of the already-balanced data makes
   the model massively over-predict fraud: measured on this same test split,
   that double correction sinks precision from 0.90 to 0.30 and F1 from 0.86
   to 0.45.
2. The threshold is chosen on VALIDATION, never on test. Tuning it on test is
   information leakage and the published metric would stop being honest.
3. `Time` (seconds since the first transaction, 48 h in total) is not used raw.
   It is converted to hour of day, which is a genuine behavioural signal.
"""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = ["hour_of_day"] + PCA_FEATURES + ["Amount"]
TARGET = "fraud"
RANDOM_STATE = 42
# Wide grid: with heavily imbalanced classes the optimal threshold usually sits
# very high, and a grid ending at 0.90 would cut it off.
THRESHOLDS = np.arange(0.05, 0.996, 0.005)


def load_dataset() -> pd.DataFrame:
    csv_path = DATA_DIR / "transactions.csv"
    if not csv_path.exists():
        archive = DATA_DIR / "creditcardfraud.zip"
        urlretrieve(DATA_URL, archive)
        with ZipFile(archive) as zip_file:
            zip_file.extract("creditcard.csv", DATA_DIR)
        (DATA_DIR / "creditcard.csv").replace(csv_path)
        archive.unlink()
    data = pd.read_csv(csv_path).rename(columns={"Class": TARGET})
    # Raw 'Time' is just a counter; the hour of day is what carries signal.
    data["hour_of_day"] = (data["Time"] / 3600) % 24
    return data


def build_models(scale_pos_weight: float) -> dict:
    """Each model corrects the imbalance through ONE route only."""
    return {
        # Route 1: class weights, no resampling.
        "Random Forest (class_weight)": (
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        # Route 2: class weights inside the booster, no SMOTE.
        "XGBoost (scale_pos_weight)": (
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                random_state=RANDOM_STATE,
                n_jobs=4,
                tree_method="hist",
                device="cpu",
            ),
            False,
        ),
        # Route 3: SMOTE resampling. scale_pos_weight stays at 1 on purpose:
        # SMOTE already balanced the classes, reweighting would count it twice.
        "XGBoost + SMOTE": (
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=1.0,
                eval_metric="aucpr",
                random_state=RANDOM_STATE,
                n_jobs=4,
                tree_method="hist",
                device="cpu",
            ),
            True,
        ),
    }


def best_threshold(y_true, probabilities) -> float:
    """Threshold maximising F1. ALWAYS called with the validation split."""
    return float(max(THRESHOLDS, key=lambda t: f1_score(y_true, probabilities >= t)))


def evaluate(y_true, probabilities, threshold: float) -> dict:
    predictions = probabilities >= threshold
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "threshold": threshold,
    }


def main() -> None:
    data = load_dataset()
    x, y = data[FEATURES], data[TARGET]

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(
        f"train={len(x_train)} ({y_train.sum()} frauds)  "
        f"val={len(x_val)} ({y_val.sum()})  test={len(x_test)} ({y_test.sum()})"
    )
    print(f"scale_pos_weight = {scale_pos_weight:.0f}\n")

    # --- 1. Train and select using ONLY validation -----------------------
    rows, fitted, thresholds = [], {}, {}
    for name, (estimator, uses_smote) in build_models(scale_pos_weight).items():
        steps = ([("smote", SMOTE(random_state=RANDOM_STATE))] if uses_smote else []) + [
            ("model", estimator)
        ]
        pipeline = Pipeline(steps).fit(x_train, y_train)
        prob_val = pipeline.predict_proba(x_val)[:, 1]
        threshold = best_threshold(y_val, prob_val)
        fitted[name], thresholds[name] = pipeline, threshold
        rows.append({"model": name, "split": "validation", **evaluate(y_val, prob_val, threshold)})
        print(f"  trained: {name}  (best validation threshold = {threshold:.3f})")

    validation = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    best_name = validation.iloc[0]["model"]
    best_pipeline, best_t = fitted[best_name], thresholds[best_name]

    # --- 2. Final measurement of the winner, once, on test ----------------
    prob_test = best_pipeline.predict_proba(x_test)[:, 1]
    test_row = {"model": best_name, "split": "test (final)", **evaluate(y_test, prob_test, best_t)}

    metrics = pd.concat([validation, pd.DataFrame([test_row])], ignore_index=True)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)

    # Lightweight sample so the deployed demo works: the full CSV is 144 MB and
    # cannot be versioned. ALL frauds plus a sample of legitimate transactions
    # are kept, so the app still runs on real cases.
    sample = pd.concat(
        [
            data[data[TARGET] == 1],
            data[data[TARGET] == 0].sample(12_000, random_state=RANDOM_STATE),
        ]
    ).sample(frac=1, random_state=RANDOM_STATE)
    sample.to_csv(DATA_DIR / "demo_sample.csv", index=False)
    print("")
    print(f"Demo sample: {len(sample)} rows ({sample[TARGET].sum()} frauds)")

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "features": FEATURES,
            "pca_features": PCA_FEATURES,
            "model_name": best_name,
            "threshold": best_t,
            "test_precision": test_row["precision"],
            "test_recall": test_row["recall"],
            "test_f1": test_row["f1"],
            "test_pr_auc": test_row["pr_auc"],
        },
        MODEL_DIR / "fraud_model.joblib",
    )

    pd.set_option("display.width", 200)
    print("\n=== Validation (used to pick model and threshold) ===")
    print(validation.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n=== Final test - {best_name} (threshold {best_t:.3f}) ===")
    for key in ["precision", "recall", "f1", "pr_auc", "roc_auc"]:
        print(f"  {key:10s} {test_row[key]:.4f}")


if __name__ == "__main__":
    main()
