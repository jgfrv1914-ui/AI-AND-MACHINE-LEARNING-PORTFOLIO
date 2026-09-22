"""Deteccion de fraude con clases muy desbalanceadas (0,17 % de positivos).

Tres decisiones que definen este proyecto:

1. El desbalanceo se corrige UNA sola vez. Aplicar SMOTE (que equilibra a 50/50)
   y ademas scale_pos_weight (~578) sobre los datos ya equilibrados hace que el
   modelo sobre-prediga fraude de forma masiva: medido sobre este mismo test,
   esa doble correccion hunde la precision de 0,90 a 0,30 y el F1 de 0,86 a 0,45.
2. El umbral se elige en VALIDACION, nunca en test. Ajustarlo sobre test es fuga
   de informacion: la metrica publicada dejaria de ser honesta.
3. `Time` (segundos desde la primera transaccion, 48 h en total) no se usa cruda.
   Se convierte a hora del dia, que si es una senal real de comportamiento.
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
FEATURES = ["hora_del_dia"] + PCA_FEATURES + ["Amount"]
RANDOM_STATE = 42
# Rejilla amplia: con clases muy desbalanceadas el umbral optimo suele estar
# muy alto, y una rejilla que termina en 0,90 lo dejaria truncado.
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
    data = pd.read_csv(csv_path).rename(columns={"Class": "fraude"})
    # 'Time' cruda es solo un contador; la hora del dia si es informativa.
    data["hora_del_dia"] = (data["Time"] / 3600) % 24
    return data


def build_models(scale_pos_weight: float) -> dict:
    """Cada modelo corrige el desbalanceo por UNA sola via."""
    return {
        # Via 1: pesos de clase, sin remuestreo.
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
        # Via 2: pesos de clase en el booster, sin SMOTE.
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
        # Via 3: remuestreo SMOTE. scale_pos_weight se queda en 1 a proposito:
        # SMOTE ya equilibro las clases, volver a pesarlas seria contarlo dos veces.
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
    """Umbral que maximiza F1. Se llama SIEMPRE con la particion de validacion."""
    return float(max(THRESHOLDS, key=lambda t: f1_score(y_true, probabilities >= t)))


def evaluate(y_true, probabilities, threshold: float) -> dict:
    predictions = probabilities >= threshold
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "umbral": threshold,
    }


def main() -> None:
    data = load_dataset()
    x, y = data[FEATURES], data["fraude"]

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(
        f"train={len(x_train)} ({y_train.sum()} fraudes)  "
        f"val={len(x_val)} ({y_val.sum()})  test={len(x_test)} ({y_test.sum()})"
    )
    print(f"scale_pos_weight = {scale_pos_weight:.0f}\n")

    # --- 1. Entrenar y elegir usando SOLO validacion ---------------------
    rows, fitted, thresholds = [], {}, {}
    for name, (estimator, usa_smote) in build_models(scale_pos_weight).items():
        steps = ([("smote", SMOTE(random_state=RANDOM_STATE))] if usa_smote else []) + [
            ("model", estimator)
        ]
        pipeline = Pipeline(steps).fit(x_train, y_train)
        prob_val = pipeline.predict_proba(x_val)[:, 1]
        threshold = best_threshold(y_val, prob_val)
        fitted[name], thresholds[name] = pipeline, threshold
        rows.append({"modelo": name, "particion": "validación", **evaluate(y_val, prob_val, threshold)})
        print(f"  entrenado: {name}  (umbral óptimo en validación = {threshold:.3f})")

    validation = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    best_name = validation.iloc[0]["modelo"]
    best_pipeline, best_t = fitted[best_name], thresholds[best_name]

    # --- 2. Medicion final del ganador, una sola vez, sobre test ---------
    prob_test = best_pipeline.predict_proba(x_test)[:, 1]
    test_row = {"modelo": best_name, "particion": "test (final)", **evaluate(y_test, prob_test, best_t)}

    metrics = pd.concat([validation, pd.DataFrame([test_row])], ignore_index=True)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)

    # Muestra ligera para que la demo funcione desplegada: el CSV completo pesa
    # 144 MB y no puede versionarse. Se guardan TODOS los fraudes y una muestra
    # de transacciones legitimas, de modo que la app siga teniendo casos reales.
    muestra = pd.concat(
        [
            data[data["fraude"] == 1],
            data[data["fraude"] == 0].sample(12_000, random_state=RANDOM_STATE),
        ]
    ).sample(frac=1, random_state=RANDOM_STATE)
    muestra.to_csv(DATA_DIR / "muestra_demo.csv", index=False)
    print("")
    print(f"Muestra para la demo: {len(muestra)} filas ({muestra['fraude'].sum()} fraudes)")

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
    print("\n=== Validación (para elegir modelo y umbral) ===")
    print(validation.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n=== Test final — {best_name} (umbral {best_t:.3f}) ===")
    for key in ["precision", "recall", "f1", "pr_auc", "roc_auc"]:
        print(f"  {key:10s} {test_row[key]:.4f}")


if __name__ == "__main__":
    main()
