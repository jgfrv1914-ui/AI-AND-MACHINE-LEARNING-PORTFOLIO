"""Clasificacion de texto en 20 temas (20 Newsgroups).

Dos decisiones que condicionan todo el proyecto:

1. Se eliminan cabeceras, firmas y citas (`remove=("headers", "footers", "quotes")`).
   Las cabeceras originales incluyen el nombre del grupo al que pertenece el
   mensaje, asi que dejarlas convierte la tarea en copiar la respuesta: la
   exactitud sube por encima del 0,90 sin que el modelo aprenda nada del texto.
   Con el texto limpio la tarea es real y las cifras son mucho mas bajas, pero
   significan algo.
2. La particion train/test es la oficial del dataset y es TEMPORAL: el test son
   mensajes posteriores a los de entrenamiento. Es mas exigente que una particion
   aleatoria y se parece mas a como se usaria el modelo en produccion.

De los datos de entrenamiento se aparta un 20 % de validacion para elegir modelo.
El test se usa una sola vez, al final, solo con el ganador.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
# Sin cabeceras ni firmas: si no, el nombre del grupo filtra la respuesta.
LIMPIEZA = ("headers", "footers", "quotes")


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        sublinear_tf=True,      # amortigua terminos muy repetidos
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),     # unigramas y bigramas
        min_df=3,               # descarta rarezas y erratas
        max_df=0.7,             # descarta terminos casi universales
    )


def build_models() -> dict:
    return {
        "Naive Bayes complementario": ComplementNB(alpha=0.3),
        "Regresión logística": LogisticRegression(
            C=6.0, max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE
        ),
        # LinearSVC no da probabilidades; se calibra para poder mostrar
        # confianza en la demo. ensemble=False guarda un unico estimador en
        # lugar de uno por pliegue: mismo resultado y un tercio del tamano en
        # disco, que importa para poder versionar el modelo en el repositorio.
        "SVM lineal (calibrada)": CalibratedClassifierCV(
            LinearSVC(C=0.6, random_state=RANDOM_STATE), cv=3, ensemble=False
        ),
    }


def score(y_true, y_pred) -> dict:
    return {
        "exactitud": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_ponderado": f1_score(y_true, y_pred, average="weighted"),
    }


def main() -> None:
    train_raw = fetch_20newsgroups(subset="train", remove=LIMPIEZA, random_state=RANDOM_STATE)
    test_raw = fetch_20newsgroups(subset="test", remove=LIMPIEZA, random_state=RANDOM_STATE)
    temas = list(train_raw.target_names)

    # Algunos mensajes quedan vacios tras quitar cabeceras y citas.
    def limpiar(textos, etiquetas):
        pares = [(t, e) for t, e in zip(textos, etiquetas) if t.strip()]
        return [p[0] for p in pares], np.array([p[1] for p in pares])

    x_full, y_full = limpiar(train_raw.data, train_raw.target)
    x_test, y_test = limpiar(test_raw.data, test_raw.target)

    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=0.20, stratify=y_full, random_state=RANDOM_STATE
    )
    print(f"train={len(x_train)}  val={len(x_val)}  test={len(x_test)}  temas={len(temas)}\n")

    # --- 1. Eleccion del modelo usando SOLO validacion -------------------
    rows, fitted = [], {}
    for name, estimator in build_models().items():
        pipeline = Pipeline([("tfidf", build_vectorizer()), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        rows.append({"modelo": name, "particion": "validación", **score(y_val, pipeline.predict(x_val))})
        print(f"  entrenado: {name}")

    validation = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    best_name = validation.iloc[0]["modelo"]

    # --- 2. Reentrenar el ganador con train+val y medir en test ----------
    best = Pipeline([("tfidf", build_vectorizer()), ("model", build_models()[best_name])])
    best.fit(x_full, y_full)
    y_pred = best.predict(x_test)
    test_score = score(y_test, y_pred)

    metrics = pd.concat(
        [validation, pd.DataFrame([{"modelo": best_name, "particion": "test (final)", **test_score}])],
        ignore_index=True,
    )
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)

    # Matriz de confusion del ganador, para ver que temas se confunden.
    pd.DataFrame(confusion_matrix(y_test, y_pred), index=temas, columns=temas).to_csv(
        MODEL_DIR / "confusion_matrix.csv"
    )
    # F1 por tema: revela que el modelo no es igual de bueno en todos.
    por_tema = pd.DataFrame(
        {
            "tema": temas,
            "f1": f1_score(y_test, y_pred, average=None, labels=range(len(temas))),
            "mensajes_test": np.bincount(y_test, minlength=len(temas)),
        }
    ).sort_values("f1", ascending=False)
    por_tema.to_csv(MODEL_DIR / "f1_por_tema.csv", index=False)

    joblib.dump(
        {
            "pipeline": best,
            "temas": temas,
            "model_name": best_name,
            "test_accuracy": test_score["exactitud"],
            "test_f1_macro": test_score["f1_macro"],
            "n_train": len(x_full),
            "n_test": len(x_test),
        },
        MODEL_DIR / "text_model.joblib",
    )
    (MODEL_DIR / "temas.json").write_text(json.dumps(temas, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n=== Validación (para elegir modelo) ===")
    print(validation.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n=== Test final — {best_name} ===")
    for key, value in test_score.items():
        print(f"  {key:14s} {value:.4f}")
    print("\n=== Mejores y peores temas (F1 en test) ===")
    print(por_tema.head(3).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("  ...")
    print(por_tema.tail(3).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
