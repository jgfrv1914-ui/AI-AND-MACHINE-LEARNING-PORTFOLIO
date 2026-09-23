"""Text classification into 20 topics (20 Newsgroups).

Two decisions shape the whole project:

1. Headers, signatures and quotes are stripped (`remove=("headers", "footers",
   "quotes")`). The original headers contain the name of the newsgroup the
   message belongs to, so leaving them in turns the task into copying the
   answer: accuracy climbs above 0.90 without the model learning anything from
   the text. With clean text the task is real and the figures are much lower,
   but they mean something.
2. The train/test split is the dataset's official one and it is TEMPORAL: the
   test set contains messages posted after the training ones. It is harder than
   a random split and closer to how the model would be used in production.

A 20 % validation slice is held out from the training data to pick the model.
The test set is used once, at the end, only with the winner.
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
# No headers or signatures: otherwise the group name leaks the answer.
STRIP = ("headers", "footers", "quotes")


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        sublinear_tf=True,      # dampens heavily repeated terms
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),     # unigrams and bigrams
        min_df=3,               # drops one-off terms and typos
        max_df=0.7,             # drops near-universal terms
    )


def build_models() -> dict:
    return {
        "Complement Naive Bayes": ComplementNB(alpha=0.3),
        "Logistic regression": LogisticRegression(
            C=6.0, max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE
        ),
        # LinearSVC gives no probabilities; it is calibrated so the demo can
        # show confidence. ensemble=False stores a single estimator instead of
        # one per fold: same result, a third of the size on disk, which matters
        # for being able to version the model in the repository.
        "Linear SVM (calibrated)": CalibratedClassifierCV(
            LinearSVC(C=0.6, random_state=RANDOM_STATE), cv=3, ensemble=False
        ),
    }


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def main() -> None:
    train_raw = fetch_20newsgroups(subset="train", remove=STRIP, random_state=RANDOM_STATE)
    test_raw = fetch_20newsgroups(subset="test", remove=STRIP, random_state=RANDOM_STATE)
    topics = list(train_raw.target_names)

    # Some messages end up empty once headers and quotes are removed.
    def drop_empty(texts, labels):
        pairs = [(t, e) for t, e in zip(texts, labels) if t.strip()]
        return [p[0] for p in pairs], np.array([p[1] for p in pairs])

    x_full, y_full = drop_empty(train_raw.data, train_raw.target)
    x_test, y_test = drop_empty(test_raw.data, test_raw.target)

    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=0.20, stratify=y_full, random_state=RANDOM_STATE
    )
    print(f"train={len(x_train)}  val={len(x_val)}  test={len(x_test)}  topics={len(topics)}\n")

    # --- 1. Model selection using ONLY the validation split --------------
    rows, fitted = [], {}
    for name, estimator in build_models().items():
        pipeline = Pipeline([("tfidf", build_vectorizer()), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        rows.append({"model": name, "split": "validation", **score(y_val, pipeline.predict(x_val))})
        print(f"  trained: {name}")

    validation = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    best_name = validation.iloc[0]["model"]

    # --- 2. Refit the winner on train+val and measure on test -------------
    best = Pipeline([("tfidf", build_vectorizer()), ("model", build_models()[best_name])])
    best.fit(x_full, y_full)
    y_pred = best.predict(x_test)
    test_score = score(y_test, y_pred)

    metrics = pd.concat(
        [validation, pd.DataFrame([{"model": best_name, "split": "test (final)", **test_score}])],
        ignore_index=True,
    )
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)

    # Confusion matrix of the winner, to see which topics get mixed up.
    pd.DataFrame(confusion_matrix(y_test, y_pred), index=topics, columns=topics).to_csv(
        MODEL_DIR / "confusion_matrix.csv"
    )
    # F1 per topic: shows the model is not equally good across all of them.
    per_topic = pd.DataFrame(
        {
            "topic": topics,
            "f1": f1_score(y_test, y_pred, average=None, labels=range(len(topics))),
            "test_messages": np.bincount(y_test, minlength=len(topics)),
        }
    ).sort_values("f1", ascending=False)
    per_topic.to_csv(MODEL_DIR / "f1_per_topic.csv", index=False)

    joblib.dump(
        {
            "pipeline": best,
            "topics": topics,
            "model_name": best_name,
            "test_accuracy": test_score["accuracy"],
            "test_f1_macro": test_score["f1_macro"],
            "n_train": len(x_full),
            "n_test": len(x_test),
        },
        MODEL_DIR / "text_model.joblib",
    )
    (MODEL_DIR / "topics.json").write_text(json.dumps(topics, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n=== Validation (used to pick the model) ===")
    print(validation.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n=== Final test - {best_name} ===")
    for key, value in test_score.items():
        print(f"  {key:14s} {value:.4f}")
    print("\n=== Best and worst topics (test F1) ===")
    print(per_topic.head(3).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("  ...")
    print(per_topic.tail(3).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
