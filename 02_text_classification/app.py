"""Streamlit demo for the 20 Newsgroups topic classifier."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "text_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.csv"
PER_TOPIC_PATH = ROOT / "models" / "f1_per_topic.csv"
CONFUSION_PATH = ROOT / "models" / "confusion_matrix.csv"

EXAMPLES = {
    "Hardware / graphics": (
        "My new graphics card keeps crashing the X server when I enable "
        "hardware acceleration. I already updated the driver and swapped the "
        "PCI slot, but the monitor still goes black after a few minutes."
    ),
    "Space": (
        "The shuttle launch window depends on the orbital inclination and the "
        "amount of fuel required to reach the space station. NASA published the "
        "trajectory data for the next mission last week."
    ),
    "Sport": (
        "The goalie played an incredible game last night, he stopped almost "
        "every shot in the third period and the team won the series. Best "
        "playoff hockey I have watched in years."
    ),
    "Cryptography": (
        "If the encryption key is only 40 bits long, a brute force attack is "
        "entirely feasible. The government proposal to escrow private keys "
        "would weaken the security of every citizen."
    ),
    "Medicine": (
        "The patient was treated with antibiotics for two weeks but the "
        "infection came back. The doctor suggested a different diagnosis and "
        "ordered more blood tests before changing the medication."
    ),
}

# Keys the artifact must carry; if any is missing the model is retrained.
REQUIRED_KEYS = {
    "pipeline", "topics", "model_name",
    "test_accuracy", "test_f1_macro", "n_train", "n_test",
}

st.set_page_config(page_title="TextSort AI", page_icon="🗂️", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    [data-testid="stMetricValue"] { color: #6d28d9; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the model, retraining it if it is missing or out of date.

    The key check avoids a subtle failure: if the artifact on disk comes from an
    older version of the training script, the newer code would blow up with a
    KeyError. Here that is detected and the model is regenerated instead.
    """
    def _valid(path):
        if not path.exists():
            return None
        artifact = joblib.load(path)
        if not REQUIRED_KEYS.issubset(artifact):
            return None
        return artifact

    artifact = _valid(MODEL_PATH)
    if artifact is None:
        from train_model import main

        main()
        artifact = joblib.load(MODEL_PATH)
    return artifact


@st.cache_data
def load_tables():
    return (
        pd.read_csv(METRICS_PATH),
        pd.read_csv(PER_TOPIC_PATH),
        pd.read_csv(CONFUSION_PATH, index_col=0),
    )


def coefficients(model) -> np.ndarray | None:
    """Extract the weight matrix, including when the model is calibrated."""
    if hasattr(model, "coef_"):
        return model.coef_
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        weights = [c.estimator.coef_ for c in calibrated if hasattr(c.estimator, "coef_")]
        if weights:
            return np.mean(weights, axis=0)
    if hasattr(model, "feature_log_prob_"):
        return model.feature_log_prob_
    return None


st.title("🗂️ TextSort AI")
st.caption("Automatic text classification into 20 topics using TF-IDF and linear models")

artifact = load_model()
metrics, per_topic, confusion = load_tables()
pipeline = artifact["pipeline"]
topics = artifact["topics"]

cols = st.columns(4)
cols[0].metric("Topics", len(topics))
cols[1].metric("Active model", artifact["model_name"])
cols[2].metric("Accuracy (test)", f"{artifact['test_accuracy']:.3f}")
cols[3].metric("Macro F1 (test)", f"{artifact['test_f1_macro']:.3f}")

st.info(
    f"Trained on {artifact['n_train']:,} messages and evaluated on {artifact['n_test']:,} "
    "later ones in time. Headers, signatures and quotes were removed: without that "
    "cleanup the group name appears in the text itself and accuracy climbs "
    "artificially above 0.90.",
    icon="🧠",
)

st.divider()
left, right = st.columns([1, 1.15])

with left:
    st.subheader("Classify a text")
    example = st.selectbox("Load an example", ["(write my own text)"] + list(EXAMPLES))
    text = st.text_area(
        "Text in English",
        value=EXAMPLES.get(example, ""),
        height=190,
        placeholder="Paste an English text here...",
        help="The training corpus is in English, so the model only works in that language.",
    )

    if text.strip():
        probabilities = pipeline.predict_proba([text])[0]
        order = np.argsort(probabilities)[::-1]
        winner = order[0]

        st.success(f"Predicted topic: **{topics[winner]}** — confidence {probabilities[winner]:.1%}")
        if probabilities[winner] < 0.35:
            st.warning(
                "Low confidence: the text is short or ambiguous, or it covers a "
                "topic that is not among the 20 in the corpus.",
                icon="⚠️",
            )

        st.caption("Five most likely topics")
        st.bar_chart(
            pd.DataFrame(
                {"probability": probabilities[order[:5]]},
                index=[topics[i] for i in order[:5]],
            )
        )

        weights = coefficients(pipeline.named_steps["model"])
        if weights is not None:
            vectorizer = pipeline.named_steps["tfidf"]
            vector = vectorizer.transform([text])
            names = vectorizer.get_feature_names_out()
            present = vector.nonzero()[1]
            if len(present):
                contribution = pd.DataFrame(
                    {
                        "term": names[present],
                        "contribution": vector.toarray()[0][present] * weights[winner][present],
                    }
                ).sort_values("contribution", ascending=False)
                st.caption(f"Terms that pushed hardest towards “{topics[winner]}”")
                st.bar_chart(contribution.head(10).set_index("term")["contribution"])
    else:
        st.caption("Write or load a text to see the prediction.")

with right:
    st.subheader("Model performance")
    st.dataframe(
        metrics.style.format({c: "{:.3f}" for c in ["accuracy", "f1_macro", "f1_weighted"]}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "The validation rows were used to pick the model; the test row is the "
        "final measurement. The drop between them is real: the test set contains "
        "later messages, and forum language changes over time."
    )

    st.subheader("F1 per topic (test)")
    fig, ax = plt.subplots(figsize=(7, 6))
    rows = per_topic.sort_values("f1")
    ax.barh(rows["topic"], rows["f1"], color="#6d28d9")
    ax.axvline(artifact["test_f1_macro"], color="#be123c", linestyle="--", linewidth=1)
    ax.set_xlabel("F1")
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    st.caption(
        "Sports and technical topics separate cleanly. Religion and politics get "
        "confused with each other: they share almost all of their vocabulary."
    )

    with st.expander("Full confusion matrix"):
        st.dataframe(
            confusion.style.background_gradient(cmap="Purples", axis=None),
            use_container_width=True,
        )

st.caption(
    "Real data from the 20 Newsgroups corpus (Usenet messages, ~1995, in English), "
    "distributed with scikit-learn. Demonstration project for a text classification "
    "workflow: vectorisation, model comparison and explanation."
)
