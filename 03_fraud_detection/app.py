"""Streamlit demo for the credit card fraud detector."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.csv"
DATA_PATH = ROOT / "data" / "transactions.csv"
SAMPLE_PATH = ROOT / "data" / "demo_sample.csv"

# Keys the artifact must carry; if any is missing the model is retrained.
REQUIRED_KEYS = {
    "pipeline", "features", "pca_features", "model_name", "threshold",
    "test_precision", "test_recall", "test_f1", "test_pr_auc",
}

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    [data-testid="stMetricValue"] { color: #be123c; }
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
def load_data():
    """Full CSV when it has been downloaded, otherwise the versioned sample.

    The whole dataset is 144 MB and does not travel in the repository, so on a
    deployment the app runs on the sample (every fraud + 12,000 legitimate
    transactions).
    """
    path = DATA_PATH if DATA_PATH.exists() else SAMPLE_PATH
    data = pd.read_csv(path).rename(columns={"Class": "fraud"})
    if "hour_of_day" not in data.columns:
        data["hour_of_day"] = (data["Time"] / 3600) % 24
    return data, path == DATA_PATH


st.title("🛡️ FraudShield AI")
st.caption("Risk scoring for transactions with heavily imbalanced classes")

artifact = load_model()
data, is_full = load_data()
metrics = pd.read_csv(METRICS_PATH)
features = artifact["features"]

cols = st.columns(5)
cols[0].metric("Transactions", f"{len(data):,}")
cols[1].metric(
    "Frauds", f"{data['fraud'].sum():,}",
    delta=f"{data['fraud'].mean():.3%}", delta_color="off",
)
cols[2].metric("Precision (test)", f"{artifact['test_precision']:.3f}")
cols[3].metric("Recall (test)", f"{artifact['test_recall']:.3f}")
cols[4].metric("PR-AUC (test)", f"{artifact['test_pr_auc']:.3f}")

st.info(
    f"Active model: **{artifact['model_name']}** · threshold **{artifact['threshold']:.3f}**, "
    "chosen on the validation split and measured once on test.",
    icon="🧠",
)

if not is_full:
    st.warning(
        "Running on the sample included in the repository (every fraud + 12,000 "
        "legitimate transactions). The metrics shown are still those of the full "
        "test split. To load all 284,807 transactions, run `train_model.py` locally.",
        icon="📦",
    )

st.divider()
left, right = st.columns([1, 1.3])

with left:
    st.subheader("Analyse a transaction")
    st.caption(
        "Variables V1–V28 are anonymised PCA components: they carry no individual "
        "meaning, so the useful starting point is a real transaction."
    )

    source = st.radio(
        "Transaction to evaluate",
        ["Real fraud (random)", "Legitimate transaction (random)", "Enter values"],
        label_visibility="collapsed",
    )

    if "seed" not in st.session_state:
        st.session_state.seed = 0
    if st.button("🎲 Pick another transaction", use_container_width=True):
        st.session_state.seed += 1

    if source == "Enter values":
        base = {f: 0.0 for f in features}
        base["Amount"] = float(data["Amount"].median())
        base["hour_of_day"] = 12.0
        with st.expander("PCA components V1–V28", expanded=False):
            for f in artifact["pca_features"]:
                base[f] = st.number_input(f, value=0.0, format="%.5f")
        base["Amount"] = st.number_input("Amount", min_value=0.0, value=base["Amount"])
        base["hour_of_day"] = st.slider("Hour of day", 0.0, 24.0, 12.0, 0.5)
        row = pd.DataFrame([base])[features]
    else:
        target = 1 if source.startswith("Real fraud") else 0
        pool = data[data["fraud"] == target]
        idx = pool.index[st.session_state.seed % len(pool)]
        row = data.loc[[idx], features]
        st.caption(
            f"Transaction #{idx} — amount ${data.loc[idx, 'Amount']:,.2f}, "
            f"hour {data.loc[idx, 'hour_of_day']:.1f}h · "
            f"true label: **{'FRAUD' if target else 'legitimate'}**"
        )

    threshold = st.slider(
        "Alert threshold", 0.05, 0.995, float(artifact["threshold"]), 0.005,
        help="Lowering it catches more fraud but raises more false alarms.",
    )

    probability = artifact["pipeline"].predict_proba(row)[0, 1]
    if probability >= threshold:
        st.error(f"🚨 Fraud alert — probability **{probability:.2%}**")
    else:
        st.success(f"✅ Low risk — probability **{probability:.2%}**")

    st.subheader("Why this decision (SHAP)")
    model = artifact["pipeline"].named_steps["model"]
    values = shap.TreeExplainer(model).shap_values(row)
    if isinstance(values, list):
        contributions = values[1][0]
    elif getattr(values, "ndim", 0) == 3:
        contributions = values[0, :, 1]
    else:
        contributions = values[0]
    impact = (
        pd.DataFrame({"variable": features, "impact": np.asarray(contributions).ravel()})
        .sort_values("impact", key=abs, ascending=False)
        .head(12)
        .set_index("variable")
    )
    st.bar_chart(impact["impact"])
    st.caption("Positive values push towards fraud; negative ones towards a legitimate transaction.")

with right:
    st.subheader("Model performance")
    st.dataframe(
        metrics.style.format(
            {c: "{:.3f}" for c in ["precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"]}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "The validation rows were used to pick the model and the threshold. "
        "The test row is the final measurement and took part in no decision."
    )

    st.subheader("Precision / recall trade-off")
    n_sample = min(40_000, len(data))
    sub = data.sample(n_sample, random_state=0)
    prob_sample = artifact["pipeline"].predict_proba(sub[features])[:, 1]
    y_sample = sub["fraud"].to_numpy()
    grid = np.arange(0.05, 0.996, 0.01)
    curve = pd.DataFrame(
        {
            "threshold": grid,
            "precision": [
                (y_sample[prob_sample >= t].mean() if (prob_sample >= t).any() else np.nan)
                for t in grid
            ],
            "recall": [
                (prob_sample[y_sample == 1] >= t).mean() for t in grid
            ],
        }
    ).set_index("threshold")
    st.line_chart(curve)

    st.subheader("Global feature importance")
    importances = pd.Series(model.feature_importances_, index=features).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot.barh(ax=ax, color="#be123c")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)

st.caption(
    "Real data from the Credit Card Fraud Detection dataset (ULB, Kaggle): 284,807 "
    "European transactions from 2013, 492 frauds. Demonstration project: a real "
    "system needs human review, business rules and drift monitoring."
)
