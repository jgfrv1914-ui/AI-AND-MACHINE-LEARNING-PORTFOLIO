"""Streamlit demo for the California housing price regressor."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "housing_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.csv"
DATA_PATH = ROOT / "data" / "california_housing.csv"

LABELS = {
    "MedInc": "Median block income (tens of thousands USD)",
    "HouseAge": "Median age of the houses (years)",
    "AveRooms": "Rooms per dwelling",
    "AveBedrms": "Bedrooms per dwelling",
    "Population": "Block population",
    "AveOccup": "Occupants per dwelling",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}

# Keys the artifact must carry; if any is missing the model is retrained.
REQUIRED_KEYS = {
    "pipeline", "features", "target", "model_name",
    "test_r2", "test_mae", "cv_r2_mean", "cv_r2_std",
}

st.set_page_config(page_title="HouseValue AI", page_icon="🏠", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    [data-testid="stMetricValue"] { color: #0f766e; }
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
    """Read the cached CSV, downloading the dataset if it is not there.

    The CSV is not versioned, so on a fresh deployment it does not exist.
    `load_dataset` fetches it from scikit-learn and writes the cache, which
    keeps the app working without a manual training run first.
    """
    if not DATA_PATH.exists():
        from train_model import load_dataset

        return load_dataset()
    return pd.read_csv(DATA_PATH)


st.title("🏠 HouseValue AI")
st.caption("Median house value estimation per California census district")

artifact = load_model()
data = load_data()
metrics = pd.read_csv(METRICS_PATH)

cols = st.columns(4)
cols[0].metric("Districts", f"{len(data):,}")
cols[1].metric("Active model", artifact["model_name"])
cols[2].metric("Test R²", f"{artifact['test_r2']:.3f}")
cols[3].metric(
    "Cross-validated R²",
    f"{artifact['cv_r2_mean']:.3f}",
    delta=f"± {artifact['cv_r2_std']:.3f}",
    delta_color="off",
)

st.divider()
left, right = st.columns([1, 1.4])

with left:
    st.subheader("District features")
    with st.form("prediction_form"):
        values = {}
        for feature in artifact["features"]:
            series = data[feature]
            values[feature] = st.number_input(
                LABELS.get(feature, feature),
                float(series.min()),
                float(series.max()),
                float(series.median()),
            )
        submitted = st.form_submit_button(
            "Estimate value", type="primary", use_container_width=True
        )

    if submitted:
        row = pd.DataFrame([values])[artifact["features"]]
        prediction = artifact["pipeline"].predict(row)[0]
        st.success(f"Estimated value: **${prediction * 100_000:,.0f}**")
        margin = artifact["test_mae"] * 100_000
        st.caption(
            f"Mean absolute error of the model on test: ±${margin:,.0f}. "
            "The dataset target is capped at $500,000, so the most expensive "
            "districts are underestimated by design."
        )

with right:
    st.subheader("Model comparison (validation split)")
    validation = metrics[metrics["split"] == "validation"]
    st.bar_chart(validation.set_index("model")[["MAE", "RMSE"]])

    st.subheader("Full metrics")
    st.dataframe(
        metrics.style.format(
            {c: "{:.3f}" for c in ["MAE", "RMSE", "R2", "R2_cv_mean", "R2_cv_std"]},
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Value distribution")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(data[artifact["target"]] * 100_000, bins=40, color="#0f766e", alpha=0.85)
    ax.set_xlabel("Median house value (USD)")
    ax.set_ylabel("Districts")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)

st.caption(
    "Real data from the 1990 California census (scikit-learn). Demonstration "
    "project: this is not an appraisal and must not be used for purchase decisions."
)
