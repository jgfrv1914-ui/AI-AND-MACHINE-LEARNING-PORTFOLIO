from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "housing_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.csv"
DATA_PATH = ROOT / "data" / "california_housing.csv"

ETIQUETAS = {
    "MedInc": "Ingreso mediano del bloque (decenas de miles USD)",
    "HouseAge": "Antigüedad mediana de las viviendas (años)",
    "AveRooms": "Habitaciones por vivienda",
    "AveBedrms": "Dormitorios por vivienda",
    "Population": "Población del bloque",
    "AveOccup": "Ocupantes por vivienda",
    "Latitude": "Latitud",
    "Longitude": "Longitud",
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
    if not MODEL_PATH.exists():
        from train_model import main

        main()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


st.title("🏠 HouseValue AI")
st.caption("Estimación del valor mediano de la vivienda por distrito censal de California")

artifact = load_model()
data = load_data()
metrics = pd.read_csv(METRICS_PATH)

cols = st.columns(4)
cols[0].metric("Distritos", f"{len(data):,}")
cols[1].metric("Modelo activo", artifact["model_name"])
cols[2].metric("R² en test", f"{artifact['test_r2']:.3f}")
cols[3].metric(
    "R² validación cruzada",
    f"{artifact['cv_r2_mean']:.3f}",
    delta=f"± {artifact['cv_r2_std']:.3f}",
    delta_color="off",
)

st.divider()
left, right = st.columns([1, 1.4])

with left:
    st.subheader("Características del distrito")
    with st.form("prediction_form"):
        values = {}
        for feature in artifact["features"]:
            serie = data[feature]
            values[feature] = st.number_input(
                ETIQUETAS.get(feature, feature),
                float(serie.min()),
                float(serie.max()),
                float(serie.median()),
            )
        submitted = st.form_submit_button(
            "Estimar valor", type="primary", use_container_width=True
        )

    if submitted:
        row = pd.DataFrame([values])[artifact["features"]]
        prediction = artifact["pipeline"].predict(row)[0]
        st.success(f"Valor estimado: **${prediction * 100_000:,.0f}**")
        margen = artifact["test_mae"] * 100_000
        st.caption(
            f"Error absoluto medio del modelo en test: ±${margen:,.0f}. "
            "El objetivo del dataset está acotado en $500.000, así que los "
            "distritos más caros se subestiman por diseño."
        )

with right:
    st.subheader("Comparación de modelos (partición de validación)")
    validacion = metrics[metrics["particion"] == "validación"]
    st.bar_chart(validacion.set_index("modelo")[["MAE", "RMSE"]])

    st.subheader("Métricas completas")
    st.dataframe(
        metrics.style.format(
            {c: "{:.3f}" for c in ["MAE", "RMSE", "R2", "R2_cv_media", "R2_cv_desv"]},
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Distribución de valores")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(data[artifact["target"]] * 100_000, bins=40, color="#0f766e", alpha=0.85)
    ax.set_xlabel("Valor mediano de la vivienda (USD)")
    ax.set_ylabel("Distritos")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)

st.caption(
    "Datos reales del censo de California de 1990 (scikit-learn). Proyecto de "
    "demostración: no es una tasación y no debe usarse para decisiones de compra."
)
