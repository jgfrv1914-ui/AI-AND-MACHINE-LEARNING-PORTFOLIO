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
MUESTRA_PATH = ROOT / "data" / "muestra_demo.csv"

# Claves que el artefacto debe traer; si falta alguna, se reentrena.
CLAVES = {"pipeline", "features", "pca_features", "model_name", "threshold", "test_precision", "test_recall", "test_f1", "test_pr_auc"}

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
    """Carga el modelo y lo reentrena si falta o si quedó desactualizado.

    La comprobación de claves evita un fallo sutil: si el artefacto en disco
    viene de una versión anterior del entrenamiento, el código nuevo reventaría
    con un KeyError. Aquí se detecta y se regenera.
    """
    def _valido(ruta):
        if not ruta.exists():
            return None
        artefacto = joblib.load(ruta)
        if not CLAVES.issubset(artefacto):
            return None
        return artefacto

    artefacto = _valido(MODEL_PATH)
    if artefacto is None:
        from train_model import main

        main()
        artefacto = joblib.load(MODEL_PATH)
    return artefacto


@st.cache_data
def load_data():
    """CSV completo si está descargado; si no, la muestra versionada.

    El dataset entero pesa 144 MB y no viaja en el repositorio, así que en un
    despliegue la app funciona con la muestra (todos los fraudes + 12.000
    transacciones legítimas).
    """
    ruta = DATA_PATH if DATA_PATH.exists() else MUESTRA_PATH
    data = pd.read_csv(ruta).rename(columns={"Class": "fraude"})
    if "hora_del_dia" not in data.columns:
        data["hora_del_dia"] = (data["Time"] / 3600) % 24
    return data, ruta == DATA_PATH


st.title("🛡️ FraudShield AI")
st.caption("Evaluación de riesgo sobre transacciones con clases muy desbalanceadas")

artifact = load_model()
data, es_completo = load_data()
metrics = pd.read_csv(METRICS_PATH)
features = artifact["features"]

cols = st.columns(5)
cols[0].metric("Transacciones", f"{len(data):,}")
cols[1].metric("Fraudes", f"{data['fraude'].sum():,}", delta=f"{data['fraude'].mean():.3%}", delta_color="off")
cols[2].metric("Precisión (test)", f"{artifact['test_precision']:.3f}")
cols[3].metric("Recall (test)", f"{artifact['test_recall']:.3f}")
cols[4].metric("PR-AUC (test)", f"{artifact['test_pr_auc']:.3f}")

st.info(
    f"Modelo activo: **{artifact['model_name']}** · umbral **{artifact['threshold']:.3f}**, "
    "elegido sobre la partición de validación y medido una sola vez en test.",
    icon="🧠",
)

if not es_completo:
    st.warning(
        "Ejecutando sobre la muestra incluida en el repositorio (todos los "
        "fraudes + 12.000 transacciones legítimas). Las métricas mostradas "
        "siguen siendo las del test completo. Para cargar las 284.807 "
        "transacciones, ejecuta `train_model.py` en local.",
        icon="📦",
    )

st.divider()
left, right = st.columns([1, 1.3])

with left:
    st.subheader("Analizar una transacción")
    st.caption(
        "Las variables V1–V28 son componentes PCA anonimizados: no tienen "
        "significado individual, así que lo útil es partir de una transacción real."
    )

    origen = st.radio(
        "Transacción a evaluar",
        ["Fraude real (aleatorio)", "Transacción legítima (aleatoria)", "Introducir valores"],
        label_visibility="collapsed",
    )

    if "semilla" not in st.session_state:
        st.session_state.semilla = 0
    if st.button("🎲 Tomar otra transacción", use_container_width=True):
        st.session_state.semilla += 1

    if origen == "Introducir valores":
        base = {f: 0.0 for f in features}
        base["Amount"] = float(data["Amount"].median())
        base["hora_del_dia"] = 12.0
        with st.expander("Componentes PCA V1–V28", expanded=False):
            for f in artifact["pca_features"]:
                base[f] = st.number_input(f, value=0.0, format="%.5f")
        base["Amount"] = st.number_input("Monto", min_value=0.0, value=base["Amount"])
        base["hora_del_dia"] = st.slider("Hora del día", 0.0, 24.0, 12.0, 0.5)
        row = pd.DataFrame([base])[features]
    else:
        objetivo = 1 if origen.startswith("Fraude") else 0
        pool = data[data["fraude"] == objetivo]
        idx = pool.index[st.session_state.semilla % len(pool)]
        row = data.loc[[idx], features]
        st.caption(
            f"Transacción #{idx} — monto ${data.loc[idx, 'Amount']:,.2f}, "
            f"hora {data.loc[idx, 'hora_del_dia']:.1f}h · "
            f"etiqueta real: **{'FRAUDE' if objetivo else 'legítima'}**"
        )

    umbral = st.slider(
        "Umbral de alerta", 0.05, 0.995, float(artifact["threshold"]), 0.005,
        help="Bajarlo detecta más fraude pero genera más falsas alarmas.",
    )

    probabilidad = artifact["pipeline"].predict_proba(row)[0, 1]
    if probabilidad >= umbral:
        st.error(f"🚨 Alerta de fraude — probabilidad **{probabilidad:.2%}**")
    else:
        st.success(f"✅ Riesgo bajo — probabilidad **{probabilidad:.2%}**")

    st.subheader("Por qué esta decisión (SHAP)")
    modelo = artifact["pipeline"].named_steps["model"]
    valores = shap.TreeExplainer(modelo).shap_values(row)
    if isinstance(valores, list):
        contribuciones = valores[1][0]
    elif getattr(valores, "ndim", 0) == 3:
        contribuciones = valores[0, :, 1]
    else:
        contribuciones = valores[0]
    aporte = (
        pd.DataFrame({"variable": features, "impacto": np.asarray(contribuciones).ravel()})
        .sort_values("impacto", key=abs, ascending=False)
        .head(12)
        .set_index("variable")
    )
    st.bar_chart(aporte["impacto"])
    st.caption("Valores positivos empujan hacia fraude; negativos, hacia transacción legítima.")

with right:
    st.subheader("Rendimiento de los modelos")
    st.dataframe(
        metrics.style.format(
            {c: "{:.3f}" for c in ["precision", "recall", "f1", "roc_auc", "pr_auc", "umbral"]}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Las filas de validación sirvieron para elegir modelo y umbral. "
        "La fila de test es la medición final y no intervino en ninguna decisión."
    )

    st.subheader("Compromiso precisión / recall")
    n_muestra = min(40_000, len(data))
    sub = data.sample(n_muestra, random_state=0)
    prob_muestra = artifact["pipeline"].predict_proba(sub[features])[:, 1]
    y_muestra = sub["fraude"].to_numpy()
    rejilla = np.arange(0.05, 0.996, 0.01)
    curva = pd.DataFrame(
        {
            "umbral": rejilla,
            "precisión": [
                (y_muestra[prob_muestra >= t].mean() if (prob_muestra >= t).any() else np.nan)
                for t in rejilla
            ],
            "recall": [
                (prob_muestra[y_muestra == 1] >= t).mean() for t in rejilla
            ],
        }
    ).set_index("umbral")
    st.line_chart(curva)

    st.subheader("Importancia global de variables")
    importancias = pd.Series(modelo.feature_importances_, index=features).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    importancias.plot.barh(ax=ax, color="#be123c")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)

st.caption(
    "Datos reales del dataset Credit Card Fraud Detection (ULB, Kaggle): 284.807 "
    "transacciones europeas de 2013, 492 fraudes. Proyecto de demostración: un "
    "sistema real exige revisión humana, reglas de negocio y monitoreo de deriva."
)
