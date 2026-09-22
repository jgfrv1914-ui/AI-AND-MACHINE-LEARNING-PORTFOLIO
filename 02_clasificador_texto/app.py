from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "text_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.csv"
POR_TEMA_PATH = ROOT / "models" / "f1_por_tema.csv"
CONFUSION_PATH = ROOT / "models" / "confusion_matrix.csv"

EJEMPLOS = {
    "Hardware / gráficos": (
        "My new graphics card keeps crashing the X server when I enable "
        "hardware acceleration. I already updated the driver and swapped the "
        "PCI slot, but the monitor still goes black after a few minutes."
    ),
    "Espacio": (
        "The shuttle launch window depends on the orbital inclination and the "
        "amount of fuel required to reach the space station. NASA published the "
        "trajectory data for the next mission last week."
    ),
    "Deporte": (
        "The goalie played an incredible game last night, he stopped almost "
        "every shot in the third period and the team won the series. Best "
        "playoff hockey I have watched in years."
    ),
    "Criptografía": (
        "If the encryption key is only 40 bits long, a brute force attack is "
        "entirely feasible. The government proposal to escrow private keys "
        "would weaken the security of every citizen."
    ),
    "Medicina": (
        "The patient was treated with antibiotics for two weeks but the "
        "infection came back. The doctor suggested a different diagnosis and "
        "ordered more blood tests before changing the medication."
    ),
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
    if not MODEL_PATH.exists():
        from train_model import main

        main()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_tables():
    return pd.read_csv(METRICS_PATH), pd.read_csv(POR_TEMA_PATH), pd.read_csv(CONFUSION_PATH, index_col=0)


def coeficientes(modelo) -> np.ndarray | None:
    """Extrae la matriz de pesos, tambien si el modelo esta calibrado."""
    if hasattr(modelo, "coef_"):
        return modelo.coef_
    calibrados = getattr(modelo, "calibrated_classifiers_", None)
    if calibrados:
        pesos = [c.estimator.coef_ for c in calibrados if hasattr(c.estimator, "coef_")]
        if pesos:
            return np.mean(pesos, axis=0)
    if hasattr(modelo, "feature_log_prob_"):
        return modelo.feature_log_prob_
    return None


st.title("🗂️ TextSort AI")
st.caption("Clasificación automática de texto en 20 temas mediante TF-IDF y modelos lineales")

artifact = load_model()
metrics, por_tema, confusion = load_tables()
pipeline = artifact["pipeline"]
temas = artifact["temas"]

cols = st.columns(4)
cols[0].metric("Temas", len(temas))
cols[1].metric("Modelo activo", artifact["model_name"])
cols[2].metric("Exactitud (test)", f"{artifact['test_accuracy']:.3f}")
cols[3].metric("F1 macro (test)", f"{artifact['test_f1_macro']:.3f}")

st.info(
    f"Entrenado con {artifact['n_train']:,} mensajes y evaluado sobre {artifact['n_test']:,} "
    "posteriores en el tiempo. Se eliminaron cabeceras, firmas y citas: sin esa "
    "limpieza el nombre del grupo aparece en el propio texto y la exactitud sube "
    "artificialmente por encima de 0,90.",
    icon="🧠",
)

st.divider()
left, right = st.columns([1, 1.15])

with left:
    st.subheader("Clasificar un texto")
    ejemplo = st.selectbox("Cargar un ejemplo", ["(escribir mi propio texto)"] + list(EJEMPLOS))
    texto = st.text_area(
        "Texto en inglés",
        value=EJEMPLOS.get(ejemplo, ""),
        height=190,
        placeholder="Pega aquí un texto en inglés...",
        help="El corpus de entrenamiento está en inglés, así que el modelo solo funciona en ese idioma.",
    )

    if texto.strip():
        probabilidades = pipeline.predict_proba([texto])[0]
        orden = np.argsort(probabilidades)[::-1]
        ganador = orden[0]

        st.success(f"Tema predicho: **{temas[ganador]}** — confianza {probabilidades[ganador]:.1%}")
        if probabilidades[ganador] < 0.35:
            st.warning(
                "Confianza baja: el texto es corto o ambiguo, o trata un tema "
                "que no está entre los 20 del corpus.",
                icon="⚠️",
            )

        st.caption("Cinco temas más probables")
        st.bar_chart(
            pd.DataFrame(
                {"probabilidad": probabilidades[orden[:5]]},
                index=[temas[i] for i in orden[:5]],
            )
        )

        pesos = coeficientes(pipeline.named_steps["model"])
        if pesos is not None:
            vectorizador = pipeline.named_steps["tfidf"]
            vector = vectorizador.transform([texto])
            nombres = vectorizador.get_feature_names_out()
            presentes = vector.nonzero()[1]
            if len(presentes):
                aporte = pd.DataFrame(
                    {
                        "término": nombres[presentes],
                        "aporte": vector.toarray()[0][presentes] * pesos[ganador][presentes],
                    }
                ).sort_values("aporte", ascending=False)
                st.caption(f"Términos que más empujaron hacia «{temas[ganador]}»")
                st.bar_chart(aporte.head(10).set_index("término")["aporte"])
    else:
        st.caption("Escribe o carga un texto para ver la predicción.")

with right:
    st.subheader("Rendimiento de los modelos")
    st.dataframe(
        metrics.style.format({c: "{:.3f}" for c in ["exactitud", "f1_macro", "f1_ponderado"]}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Las filas de validación sirvieron para elegir el modelo; la de test es "
        "la medición final. La caída entre ambas es real: el test son mensajes "
        "posteriores, y el lenguaje de los foros cambia con el tiempo."
    )

    st.subheader("F1 por tema (test)")
    fig, ax = plt.subplots(figsize=(7, 6))
    datos = por_tema.sort_values("f1")
    ax.barh(datos["tema"], datos["f1"], color="#6d28d9")
    ax.axvline(artifact["test_f1_macro"], color="#be123c", linestyle="--", linewidth=1)
    ax.set_xlabel("F1")
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    st.caption(
        "Los temas deportivos y técnicos se separan bien. Los de religión y "
        "política se confunden entre sí: comparten vocabulario casi por completo."
    )

    with st.expander("Matriz de confusión completa"):
        st.dataframe(
            confusion.style.background_gradient(cmap="Purples", axis=None),
            use_container_width=True,
        )

st.caption(
    "Datos reales del corpus 20 Newsgroups (mensajes de Usenet, ~1995, en inglés), "
    "distribuido con scikit-learn. Proyecto de demostración de un flujo de "
    "clasificación de texto: vectorización, comparación de modelos y explicación."
)
