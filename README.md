# Portafolio de Machine Learning e IA — Fernando Tafurt Pinto

Tres proyectos de Machine Learning reproducibles, cada uno con su entrenamiento, sus métricas y una interfaz interactiva en Streamlit.

| # | Proyecto | Tipo de problema | Métrica principal |
|---|---|---|---|
| 01 | [Predicción de precios de vivienda](./01_prediccion_viviendas) | Regresión tabular | R² **0.850** (test) · 0.847 ± 0.007 (CV) |
| 02 | [Clasificador de texto por tema](./02_clasificador_texto) | NLP · 20 clases | F1 macro **0.706** (azar: 0.05) |
| 03 | [Detección de fraude](./03_deteccion_fraude) | Clasificación desbalanceada | PR-AUC **0.874** · precisión 0.931 |

## Metodología común

Los tres proyectos siguen las mismas tres reglas, y son la razón de que las cifras sean más bajas de lo que podrían parecer:

**1. El conjunto de test se usa una sola vez.** Partición 60/20/20. El modelo, los hiperparámetros y el umbral de decisión se eligen mirando **validación**. El test se reserva para la medición final del ganador y no interviene en ninguna decisión. Elegir mirando el test es la forma más común de publicar un número inflado sin darse cuenta.

**2. Cada número va con su desviación.** Una cifra de una sola partición no dice nada sobre estabilidad. Donde tiene sentido se reporta validación cruzada con su desviación típica.

**3. La métrica se elige según el problema.** Con un 0,17 % de positivos, ROC-AUC se ve bien casi siempre y engaña: el proyecto 03 se juzga con PR-AUC. Con 20 clases desbalanceadas, la exactitud premia a las clases grandes: el proyecto 02 se juzga con F1 macro.

Cada proyecto documenta además sus **limitaciones** en su propio README.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Ejecutar

```bash
# Entrenar (descarga los datos la primera vez)
python 01_prediccion_viviendas/train_model.py
python 02_clasificador_texto/train_model.py
python 03_deteccion_fraude/train_model.py

# Abrir cualquier demo
streamlit run 01_prediccion_viviendas/app.py
streamlit run 02_clasificador_texto/app.py
streamlit run 03_deteccion_fraude/app.py
```

Los modelos entrenados están versionados, así que las demos funcionan sin entrenar nada.

## Datos

| Proyecto | Fuente | Descarga |
|---|---|---|
| 01 | California Housing (censo EE. UU. 1990) | automática, vía scikit-learn |
| 02 | 20 Newsgroups | automática, vía scikit-learn |
| 03 | Credit Card Fraud Detection (ULB) | automática, desde Kaggle (144 MB) |

El dataset de fraude no se versiona por tamaño. El repositorio incluye una **muestra** (`muestra_demo.csv`: los 492 fraudes + 12.000 transacciones legítimas) para que la demo funcione desplegada.

## Web del portafolio

`index.html` es la página del portafolio. Antes de publicarla, edita el bloque `CONFIG` al inicio de `script.js`:

```js
const CONFIG = {
  github: "https://github.com/TU-USUARIO/portafolio-ia-ml",   // ← tu usuario
  proyectos: {
    viviendas: { carpeta: "01_prediccion_viviendas", demo: "" },  // ← URL al desplegar
    ...
  },
};
```

Mientras una demo no tenga URL, la tarjeta muestra «Ejecutar en local» y enlaza al README en lugar de a un enlace roto.
