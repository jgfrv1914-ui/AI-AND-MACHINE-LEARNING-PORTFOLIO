# 01 · Predicción de precios de vivienda

Regresión sobre **California Housing** (censo de California de 1990, 20.640 distritos, 8 variables). El objetivo es el valor mediano de la vivienda por distrito.

## Resultados

| Métrica | Valor |
|---|---|
| R² en test | **0.850** |
| R² validación cruzada 5-fold | **0.847 ± 0.007** |
| MAE en test | 0.296 (≈ $29.600) |
| Modelo elegido | XGBoost |

Comparados: regresión lineal (R² 0.606), Random Forest (0.804) y XGBoost (0.847).

## Metodología

Partición **60/20/20**. El modelo se elige sobre validación y solo el ganador se mide una vez sobre test, así que la cifra publicada no está inflada por haber elegido mirando el resultado. Además se reporta R² con validación cruzada para acompañar cada número de su desviación.

## Ejecutar

```bash
python 01_prediccion_viviendas/train_model.py
streamlit run 01_prediccion_viviendas/app.py
```

## Limitaciones

- El objetivo está **acotado en $500.000** en el dataset original: los distritos más caros aparecen truncados y el modelo los subestima por diseño.
- Son datos de 1990. Sirve para demostrar el flujo de trabajo, no para tasar nada hoy.
- `Latitude` y `Longitude` aportan mucha señal, así que el modelo generaliza mal fuera de California.

## Fuente

`sklearn.datasets.fetch_california_housing` — derivado del censo de EE. UU. de 1990.

> **Nota:** este proyecto usaba antes *Boston Housing*, retirado de scikit-learn en la versión 1.2 por contener una variable construida como proxy racial. California Housing es el reemplazo recomendado por la propia biblioteca.
