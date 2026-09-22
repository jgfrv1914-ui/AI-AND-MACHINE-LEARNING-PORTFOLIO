# 03 · Detección de fraude en transacciones

Clasificación con clases **muy desbalanceadas** (0,173 % de positivos) sobre el dataset real *Credit Card Fraud Detection* de la ULB: 284.807 transacciones europeas de 2013, de las cuales 492 son fraude. `V1`–`V28` son componentes PCA anonimizados.

## Resultados (test, medido una sola vez)

| Métrica | Valor |
|---|---|
| Precisión | **0.931** |
| Recall | **0.818** |
| F1 | **0.871** |
| PR-AUC | **0.874** |
| ROC-AUC | 0.978 |
| Modelo elegido | XGBoost con `scale_pos_weight` |
| Umbral | 0.830 (elegido en validación) |

Con 0,17 % de positivos, **PR-AUC es la métrica honesta**; ROC-AUC se ve bien casi siempre en este tipo de problema y engaña.

## El error que costaba la mitad del F1

La versión anterior corregía el desbalanceo **dos veces**: aplicaba `SMOTE` (que equilibra las clases a 50/50) y además `scale_pos_weight ≈ 578` sobre los datos ya equilibrados. El modelo sobre-predecía fraude de forma masiva.

Medido sobre la misma partición de test:

| Configuración | Precisión | Recall | F1 |
|---|---|---|---|
| SMOTE + `scale_pos_weight` (doble corrección) | 0.301 | 0.879 | **0.449** |
| Solo SMOTE (`scale_pos_weight=1`) | 0.901 | 0.828 | **0.863** |
| Solo `scale_pos_weight` (sin SMOTE) | 0.880 | 0.818 | **0.848** |

**El desbalanceo se corrige una vez, por una sola vía.** Cada modelo de este proyecto elige una.

Había un segundo problema encadenado: la rejilla de umbrales terminaba en 0.90, pero el óptimo de la configuración con doble corrección estaba por encima de 0.95, así que quedaba truncado. Ahora la rejilla llega a 0.995.

## Metodología

- Partición **60/20/20** estratificada.
- **El umbral se elige en validación, nunca en test.** Ajustarlo sobre test es fuga de información y convierte la métrica publicada en propaganda.
- `Time` (segundos desde la primera transacción, 48 h en total) no se usa cruda: se convierte a **hora del día**, que sí es una señal de comportamiento.

## Ejecutar

```bash
python 03_deteccion_fraude/train_model.py
streamlit run 03_deteccion_fraude/app.py
```

La app carga transacciones reales del dataset (fraudulentas o legítimas), permite mover el umbral y explica cada decisión con **SHAP**.

## Limitaciones

- La partición es aleatoria sobre datos que son temporales. Una partición por tiempo sería más realista; con solo 48 h de datos habría muy pocos fraudes en el tramo final.
- `V1`–`V28` son PCA: la explicación SHAP indica *qué componente* pesó, pero no puede traducirse a una causa de negocio.
- Un sistema real necesita revisión humana, reglas de negocio, coste asimétrico de falsos positivos/negativos y monitoreo de deriva.

## Fuente

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
