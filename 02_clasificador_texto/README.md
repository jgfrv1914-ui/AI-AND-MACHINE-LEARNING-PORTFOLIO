# 02 · Clasificador de texto por tema (NLP)

Clasificación de mensajes en **20 categorías** usando TF-IDF y modelos lineales. Corpus: *20 Newsgroups* (mensajes de Usenet, en inglés).

## Resultados

| Métrica | Valor |
|---|---|
| Exactitud en test | **0.719** |
| F1 macro en test | **0.706** |
| Modelo elegido | SVM lineal calibrada |
| Azar | 0.05 |

Comparados en validación: SVM lineal calibrada (F1 macro 0.779), Naive Bayes complementario (0.774) y regresión logística (0.769).

## Las dos decisiones que definen el proyecto

**1. Se eliminan cabeceras, firmas y citas.** Las cabeceras originales contienen el nombre del grupo al que pertenece el mensaje. Si se dejan, el modelo simplemente copia la respuesta y la exactitud sube por encima de 0.90 sin haber aprendido nada del texto. Con el texto limpio las cifras son bastante más bajas, pero miden algo real.

**2. La partición train/test es temporal, no aleatoria.** Es la partición oficial del dataset: el test son mensajes *posteriores* a los de entrenamiento. Por eso el rendimiento cae de 0.779 en validación a 0.706 en test — esa caída es el coste real del paso del tiempo, y se parece mucho más a producción que una partición aleatoria.

## Rendimiento por tema

No es uniforme, y eso es lo interesante:

- **Se separan bien:** `rec.sport.hockey` (F1 0.905), `rec.sport.baseball` (0.853), `talk.politics.mideast` (0.814) — vocabulario muy específico.
- **Se confunden:** `talk.religion.misc` (0.347), `talk.politics.misc` (0.507), `alt.atheism` (0.519) — comparten casi todo el vocabulario entre sí.

La app incluye la matriz de confusión completa para inspeccionarlo.

## Ejecutar

```bash
python 02_clasificador_texto/train_model.py
streamlit run 02_clasificador_texto/app.py
```

La app permite pegar un texto, ver los cinco temas más probables y **qué términos concretos empujaron la decisión** (peso del modelo lineal × valor TF-IDF).

## Limitaciones

- **Solo funciona en inglés**: el corpus está en inglés.
- Los mensajes son de ~1995. El vocabulario técnico ha envejecido mucho.
- Con 20 clases y temas solapados, hay un techo natural: los propios humanos no distinguirían de forma fiable `talk.religion.misc` de `alt.atheism`.

## Fuente

`sklearn.datasets.fetch_20newsgroups`.
