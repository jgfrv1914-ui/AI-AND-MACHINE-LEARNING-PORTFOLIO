# AI & Machine Learning Portfolio — Fernando Tafurt Pinto

Three reproducible Machine Learning projects, each with its own training script, its metrics and an interactive Streamlit interface, plus the portfolio website that presents them.

**Live site:** https://jgfrv1914-ui.github.io/AI-AND-MACHINE-LEARNING-PORTFOLIO/

| # | Project | Problem type | Headline metric |
|---|---|---|---|
| 01 | [Housing price prediction](./01_housing_price_prediction) | Tabular regression | R² **0.850** (test) · 0.847 ± 0.007 (CV) |
| 02 | [Topic text classifier](./02_text_classification) | NLP · 20 classes | Macro F1 **0.706** (chance: 0.05) |
| 03 | [Fraud detection](./03_fraud_detection) | Imbalanced classification | PR-AUC **0.874** · precision 0.931 |

## Shared methodology

All three projects follow the same three rules, and those rules are the reason the figures are lower than they might otherwise look:

**1. The test set is used once.** A 60/20/20 split. The model, the hyperparameters and the decision threshold are all chosen by looking at **validation**. The test set is reserved for the final measurement of the winner and takes part in no decision. Choosing while looking at the test set is the most common way to publish an inflated number without noticing.

**2. Every number comes with its spread.** A figure from a single split says nothing about stability. Where it makes sense, cross-validation is reported together with its standard deviation.

**3. The metric is chosen to fit the problem.** With 0.17 % positives, ROC-AUC looks good almost always and misleads: project 03 is judged on PR-AUC. With 20 imbalanced classes, accuracy rewards the large ones: project 02 is judged on macro F1.

Each project additionally documents its **limitations** in its own README.

## Repository structure

```
.
├── index.html                      Portfolio website (GitHub Pages entry point)
├── styles.css                      Website styles
├── script.js                       Interactions, ES/EN i18n and project config
├── robot3d.js                      Three.js animated 3D robot avatar in the hero
├── assets/                         Images, certificates and background video
├── requirements.txt                Python dependencies for the three projects
│
├── 01_housing_price_prediction/
│   ├── train_model.py              Training, model selection and metrics
│   ├── app.py                      Streamlit demo
│   ├── data/                       Dataset (downloaded on first run, not versioned)
│   ├── models/                     Trained model + metrics.csv
│   └── README.md
│
├── 02_text_classification/         Same layout
└── 03_fraud_detection/             Same layout
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Running

```bash
# Train (downloads the data on first run)
python 01_housing_price_prediction/train_model.py
python 02_text_classification/train_model.py
python 03_fraud_detection/train_model.py

# Open any demo
streamlit run 01_housing_price_prediction/app.py
streamlit run 02_text_classification/app.py
streamlit run 03_fraud_detection/app.py
```

The trained models are versioned, so the demos run without training anything first.

## Data

| Project | Source | Download |
|---|---|---|
| 01 | California Housing (1990 US census) | automatic, via scikit-learn |
| 02 | 20 Newsgroups | automatic, via scikit-learn |
| 03 | Credit Card Fraud Detection (ULB) | automatic, from Kaggle (144 MB) |

The fraud dataset is not versioned because of its size. The repository includes a **sample** (`demo_sample.csv`: all 492 frauds + 12,000 legitimate transactions) so the demo works when deployed.

## The website

`index.html` is the portfolio page, served by GitHub Pages from the repository root. It is bilingual (English / Spanish) via the switch in the header, and the hero shows a 3D robot avatar built procedurally with Three.js — no external model file.

To point a project card at a deployed demo, edit the `CONFIG` block at the top of `script.js`:

```js
const CONFIG = {
  github: "https://github.com/jgfrv1914-ui/AI-AND-MACHINE-LEARNING-PORTFOLIO",
  projects: {
    housing: { folder: "01_housing_price_prediction", demo: "" },  // ← URL once deployed
    ...
  },
};
```

While a demo has no URL, its card shows “Run locally” and links to the project README instead of a broken link.

## Author

**Fernando Tafurt Pinto** — [LinkedIn](https://www.linkedin.com/in/fernandotafurtag9a00/)

## License

[MIT](./LICENSE)
