# 03 · Transaction fraud detection

Classification with **heavily imbalanced** classes (0.173 % positives) on the real *Credit Card Fraud Detection* dataset from ULB: 284,807 European transactions from 2013, of which 492 are fraud. `V1`–`V28` are anonymised PCA components.

## Results (test, measured once)

| Metric | Value |
|---|---|
| Precision | **0.931** |
| Recall | **0.818** |
| F1 | **0.871** |
| PR-AUC | **0.874** |
| ROC-AUC | 0.978 |
| Selected model | XGBoost with `scale_pos_weight` |
| Threshold | 0.830 (chosen on validation) |

With 0.17 % positives, **PR-AUC is the honest metric**; ROC-AUC looks good almost always in this kind of problem and is misleading.

## The bug that cost half the F1

The earlier version corrected the imbalance **twice**: it applied `SMOTE` (which rebalances the classes to 50/50) and then `scale_pos_weight ≈ 578` on top of the already-balanced data. The model massively over-predicted fraud.

Measured on the same test split:

| Configuration | Precision | Recall | F1 |
|---|---|---|---|
| SMOTE + `scale_pos_weight` (double correction) | 0.301 | 0.879 | **0.449** |
| SMOTE only (`scale_pos_weight=1`) | 0.901 | 0.828 | **0.863** |
| `scale_pos_weight` only (no SMOTE) | 0.880 | 0.818 | **0.848** |

**The imbalance is corrected once, through one route only.** Each model in this project picks one.

There was a second, chained problem: the threshold grid stopped at 0.90, but the optimum for the double-corrected configuration sat above 0.95, so it was being cut off. The grid now reaches 0.995.

## Methodology

- Stratified **60/20/20** split.
- **The threshold is chosen on validation, never on test.** Tuning it on test is information leakage and turns the published metric into marketing.
- `Time` (seconds since the first transaction, 48 h in total) is not used raw: it is converted to **hour of day**, which is a genuine behavioural signal.

## Run

```bash
python 03_fraud_detection/train_model.py
streamlit run 03_fraud_detection/app.py
```

The app loads real transactions from the dataset (fraudulent or legitimate), lets you move the threshold and explains each decision with **SHAP**.

## Limitations

- The split is random over data that is temporal. A time-based split would be more realistic; with only 48 h of data there would be very few frauds in the final stretch.
- `V1`–`V28` are PCA components: the SHAP explanation shows *which component* weighed in, but it cannot be translated into a business cause.
- A real system needs human review, business rules, asymmetric costs for false positives/negatives and drift monitoring.

## Data

The full CSV (144 MB) is not versioned. `train_model.py` downloads it automatically on first run. The repository ships `data/demo_sample.csv` (all 492 frauds + 12,000 legitimate transactions) so the deployed demo works without it.

## Source

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
