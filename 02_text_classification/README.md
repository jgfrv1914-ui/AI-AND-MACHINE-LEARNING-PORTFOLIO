# 02 · Topic text classifier (NLP)

Classification of messages into **20 categories** using TF-IDF and linear models. Corpus: *20 Newsgroups* (Usenet messages, in English).

## Results

| Metric | Value |
|---|---|
| Test accuracy | **0.719** |
| Test macro F1 | **0.706** |
| Selected model | Calibrated linear SVM |
| Chance level | 0.05 |

Compared on validation: calibrated linear SVM (macro F1 0.779), Complement Naive Bayes (0.774) and logistic regression (0.769).

## The two decisions that define the project

**1. Headers, signatures and quotes are removed.** The original headers contain the name of the newsgroup the message belongs to. Leaving them in means the model simply copies the answer and accuracy climbs above 0.90 without having learned anything from the text. With clean text the figures are considerably lower, but they measure something real.

**2. The train/test split is temporal, not random.** It is the dataset's official split: the test set contains messages posted *after* the training ones. That is why performance drops from 0.779 on validation to 0.706 on test — that drop is the real cost of time passing, and it resembles production far more than a random split does.

## Per-topic performance

It is not uniform, and that is the interesting part:

- **Separate cleanly:** `rec.sport.hockey` (F1 0.905), `rec.sport.baseball` (0.853), `talk.politics.mideast` (0.814) — highly specific vocabulary.
- **Get confused:** `talk.religion.misc` (0.347), `talk.politics.misc` (0.507), `alt.atheism` (0.519) — they share almost all of their vocabulary.

The app includes the full confusion matrix so this can be inspected.

## Run

```bash
python 02_text_classification/train_model.py
streamlit run 02_text_classification/app.py
```

The app lets you paste a text, see the five most likely topics and **which specific terms pushed the decision** (linear model weight × TF-IDF value).

## Limitations

- **English only**: the corpus is in English.
- The messages date from ~1995. The technical vocabulary has aged considerably.
- With 20 classes and overlapping topics there is a natural ceiling: humans themselves would not reliably tell `talk.religion.misc` from `alt.atheism`.

## Source

`sklearn.datasets.fetch_20newsgroups`.
