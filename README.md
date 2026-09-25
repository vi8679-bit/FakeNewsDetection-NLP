# Fake News Detection with NLP

Classifying news articles as real or fake using TF-IDF text features and linear models, on 38,644 deduplicated articles. The project goes beyond the accuracy score to ask **what the model actually learned**, and finds that much of it is publisher writing style rather than an understanding of misinformation.

## Results

| Model | 5-fold CV accuracy | Test accuracy | Real-news F1 | Fake-news F1 |
|---|---|---|---|---|
| Multinomial Naive Bayes | 0.933 (± 0.002) | 0.933 | 0.940 | 0.924 |
| **Logistic Regression** | **0.983 (± 0.001)** | **0.983** | **0.985** | **0.981** |

*Test set: 7,729 held-out articles (4,238 real, 3,491 fake).*

<img width="1089" height="440" alt="confusion_matrices" src="https://github.com/user-attachments/assets/69e69247-276b-4d11-98b6-dac3ccb03697" />


Logistic Regression misclassified 130 of 7,729 test articles: 41 real articles flagged as fake and 89 fake articles passed as real.

## What the model learned

<img width="1189" height="495" alt="top_words_original" src="https://github.com/user-attachments/assets/a5d44ebf-c1c4-4d3f-8863-c2ec94aeef86" />


The strongest signals are not about truthfulness. They're **publisher fingerprints**:

- **Real news:** `reuter`, `said`, `washington`. Every real article in this dataset comes from Reuters, and most open with a dateline like *"WASHINGTON (Reuters) –"*.
- **Fake news:** `via`, `imag`, `featur`, `getti`, `pic`. These come from photo credits such as *"Featured image via Getty Images"* and embedded Twitter links.

## Experiment: remove the fingerprints and re-train

To test how much the score depends on these artifacts, I stripped Reuters datelines and mentions, image credits, URLs and Twitter handles from every article, then re-trained and re-evaluated the same model on the same split.

| | Test accuracy |
|---|---|
| Original text | 0.9832 |
| Publisher fingerprints removed | 0.9765 |

<img width="1189" height="495" alt="top_words_fingerprints_removed" src="https://github.com/user-attachments/assets/6bf384b6-c9e8-4e14-b95b-5d7c097d3698" />

**Finding:** accuracy dropped less than one point, but the model didn't switch to learning content. It shifted to **the next layer of house style**:

- **Real news:** `said`, `told`, `spokesman`, `statement`, and weekday names. This is wire-service reporting convention ("*said in a statement on Wednesday*").
- **Fake news:** `via`, `read`, `watch`, `sen`, `rep`. These reflect blog conventions ("*Watch:*", "*Read more*", "*Sen.*" abbreviations).

**Takeaway:** a 98% benchmark score here mostly measures *"does this read like a Reuters wire story?"* rather than *"is this true?"*. Removing obvious artifacts isn't enough, because style runs through the whole text. A model like this would likely perform much worse on real-world articles from outlets it hasn't seen. Checking what a model learned, not just its score, is what reveals this.

## Approach

1. **Data cleaning:** merged the Fake and True sets (44,898 articles) and removed 631 empty articles and 5,623 duplicates. Duplicates split across train and test would inflate scores, since the model would be tested on text it had already seen.
2. **Text preprocessing:** lowercasing, removing non-letters and English stopwords, Porter stemming (NLTK).
3. **Leak-free modeling:** stratified 80/20 split *before* vectorization; TF-IDF and the classifier are wrapped in a scikit-learn `Pipeline`, so TF-IDF is fit on training data only, including within each cross-validation fold.
4. **Models:** Multinomial Naive Bayes and Logistic Regression, compared with stratified 5-fold cross-validation.
5. **Interpretation:** Logistic Regression coefficients to identify the most influential words.
6. **Ablation experiment:** removed source-specific boilerplate with regular expressions and re-evaluated.

## Dataset

[Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset): political and world news articles, 2015–2018. Real articles are from Reuters; fake articles come from sites flagged by fact-checkers.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook FakeNews.ipynb
```

The notebook downloads the data automatically with `kagglehub`.

## Tech stack

Python · pandas · scikit-learn · NLTK · Matplotlib · Seaborn

## Next steps

- Evaluate on articles from outlets and time periods not seen in training, the real test of generalization.
- Compare against a transformer model (e.g., DistilBERT) on the cleaned text.

---
**Author:** Indraneel Mannava · [LinkedIn](https://www.linkedin.com/in/indraneel-sarma-mannava/)
