# Fake News Detection with NLP

Classifying news articles as real or fake using TF-IDF text features and linear models, trained on 44,898 articles. The project also looks at *what the model actually learned*, and finds that high accuracy on this dataset largely comes from source-specific patterns rather than an understanding of misinformation.

## Results

| Model | 5-fold CV accuracy | Test accuracy |
|---|---|---|
| Multinomial Naive Bayes | 0.933 | — |
| **Logistic Regression** | **0.984** | **0.99** (precision, recall, F1 all ≈ 0.99) |

## What the model learned (and why 99% should be read carefully)

Inspecting the Logistic Regression coefficients shows the strongest signals are:

- **Real news:** `reuter`, `said`, `washington`, and weekday names such as `wednesday` and `tuesday`
- **Fake news:** `via`, `imag` (image), `read`, `featur`, `com`, `gop`, `hillari`

Every "real" article in this dataset comes from Reuters, and most begin with a dateline like *"WASHINGTON (Reuters) –"*. The fake articles come from other sites with their own habits ("Featured image via…", "Read more"). So the model is largely recognizing **which outlet wrote the article**, not whether it's true.

**Takeaway:** a model this accurate on a benchmark could still fail on real-world articles from new sources. Checking coefficients or feature importances before trusting a score is essential. The next version of this project removes the Reuters dateline and publisher boilerplate and re-evaluates to measure how much genuine signal remains.

## Approach

1. **Data:** combined the Fake and True article sets (23,481 fake + 21,417 real) with a binary label.
2. **Text preprocessing:** lowercasing, removing non-letters and English stopwords, Porter stemming (NLTK).
3. **Features:** TF-IDF vectorization (89,633 terms).
4. **Models:** Multinomial Naive Bayes and Logistic Regression, compared with stratified 5-fold cross-validation.
5. **Evaluation:** classification reports and confusion matrices on a stratified 20% hold-out set.
6. **Interpretation:** top positive and negative Logistic Regression coefficients.

## Dataset

[Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset): political and world news articles, 2015–2018, with title, text, subject, and date.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook FakeNews.ipynb
```

The notebook downloads the data automatically with `kagglehub`.

## Tech stack

Python · pandas · NumPy · scikit-learn · NLTK · Matplotlib · Seaborn

## Next steps

- Strip datelines, "(Reuters)", URLs, and "Featured image via…" boilerplate, then re-train to measure performance without source leakage.
- Test on articles from a different time period or outlets not seen in training.
- Compare against a transformer model (e.g., DistilBERT) on the cleaned text.

---
**Author:** Indraneel Mannava · [LinkedIn](https://www.linkedin.com/in/indraneel-sarma-mannava/)
