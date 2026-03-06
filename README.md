# Fake News Detection using Natural Language Processing

## Overview
Fake news detection is an important problem in modern information systems where misleading or fabricated news articles can spread rapidly online. Machine learning and natural language processing techniques can help automatically identify misinformation by analyzing textual patterns.

This project builds a machine learning pipeline to classify news articles as **real** or **fake** using natural language processing and supervised learning models.

The workflow includes data preprocessing, TF-IDF feature extraction, model training, cross-validation, and model interpretation.


## Problem Statement
The goal of this project is to develop a classification model that can automatically distinguish between real and fake news articles based on their textual content.

Detecting misinformation is important for improving information reliability on digital platforms.


## Dataset
Dataset: Fake and Real News Dataset (Kaggle)

The dataset contains two sets of news articles:

- Fake news articles
- Real news articles

After combining both datasets, the final dataset contains:

**44,898 news articles**

Each article includes:

- title
- text
- subject
- date

Target variable:

- **0 - Real news**
- **1 - Fake news**


## Data Preprocessing
Before model training, the text data was cleaned and normalized.

Preprocessing steps included:

- converting text to lowercase
- removing punctuation and non-alphabetic characters
- removing stopwords
- stemming words using Porter Stemmer

These steps reduce noise and improve feature representation.


## Feature Extraction
Text data was converted into numerical features using **TF-IDF (Term Frequency – Inverse Document Frequency)**.

TF-IDF captures the importance of words within each document relative to the entire corpus.

Feature matrix size:

**(44,898 documents, 89,633 features)**


## Machine Learning Models
Two classification algorithms were evaluated:

- Multinomial Naive Bayes
- Logistic Regression

Both models are commonly used for text classification tasks.


## Cross-Validation Results

| Model | Mean CV Accuracy |
|------|------|
| Multinomial Naive Bayes | 0.9332 |
| Logistic Regression | 0.9844 |

Logistic Regression achieved the best performance during cross-validation.

## Test Set Performance (Logistic Regression)

| Metric | Score |
|------|------|
| Accuracy | 0.99 |
| Precision | 0.99 |
| Recall | 0.99 |
| F1 Score | 0.99 |

The confusion matrix shows that the model correctly classifies the majority of both real and fake news articles.

## Model Interpretation
To understand which words contribute most strongly to predictions, the coefficients of the Logistic Regression model were analyzed.

Examples of words strongly associated with **fake news** include:

- via
- gop
- hillari
- imag

Examples of words strongly associated with **real news** include:

- reuter
- washington
- monday
- said

This analysis helps interpret the linguistic patterns learned by the model.

## Technologies Used
Python  
pandas  
NumPy  
scikit-learn  
NLTK  
Matplotlib  
Seaborn  
