# Fake News Detection

![Fake News Detection](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Completed-success)

## Overview

This project implements a machine learning-based system for detecting fake news using the LIAR dataset. It incorporates natural language processing for text preprocessing, TF-IDF vectorization for feature extraction, and logistic regression for binary classification of statements as real or fake. The model is evaluated on a test set, providing insights into its effectiveness for misinformation detection.

## Problem Statement

The dissemination of false information through digital platforms has become a pervasive issue, undermining public trust in media and influencing societal outcomes. This project addresses the challenge of detecting fake news by utilizing the LIAR dataset, which consists of labeled political statements evaluated for truthfulness. Through natural language processing techniques for text preprocessing and feature extraction via TF-IDF, coupled with logistic regression for binary classification, the model seeks to differentiate between genuine and fabricated content. By simplifying multi-class labels into binary categories and evaluating performance metrics such as accuracy, precision, recall, and F1-score, this initiative aims to enhance the reliability of automated misinformation detection while accounting for linguistic complexities and dataset limitations.

## Dataset

- **Source**: The LIAR dataset is downloaded from https://www.cs.ucsb.edu/~william/data/liar_dataset.zip, containing train, validation, and test TSV files with political statements labeled for truthfulness.
- **Size**: Approximately 12,836 statements across train (10,269), validation (1,284), and test (1,283) sets.
- **Key Features**:
  - Textual: Statement (the news content).
  - Categorical: Label (multi-class, simplified to binary: 1 for real [true, mostly-true, half-true], 0 for fake [others]).
  - Additional metadata (e.g., speaker, context) not used in this model.
- **Preprocessing Steps**:
  - Lowercasing, tokenization, stopword removal, lemmatization.
  - Concatenation of train and validation sets for training.
  - Binary label simplification.

The dataset is automatically downloaded and extracted in the notebook if not present.

## Technologies Used

- **Programming Language**: Python 3.12+
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - NLP: `nltk` (for tokenization, stopwords, lemmatization)
  - Machine Learning: `scikit-learn` (for train-test split, TF-IDF, LogisticRegression, metrics)
  - Utilities: `re`, `urllib.request`, `zipfile`, `os`
- **Environment**: Jupyter Notebook (Colab compatible)
