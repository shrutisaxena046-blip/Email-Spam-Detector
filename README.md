# Email Spam Detector

### Machine Learning-Based Spam Message Classification

A complete Natural Language Processing and Machine Learning project that classifies messages as **Spam** or **Not Spam** using TF-IDF feature extraction and Logistic Regression.

<p align="center">
  <strong>Python</strong> ·
  <strong>Machine Learning</strong> ·
  <strong>NLP</strong> ·
  <strong>Scikit-learn</strong> ·
  <strong>Streamlit</strong>
</p>

---

## Project Overview

The Email Spam Detector is a machine learning application designed to analyze the content of a message and determine whether it is likely to be **Spam** or **Not Spam**.

The project combines text preprocessing, TF-IDF feature extraction, machine learning classification, and a Streamlit-based interface into a complete end-to-end application.

The application also provides probability scores, model confidence, prediction history, and a dashboard for analyzing model performance.

> **Dataset Note:** The model is trained using the SMS Spam Collection dataset, which contains SMS-style messages rather than a large collection of real-world emails.

---

## Key Features

| Feature | Description |
|---|---|
| Message Classification | Predicts whether a message is Spam or Not Spam |
| TF-IDF Processing | Converts text into numerical machine learning features |
| Model Comparison | Compares Multinomial Naive Bayes and Logistic Regression |
| Probability Analysis | Displays Spam and Not Spam probabilities |
| Confidence Score | Shows the model's prediction confidence |
| Dashboard | Provides dataset and model performance statistics |
| Prediction History | Tracks messages analyzed during the current session |
| Streamlit Interface | Provides a simple browser-based interface |

---

## Machine Learning Pipeline

```text
                 Input Message
                       |
                       v
              Text Preprocessing
                       |
                       v
              TF-IDF Vectorization
                       |
                       v
              Machine Learning Model
                       |
             +---------+---------+
             |                   |
             v                   v
          Spam              Not Spam