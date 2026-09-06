# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


# ============================================================
# 2. LOAD THE DATASET
# ============================================================

data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


# ============================================================
# 3. EXPLORE THE DATASET
# ============================================================

print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nLabel counts:")
print(data["label"].value_counts())

print("\nSpam messages:")
print(data[data["label"] == "spam"].head())

print("\nHam messages:")
print(data[data["label"] == "ham"].head())


# ============================================================
# 4. TEXT CLEANING - CONVERT TO LOWERCASE
# ============================================================

data["clean_message"] = data["message"].str.lower()


# ============================================================
# 5. TEXT CLEANING - REMOVE PUNCTUATION
# ============================================================

data["clean_message"] = data["clean_message"].apply(
    lambda x: x.translate(
        str.maketrans("", "", string.punctuation)
    )
)


# ============================================================
# 6. SEPARATE FEATURES (X) AND LABELS (y)
# ============================================================

X = data["clean_message"]
y = data["label"]

print("\nFeatures (X):")
print(X.head())

print("\nLabels (y):")
print(y.head())


# ============================================================
# 7. SPLIT DATA INTO TRAINING AND TESTING DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data size:")
print(X_train.shape)

print("\nTesting data size:")
print(X_test.shape)


# ============================================================
# 8. CONVERT TEXT INTO NUMERICAL FEATURES USING TF-IDF
# ============================================================

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF testing shape:")
print(X_test_tfidf.shape)


# ============================================================
# 9. TRAIN MODEL 1 - MULTINOMIAL NAIVE BAYES
# ============================================================

print("\n" + "=" * 60)
print("MODEL 1: MULTINOMIAL NAIVE BAYES")
print("=" * 60)

nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_predictions = nb_model.predict(
    X_test_tfidf
)

nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)

print("\nNaive Bayes Accuracy:")
print(f"{nb_accuracy * 100:.2f}%")

print("\nNaive Bayes Classification Report:")
print(
    classification_report(
        y_test,
        nb_predictions
    )
)


# ============================================================
# 10. TRAIN MODEL 2 - LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2: LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(
    X_train_tfidf,
    y_train
)

lr_predictions = lr_model.predict(
    X_test_tfidf
)

lr_accuracy = accuracy_score(
    y_test,
    lr_predictions
)

print("\nLogistic Regression Accuracy:")
print(f"{lr_accuracy * 100:.2f}%")

print("\nLogistic Regression Classification Report:")
print(
    classification_report(
        y_test,
        lr_predictions
    )
)


# ============================================================
# 11. COMPARE BOTH MODELS
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    f"\nMultinomial Naive Bayes: "
    f"{nb_accuracy * 100:.2f}%"
)

print(
    f"Logistic Regression: "
    f"{lr_accuracy * 100:.2f}%"
)


# ============================================================
# 12. SELECT THE BETTER MODEL
# ============================================================

if lr_accuracy > nb_accuracy:

    print("\nBetter model: Logistic Regression")

else:

    print("\nBetter model: Multinomial Naive Bayes")


# ============================================================
# 13. SAVE THE NAIVE BAYES MODEL
# ============================================================
# We keep Naive Bayes as the deployed model for now
# because your existing detector.py already uses it.

with open("spam_model.pkl", "wb") as file:
    pickle.dump(nb_model, file)


# ============================================================
# 14. SAVE THE TF-IDF VECTORIZER
# ============================================================

with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


# ============================================================
# 15. TRAINING COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nSaved files:")
print("✓ spam_model.pkl")
print("✓ tfidf_vectorizer.pkl")