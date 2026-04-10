"""
train.py — Train and Save the SMS Spam Classifier
==================================================
HTB Academy: Applications of AI in InfoSec
Author: lyethar (Fabian)

This script loads the SMS Spam Collection dataset, preprocesses it,
trains a TF-IDF + Naive Bayes pipeline, and saves the model to disk.

Run:
    python src/train.py
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────────────────
# STEP 1: LOAD THE DATASET
# ─────────────────────────────────────────────────────────
# The UCI SMS Spam Collection dataset is a tab-separated file.
# Each row has two fields: label (ham/spam) and the message text.
#
# pd.read_csv with sep='\t' tells pandas the columns are separated
# by tabs. names=['label', 'message'] assigns column headers since
# the raw file has none. encoding='latin-1' handles special characters
# in some older SMS messages.

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'spam.csv')

print("[*] Loading dataset...")
df = pd.read_csv(DATA_PATH, sep='\t', names=['label', 'message'], encoding='latin-1')

print(f"[+] Dataset loaded: {len(df)} messages")
print(f"    Spam:  {len(df[df['label'] == 'spam'])} messages")
print(f"    Ham:   {len(df[df['label'] == 'ham'])} messages")

# ─────────────────────────────────────────────────────────
# STEP 2: ENCODE LABELS
# ─────────────────────────────────────────────────────────
# Machine learning models work with numbers, not strings.
# We convert 'spam' → 1 and 'ham' → 0 using a simple map.
# This is called binary label encoding.

df['label_encoded'] = df['label'].map({'spam': 1, 'ham': 0})

X = df['message']           # Features: the raw SMS text
y = df['label_encoded']     # Target: 0 (ham) or 1 (spam)

# ─────────────────────────────────────────────────────────
# STEP 3: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────
# We split our data: 80% to train the model, 20% to test it.
# The model NEVER sees the test set during training — this is
# how we simulate real-world performance on unseen messages.
#
# test_size=0.2       → 20% of data for testing
# random_state=42     → Ensures reproducible results (same split every run)
# stratify=y          → Ensures both spam and ham are proportionally
#                       represented in both train and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[*] Train/Test split complete")
print(f"    Training samples: {len(X_train)}")
print(f"    Test samples:     {len(X_test)}")

# ─────────────────────────────────────────────────────────
# STEP 4: BUILD THE PIPELINE
# ─────────────────────────────────────────────────────────
# A Pipeline chains multiple steps so they execute in sequence.
# This ensures the TF-IDF vectoriser learns ONLY from training data
# and applies the same transformation to test data — preventing
# data leakage.
#
# Step 1 — TfidfVectorizer:
#   Converts raw text into a numerical matrix of TF-IDF scores.
#   stop_words='english' removes common words like "the", "and", "is"
#   that don't carry spam/ham signal. Max 5000 features keeps it lean.
#
# Step 2 — MultinomialNB:
#   Multinomial Naive Bayes classifier. Works well with word count
#   features. Calculates P(spam | words) using Bayes' theorem.
#   alpha=0.1 is the smoothing parameter — prevents zero probabilities
#   for unseen words (Laplace smoothing).

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)   # Include both single words AND word pairs
                              # e.g., "free" AND "free entry" are features
    )),
    ('classifier', MultinomialNB(alpha=0.1))
])

# ─────────────────────────────────────────────────────────
# STEP 5: TRAIN THE MODEL
# ─────────────────────────────────────────────────────────
# .fit() is where the actual learning happens.
# The TF-IDF vectoriser builds its vocabulary from X_train.
# The Naive Bayes classifier calculates word probabilities
# for spam vs. ham from the training labels.

print("\n[*] Training model...")
pipeline.fit(X_train, y_train)
print("[+] Training complete.")

# ─────────────────────────────────────────────────────────
# STEP 6: QUICK EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────
# We make predictions on the held-out test set and measure
# how well the model generalised to messages it never saw.

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n[+] Test Accuracy: {accuracy * 100:.2f}%")
print("\n[+] Classification Report:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# ─────────────────────────────────────────────────────────
# STEP 7: SAVE THE MODEL
# ─────────────────────────────────────────────────────────
# joblib efficiently serialises the entire pipeline (vectoriser +
# classifier) into a single file. This means we can load it
# later in predict.py without re-training.

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'spam_classifier.pkl')

joblib.dump(pipeline, MODEL_PATH)
print(f"\n[+] Model saved to: {MODEL_PATH}")
print("\n[*] Done. Run 'python src/evaluate.py' for detailed metrics.")
