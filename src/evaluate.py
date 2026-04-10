"""
evaluate.py — Detailed Model Evaluation with Confusion Matrix
=============================================================
HTB Academy: Applications of AI in InfoSec
Author: lyethar (Fabian)

Produces a full evaluation report:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix (visualised)
  - Top spam/ham indicator words (model interpretability)
  - Adversarial probe: which words carry the most spam weight

Run:
    python src/evaluate.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────────────────
# LOAD DATA AND MODEL
# ─────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'spam.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'spam_classifier.pkl')

if not os.path.exists(MODEL_PATH):
    print("[!] Model not found. Run 'python src/train.py' first.")
    exit(1)

df = pd.read_csv(DATA_PATH, sep='\t', names=['label', 'message'], encoding='latin-1')
df['label_encoded'] = df['label'].map({'spam': 1, 'ham': 0})

X = df['message']
y = df['label_encoded']

# Use the same random_state=42 split as train.py so we evaluate
# on the exact same test set the model was never trained on
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = joblib.load(MODEL_PATH)
y_pred = pipeline.predict(X_test)

# ─────────────────────────────────────────────────────────
# PRINT METRICS
# ─────────────────────────────────────────────────────────
# Accuracy  = (TP + TN) / Total — overall correct rate
# Precision = TP / (TP + FP)  — "when it says spam, how often is it right?"
# Recall    = TP / (TP + FN)  — "how many real spam messages did it catch?"
# F1-Score  = harmonic mean of Precision and Recall
#
# For spam filtering, RECALL matters more — missing spam (FN) is
# worse than accidentally flagging a ham message (FP).

print("\n" + "=" * 60)
print("  MODEL EVALUATION REPORT")
print("=" * 60)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"  Precision : {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"  Recall    : {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"  F1-Score  : {f1_score(y_test, y_pred) * 100:.2f}%")
print("=" * 60)
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# ─────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────
# The confusion matrix shows:
#   TN (top-left)  : Ham correctly identified as ham
#   FP (top-right) : Ham incorrectly flagged as spam (false alarm)
#   FN (bottom-left): Spam that got through (missed — dangerous)
#   TP (bottom-right): Spam correctly caught

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Predicted Ham', 'Predicted Spam'],
    yticklabels=['Actual Ham', 'Actual Spam']
)
plt.title('Confusion Matrix — SMS Spam Classifier', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Annotate what each quadrant means
plt.tight_layout()
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
print(f"\n[+] Confusion matrix saved to outputs/confusion_matrix.png")

# ─────────────────────────────────────────────────────────
# MODEL INTERPRETABILITY — TOP SPAM WORDS
# ─────────────────────────────────────────────────────────
# Naive Bayes stores log-probabilities for each word per class.
# By extracting feature_log_prob_, we can see which words are
# the strongest spam predictors — this is also how an attacker
# would probe the model to discover words to AVOID.

vectoriser = pipeline.named_steps['tfidf']
classifier = pipeline.named_steps['classifier']
feature_names = vectoriser.get_feature_names_out()

# Difference in log-probabilities between spam (class 1) and ham (class 0)
# Higher score = stronger spam indicator
spam_score = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]

top_n = 20
top_spam_idx = np.argsort(spam_score)[-top_n:][::-1]
top_ham_idx = np.argsort(spam_score)[:top_n]

print("\n" + "=" * 60)
print("  🔴 TOP SPAM INDICATOR WORDS (Red Team Intel)")
print("  Words with highest spam log-probability weight")
print("=" * 60)
for idx in top_spam_idx:
    print(f"  {feature_names[idx]:<25} score: {spam_score[idx]:.3f}")

print("\n" + "=" * 60)
print("  🟢 TOP HAM INDICATOR WORDS")
print("=" * 60)
for idx in top_ham_idx:
    print(f"  {feature_names[idx]:<25} score: {spam_score[idx]:.3f}")

# ─────────────────────────────────────────────────────────
# RED TEAM NOTE
# ─────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════╗
║  🔴 RED TEAM NOTE                                        ║
║                                                          ║
║  The spam indicator words above reveal the exact         ║
║  features this model uses to detect spam. An attacker    ║
║  performing adversarial evasion would:                   ║
║                                                          ║
║  1. Identify high-weight spam words (from this output)   ║
║  2. Avoid or obfuscate those words in crafted messages   ║
║  3. Pad the message with high-weight ham words           ║
║  4. Achieve misclassification without raising suspicion  ║
║                                                          ║
║  This is why model interpretability tools must be        ║
║  access-controlled in production AI systems.             ║
╚══════════════════════════════════════════════════════════╝
""")
