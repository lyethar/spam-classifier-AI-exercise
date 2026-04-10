"""
predict.py — Classify New SMS Messages as Spam or Ham
======================================================
HTB Academy: Applications of AI in InfoSec
Author: lyethar (Fabian)

Loads the saved model and classifies messages passed via command line
or entered interactively. Also shows the model's confidence score.

Usage:
    python src/predict.py "Your message here"
    python src/predict.py                      (interactive mode)
"""

import os
import sys
import joblib

# ─────────────────────────────────────────────────────────
# LOAD THE SAVED MODEL
# ─────────────────────────────────────────────────────────
# joblib.load() deserialises the entire pipeline from disk.
# The same TF-IDF vectoriser and Naive Bayes classifier trained
# in train.py are restored — no retraining needed.

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'spam_classifier.pkl')

if not os.path.exists(MODEL_PATH):
    print("[!] Model not found. Please run 'python src/train.py' first.")
    sys.exit(1)

pipeline = joblib.load(MODEL_PATH)
print("[+] Model loaded successfully.\n")


def classify(message: str) -> dict:
    """
    Classify a single SMS message.

    Returns a dict with:
        - prediction: 'SPAM' or 'HAM'
        - confidence: probability score (0.0 - 1.0)
        - raw_proba: [P(ham), P(spam)] array

    How predict_proba works:
        The Naive Bayes classifier calculates the probability of
        each class. predict_proba returns [P(ham), P(spam)] for
        a given message. We report the confidence of the final
        prediction, not always the spam probability.
    """
    proba = pipeline.predict_proba([message])[0]
    prediction_idx = proba.argmax()

    label_map = {0: 'HAM ✅', 1: 'SPAM 🚨'}
    prediction = label_map[prediction_idx]
    confidence = proba[prediction_idx]

    return {
        'prediction': prediction,
        'confidence': confidence,
        'p_ham': proba[0],
        'p_spam': proba[1]
    }


def display_result(message: str, result: dict):
    """Pretty-print classification results."""
    print("─" * 60)
    print(f"  Message    : {message[:80]}{'...' if len(message) > 80 else ''}")
    print(f"  Prediction : {result['prediction']}")
    print(f"  Confidence : {result['confidence'] * 100:.1f}%")
    print(f"  P(ham)     : {result['p_ham'] * 100:.1f}%")
    print(f"  P(spam)    : {result['p_spam'] * 100:.1f}%")
    print("─" * 60)


# ─────────────────────────────────────────────────────────
# MAIN — COMMAND LINE OR INTERACTIVE MODE
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Command-line mode: message passed as argument
        message = ' '.join(sys.argv[1:])
        result = classify(message)
        display_result(message, result)

    else:
        # Interactive mode: prompt for messages in a loop
        print("🔍 SMS Spam Classifier — Interactive Mode")
        print("Type a message to classify it. Press Ctrl+C to exit.\n")
        try:
            while True:
                message = input("Enter SMS message: ").strip()
                if not message:
                    continue
                result = classify(message)
                display_result(message, result)
                print()
        except KeyboardInterrupt:
            print("\n\n[*] Exiting classifier.")
