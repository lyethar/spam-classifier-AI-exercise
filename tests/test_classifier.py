"""
test_classifier.py — Unit Tests for the Spam Classifier
========================================================
HTB Academy: Applications of AI in InfoSec
Author: lyethar (Fabian)

Tests model loading, prediction shape, and basic sanity checks.

Run:
    python -m pytest tests/ -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'spam_classifier.pkl')


# ─────────────────────────────────────────────────────────
# FIXTURE — Load model once for all tests
# ─────────────────────────────────────────────────────────
@pytest.fixture(scope='module')
def pipeline():
    """Load the trained pipeline. Skip all tests if model not found."""
    import joblib
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not trained yet. Run 'python src/train.py' first.")
    return joblib.load(MODEL_PATH)


# ─────────────────────────────────────────────────────────
# TEST 1: Model loads without errors
# ─────────────────────────────────────────────────────────
def test_model_loads(pipeline):
    """Verify the pipeline loads and has the expected steps."""
    assert 'tfidf' in pipeline.named_steps
    assert 'classifier' in pipeline.named_steps


# ─────────────────────────────────────────────────────────
# TEST 2: Obvious spam is classified as spam
# ─────────────────────────────────────────────────────────
def test_obvious_spam(pipeline):
    """Classic spam message should be classified as spam (label=1)."""
    spam_message = "WINNER!! FREE prize claim your £1000 now! Call URGENT"
    prediction = pipeline.predict([spam_message])[0]
    assert prediction == 1, f"Expected spam (1), got {prediction}"


# ─────────────────────────────────────────────────────────
# TEST 3: Obvious ham is classified as ham
# ─────────────────────────────────────────────────────────
def test_obvious_ham(pipeline):
    """Normal conversational message should be classified as ham (label=0)."""
    ham_message = "Hey, are you coming to the meeting at 3pm tomorrow?"
    prediction = pipeline.predict([ham_message])[0]
    assert prediction == 0, f"Expected ham (0), got {prediction}"


# ─────────────────────────────────────────────────────────
# TEST 4: Predict_proba returns valid probabilities
# ─────────────────────────────────────────────────────────
def test_probability_output(pipeline):
    """Probabilities must sum to ~1.0 and be between 0 and 1."""
    message = "Hello there"
    proba = pipeline.predict_proba([message])[0]
    assert len(proba) == 2
    assert abs(sum(proba) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in proba)


# ─────────────────────────────────────────────────────────
# TEST 5: Empty string handling
# ─────────────────────────────────────────────────────────
def test_empty_message(pipeline):
    """Empty string should not raise an exception."""
    try:
        prediction = pipeline.predict([""])[0]
        assert prediction in [0, 1]
    except Exception as e:
        pytest.fail(f"Empty message raised exception: {e}")


# ─────────────────────────────────────────────────────────
# TEST 6: Batch prediction
# ─────────────────────────────────────────────────────────
def test_batch_prediction(pipeline):
    """Model should handle multiple messages at once."""
    messages = [
        "Free entry! Win prizes now!",
        "I'll see you at the gym tomorrow",
        "Claim your reward immediately",
        "Can you pick up some milk?"
    ]
    predictions = pipeline.predict(messages)
    assert len(predictions) == 4
    assert all(p in [0, 1] for p in predictions)
