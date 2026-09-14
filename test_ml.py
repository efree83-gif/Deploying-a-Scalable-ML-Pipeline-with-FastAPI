import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from ml.model import train_model, inference, compute_model_metrics


# Dummy Data
@pytest.fixture
def dummy_data():
    """Creates a minimal dataset for testing model functions."""
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])
    return X, y


# Test Functions
def test_train_model(dummy_data):
    """
    # Verifies train_model returns a fitted LogisticRegression instance.
    """
    X, y = dummy_data
    model = train_model(X, y)

    # Check that the model is the correct type and has been fitted
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "classes_")


def test_inference(dummy_data):
    """
    # Verifies inference returns an array of predictions matching input length.
    """
    X, y = dummy_data
    model = train_model(X, y)
    preds = inference(model, X)

    # Check that predictions match the number of input rows
    assert len(preds) == len(X)
    assert isinstance(preds, np.ndarray)


def test_compute_model_metrics():
    """
    Test the compute_model_metrics function to ensure numerical correctness.
    Using dummy data: True Positives=1, False Positives=0, False Negatives=1
    """
    y_true = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]

    # Expected metrics for these specific arrays:
    # Precision = 1 / (1 + 0) = 1.0
    # Recall = 1 / (1 + 1) = 0.5
    # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 0.666...

    precision, recall, f1 = compute_model_metrics(y_true, preds)

    # Asserting return types (your original checks)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)

    # Asserting numerical correctness (the evaluator's requested fix)
    assert precision == 1.0
    assert recall == 0.5
    assert round(f1, 2) == 0.67
