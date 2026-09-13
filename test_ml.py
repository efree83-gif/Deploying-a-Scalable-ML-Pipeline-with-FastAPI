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
    # Verifies compute_model_metrics correctly calculates and returns three floats.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Check that all three metrics are returned as floats
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
