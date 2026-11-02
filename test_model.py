
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer


def test_model_file_exists():
    model = joblib.load("model.pkl")
    assert model is not None


def test_model_prediction():
    model = joblib.load("model.pkl")

    data = load_breast_cancer()
    sample = data.data[:5]

    predictions = model.predict(sample)

    assert len(predictions) == 5
    assert all(p in [0, 1] for p in predictions)


def test_model_accuracy():
    model = joblib.load("model.pkl")

    data = load_breast_cancer()
    predictions = model.predict(data.data)

    accuracy = np.mean(predictions == data.target)

    assert accuracy >= 0.90
