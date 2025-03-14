import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.models.train import train_model, evaluate_model

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)

@pytest.fixture
def model_config():
    """Create sample model configuration."""
    return {
        "model": {
            "params": {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "objective": "binary:logistic"
            },
            "metrics": ["accuracy", "precision", "recall", "f1", "auc"]
        }
    }

def test_train_model(sample_data, model_config):
    """Test model training functionality."""
    X, y = sample_data
    
    # Split data into train and validation
    X_train = X[:800]
    y_train = y[:800]
    X_val = X[800:]
    y_val = y[800:]
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val, model_config)
    
    # Check if model is trained
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

def test_evaluate_model(sample_data, model_config):
    """Test model evaluation functionality."""
    X, y = sample_data
    
    # Split data into train and validation
    X_train = X[:800]
    y_train = y[:800]
    X_val = X[800:]
    y_val = y[800:]
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val, model_config)
    
    # Evaluate model
    metrics = evaluate_model(model, X_val, y_val, model_config)
    
    # Check if all metrics are present
    assert all(metric in metrics for metric in model_config["model"]["metrics"])
    
    # Check if metrics are within expected ranges
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["auc"] <= 1

def test_model_predictions(sample_data, model_config):
    """Test model predictions."""
    X, y = sample_data
    
    # Train model
    model = train_model(X, y, X, y, model_config)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Check prediction shapes
    assert len(predictions) == len(X)
    assert probabilities.shape == (len(X), 2)
    
    # Check prediction values
    assert all(pred in [0, 1] for pred in predictions)
    assert all(0 <= prob <= 1 for prob in probabilities[:, 0])
    assert all(0 <= prob <= 1 for prob in probabilities[:, 1])
    assert np.allclose(probabilities.sum(axis=1), 1.0) 