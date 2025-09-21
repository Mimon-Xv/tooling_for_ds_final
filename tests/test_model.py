import pytest
import pickle
import os
import tempfile
import shutil
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_utils import load_data, split_data, scale_data
from src.train_model import train

def test_model_training():
    """Test model training and saving functionality."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Run training
            model, scaler, test_r2, test_rmse = train()
            
            # Check that model files were created
            assert os.path.exists("app/model.pkl")
            assert os.path.exists("app/scaler.pkl")
            
            # Load saved model and scaler
            with open("app/model.pkl", "rb") as f:
                loaded_model = pickle.load(f)
            with open("app/scaler.pkl", "rb") as f:
                loaded_scaler = pickle.load(f)
            
            # Test basic prediction functionality
            X, y = load_data()
            _, X_test, _, y_test = split_data(X, y)
            X_test_scaled = loaded_scaler.transform(X_test)
            preds = loaded_model.predict(X_test_scaled)
            
            # Check prediction shape
            assert preds.shape[0] == X_test.shape[0]
            assert len(preds) == len(y_test)
            
            # Check that predictions are reasonable (not all the same value)
            assert len(set(preds)) > 1
            
            # Check that R² is reasonable (should be positive for a good model)
            assert test_r2 > 0
            
        finally:
            os.chdir(original_cwd)

def test_model_prediction_consistency():
    """Test that model predictions are consistent."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # Train a simple model for testing
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    preds1 = model.predict(X_test_scaled)
    preds2 = model.predict(X_test_scaled)
    
    # Predictions should be identical
    assert np.allclose(preds1, preds2)

def test_scaler_consistency():
    """Test that scaler works consistently."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # Transform test data again - should give same result
    X_test_scaled2 = scaler.transform(X_test)
    assert np.allclose(X_test_scaled, X_test_scaled2)
