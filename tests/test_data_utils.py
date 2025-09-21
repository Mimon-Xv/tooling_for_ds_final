import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_utils import load_data, split_data, scale_data, get_data_summary

def test_load_data_shape():
    """Test that data loading returns correct shapes."""
    X, y = load_data(as_frame=True)
    assert X.shape == (442, 10)  # 442 samples, 10 features
    assert y.shape == (442,)

def test_load_data_types():
    """Test that data loading returns correct types."""
    X, y = load_data(as_frame=True)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_split_data():
    """Test data splitting functionality."""
    X, y = load_data(as_frame=True)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)
    
    # Check shapes
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
    
    # Check that test size is approximately correct
    assert abs(X_test.shape[0] / X.shape[0] - 0.3) < 0.01

def test_scale_data():
    """Test data scaling functionality."""
    X, y = load_data(as_frame=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # Check that scaled data has mean 0 and std 1 for training set
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-10)
    
    # Check shapes are preserved
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape

def test_get_data_summary():
    """Test data summary functionality."""
    X, y = load_data(as_frame=True)
    summary = get_data_summary(X, y)
    
    assert summary['n_samples'] == 442
    assert summary['n_features'] == 10
    assert len(summary['feature_names']) == 10
    assert 'target_range' in summary
    assert 'target_mean' in summary
    assert 'target_std' in summary
