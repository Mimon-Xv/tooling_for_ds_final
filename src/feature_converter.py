"""
Feature conversion utilities for realistic user interface.
Converts between normalized (scikit-learn) values and realistic medical values.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from .feature_config import REALISTIC_RANGES

def load_diabetes_ranges():
    """Load the actual ranges from the diabetes dataset."""
    diabetes = load_diabetes()
    X = diabetes.data
    
    ranges = {}
    for i, feature in enumerate(diabetes.feature_names):
        ranges[feature] = {
            'min': float(X[:, i].min()),
            'max': float(X[:, i].max()),
            'mean': float(X[:, i].mean())
        }
    return ranges

def realistic_to_normalized(realistic_value, feature_name):
    """
    Convert realistic value to normalized value for model prediction.
    
    Args:
        realistic_value: Value in realistic units (e.g., age in years)
        feature_name: Name of the feature
    
    Returns:
        Normalized value for the model
    """
    # Get the actual ranges from the dataset
    dataset_ranges = load_diabetes_ranges()
    realistic_config = REALISTIC_RANGES[feature_name]
    
    # Get dataset min/max
    dataset_min = dataset_ranges[feature_name]['min']
    dataset_max = dataset_ranges[feature_name]['max']
    
    # Get realistic min/max
    realistic_min = realistic_config['min']
    realistic_max = realistic_config['max']
    
    # Linear interpolation
    normalized_value = dataset_min + (realistic_value - realistic_min) * (dataset_max - dataset_min) / (realistic_max - realistic_min)
    
    return normalized_value

def normalized_to_realistic(normalized_value, feature_name):
    """
    Convert normalized value to realistic value for display.
    
    Args:
        normalized_value: Normalized value from the model
        feature_name: Name of the feature
    
    Returns:
        Realistic value for display
    """
    # Get the actual ranges from the dataset
    dataset_ranges = load_diabetes_ranges()
    realistic_config = REALISTIC_RANGES[feature_name]
    
    # Get dataset min/max
    dataset_min = dataset_ranges[feature_name]['min']
    dataset_max = dataset_ranges[feature_name]['max']
    
    # Get realistic min/max
    realistic_min = realistic_config['min']
    realistic_max = realistic_config['max']
    
    # Linear interpolation
    realistic_value = realistic_min + (normalized_value - dataset_min) * (realistic_max - realistic_min) / (dataset_max - dataset_min)
    
    return realistic_value

def get_realistic_default(feature_name):
    """Get a realistic default value for a feature."""
    dataset_ranges = load_diabetes_ranges()
    realistic_config = REALISTIC_RANGES[feature_name]
    
    # Use the mean of the dataset, converted to realistic units
    dataset_mean = dataset_ranges[feature_name]['mean']
    return normalized_to_realistic(dataset_mean, feature_name)
