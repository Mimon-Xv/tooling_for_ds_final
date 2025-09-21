import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(as_frame=True):
    """Load diabetes dataset and return (X, y)."""
    data = load_diabetes(as_frame=as_frame)
    X = data.data
    y = data.target
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler, scaler.transform(X_train), scaler.transform(X_test)

def get_data_summary(X, y):
    """Get basic summary statistics of the dataset."""
    summary = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'feature_names': list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])],
        'target_range': (y.min(), y.max()),
        'target_mean': y.mean(),
        'target_std': y.std()
    }
    return summary
