import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from .data_utils import load_data, split_data, scale_data

def train():
    """Train a Random Forest model on the diabetes dataset."""
    print("Loading diabetes dataset...")
    X, y = load_data()
    print(f"Dataset shape: {X.shape}")
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    print("Scaling features...")
    scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Create app directory if it doesn't exist
    os.makedirs("app", exist_ok=True)
    
    # Save model and scaler
    print("Saving model and scaler...")
    with open("app/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("app/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print("Model training completed successfully!")
    return model, scaler, test_r2, test_rmse

if __name__ == "__main__":
    train()
