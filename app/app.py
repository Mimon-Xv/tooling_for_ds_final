import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sys
import os
from sklearn.datasets import load_diabetes

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import split_data, scale_data, get_data_summary
from src.charts import (
    feature_distribution_chart, 
    correlation_heatmap, 
    predictions_vs_actual_chart,
    feature_importance_chart,
    residual_plot
)

# Page configuration
st.set_page_config(
    page_title="Diabetes Progression Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🩺 Diabetes Progression Prediction")
st.markdown("""
This app uses machine learning to predict diabetes progression based on various physiological measurements.
The model is trained on the scikit-learn diabetes dataset with 442 samples and 10 features.
""")

# Load data
@st.cache_data
def load_cached_data():
    """Load and cache the diabetes dataset."""
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target
    return X, y

X, y = load_cached_data()

# Sidebar for user input
st.sidebar.header("🎛️ Input Features")
st.sidebar.markdown("Adjust the sliders to set feature values for prediction:")

# Get feature ranges for sliders
feature_ranges = {}
for feature in X.columns:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    feature_ranges[feature] = (min_val, max_val, mean_val)

# Create sliders for each feature
user_input = []
for feature in X.columns:
    min_val, max_val, mean_val = feature_ranges[feature]
    value = st.sidebar.slider(
        f"{feature}",
        min_val,
        max_val,
        mean_val,
        step=(max_val - min_val) / 100,
        format="%.3f",
        help=f"Range: [{min_val:.3f}, {max_val:.3f}]"
    )
    user_input.append(value)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler."""
    try:
        with open("app/model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("app/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        st.stop()

model, scaler = load_model_and_scaler()

# Make prediction
if st.sidebar.button("🔮 Predict", type="primary"):
    scaled_input = scaler.transform([user_input])
    prediction = model.predict(scaled_input)[0]
    
    st.sidebar.success(f"**Predicted diabetes progression: {prediction:.2f}**")
    
    # Show prediction interpretation
    if prediction < 100:
        interpretation = "Low progression risk"
        color = "green"
    elif prediction < 200:
        interpretation = "Moderate progression risk"
        color = "orange"
    else:
        interpretation = "High progression risk"
        color = "red"
    
    st.sidebar.markdown(f"**Interpretation:** :{color}[{interpretation}]")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "📈 Feature Analysis", "🎯 Model Performance", "🔍 Model Insights"])

with tab1:
    st.subheader("Dataset Overview")
    
    # Show dataset summary
    summary = get_data_summary(X, y)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Samples", summary['n_samples'])
    with col2:
        st.metric("Features", summary['n_features'])
    with col3:
        st.metric("Target Mean", f"{summary['target_mean']:.2f}")
    with col4:
        st.metric("Target Std", f"{summary['target_std']:.2f}")
    
    # Show raw data option
    if st.checkbox("Show raw data"):
        st.dataframe(pd.concat([X, y.rename("target")], axis=1), width='stretch')
    
    # Show basic statistics
    st.subheader("Feature Statistics")
    st.dataframe(X.describe(), width='stretch')

with tab2:
    st.subheader("Feature Distribution Analysis")
    
    # Feature distribution charts
    st.markdown("**Individual Feature Distributions:**")
    for col, fig in feature_distribution_chart(X):
        st.plotly_chart(fig, width='stretch')
    
    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    corr_fig = correlation_heatmap(X)
    st.plotly_chart(corr_fig, width='stretch')

with tab3:
    st.subheader("Model Performance on Test Set")
    
    # Get test predictions
    _, X_test, _, y_test = split_data(X, y)
    scaler2, _, X_test_scaled = scale_data(X_test, X_test)
    preds = model.predict(scaler.transform(X_test))
    
    # Performance metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.4f}")
    with col3:
        st.metric("MAE", f"{mae:.4f}")
    
    # Predictions vs Actual chart
    st.subheader("Predicted vs Actual Values")
    pvf = predictions_vs_actual_chart(y_test, preds)
    st.plotly_chart(pvf, width='stretch')
    
    # Residual plot
    st.subheader("Residual Plot")
    residual_fig = residual_plot(y_test, preds)
    st.plotly_chart(residual_fig, width='stretch')

with tab4:
    st.subheader("Model Insights")
    
    # Feature importance
    st.markdown("**Feature Importance:**")
    importance_fig = feature_importance_chart(model, X.columns)
    st.plotly_chart(importance_fig, width='stretch')
    
    # Model information
    st.subheader("Model Information")
    st.markdown(f"""
    - **Algorithm:** Random Forest Regressor
    - **Number of Trees:** {model.n_estimators}
    - **Max Depth:** {model.max_depth if model.max_depth else 'Unlimited'}
    - **Random State:** {model.random_state}
    """)
    
    # Feature descriptions
    st.subheader("Feature Descriptions")
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Gender (0: female, 1: male)',
        'bmi': 'Body mass index',
        'bp': 'Average blood pressure',
        's1': 'Total serum cholesterol',
        's2': 'Low-density lipoproteins',
        's3': 'High-density lipoproteins',
        's4': 'Total cholesterol / HDL',
        's5': 'Log of serum triglycerides level',
        's6': 'Blood sugar level'
    }
    
    for feature, description in feature_descriptions.items():
        st.markdown(f"- **{feature}:** {description}")

# Footer
st.markdown("---")
st.markdown("""
**About this app:** This application uses the scikit-learn diabetes dataset and a Random Forest Regressor 
to predict diabetes progression. The model is trained on 442 samples with 10 physiological features.
""")
