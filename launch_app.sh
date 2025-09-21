#!/bin/bash

# Diabetes Prediction App Launcher
echo "🩺 Starting Diabetes Progression Prediction App..."

# Change to the project directory
cd "/Users/simontessier/Documents/03-DSB classes/00-repos/01_tooling_for_ds/final_project"

# Activate virtual environment
source venv/bin/activate

# Check if model files exist
if [ ! -f "app/model.pkl" ] || [ ! -f "app/scaler.pkl" ]; then
    echo "📊 Training model..."
    python3 src/train_model.py
    echo "✅ Model training completed!"
else
    echo "✅ Model files found!"
fi

# Start Streamlit app
echo "🚀 Starting Streamlit app..."
echo "📱 Open your browser to http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the app"

streamlit run app/app.py --server.headless true --server.port 8501
