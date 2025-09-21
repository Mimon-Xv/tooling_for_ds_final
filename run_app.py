#!/usr/bin/env python3
"""
Simple script to run the diabetes prediction Streamlit app.
This script ensures the model is trained before starting the app.
"""

import subprocess
import sys
import os

def main():
    """Run the diabetes prediction app."""
    print("🩺 Starting Diabetes Progression Prediction App...")
    
    # Check if model files exist
    model_path = "app/model.pkl"
    scaler_path = "app/scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("📊 Training model...")
        try:
            subprocess.run([sys.executable, "src/train_model.py"], check=True)
            print("✅ Model training completed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error training model: {e}")
            sys.exit(1)
    else:
        print("✅ Model files found!")
    
    print("🚀 Starting Streamlit app...")
    print("📱 Open your browser to http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    
    try:
        subprocess.run([
            "streamlit", "run", "app/app.py", 
            "--server.headless", "true",
            "--server.port", "8501"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit app: {e}")
        print("💡 Try running manually: streamlit run app/app.py")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")

if __name__ == "__main__":
    main()
