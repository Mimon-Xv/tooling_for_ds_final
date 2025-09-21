# 🩺 Diabetes Progression Prediction App

A machine learning web application that predicts diabetes progression using physiological measurements. Built with Streamlit, scikit-learn, and deployed with Docker.

## 📋 Overview

This application uses the scikit-learn diabetes dataset (442 samples, 10 features) to train a Random Forest Regressor model that predicts diabetes progression. The app provides an interactive interface with feature sliders, data visualizations, and model performance metrics.

## 🚀 Features

- **Interactive Prediction**: Adjust feature values using sliders to get real-time predictions
- **Data Visualizations**: 
  - Feature distribution histograms
  - Correlation heatmap
  - Predicted vs actual scatter plots
  - Feature importance charts
  - Residual plots
- **Model Performance Metrics**: R² score, RMSE, MAE
- **Responsive Design**: Clean, modern UI with tabbed interface
- **Docker Support**: Easy deployment with containerization
- **CI/CD**: Automated testing with GitHub Actions

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Testing**: pytest
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

## 📊 Dataset

The application uses the [scikit-learn diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html), which contains:

- **442 samples** of diabetes patients
- **10 physiological features**:
  - `age`: Age in years
  - `sex`: Gender (0: female, 1: male)
  - `bmi`: Body mass index
  - `bp`: Average blood pressure
  - `s1`: Total serum cholesterol
  - `s2`: Low-density lipoproteins
  - `s3`: High-density lipoproteins
  - `s4`: Total cholesterol / HDL
  - `s5`: Log of serum triglycerides level
  - `s6`: Blood sugar level
- **Target**: Quantitative measure of diabetes progression one year after baseline

## 🏗️ Project Structure

```
diabetes-streamlit-app/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci.yml        # GitHub Actions
├── data/                           # (optional: cache dataset)
├── src/
│   ├── data_utils.py               # data loading & preprocessing
│   ├── train_model.py              # model training script
│   └── charts.py                   # visualization functions
├── app/
│   ├── app.py                      # Streamlit interface
│   ├── model.pkl                   # saved model (generated)
│   └── scaler.pkl                  # saved scaler (generated)
└── tests/
    ├── test_data_utils.py
    └── test_model.py
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/diabetes-streamlit-app.git
   cd diabetes-streamlit-app
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python src/train_model.py
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app/app.py
   ```

6. **Open your browser** and navigate to `http://localhost:8501`

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t diabetes-app .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 diabetes-app
   ```

3. **Access the app** at `http://localhost:8501`

### Docker Hub Deployment

1. **Build and tag the image**
   ```bash
   docker build -t <your-dockerhub-username>/diabetes-app:latest .
   ```

2. **Push to Docker Hub**
   ```bash
   docker login
   docker push <your-dockerhub-username>/diabetes-app:latest
   ```

3. **Run from Docker Hub**
   ```bash
   docker run -p 8501:8501 <your-dockerhub-username>/diabetes-app:latest
   ```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

The tests cover:
- Data loading and preprocessing functions
- Model training and saving
- Prediction consistency
- Data scaling functionality

## 📈 Usage

1. **Input Features**: Use the sliders in the sidebar to adjust feature values
2. **Get Prediction**: Click the "Predict" button to see the diabetes progression prediction
3. **Explore Data**: Navigate through the tabs to explore:
   - Dataset overview and statistics
   - Feature distributions and correlations
   - Model performance metrics
   - Feature importance and model insights

## 🔧 Configuration

The model can be configured by modifying `src/train_model.py`:

- **Algorithm**: Currently uses RandomForestRegressor
- **Test Size**: Default 30% (configurable in `split_data`)
- **Random State**: Set to 42 for reproducibility
- **Number of Trees**: 100 estimators

## 📊 Model Performance

The Random Forest model typically achieves:
- **R² Score**: ~0.45-0.55
- **RMSE**: ~50-60
- **MAE**: ~40-50

Performance may vary based on the random state and data split.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for the diabetes dataset and ML algorithms
- [Streamlit](https://streamlit.io/) for the web framework
- [Plotly](https://plotly.com/) for interactive visualizations
- [Docker](https://www.docker.com/) for containerization

## 📚 References

- [scikit-learn diabetes dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Docker best practices](https://docs.docker.com/develop/best-practices/)

## 🐛 Troubleshooting

### Common Issues

1. **Model files not found**: Run `python src/train_model.py` first
2. **Port already in use**: Change the port with `streamlit run app/app.py --server.port 8502`
3. **Docker build fails**: Ensure all dependencies are in `requirements.txt`

### Getting Help

If you encounter any issues:
1. Check the [Issues](https://github.com/<your-username>/diabetes-streamlit-app/issues) page
2. Create a new issue with detailed error information
3. Include your Python version and operating system

---

**Note**: This application is for educational and demonstration purposes. The predictions should not be used for actual medical diagnosis or treatment decisions.
