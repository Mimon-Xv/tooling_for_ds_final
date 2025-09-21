import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def feature_distribution_chart(X: pd.DataFrame):
    """Return Plotly histograms for each feature."""
    figs = []
    for col in X.columns:
        fig = px.histogram(
            X, 
            x=col, 
            nbins=20, 
            title=f"Distribution of {col}",
            labels={col: col, 'count': 'Frequency'}
        )
        fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        figs.append((col, fig))
    return figs

def correlation_heatmap(X: pd.DataFrame):
    """Create correlation heatmap of features."""
    corr = X.corr()
    fig = px.imshow(
        corr, 
        text_auto=True, 
        aspect="auto", 
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r"
    )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def predictions_vs_actual_chart(y_true, y_pred):
    """Create scatter plot of predicted vs actual values."""
    df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    fig = px.scatter(
        df, 
        x="Actual", 
        y="Predicted", 
        trendline="ols",
        title="Predicted vs Actual Values",
        labels={"Actual": "Actual Values", "Predicted": "Predicted Values"}
    )
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def feature_importance_chart(model, feature_names):
    """Create feature importance chart."""
    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        df_importance,
        x='importance',
        y='feature',
        orientation='h',
        title="Feature Importance",
        labels={'importance': 'Importance', 'feature': 'Feature'}
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def residual_plot(y_true, y_pred):
    """Create residual plot."""
    residuals = y_true - y_pred
    fig = px.scatter(
        x=y_pred,
        y=residuals,
        title="Residual Plot",
        labels={'x': 'Predicted Values', 'y': 'Residuals'}
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
