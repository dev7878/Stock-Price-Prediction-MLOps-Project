import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
import os
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_status' not in st.session_state:
    st.session_state.api_status = 'unknown'
if 'api_url' not in st.session_state:
    st.session_state.api_url = os.getenv('API_URL', 'https://stock-prediction-api.onrender.com')

# Sample data generation for demo purposes
def generate_sample_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """Generate sample stock data for demonstration."""
    np.random.seed(hash(symbol) % 2**32)  # Deterministic but different per symbol
    
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # Generate realistic stock price movements
    base_price = 100 + hash(symbol) % 200  # Different base price per symbol
    returns = np.random.normal(0.001, 0.02, days)  # Daily returns with volatility
    
    # Add some trend and seasonality
    trend = np.linspace(0, 0.1, days)  # Slight upward trend
    seasonality = 0.02 * np.sin(2 * np.pi * np.arange(days) / 252)  # Annual seasonality
    
    # Calculate prices
    prices = [base_price]
    for i in range(1, days):
        price = prices[-1] * (1 + returns[i] + trend[i] + seasonality[i])
        prices.append(max(price, 1))  # Ensure price doesn't go negative
    
    # Create DataFrame with features
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days),
        'sma_20': pd.Series(prices).rolling(20).mean(),
        'sma_50': pd.Series(prices).rolling(50).mean(),
        'rsi': np.random.uniform(20, 80, days),
        'macd': np.random.uniform(-2, 2, days),
        'bollinger_upper': pd.Series(prices).rolling(20).mean() + 2 * pd.Series(prices).rolling(20).std(),
        'bollinger_lower': pd.Series(prices).rolling(20).mean() - 2 * pd.Series(prices).rolling(20).std()
    })
    
    return df

def generate_predictions(df: pd.DataFrame, model_type: str) -> np.ndarray:
    """Generate sample predictions for demonstration."""
    np.random.seed(hash(model_type) % 2**32)
    
    # Use the last 30 days of actual data as base
    recent_prices = df['close'].tail(30).values
    
    # Add some realistic prediction noise
    if model_type == 'lstm':
        # LSTM predictions tend to be smoother
        noise = np.random.normal(0, 0.01, 30)
        predictions = recent_prices * (1 + noise)
    elif model_type == 'xgboost':
        # XGBoost predictions have moderate noise
        noise = np.random.normal(0, 0.015, 30)
        predictions = recent_prices * (1 + noise)
    else:  # lightgbm
        # LightGBM predictions have higher noise
        noise = np.random.normal(0, 0.02, 30)
        predictions = recent_prices * (1 + noise)
    
    return predictions

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate prediction metrics."""
    if len(actual) != len(predicted):
        return {}
    
    # Calculate basic metrics
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    
    # Calculate directional accuracy
    actual_direction = np.diff(actual)
    predicted_direction = np.diff(predicted)
    directional_accuracy = np.mean(np.sign(actual_direction) == np.sign(predicted_direction)) * 100
    
    # Calculate MAPE
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }

def create_actual_vs_predicted_plot(df: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> go.Figure:
    """Create actual vs predicted comparison chart."""
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=df['date'].tail(30),
        y=df['close'].tail(30),
        mode='lines+markers',
        name='Actual',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    # Add predictions
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    for i, (model_name, pred) in enumerate(predictions.items()):
        fig.add_trace(go.Scatter(
            x=df['date'].tail(30),
            y=pred,
            mode='lines+markers',
            name=f'{model_name.upper()} Prediction',
            line=dict(color=colors[i], width=2, dash='dash'),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title='Actual vs Predicted Stock Prices',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_residuals_plot(df: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> go.Figure:
    """Create rolling residuals analysis."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Rolling Residuals (30-day window)', 'Residuals Distribution'),
        vertical_spacing=0.1
    )
    
    # Calculate residuals for each model
    actual = df['close'].tail(30).values
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        residuals = actual - pred
        
        # Rolling residuals
        rolling_residuals = pd.Series(residuals).rolling(7).mean()
        fig.add_trace(
            go.Scatter(
                x=df['date'].tail(30),
                y=rolling_residuals,
                mode='lines',
                name=f'{model_name.upper()} Residuals',
                line=dict(color=colors[i], width=2)
            ),
            row=1, col=1
        )
        
        # Residuals distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name=f'{model_name.upper()} Distribution',
                marker_color=colors[i],
                opacity=0.7,
                nbinsx=20
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title='Residuals Analysis',
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Rolling Residuals', row=1, col=1)
    fig.update_xaxes(title_text='Residual Value', row=2, col=1)
    fig.update_yaxes(title_text='Frequency', row=2, col=1)
    
    return fig

def create_drift_visualization(df: pd.DataFrame) -> go.Figure:
    """Create concept drift visualization."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Feature Drift Analysis', 'Statistical Drift Metrics'),
        vertical_spacing=0.15
    )
    
    # Calculate rolling statistics for drift detection
    window = 20
    
    # Price drift
    rolling_mean = df['close'].rolling(window).mean()
    rolling_std = df['close'].rolling(window).std()
    
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=rolling_mean,
            mode='lines',
            name='Rolling Mean (20-day)',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=rolling_mean + 2*rolling_std,
            mode='lines',
            name='Upper Bound (+2σ)',
            line=dict(color='red', width=1, dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=rolling_mean - 2*rolling_std,
            mode='lines',
            name='Lower Bound (-2σ)',
            line=dict(color='red', width=1, dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add actual prices
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            name='Actual Price',
            line=dict(color='black', width=1, alpha=0.7)
        ),
        row=1, col=1
    )
    
    # Drift metrics
    drift_metrics = []
    dates = []
    
    for i in range(window, len(df)):
        recent_data = df['close'].iloc[i-window:i]
        historical_data = df['close'].iloc[:i-window]
        
        if len(historical_data) > 0:
            # Calculate PSI-like metric
            recent_mean = recent_data.mean()
            historical_mean = historical_data.mean()
            
            if historical_mean != 0:
                drift_score = abs(recent_mean - historical_mean) / historical_mean * 100
                drift_metrics.append(drift_score)
                dates.append(df['date'].iloc[i])
    
    if drift_metrics:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drift_metrics,
                mode='lines+markers',
                name='Drift Score (%)',
                line=dict(color='#d62728', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title='Concept Drift Analysis',
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Price ($)', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Drift Score (%)', row=2, col=1)
    
    return fig

def check_api_status(api_url: str) -> str:
    """Check if the external API is available."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return 'connected'
        else:
            return 'error'
    except:
        return 'disconnected'

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">⚙️ Dashboard Settings</div>', unsafe_allow_html=True)
    
    # Symbol selection
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"]
    selected_symbol = st.sidebar.selectbox("Select Stock Symbol", symbols, index=0)
    
    # Date range selection
    st.sidebar.markdown("**📅 Date Range**")
    days = st.sidebar.slider("Number of Days", min_value=30, max_value=200, value=100, step=10)
    
    # Model comparison toggle
    st.sidebar.markdown("**🔍 Analysis Options**")
    show_baseline = st.sidebar.checkbox("Compare with Naive Baseline", value=True)
    show_residuals = st.sidebar.checkbox("Show Residuals Analysis", value=True)
    show_drift = st.sidebar.checkbox("Show Drift Analysis", value=True)
    
    # API status check
    st.sidebar.markdown("**🌐 API Status**")
    api_status = check_api_status(st.session_state.api_url)
    
    if api_status == 'connected':
        st.sidebar.success("✅ API Connected")
    elif api_status == 'error':
        st.sidebar.error("⚠️ API Error")
    else:
        st.sidebar.warning("❌ API Disconnected")
    
    # API URL configuration
    api_url = st.sidebar.text_input("API Base URL", value=st.session_state.api_url)
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url
        st.rerun()
    
    # Main content
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### 📊 Analysis for {selected_symbol}")
        st.markdown(f"*Analyzing {days} days of data*")
    
    with col2:
        st.metric("Current Price", f"${np.random.randint(100, 500):.2f}")
    
    with col3:
        st.metric("Daily Change", f"{np.random.uniform(-5, 5):.2f}%")
    
    # Generate sample data
    df = generate_sample_data(selected_symbol, days)
    
    # Generate predictions
    predictions = {
        'lstm': generate_predictions(df, 'lstm'),
        'xgboost': generate_predictions(df, 'xgboost'),
        'lightgbm': generate_predictions(df, 'lightgbm')
    }
    
    # Calculate metrics
    actual = df['close'].tail(30).values
    metrics = {}
    for model_name, pred in predictions.items():
        metrics[model_name] = calculate_metrics(actual, pred)
    
    # Display metrics
    st.markdown("### 📈 Model Performance Metrics")
    
    metric_cols = st.columns(3)
    for i, (model_name, metric) in enumerate(metrics.items()):
        with metric_cols[i]:
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            st.metric(f"{model_name.upper()} RMSE", f"${metric['rmse']:.2f}")
            st.metric(f"Directional Accuracy", f"{metric['directional_accuracy']:.1f}%")
            st.metric(f"MAPE", f"{metric['mape']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Actual vs Predicted Chart
    st.markdown("### 🎯 Actual vs Predicted Comparison")
    fig_actual_pred = create_actual_vs_predicted_plot(df, predictions)
    st.plotly_chart(fig_actual_pred, use_container_width=True)
    
    # Residuals Analysis
    if show_residuals:
        st.markdown("### 📊 Residuals Analysis")
        fig_residuals = create_residuals_plot(df, predictions)
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    # Drift Analysis
    if show_drift:
        st.markdown("### 🔄 Concept Drift Analysis")
        fig_drift = create_drift_visualization(df)
        st.plotly_chart(fig_drift, use_container_width=True)
    
    # Baseline Comparison
    if show_baseline:
        st.markdown("### 🆚 Model vs Naive Baseline")
        
        # Calculate naive baseline (last price)
        naive_baseline = np.full(30, df['close'].iloc[-31])  # Last known price
        
        baseline_metrics = calculate_metrics(actual, naive_baseline)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🤖 ML Models (Ensemble)**")
            ensemble_pred = np.mean([predictions['lstm'], predictions['xgboost'], predictions['lightgbm']], axis=0)
            ensemble_metrics = calculate_metrics(actual, ensemble_pred)
            
            st.metric("RMSE", f"${ensemble_metrics['rmse']:.2f}")
            st.metric("Directional Accuracy", f"{ensemble_metrics['directional_accuracy']:.1f}%")
            st.metric("MAPE", f"{ensemble_metrics['mape']:.1f}%")
        
        with col2:
            st.markdown("**📉 Naive Baseline**")
            st.metric("RMSE", f"${baseline_metrics['rmse']:.2f}")
            st.metric("Directional Accuracy", f"{baseline_metrics['directional_accuracy']:.1f}%")
            st.metric("MAPE", f"{baseline_metrics['mape']:.1f}%")
        
        # Improvement calculation
        improvement = ((baseline_metrics['rmse'] - ensemble_metrics['rmse']) / baseline_metrics['rmse']) * 100
        st.success(f"🎉 **ML Models improve over baseline by {improvement:.1f}%**")
    
    # Information box
    st.markdown("""
    <div class="info-box">
        <h4>ℹ️ About This Dashboard</h4>
        <p>This dashboard demonstrates stock price prediction using three ML models:</p>
        <ul>
            <li><strong>LSTM:</strong> Deep learning model for time series prediction</li>
            <li><strong>XGBoost:</strong> Gradient boosting for structured data</li>
            <li><strong>LightGBM:</strong> Light gradient boosting machine</li>
        </ul>
        <p><em>Note: This is a demonstration using simulated data. For production use, connect to real market data APIs.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit, Plotly, and Python</p>
        <p>Stock Price Prediction MLOps Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
