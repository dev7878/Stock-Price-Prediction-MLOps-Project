import streamlit as st
import requests
import plotly.io as pio
import json
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

# API Configuration
API_URL = "http://127.0.0.1:8000"

def get_available_symbols():
    """Get list of available stock symbols from the API."""
    try:
        response = requests.get(f"{API_URL}/symbols")
        return response.json()["symbols"]
    except Exception as e:
        st.error(f"Error fetching symbols: {str(e)}")
        return []

def get_predictions(symbol: str, days: int):
    """Get predictions from the API."""
    try:
        response = requests.post(
            f"{API_URL}/predict/{symbol}",
            json={"symbol": symbol, "days": days}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error getting predictions: {str(e)}")
        return None

def get_plot(symbol: str, days: int):
    """Get plot from the API."""
    try:
        response = requests.get(f"{API_URL}/plot/{symbol}?days={days}")
        if response.status_code == 200:
            plot_data = response.json()
            fig = go.Figure(data=plot_data['data'], layout=plot_data['layout'])
            return fig
        else:
            st.error(f"Error fetching plot: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting plot: {str(e)}")
        return None

def format_metrics(metrics):
    """Format metrics for display."""
    return pd.DataFrame([{
        "RMSE": f"{m['rmse']:.2f}",
        "MAE": f"{m['mae']:.2f}",
        "MAPE": f"{m['mape']:.2f}%",
        "R²": f"{m['r2']:.2f}",
        "Dir. Accuracy": f"{m['directional_accuracy']:.2f}%",
        "Sharpe Ratio": f"{m['sharpe_ratio']:.2f}"
    } for m in metrics.values()], index=metrics.keys())

def main():
    st.set_page_config(
        page_title="Stock Price Prediction Dashboard",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 Stock Price Prediction Dashboard")
    st.markdown("""
    This dashboard provides stock price predictions using three different models:
    - LSTM (Deep Learning)
    - XGBoost (Gradient Boosting)
    - LightGBM (Gradient Boosting)
    """)

    # Sidebar
    st.sidebar.header("Settings")
    symbols = get_available_symbols()
    
    if not symbols:
        st.error("No symbols available. Please check if the API is running.")
        return

    selected_symbol = st.sidebar.selectbox(
        "Select Stock Symbol",
        symbols
    )

    prediction_days = st.sidebar.slider(
        "Prediction Days",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to predict"
    )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Price Predictions")
        plot = get_plot(selected_symbol, prediction_days)
        if plot:
            st.plotly_chart(plot, use_container_width=True)

    with col2:
        st.subheader("Model Performance Metrics")
        predictions = get_predictions(selected_symbol, prediction_days)
        
        if predictions:
            metrics_df = format_metrics(predictions["metrics"])
            st.dataframe(metrics_df, use_container_width=True)
            
            st.subheader("Latest Predictions")
            pred_df = pd.DataFrame(predictions["predictions"])
            st.dataframe(pred_df.tail(5), use_container_width=True)

    # Additional Analysis
    st.subheader("Model Comparison")
    if predictions:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Best RMSE Model",
                min(predictions["metrics"].items(), key=lambda x: x[1]["rmse"])[0].upper(),
                f"{min(m['rmse'] for m in predictions['metrics'].values()):.2f}"
            )
        
        with col2:
            st.metric(
                "Best Directional Accuracy",
                max(predictions["metrics"].items(), key=lambda x: x[1]["directional_accuracy"])[0].upper(),
                f"{max(m['directional_accuracy'] for m in predictions['metrics'].values()):.2f}%"
            )
        
        with col3:
            st.metric(
                "Best Sharpe Ratio",
                max(predictions["metrics"].items(), key=lambda x: x[1]["sharpe_ratio"])[0].upper(),
                f"{max(m['sharpe_ratio'] for m in predictions['metrics'].values()):.2f}"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    **Note:** This is a demonstration dashboard. Predictions are based on historical data and should not be used as financial advice.
    """)

if __name__ == "__main__":
    main() 