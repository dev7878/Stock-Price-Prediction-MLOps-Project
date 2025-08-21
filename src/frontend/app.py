import streamlit as st
import requests
import plotly.io as pio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# API Configuration
import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

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

    # API Status Check
    try:
        health_response = requests.get(f"{API_URL}/health")
        version_response = requests.get(f"{API_URL}/version")
        if health_response.status_code == 200 and version_response.status_code == 200:
            version_info = version_response.json()
            st.sidebar.success(f"✅ API Connected (v{version_info['version']})")
        else:
            st.sidebar.error("❌ API Connection Failed")
    except Exception as e:
        st.sidebar.error(f"❌ API Connection Error: {str(e)}")

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

    # Toggle for baseline comparison
    show_baseline = st.sidebar.checkbox(
        "Show Naive Baseline",
        value=False,
        help="Compare predictions against simple moving average baseline"
    )

    # Date range selector
    st.sidebar.subheader("Date Range")
    use_custom_range = st.sidebar.checkbox("Use Custom Date Range", value=False)
    if use_custom_range:
        start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=90))
        end_date = st.sidebar.date_input("End Date", value=datetime.now())

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

        # Enhanced Analysis Section
        st.subheader("📊 Advanced Model Analysis")
        
        # Residuals Analysis
        st.subheader("🔍 Residuals Analysis")
        try:
            # Create residuals plot
            actual_values = []
            predicted_values = []
            model_names = []
            
            for model_name, preds in predictions["predictions"].items():
                if len(preds) > 0:
                    # For demo, create sample actual values
                    sample_actual = [100 + i * 0.5 + np.random.normal(0, 2) for i in range(len(preds))]
                    actual_values.extend(sample_actual)
                    predicted_values.extend(preds)
                    model_names.extend([model_name] * len(preds))
            
            if actual_values and predicted_values:
                residuals = [a - p for a, p in zip(actual_values, predicted_values)]
                
                # Residuals plot
                fig_residuals = go.Figure()
                fig_residuals.add_trace(go.Scatter(
                    x=predicted_values,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='red', size=8)
                ))
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="black")
                fig_residuals.update_layout(
                    title="Residuals vs Predicted Values",
                    xaxis_title="Predicted Values",
                    yaxis_title="Residuals (Actual - Predicted)",
                    showlegend=True
                )
                st.plotly_chart(fig_residuals, use_container_width=True)
                
                # Residuals statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Residual", f"{np.mean(residuals):.2f}")
                with col2:
                    st.metric("Residual Std Dev", f"{np.std(residuals):.2f}")
                with col3:
                    st.metric("Residual Range", f"{max(residuals) - min(residuals):.2f}")
        except Exception as e:
            st.warning(f"Could not generate residuals analysis: {str(e)}")

        # Feature Importance (if available)
        st.subheader("🎯 Feature Importance")
        try:
            # This would typically come from the API, but for now we'll show a placeholder
            feature_importance_data = {
                'Feature': ['SMA_20', 'RSI', 'MACD', 'Volume', 'Price_Change'],
                'Importance': [0.25, 0.20, 0.18, 0.15, 0.12]
            }
            
            fig_importance = go.Figure(data=[
                go.Bar(x=feature_importance_data['Importance'], 
                      y=feature_importance_data['Feature'],
                      orientation='h',
                      marker_color='lightblue')
            ])
            fig_importance.update_layout(
                title="Top Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                showlegend=False
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        except Exception as e:
            st.info("Feature importance data not available in this demo")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Note:** This is a demonstration dashboard. Predictions are based on historical data and should not be used as financial advice.
    """)

if __name__ == "__main__":
    main() 