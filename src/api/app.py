from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import mlflow
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load configuration
def load_config():
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="API for stock price predictions using LSTM, XGBoost, and LightGBM models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    symbol: str
    days: int = 1

class ModelMetrics(BaseModel):
    rmse: float
    mae: float
    mape: float
    r2: float
    directional_accuracy: float
    sharpe_ratio: float

class PredictionResponse(BaseModel):
    symbol: str
    predictions: Dict[str, List[float]]
    metrics: Dict[str, ModelMetrics]

class StockPredictor:
    def __init__(self, config: dict):
        self.config = config
        self.processed_data_path = Path(config["data"]["processed_data_path"])
        self.models_path = Path(config["model"]["model_path"])
        
        # Set up MLflow
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(f"{config['mlflow']['experiment_name']}_api")

    def load_models(self, symbol: str):
        """Load all models for a given symbol."""
        model_path = self.models_path / symbol
        
        try:
            # Load LSTM model
            lstm_model = tf.keras.models.load_model(str(model_path / 'lstm_model.keras'))
            
            # Load XGBoost model
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(str(model_path / 'xgboost_model.json'))
            
            # Load LightGBM model
            lgb_model = lgb.Booster(model_file=str(model_path / 'lightgbm_model.txt'))
            
            return lstm_model, xgb_model, lgb_model
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Models for {symbol} not found: {str(e)}")

    def prepare_data(self, symbol: str):
        """Prepare data for prediction."""
        try:
            # Load processed data
            data_path = self.processed_data_path / f"{symbol}_processed.parquet"
            df = pd.read_parquet(data_path)
            
            # Drop object columns
            df = df.select_dtypes(exclude=['object'])
            df['target'] = df['close'].shift(-1)
            df = df.dropna()
            
            # Select features
            features = [col for col in df.columns if col not in ['target']]
            X = df[features].values
            y = df['target'].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            return X_scaled, y, df.index, features
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Data for {symbol} not found: {str(e)}")

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Calculate model metrics."""
        metrics = ModelMetrics(
            rmse=float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            mae=float(np.mean(np.abs(y_true - y_pred))),
            mape=float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            r2=float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)),
            directional_accuracy=float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100),
            sharpe_ratio=float(np.mean(np.diff(y_pred) / y_pred[:-1]) / np.std(np.diff(y_pred) / y_pred[:-1]) if len(y_pred) > 1 else 0)
        )
        return metrics

    def make_predictions(self, symbol: str, days: int = 1):
        """Make predictions using all models."""
        # Load data and models
        X, y, dates, features = self.prepare_data(symbol)
        lstm_model, xgb_model, lgb_model = self.load_models(symbol)
        
        # Get recent data for prediction
        X_recent = X[-days:]
        y_recent = y[-days:]
        
        # Prepare data for LSTM
        X_lstm = X_recent.reshape((X_recent.shape[0], 1, X_recent.shape[1]))
        
        # Make predictions
        predictions = {
            'lstm': lstm_model.predict(X_lstm).flatten().tolist(),
            'xgboost': xgb_model.predict(X_recent).tolist(),
            'lightgbm': lgb_model.predict(X_recent).tolist()
        }
        
        # Calculate metrics
        metrics = {
            'lstm': self.calculate_metrics(y_recent, predictions['lstm']),
            'xgboost': self.calculate_metrics(y_recent, predictions['xgboost']),
            'lightgbm': self.calculate_metrics(y_recent, predictions['lightgbm'])
        }
        
        return predictions, metrics

    def create_prediction_plot(self, symbol: str, days: int = 30):
        """Create interactive plot using plotly."""
        X, y, dates, features = self.prepare_data(symbol)
        lstm_model, xgb_model, lgb_model = self.load_models(symbol)
        
        # Get recent data
        X_recent = X[-days:]
        y_recent = y[-days:]
        dates_recent = dates[-days:]
        
        # Prepare data for LSTM
        X_lstm = X_recent.reshape((X_recent.shape[0], 1, X_recent.shape[1]))
        
        # Make predictions
        lstm_pred = lstm_model.predict(X_lstm).flatten()
        xgb_pred = xgb_model.predict(X_recent)
        lgb_pred = lgb_model.predict(X_recent)
        
        # Create plot
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            x=dates_recent,
            y=y_recent,
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=dates_recent,
            y=lstm_pred,
            name='LSTM',
            line=dict(color='red', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates_recent,
            y=xgb_pred,
            name='XGBoost',
            line=dict(color='green', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates_recent,
            y=lgb_pred,
            name='LightGBM',
            line=dict(color='blue', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Stock Price Predictions for {symbol}',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )
        
        return fig

# Initialize predictor
predictor = StockPredictor(config)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Stock Price Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/symbols")
async def get_symbols():
    """Get list of available symbols."""
    return {"symbols": config["data"]["symbols"]}

@app.post("/predict/{symbol}", response_model=PredictionResponse)
async def predict(symbol: str, request: PredictionRequest):
    """Make predictions for a symbol."""
    if symbol not in config["data"]["symbols"]:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    predictions, metrics = predictor.make_predictions(symbol, request.days)
    
    return {
        "symbol": symbol,
        "predictions": predictions,
        "metrics": metrics
    }

@app.get("/plot/{symbol}")
async def get_plot(symbol: str, days: int = 30):
    """Get interactive plot for a symbol."""
    if symbol not in config["data"]["symbols"]:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    try:
        X, y, dates, features = predictor.prepare_data(symbol)
        lstm_model, xgb_model, lgb_model = predictor.load_models(symbol)
        
        # Get recent data
        X_recent = X[-days:]
        y_recent = y[-days:]
        dates_recent = dates[-days:]
        
        # Prepare data for LSTM
        X_lstm = X_recent.reshape((X_recent.shape[0], 1, X_recent.shape[1]))
        
        # Make predictions
        lstm_pred = lstm_model.predict(X_lstm).flatten()
        xgb_pred = xgb_model.predict(X_recent)
        lgb_pred = lgb_model.predict(X_recent)
        
        # Convert dates to string format
        dates_str = [d.strftime('%Y-%m-%d') for d in dates_recent]
        
        # Create plot data
        plot_data = {
            'data': [
                {
                    'x': dates_str,
                    'y': y_recent.tolist(),
                    'name': 'Actual',
                    'type': 'scatter',
                    'line': {'color': 'black', 'width': 2}
                },
                {
                    'x': dates_str,
                    'y': lstm_pred.tolist(),
                    'name': 'LSTM',
                    'type': 'scatter',
                    'line': {'color': 'red', 'dash': 'dash'}
                },
                {
                    'x': dates_str,
                    'y': xgb_pred.tolist(),
                    'name': 'XGBoost',
                    'type': 'scatter',
                    'line': {'color': 'green', 'dash': 'dash'}
                },
                {
                    'x': dates_str,
                    'y': lgb_pred.tolist(),
                    'name': 'LightGBM',
                    'type': 'scatter',
                    'line': {'color': 'blue', 'dash': 'dash'}
                }
            ],
            'layout': {
                'title': f'Stock Price Predictions for {symbol}',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Price'},
                'hovermode': 'x unified'
            }
        }
        
        return plot_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating plot: {str(e)}")

@app.get("/metrics/{symbol}")
async def get_metrics(symbol: str):
    """Get model metrics for a symbol."""
    if symbol not in config["data"]["symbols"]:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    _, metrics = predictor.make_predictions(symbol)
    return metrics

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"]) 