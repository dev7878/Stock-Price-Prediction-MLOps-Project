import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Configure logging for cloud deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to Python path for cloud deployment
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load configuration
def load_config():
    """Load configuration with environment variable overrides for cloud deployment."""
    try:
        config_path = Path("configs/config.yaml")
        if not config_path.exists():
            # Fallback for cloud deployment
            logger.warning("Config file not found, using environment variables")
            return {
                "api": {
                    "host": os.getenv("API_HOST", "0.0.0.0"),
                    "port": int(os.getenv("PORT", "8000"))
                },
                "mlflow": {
                    "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
                    "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "stock_price_prediction")
                },
                "data_sources": {
                    "alpha_vantage": {"api_key": os.getenv("ALPHA_VANTAGE_API_KEY", "")},
                    "polygon": {"api_key": os.getenv("POLYGON_API_KEY", "")}
                },
                "model": {"model_path": os.getenv("MODEL_PATH", "./models")}
            }
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Override config values with environment variables for cloud deployment
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            config["data_sources"]["alpha_vantage"]["api_key"] = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        if os.getenv("POLYGON_API_KEY"):
            config["data_sources"]["polygon"]["api_key"] = os.getenv("POLYGON_API_KEY")
        
        # Override MLflow tracking URI for external deployment
        if os.getenv("MLFLOW_TRACKING_URI"):
            config["mlflow"]["tracking_uri"] = os.getenv("MLFLOW_TRACKING_URI")
        
        if os.getenv("MLFLOW_EXPERIMENT_NAME"):
            config["mlflow"]["experiment_name"] = os.getenv("MLFLOW_EXPERIMENT_NAME")
        
        if os.getenv("API_HOST"):
            config["api"]["host"] = os.getenv("API_HOST")
        
        if os.getenv("PORT"):
            try:
                config["api"]["port"] = int(os.getenv("PORT"))
            except (ValueError, TypeError):
                logger.warning(f"Invalid PORT env var: {os.getenv('PORT')}, using default")
                pass
        
        if os.getenv("MODEL_PATH"):
            config["model"]["model_path"] = os.getenv("MODEL_PATH")
        
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Return minimal config for cloud deployment
        return {
            "api": {"host": "0.0.0.0", "port": 8000},
            "mlflow": {"tracking_uri": "sqlite:///mlflow.db", "experiment_name": "stock_price_prediction"},
            "data_sources": {"alpha_vantage": {"api_key": ""}, "polygon": {"api_key": ""}},
            "model": {"model_path": "./models"}
        }

config = load_config()
logger.info(f"Configuration loaded: API={config['api']}, MLflow={config['mlflow']}")

# Initialize FastAPI app with CORS for UI integration
app = FastAPI(
    title="Stock Price Prediction API",
    description="ML-powered stock price prediction service",
    version="1.0.0"
)

# Configure CORS for UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your UI domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    days: int = 30

class PredictionResponse(BaseModel):
    symbol: str
    predictions: list
    metrics: dict

# Stock Predictor Class
class StockPredictor:
    def __init__(self, config: dict):
        """Initialize predictor with external MLflow tracking."""
        self.config = config
        self.models_path = Path(config["model"]["model_path"])
        
        # Set up MLflow with external tracking
        try:
            logger.info(f"Setting up MLflow with tracking URI: {config['mlflow']['tracking_uri']}")
            mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
            mlflow.set_experiment(f"{config['mlflow']['experiment_name']}_api")
            logger.info("MLflow setup successful")
        except Exception as e:
            logger.warning(f"MLflow setup failed, continuing without tracking: {e}")
            # Set a local fallback
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Load models (with error handling)
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models with error handling."""
        try:
            # Load LSTM model
            if (self.models_path / "lstm_model.h5").exists():
                self.models["lstm"] = tf.keras.models.load_model(self.models_path / "lstm_model.h5")
                logger.info("LSTM model loaded successfully")
            
            # Load XGBoost model
            if (self.models_path / "xgboost_model.pkl").exists():
                self.models["xgboost"] = xgb.Booster()
                self.models["xgboost"].load_model(str(self.models_path / "xgboost_model.pkl"))
                logger.info("XGBoost model loaded successfully")
            
            # Load LightGBM model
            if (self.models_path / "lightgbm_model.pkl").exists():
                self.models["lightgbm"] = lgb.Booster(model_file=str(self.models_path / "lightgbm_model.pkl"))
                logger.info("LightGBM model loaded successfully")
            
            logger.info(f"Loaded {len(self.models)} models")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.models = {}
    
    def prepare_data(self, symbol: str):
        """Prepare data for prediction (simplified for demo)."""
        # For demo purposes, generate sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)  # For reproducible demo
        
        # Generate sample features
        features = np.random.randn(100, 10)
        target = np.random.randn(100)
        
        return features, target, dates, [f"feature_{i}" for i in range(10)]
    
    def make_predictions(self, symbol: str, days: int = 30):
        """Make predictions using available models."""
        try:
            X, y, dates, feature_names = self.prepare_data(symbol)
            
            predictions = {}
            metrics = {}
            
            # Make predictions with each available model
            for model_name, model in self.models.items():
                try:
                    if model_name == "lstm":
                        # LSTM expects 3D input
                        X_reshaped = X.reshape(1, X.shape[0], X.shape[1])
                        pred = model.predict(X_reshaped).flatten()
                    elif model_name == "xgboost":
                        dmatrix = xgb.DMatrix(X)
                        pred = model.predict(dmatrix)
                    elif model_name == "lightgbm":
                        pred = model.predict(X)
                    else:
                        continue
                    
                    predictions[model_name] = pred.tolist()
                    
                    # Calculate metrics
                    if len(pred) == len(y):
                        mse = np.mean((pred - y) ** 2)
                        rmse = np.sqrt(mse)
                        mae = np.mean(np.abs(pred - y))
                        
                        metrics[model_name] = {
                            "rmse": float(rmse),
                            "mae": float(mae),
                            "mse": float(mse)
                        }
                        
                        # Log to MLflow
                        try:
                            with mlflow.start_run():
                                mlflow.log_metric(f"{model_name}_rmse", rmse)
                                mlflow.log_metric(f"{model_name}_mae", mae)
                                mlflow.log_metric(f"{model_name}_mse", mse)
                        except Exception as e:
                            logger.warning(f"Failed to log to MLflow: {e}")
                
                except Exception as e:
                    logger.error(f"Prediction failed for {model_name}: {e}")
                    continue
            
            return predictions, metrics
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def create_plot(self, symbol: str):
        """Create visualization plot."""
        try:
            X, y, dates, feature_names = self.prepare_data(symbol)
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Stock Price Prediction', 'Feature Importance'),
                vertical_spacing=0.1
            )
            
            # Add actual vs predicted
            fig.add_trace(
                go.Scatter(x=dates, y=y, mode='lines', name='Actual', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Add predictions if models are available
            for model_name, model in self.models.items():
                try:
                    if model_name == "lstm":
                        X_reshaped = X.reshape(1, X.shape[0], X.shape[1])
                        pred = model.predict(X_reshaped).flatten()
                    elif model_name == "xgboost":
                        dmatrix = xgb.DMatrix(X)
                        pred = model.predict(dmatrix)
                    elif model_name == "lightgbm":
                        pred = model.predict(X)
                    else:
                        continue
                    
                    fig.add_trace(
                        go.Scatter(x=dates, y=pred, mode='lines', name=f'{model_name.upper()} Prediction'),
                        row=1, col=1
                    )
                except Exception as e:
                    logger.warning(f"Failed to add {model_name} prediction to plot: {e}")
                    continue
            
            # Add feature importance (placeholder)
            if feature_names:
                importance = np.random.rand(len(feature_names))  # Placeholder
                fig.add_trace(
                    go.Bar(x=feature_names, y=importance, name='Feature Importance'),
                    row=2, col=1
                )
            
            fig.update_layout(
                title=f'Stock Price Prediction for {symbol}',
                xaxis_title='Date',
                yaxis_title='Price',
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create plot: {e}")
            raise

# API Key Authentication (Optional)
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if enabled."""
    api_key_enabled = os.getenv("API_KEY_ENABLED", "false").lower() == "true"
    if not api_key_enabled:
        return None
    
    api_key_value = os.getenv("API_KEY_VALUE")
    if not api_key_value:
        return None
    
    if x_api_key != api_key_value:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key

# Initialize predictor (with error handling)
logger.info("Starting predictor initialization...")
try:
    logger.info("Initializing StockPredictor...")
    predictor = StockPredictor(config)
    logger.info("StockPredictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    logger.warning("Continuing without prediction capabilities...")
    predictor = None

logger.info("FastAPI app initialization complete")
logger.info(f"Available endpoints: /, /health, /version, /symbols, /predict/{{symbol}}, /plot/{{symbol}}, /metrics/{{symbol}}")
logger.info("API server ready to start...")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Stock Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/version", 
            "/symbols",
            "/predict/{symbol}",
            "/plot/{symbol}",
            "/metrics/{symbol}"
        ],
        "models_loaded": len(predictor.models) if predictor else 0,
        "mlflow_tracking": config["mlflow"]["tracking_uri"] if config else "unknown"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "models_available": len(predictor.models) if predictor else 0,
        "mlflow_status": "connected" if config and "mlflow" in config else "unknown"
    }

@app.get("/version")
async def get_version():
    """Get API version information."""
    return {
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(predictor.models) if predictor else 0,
        "mlflow_tracking": config["mlflow"]["tracking_uri"] if config else "unknown"
    }

@app.get("/symbols")
async def get_symbols():
    """Get available stock symbols."""
    return {"symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"]}

@app.post("/predict/{symbol}", response_model=PredictionResponse)
async def predict_stock(
    symbol: str, 
    request: PredictionRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Predict stock prices for a given symbol."""
    symbol = symbol.upper()
    
    if symbol not in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"]:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction service temporarily unavailable")
    
    try:
        predictions, metrics = predictor.make_predictions(symbol, request.days)
        
        return {
            "symbol": symbol,
            "predictions": predictions,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Prediction failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/plot/{symbol}")
async def get_plot(symbol: str):
    """Get prediction plot for a symbol."""
    symbol = symbol.upper()
    
    if symbol not in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"]:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction service temporarily unavailable")
    
    try:
        fig = predictor.create_plot(symbol)
        return fig.to_json()
    except Exception as e:
        logger.error(f"Failed to create plot for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create plot: {str(e)}")

@app.get("/metrics/{symbol}")
async def get_metrics(symbol: str):
    """Get prediction metrics for a symbol."""
    symbol = symbol.upper()
    
    if symbol not in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"]:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction service temporarily unavailable")
    
    try:
        _, metrics = predictor.make_predictions(symbol)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

# Main execution for local development
if __name__ == "__main__":
    import uvicorn
    
    # Use environment variable PORT if available (for cloud deployment), otherwise config
    port = int(os.getenv("PORT", config["api"]["port"]))
    host = os.getenv("HOST", config["api"]["host"])
    
    logger.info(f"Starting API server on {host}:{port}")
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise 