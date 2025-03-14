import logging
import time
from pathlib import Path
from typing import Dict, List

import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Finance ML API")

# Initialize Prometheus metrics
PREDICTION_COUNTER = Counter(
    "prediction_count",
    "Number of predictions made",
    ["model_version"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction",
    ["model_version"]
)

# Load configuration
def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load model
def load_model(config: dict):
    """Load the latest model from MLflow."""
    try:
        model = mlflow.pyfunc.load_model(
            f"models:/{config['mlflow']['model_name']}/latest"
        )
        logger.info("Successfully loaded model from MLflow")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Initialize model and config
config = load_config()
model = load_model(config)

# Initialize Prometheus instrumentator
Instrumentator().instrument(app).expose(app)

class PredictionRequest(BaseModel):
    """Request model for prediction."""
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    """Response model for prediction."""
    prediction: float
    probability: float
    model_version: str
    prediction_time: float

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/model-info")
async def model_info():
    """Get information about the current model."""
    return {
        "model_name": config["mlflow"]["model_name"],
        "model_type": config["model"]["name"],
        "features": config["features"]["numerical_features"] + config["features"]["categorical_features"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the loaded model."""
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.features])
        
        # Validate features
        expected_features = set(config["features"]["numerical_features"] + config["features"]["categorical_features"])
        provided_features = set(request.features.keys())
        missing_features = expected_features - provided_features
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )
        
        # Record start time
        start_time = time.time()
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Calculate prediction time
        prediction_time = time.time() - start_time
        
        # Update metrics
        model_version = mlflow.get_run(mlflow.active_run().info.run_id).data.tags.get("version", "unknown")
        PREDICTION_COUNTER.labels(model_version=model_version).inc()
        PREDICTION_LATENCY.labels(model_version=model_version).observe(prediction_time)
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=float(probability),
            model_version=model_version,
            prediction_time=prediction_time
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/metrics")
async def metrics():
    """Get current model metrics."""
    try:
        # Get latest run metrics from MLflow
        latest_run = mlflow.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"]).experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        ).iloc[0]
        
        return {
            "metrics": latest_run.data.metrics,
            "parameters": latest_run.data.params,
            "model_version": latest_run.data.tags.get("version", "unknown")
        }
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching metrics: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["api"]["debug"],
        workers=config["api"]["workers"]
    ) 