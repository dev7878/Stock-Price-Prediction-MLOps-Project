# Lightweight FastAPI app - Gradual feature addition
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Prediction API - Lightweight Version")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str
    days: Optional[int] = 30

class PredictionResponse(BaseModel):
    symbol: str
    prediction: float
    confidence: float
    message: str

# Sample data for demo
SAMPLE_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"]

@app.get("/")
async def root():
    return {
        "message": "Stock Prediction API - Lightweight Version",
        "status": "running",
        "version": "1.0.0",
        "features": ["health", "version", "symbols", "predict_sample"]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "service": "stock-prediction-api", "version": "lightweight"}

@app.get("/version")
async def version():
    return {
        "version": "1.0.0", 
        "environment": "lightweight",
        "mlflow_status": "disabled",
        "models_loaded": 0
    }

@app.get("/symbols")
async def get_symbols():
    """Get available stock symbols."""
    return {
        "symbols": SAMPLE_SYMBOLS,
        "count": len(SAMPLE_SYMBOLS),
        "message": "Sample symbols for demo purposes"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """Generate sample prediction (no ML models loaded)."""
    if request.symbol not in SAMPLE_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Symbol {request.symbol} not supported")
    
    # Generate sample prediction
    import random
    base_price = 100.0
    prediction = base_price + random.uniform(-20, 30)
    confidence = random.uniform(0.6, 0.9)
    
    return PredictionResponse(
        symbol=request.symbol,
        prediction=round(prediction, 2),
        confidence=round(confidence, 2),
        message="Sample prediction generated (ML models not loaded)"
    )

@app.get("/predict_sample")
async def predict_sample():
    """Get a sample prediction without ML models."""
    return {
        "symbol": "AAPL",
        "prediction": 150.25,
        "confidence": 0.85,
        "message": "Sample prediction - ML models not loaded",
        "note": "This is demo data, not actual ML predictions"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
