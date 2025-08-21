# 🏗️ Architecture Overview

This document provides a comprehensive overview of the Stock Price Prediction MLOps project architecture, including system design, data flow, and component interactions.

## 🎯 System Overview

The project implements a complete MLOps pipeline for stock price prediction, consisting of three main components:

1. **Data Pipeline**: Ingestion, processing, and feature engineering
2. **ML Pipeline**: Model training, evaluation, and deployment
3. **Serving Layer**: API service and interactive dashboard

## 🏛️ High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Feature Store  │    │   ML Models     │
│                 │    │                 │    │                 │
│ • Yahoo Finance │───▶│ • Technical     │───▶│ • LSTM          │
│ • Alpha Vantage │    │   Indicators    │    │ • XGBoost       │
│ • Polygon       │    │ • Market Data   │    │ • LightGBM      │
│ • News/Sentiment│    │ • Sentiment     │    │ • Ensemble      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   MLflow        │    │   Model         │
                       │                 │    │   Registry      │
                       │ • Experiment    │    │                 │
                       │   Tracking      │    │ • Versioning    │
                       │ • Model         │    │ • Artifacts     │
                       │   Registry      │    │ • Deployment    │
                       └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   FastAPI       │    │   Streamlit     │
                       │   Service       │    │   Dashboard     │
                       │                 │    │                 │
                       │ • REST API      │    │ • Interactive   │
                       │ • Predictions   │    │   Charts        │
                       │ • Health Checks │    │ • Model         │
                       │ • Monitoring    │    │   Comparison    │
                       └─────────────────┘    └─────────────────┘
```

## 🔄 Data Flow

### 1. Data Ingestion

```
External Sources → Data Collectors → Raw Data Storage
```

- **Sources**: Yahoo Finance, Alpha Vantage, Polygon, News APIs
- **Collectors**: Scheduled Python scripts using `yfinance`, `alpha_vantage`
- **Storage**: Parquet files in `data/raw/` directory

### 2. Data Processing

```
Raw Data → Feature Engineering → Processed Data
```

- **Cleaning**: Handle missing values, outliers, data validation
- **Feature Engineering**: Technical indicators, market data, sentiment
- **Storage**: Processed data in `data/processed/` directory

### 3. Model Training

```
Processed Data → Model Training → Model Artifacts
```

- **Training**: LSTM, XGBoost, LightGBM models
- **Evaluation**: Cross-validation, backtesting, performance metrics
- **Storage**: Models saved in `models/` directory

### 4. Model Serving

```
Model Artifacts → API Service → Predictions
```

- **Loading**: Models loaded into memory on API startup
- **Inference**: Real-time predictions via REST API
- **Caching**: Model predictions cached for performance

## 🏗️ Component Architecture

### Data Layer

#### Data Collectors (`src/data/`)

- **`collect_data.py`**: Fetches data from external APIs
- **`process_data.py`**: Cleans and transforms raw data
- **Scheduled Execution**: Using cron jobs or GitHub Actions

#### Feature Engineering (`src/features/`)

- **`technical_indicators.py`**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Market Indicators**: S&P 500, NASDAQ, VIX integration
- **Sentiment Analysis**: News and social media sentiment scoring

### ML Layer

#### Model Training (`src/models/`)

- **`train.py`**: Orchestrates model training pipeline
- **`evaluate.py`**: Model performance evaluation
- **`predict.py`**: Prediction logic and ensemble methods

#### MLflow Integration

- **Experiment Tracking**: Model parameters, metrics, artifacts
- **Model Registry**: Versioned model storage and deployment
- **Artifact Storage**: Feature importance plots, predictions

### Serving Layer

#### FastAPI Service (`src/api/`)

- **REST API**: Prediction endpoints, health checks
- **Model Loading**: Efficient model loading and caching
- **Error Handling**: Comprehensive error handling and logging
- **CORS Support**: Cross-origin resource sharing for UI

#### Streamlit Dashboard (`src/frontend/`)

- **Interactive Charts**: Plotly-based visualizations
- **Real-time Updates**: Live data from API
- **Model Comparison**: Side-by-side model performance
- **Responsive Design**: Mobile-friendly interface

## 🔌 API Design

### Endpoints

```
GET  /                    # API information
GET  /health             # Health check
GET  /version            # Version information
GET  /symbols            # Available stock symbols
POST /predict/{symbol}   # Get predictions
GET  /plot/{symbol}      # Get visualization data
GET  /metrics/{symbol}   # Get model metrics
```

### Request/Response Models

```python
# Prediction Request
class PredictionRequest(BaseModel):
    symbol: str
    days: int = 1

# Prediction Response
class PredictionResponse(BaseModel):
    symbol: str
    predictions: Dict[str, List[float]]
    metrics: Dict[str, ModelMetrics]
```

## 🐳 Container Architecture

### Docker Services

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLflow        │    │   FastAPI       │    │   Streamlit     │
│   Service       │    │   Service       │    │   Dashboard     │
│                 │    │                 │    │                 │
│ Port: 5000      │    │ Port: 8000      │    │ Port: 8501      │
│ Health: /health │    │ Health: /health │    │ Health: /_stcore/health
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     Docker Network        │
                    │   (mlops-network)        │
                    └───────────────────────────┘
```

### Service Dependencies

1. **MLflow**: No dependencies, starts first
2. **API Service**: Depends on MLflow health
3. **UI Service**: Depends on API service health

## 🔒 Security Architecture

### Authentication

- **API Key Support**: Optional API key authentication via headers
- **CORS Configuration**: Configurable cross-origin policies
- **Rate Limiting**: Configurable request rate limits

### Data Security

- **Environment Variables**: Sensitive configuration via `.env`
- **No Secrets in Code**: All secrets externalized
- **Input Validation**: Pydantic models for request validation

## 📊 Monitoring & Observability

### Health Checks

- **Service Health**: `/health` endpoints for all services
- **Dependency Health**: MLflow connection status
- **Model Health**: Model loading and prediction status

### Metrics Collection

- **API Metrics**: Response times, error rates, throughput
- **Model Metrics**: Prediction accuracy, latency, drift
- **System Metrics**: CPU, memory, disk usage

### Logging

- **Structured Logging**: JSON-formatted logs
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Centralized Logging**: Docker logs aggregation

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

#### CI Pipeline

- **Python Matrix**: 3.10 and 3.11 testing
- **Code Quality**: Ruff, Black, MyPy
- **Testing**: Pytest with coverage
- **Security**: Bandit security scanning
- **Docker Build**: Image building and caching

#### Release Pipeline

- **Tag-based**: Triggers on version tags
- **Image Publishing**: GitHub Container Registry
- **Release Creation**: Automated GitHub releases

## 📈 Scalability Considerations

### Horizontal Scaling

- **API Service**: Stateless design, easy to scale
- **Load Balancing**: Multiple API instances behind load balancer
- **Database**: External MLflow backend for production

### Performance Optimization

- **Model Caching**: Models loaded once on startup
- **Response Caching**: Prediction results cached
- **Async Processing**: FastAPI async endpoints
- **Connection Pooling**: Efficient database connections

## 🚀 Deployment Options

### 1. Local Development

- **Docker Compose**: All services locally
- **Virtual Environment**: Python dependencies
- **Hot Reload**: Development server with auto-reload

### 2. Cloud Deployment

- **Render**: Free tier hosting
- **Heroku**: Container-based deployment
- **AWS/GCP**: Production-grade infrastructure

### 3. Kubernetes

- **Helm Charts**: Application packaging
- **Ingress**: Load balancing and routing
- **HPA**: Horizontal Pod Autoscaling

## 🔧 Configuration Management

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=stock_price_prediction

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Configuration Files

- **`configs/config.yaml`**: Main application configuration
- **`configs/model_config.yaml`**: Model-specific parameters
- **`configs/feature_config.yaml`**: Feature engineering settings

## 📋 System Requirements

### Minimum Requirements

- **CPU**: 2 cores
- **Memory**: 4GB RAM
- **Storage**: 10GB disk space
- **Python**: 3.10+

### Recommended Requirements

- **CPU**: 4+ cores
- **Memory**: 8GB+ RAM
- **Storage**: 50GB+ disk space
- **GPU**: NVIDIA GPU for LSTM training (optional)

## 🔮 Future Enhancements

### Planned Features

- **Real-time Streaming**: WebSocket support for live data
- **Advanced Monitoring**: Prometheus metrics, Grafana dashboards
- **Model Drift Detection**: Automated drift detection and retraining
- **A/B Testing**: Model comparison in production
- **Multi-tenancy**: Support for multiple organizations

### Technology Upgrades

- **Async MLflow**: Non-blocking MLflow operations
- **Vector Database**: Efficient feature storage and retrieval
- **Model Serving**: TorchServe or TensorFlow Serving integration
- **Event Streaming**: Kafka integration for real-time processing
