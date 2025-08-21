# 📈 Stock Price Prediction MLOps Project

A comprehensive MLOps project for stock price prediction using multiple machine learning models (LSTM, XGBoost, and LightGBM) with a complete pipeline from data ingestion to model deployment and monitoring.

[![CI](https://github.com/yourusername/stock-prediction-mlops/workflows/CI%20Pipeline/badge.svg)](https://github.com/yourusername/stock-prediction-mlops/actions)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://hub.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 Features

- **Multi-Model Prediction System**

  - LSTM (Deep Learning)
  - XGBoost (Gradient Boosting)
  - LightGBM (Gradient Boosting)
  - Ensemble predictions for improved accuracy

- **Advanced Feature Engineering**

  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
  - Market indicators integration
  - Sentiment analysis from multiple sources
  - Automated feature selection

- **MLOps Infrastructure**

  - Automated data pipeline with DVC
  - Model versioning and experiment tracking with MLflow
  - Model serving via FastAPI
  - Interactive dashboard using Streamlit
  - Continuous model monitoring and retraining

- **Production-Ready Architecture**
  - RESTful API service with health checks
  - Real-time predictions
  - Performance monitoring
  - Scalable design
  - Docker containerization
  - CI/CD pipeline

## 🏗️ Architecture

```
src/
├── data/           # Data ingestion and processing
├── features/       # Feature engineering
├── models/         # Model training and evaluation
├── api/           # FastAPI service
├── frontend/      # Streamlit dashboard
└── monitoring/    # Model monitoring

configs/           # Configuration files
tests/             # Unit and integration tests
notebooks/         # Jupyter notebooks for analysis
```

## 🌟 **Live Demo (Deployed on Render)**

🎯 **Try it now! Your project is live and accessible to anyone:**

- **📊 Interactive Dashboard**: [https://stock-prediction-ui.onrender.com](https://stock-prediction-ui.onrender.com)
- **🔌 API Documentation**: [https://stock-prediction-api.onrender.com/docs](https://stock-prediction-api.onrender.com/docs)
- **📈 MLflow Tracking**: [https://stock-prediction-mlflow.onrender.com](https://stock-prediction-mlflow.onrender.com)

> **💡 Perfect for resumes and portfolios!** Share these links with recruiters and hiring managers.

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

1. **Clone the repository**

```bash
git clone https://github.com/dev7878/Stock-Price-Prediction-MLOps-Project.git
```

2. **Start all services with Docker Compose**

```bash
docker compose up -d
```

3. **Access the services**

- MLflow UI: http://localhost:5000
- API Documentation: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Health Check: http://localhost:8000/health

### Option 2: Local Development

1. **Set up the environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Start the services**

```bash
# Start MLflow server
mlflow server --host 127.0.0.1 --port 5000

# Start FastAPI server
python src/api/app.py

# Start Streamlit dashboard
streamlit run src/frontend/app.py
```

## 🐳 Docker Services

The project includes three main services:

- **MLflow Server**: Experiment tracking and model registry
- **FastAPI Service**: RESTful API for predictions and model serving
- **Streamlit UI**: Interactive dashboard for analysis and visualization

### Docker Commands

```bash
# Build and start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down

# Rebuild and restart
docker compose up -d --build

# Check service status
docker compose ps
```

## 📊 Model Performance

Current model performance metrics for supported stocks:

| Model    | RMSE (avg) | MAPE (avg) | Directional Accuracy |
| -------- | ---------- | ---------- | -------------------- |
| LSTM     | 32.73      | 16.19%     | 42%                  |
| XGBoost  | 29.97      | 20.40%     | 38%                  |
| LightGBM | 31.28      | 18.75%     | 40%                  |

## 🔧 Configuration

The project uses YAML configuration files for easy customization:

- `configs/config.yaml`: Main configuration file
- `configs/model_config.yaml`: Model-specific parameters
- `configs/feature_config.yaml`: Feature engineering settings

Copy `env.example` to `.env` and customize your environment variables:

```bash
cp env.example .env
# Edit .env with your configuration
```

## 📈 API Endpoints

- `GET /`: API information
- `GET /health`: Health check endpoint
- `GET /version`: API version information
- `GET /symbols`: List available stock symbols
- `POST /predict/{symbol}`: Get price predictions
- `GET /plot/{symbol}`: Get interactive visualizations
- `GET /metrics/{symbol}`: Get model performance metrics

## 🔍 Monitoring

The system includes:

- Model performance monitoring
- Data drift detection
- Automated retraining triggers
- Performance alerts
- Resource utilization tracking
- Health checks for all services

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run linting
ruff check src/ tests/
black --check src/ tests/
mypy src/
```

## 🚀 Cloud Deployment

### Render (Free Tier)

1. **Fork this repository**
2. **Connect to Render**:

   - Go to [render.com](https://render.com)
   - Create new Web Service
   - Connect your GitHub repository
   - Select the `render.yaml` configuration

3. **Environment Variables**:

   - Set `API_URL` for the UI service
   - Configure MLflow tracking URI
   - Add any required API keys

4. **Deploy**:
   - Render will automatically build and deploy both services
   - Access your live demo at the provided URLs

### Environment Variables for Cloud

```bash
# API Service
MLFLOW_TRACKING_URI=https://your-mlflow-service.onrender.com
API_HOST=0.0.0.0
PORT=8000

# UI Service
API_URL=https://your-api-service.onrender.com
PORT=8501
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🛠️ Development

1. **Install development dependencies**:

```bash
pip install -r requirements.txt
pip install ruff black mypy pytest pytest-cov
```

2. **Set up pre-commit hooks**:

```bash
pre-commit install
```

3. **Follow the [contribution guidelines](CONTRIBUTING.md)**

## 📝 Documentation

Detailed documentation is available in the `docs/` directory:

- [Setup Guide](docs/setup.md)
- [API Documentation](docs/api.md)
- [Model Documentation](docs/models.md)
- [Monitoring Guide](docs/monitoring.md)
- [Deployment Guide](docs/DEPLOY.md)
- [Architecture Overview](docs/ARCHITECTURE.md)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for financial data
- [MLflow](https://mlflow.org/) for experiment tracking
- [FastAPI](https://fastapi.tiangolo.com/) for API development
- [Streamlit](https://streamlit.io/) for dashboard development
- [Render](https://render.com/) for free cloud hosting

## 📧 Contact

- Your Name - [devpatel5578@gmail.com](mailto:devpatel5578@gmail.com)
- Project Link: [https://github.com/dev7878/Stock-Price-Prediction-MLOps-Project](https://github.com/dev7878/Stock-Price-Prediction-MLOps-Project)

