# 📈 Stock Price Prediction MLOps Project

A comprehensive MLOps project for stock price prediction using multiple machine learning models (LSTM, XGBoost, and LightGBM) with a complete pipeline from data ingestion to model deployment and monitoring.

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
  - RESTful API service
  - Real-time predictions
  - Performance monitoring
  - Scalable design

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

## 🚀 Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/stock-prediction-mlops.git](https://github.com/dev7878/Stock-Price-Prediction-MLOps-Project.git
cd stock-prediction-mlops
```

2. **Set up the environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Start the services**

```bash
# Start MLflow server
mlflow server --host 127.0.0.1 --port 5000

# Start FastAPI server
python src/api/app.py

# Start Streamlit dashboard
streamlit run src/frontend/app.py
```

4. **Access the services**

- MLflow UI: http://127.0.0.1:5000
- API Documentation: http://127.0.0.1:8000/docs
- Dashboard: http://localhost:8501

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

## 📈 API Endpoints

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

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 📝 Documentation

Detailed documentation is available in the `docs/` directory:

- [Setup Guide](docs/setup.md)
- [API Documentation](docs/api.md)
- [Model Documentation](docs/models.md)
- [Monitoring Guide](docs/monitoring.md)

## 🛠️ Development

1. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

2. Set up pre-commit hooks:

```bash
pre-commit install
```

3. Follow the [contribution guidelines](CONTRIBUTING.md)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for financial data
- [MLflow](https://mlflow.org/) for experiment tracking
- [FastAPI](https://fastapi.tiangolo.com/) for API development
- [Streamlit](https://streamlit.io/) for dashboard development

## 📧 Contact

- Your Name - [devpatel5578@gmail.com](mailto:devpatel5578@gmail.com)
- Project Link: [https://github.com/dev7878/Stock-Price-Prediction-MLOps-Project](https://github.com/dev7878/Stock-Price-Prediction-MLOps-Project)

