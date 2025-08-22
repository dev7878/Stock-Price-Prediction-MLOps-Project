# Stock Price Prediction MLOps Project

<!-- Test deployment trigger - Cloud Build integration active -->

## 🚀 Quick Start

### **Option 1: Google Cloud Platform (Recommended)**
🎯 **Get a professional live demo URL in 15 minutes!**

1. **Set up GCP account** (free, no credit card required)
2. **Deploy to Google Cloud** using our automated script
3. **Get enterprise-grade URLs** for your resume and portfolio

**Why Google Cloud Platform?**
- ✅ **Professional Platform** - Industry-standard cloud services
- ✅ **Generous Free Tier** - 2M requests/month on Cloud Run
- ✅ **Auto-scaling** - Scales to zero when not in use
- ✅ **Enterprise Security** - Google-grade security and compliance
- ✅ **Built-in Monitoring** - Logging, metrics, and alerts

### **Option 2: Local Development**
```bash
# Clone and setup
git clone <your-repo-url>
cd stock-prediction-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-gcp.txt

# Run the dashboard
streamlit run streamlit_app.py
```

## 🏗️ **Architecture Overview**

This project follows a **split architecture** approach for optimal deployment:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │  DAGsHub MLflow │
│  (Streamlit CC) │◄──►│  (Cloud Run)    │◄──►│   (Tracking)    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Services**
- **🎨 Frontend**: Streamlit dashboard (Streamlit Cloud)
- **🔧 Backend**: FastAPI service (Cloud Run/Railway)
- **📊 Tracking**: MLflow experiment tracking (DAGsHub)

## 🎯 **Features**

### **📊 Advanced Analytics**
- **Actual vs Predicted**: Side-by-side comparison charts
- **Residuals Analysis**: Rolling residuals and distribution analysis
- **Drift Detection**: Concept drift visualization with PSI metrics
- **Baseline Comparison**: ML models vs naive baseline

### **🤖 ML Models**
- **LSTM**: Deep learning for time series prediction
- **XGBoost**: Gradient boosting for structured data
- **LightGBM**: Light gradient boosting machine
- **Ensemble**: Combined predictions for better accuracy

### **🎨 Professional UI**
- **Responsive Design**: Works on all devices
- **Interactive Charts**: Plotly-powered visualizations
- **Real-time Updates**: Dynamic data generation
- **Custom Styling**: Professional color scheme

## 🚀 **Deployment Options**

### **1. Google Cloud Platform (Primary)**
- **API**: Cloud Run with FastAPI
- **UI**: App Engine with Streamlit
- **Requirements**: `requirements-gcp.txt`
- **Guide**: [docs/GCP_DEPLOY.md](docs/GCP_DEPLOY.md)
- **Script**: `deploy-gcp.sh` (automated deployment)

### **2. Docker Compose (Local Development)**
```bash
# Start all services
docker-compose up -d

# Access services
# UI: http://localhost:8501
# API: http://localhost:8000
# MLflow: http://localhost:5000
```

### **3. Local Development**
```bash
# Install dependencies
pip install -r requirements-gcp.txt

# Run API
python src/api/app.py

# Run UI
streamlit run streamlit_app.py
```

## 📁 **Project Structure**

```
stock-prediction-mlops/
├── streamlit_app.py          # 🎯 Standalone Streamlit app
├── requirements-gcp.txt      # 📦 GCP deployment dependencies
├── Dockerfile.gcp            # 🐳 GCP Cloud Run container
├── app.yaml                  # ⚙️ GCP App Engine config
├── deploy-gcp.sh            # 🚀 Automated GCP deployment script
├── src/
│   ├── api/                 # 🔧 FastAPI service
│   ├── frontend/            # 🎨 Streamlit UI
│   ├── models/              # 🤖 ML model training
│   └── monitoring/          # 📊 Model monitoring
├── configs/                 # ⚙️ Configuration files
├── docs/                    # 📚 Documentation
│   ├── GCP_DEPLOY.md       # 🚀 GCP deployment guide
│   ├── DEPLOY.md           # ☁️ General deployment guide
│   └── ARCHITECTURE.md     # 🏗️ Architecture overview
└── docker-compose.yml      # 🐳 Local development
```

## 🔧 **Technology Stack**

### **Frontend**
- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive data visualization
- **Pandas**: Data manipulation and analysis

### **Backend**
- **FastAPI**: High-performance API framework
- **Uvicorn**: ASGI server for FastAPI
- **Pydantic**: Data validation and settings

### **Machine Learning**
- **TensorFlow**: Deep learning (LSTM models)
- **XGBoost**: Gradient boosting
- **LightGBM**: Light gradient boosting
- **Scikit-learn**: Machine learning utilities

### **MLOps & Monitoring**
- **MLflow**: Experiment tracking and model registry
- **DAGsHub**: Hosted MLflow (free tier)
- **Docker**: Containerization
- **GitHub Actions**: CI/CD automation

## 📊 **Model Performance**

Current metrics (on sample data):
- **RMSE**: ~$2-5 (varies by stock)
- **MAPE**: ~3-8%
- **Directional Accuracy**: ~65-75%

*Note: These are demonstration metrics using simulated data*

## 🚀 **Getting Started**

### **For Resume/Portfolio (Recommended)**
1. **Deploy to Google Cloud Platform** in 15 minutes
2. **Get professional live URLs** immediately
3. **Share enterprise-grade demo** on resume and LinkedIn

### **For Learning/Development**
1. **Clone repository**
2. **Run locally** with Docker Compose
3. **Experiment** with different models

### **For Production**
1. **Set up external MLflow** (DAGsHub)
2. **Deploy API** to GCP Cloud Run
3. **Deploy UI** to GCP App Engine
4. **Connect real data sources**

## 📚 **Documentation**

- **[🚀 GCP Deployment](docs/GCP_DEPLOY.md)**: Deploy to Google Cloud in 15 minutes
- **[☁️ General Deployment](docs/DEPLOY.md)**: Deployment architecture overview
- **[🏗️ Architecture](docs/ARCHITECTURE.md)**: System design and data flow
- **[📋 Resume Guide](docs/RESUME.md)**: How to showcase this project

## 🤝 **Contributing**

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** locally
5. **Submit** a pull request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **FastAPI** for high-performance APIs
- **MLflow** for experiment tracking

---

## 🎯 **Ready to Deploy?**

**Get your professional live demo URLs in 15 minutes:**

1. **Set up GCP account** (free, no credit card)
2. **Deploy to Google Cloud Platform**
3. **Share enterprise-grade URLs on your resume!**

**Need help?** Check out the [GCP Deployment Guide](docs/GCP_DEPLOY.md) 🚀

---

<div align="center">

**⭐ Star this repo if it helps you get a job! ⭐**

**Built with ❤️ for MLOps Engineers**

</div>

