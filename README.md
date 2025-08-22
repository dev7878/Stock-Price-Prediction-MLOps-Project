# 📈 Stock Price Prediction MLOps Project

A comprehensive MLOps project demonstrating end-to-end machine learning operations for stock price prediction using **LSTM**, **XGBoost**, and **LightGBM** models.

## 🚀 **Quick Start - Live Demo**

### **Option 1: Streamlit Cloud (Recommended)**
🎯 **Get a live demo URL in 5 minutes!**

1. **Push code to GitHub** (see deployment guide below)
2. **Deploy to Streamlit Cloud** - [share.streamlit.io](https://share.streamlit.io)
3. **Get live URL** for your resume and portfolio

**Why Streamlit Cloud?**
- ✅ **100% Free** - No credit card required
- ✅ **Auto-deploy** from GitHub
- ✅ **Professional hosting** with HTTPS
- ✅ **Mobile responsive** design
- ✅ **No server maintenance**

### **Option 2: Local Development**
```bash
# Clone and setup
git clone <your-repo-url>
cd stock-prediction-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-streamlit.txt

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

### **1. Streamlit Cloud (Primary)**
- **File**: `streamlit_app.py`
- **Requirements**: `requirements-streamlit.txt`
- **Guide**: [docs/STREAMLIT_DEPLOY.md](docs/STREAMLIT_DEPLOY.md)

### **2. Docker Compose (Local)**
```bash
# Start all services
docker-compose up -d

# Access services
# UI: http://localhost:8501
# API: http://localhost:8000
# MLflow: http://localhost:5000
```

### **3. Cloud Deployment (Advanced)**
- **API**: Google Cloud Run / Railway
- **UI**: Streamlit Cloud / Hugging Face Spaces
- **Tracking**: DAGsHub MLflow / Weights & Biases

## 📁 **Project Structure**

```
stock-prediction-mlops/
├── streamlit_app.py          # 🎯 Standalone Streamlit app
├── requirements-streamlit.txt # 📦 Streamlit dependencies
├── src/
│   ├── api/                 # 🔧 FastAPI service
│   ├── frontend/            # 🎨 Streamlit UI
│   ├── models/              # 🤖 ML model training
│   └── monitoring/          # 📊 Model monitoring
├── configs/                 # ⚙️ Configuration files
├── docs/                    # 📚 Documentation
│   ├── STREAMLIT_DEPLOY.md # 🚀 Streamlit deployment guide
│   ├── DEPLOY.md           # ☁️ Cloud deployment guide
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
1. **Deploy to Streamlit Cloud** in 5 minutes
2. **Get live demo URL** immediately
3. **Share on resume** and LinkedIn

### **For Learning/Development**
1. **Clone repository**
2. **Run locally** with Docker Compose
3. **Experiment** with different models

### **For Production**
1. **Set up external MLflow** (DAGsHub)
2. **Deploy API** to Cloud Run/Railway
3. **Connect real data sources**

## 📚 **Documentation**

- **[🚀 Streamlit Deployment](docs/STREAMLIT_DEPLOY.md)**: Get live demo in 5 minutes
- **[☁️ Cloud Deployment](docs/DEPLOY.md)**: Full cloud architecture setup
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

**Get your live demo URL in 5 minutes:**

1. **Push code to GitHub**
2. **Deploy to Streamlit Cloud**
3. **Share on your resume!**

**Need help?** Check out the [Streamlit Deployment Guide](docs/STREAMLIT_DEPLOY.md) 🚀

---

<div align="center">

**⭐ Star this repo if it helps you get a job! ⭐**

**Built with ❤️ for MLOps Engineers**

</div>

