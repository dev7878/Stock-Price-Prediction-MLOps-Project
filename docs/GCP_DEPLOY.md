# 🚀 GCP Deployment Guide - Complete MLOps Stack

This guide will help you deploy your **Stock Price Prediction MLOps Project** to Google Cloud Platform using **free tier services**. Get a professional, scalable deployment that's perfect for your resume!

## ✨ **Why GCP Free Tier?**

- **🆓 Generous Free Tier** - 2M requests/month on Cloud Run
- **🚀 Professional Platform** - Industry-standard cloud services
- **📱 Auto-scaling** - Scales to zero when not in use
- **🔒 Enterprise Security** - Google-grade security and compliance
- **📊 Monitoring** - Built-in logging and monitoring
- **🌍 Global CDN** - Fast access worldwide

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │  DAGsHub MLflow │
│  (App Engine)   │◄──►│  (Cloud Run)    │◄──►│   (Tracking)    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Services Used**
- **🎨 Frontend**: Google App Engine (Streamlit)
- **🔧 Backend**: Google Cloud Run (FastAPI)
- **📊 Tracking**: DAGsHub MLflow (external, free)
- **🐳 Container**: Google Container Registry

## 📋 **Prerequisites**

### **1. Google Cloud Account**
- [Sign up for GCP](https://cloud.google.com/free) (free $300 credit)
- **No credit card required** for free tier
- **Free tier includes**:
  - Cloud Run: 2M requests/month
  - App Engine: 28 instance hours/day
  - Cloud Build: 120 build-minutes/day

### **2. Install Google Cloud CLI**
```bash
# Windows (PowerShell)
# Download from: https://cloud.google.com/sdk/docs/install

# macOS/Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### **3. Authenticate with GCP**
```bash
gcloud auth login
gcloud auth application-default login
```

## 🚀 **Step-by-Step Deployment**

### **Step 1: Create GCP Project**
```bash
# Create new project (or use existing)
gcloud projects create stock-prediction-mlops --name="Stock Prediction MLOps"

# Set as default project
gcloud config set project stock-prediction-mlops

# Enable billing (required for free tier)
# Go to: https://console.cloud.google.com/billing
```

### **Step 2: Enable Required APIs**
```bash
# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Enable App Engine API
gcloud services enable appengine.googleapis.com

# Enable Cloud Build API
gcloud services enable cloudbuild.googleapis.com

# Enable Container Registry API
gcloud services enable containerregistry.googleapis.com
```

### **Step 3: Deploy API to Cloud Run**
```bash
# Build and deploy the API
gcloud builds submit --tag gcr.io/stock-prediction-mlops/stock-prediction-api

# Deploy to Cloud Run
gcloud run deploy stock-prediction-api \
    --image gcr.io/stock-prediction-mlops/stock-prediction-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --port 8080 \
    --set-env-vars="MLFLOW_TRACKING_URI=sqlite:///mlflow.db" \
    --set-env-vars="API_HOST=0.0.0.0"
```

### **Step 4: Deploy UI to App Engine**
```bash
# Deploy Streamlit app to App Engine
gcloud app deploy app.yaml
```

### **Step 5: Get Your URLs**
```bash
# Get API URL
gcloud run services describe stock-prediction-api \
    --platform managed \
    --region us-central1 \
    --format="value(status.url)"

# Get App Engine URL
gcloud app browse --no-launch-browser
```

## 🔧 **Configuration Files**

### **Dockerfile.gcp**
- Optimized for GCP Cloud Run
- Multi-stage build for efficiency
- Health checks included

### **app.yaml**
- App Engine configuration
- Auto-scaling settings
- Resource limits

### **deploy-gcp.sh**
- Automated deployment script
- Handles both services
- Error checking and validation

## 📊 **Free Tier Limits & Costs**

### **Cloud Run (API)**
- **Free**: 2 million requests/month
- **Memory**: 512MB included
- **CPU**: 1 vCPU included
- **Scaling**: Scales to zero (no cost when idle)

### **App Engine (UI)**
- **Free**: 28 instance hours/day
- **Memory**: 256MB included
- **Storage**: 1GB included
- **Scaling**: Automatic scaling

### **Total Monthly Cost**
- **Free tier**: $0/month
- **Beyond free tier**: ~$5-15/month (very affordable)

## 🎯 **Deployment Commands (Quick Start)**

### **Option 1: Automated Script**
```bash
# Make script executable
chmod +x deploy-gcp.sh

# Edit PROJECT_ID in script
nano deploy-gcp.sh

# Run deployment
./deploy-gcp.sh
```

### **Option 2: Manual Commands**
```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Deploy API
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/stock-prediction-api
gcloud run deploy stock-prediction-api --image gcr.io/YOUR_PROJECT_ID/stock-prediction-api --platform managed --region us-central1 --allow-unauthenticated

# Deploy UI
gcloud app deploy app.yaml
```

## 🔍 **Testing Your Deployment**

### **API Health Check**
```bash
# Test API health
curl https://YOUR_API_URL/health

# Expected response:
# {"status": "ok", "timestamp": "2024-01-01T00:00:00", ...}
```

### **UI Dashboard**
- Open your App Engine URL
- Test all features
- Verify API connectivity

## 📈 **Monitoring & Analytics**

### **Cloud Run Monitoring**
- **Metrics**: Request count, latency, errors
- **Logs**: Application logs and errors
- **Alerts**: Set up notifications for issues

### **App Engine Monitoring**
- **Instances**: Active instance count
- **Performance**: Response times and throughput
- **Costs**: Usage and billing alerts

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Build Failures**
```bash
# Check build logs
gcloud builds log BUILD_ID

# Verify Dockerfile syntax
docker build -f Dockerfile.gcp .
```

#### **Deployment Failures**
```bash
# Check service status
gcloud run services describe stock-prediction-api --region us-central1

# View logs
gcloud logs read --service=stock-prediction-api
```

#### **API Connection Issues**
- Verify CORS settings
- Check environment variables
- Test health endpoint

### **Getting Help**
- **GCP Documentation**: [cloud.google.com/docs](https://cloud.google.com/docs)
- **GCP Community**: [stackoverflow.com/questions/tagged/google-cloud-platform](https://stackoverflow.com/questions/tagged/google-cloud-platform)
- **GCP Support**: Available in console

## 🎉 **Success! Your App is Live**

### **What You've Accomplished**
- ✅ **Professional deployment** on Google Cloud
- ✅ **Scalable architecture** that grows with you
- ✅ **Enterprise-grade security** and reliability
- ✅ **Cost-effective** (free tier + minimal costs)
- ✅ **Resume-worthy** MLOps project

### **Next Steps**
1. **Test all functionality**
2. **Add URLs to your resume**
3. **Share with colleagues**
4. **Monitor performance**
5. **Plan future enhancements**

## 🔮 **Future Enhancements**

### **Advanced Features**
- **Custom Domain**: Add your own domain
- **SSL Certificates**: Automatic HTTPS
- **Load Balancing**: Distribute traffic
- **CDN**: Global content delivery

### **MLOps Features**
- **Model Registry**: Store trained models
- **A/B Testing**: Compare model versions
- **Automated Retraining**: Schedule model updates
- **Performance Monitoring**: Track model drift

---

## 🎯 **Ready to Deploy?**

**Your GCP deployment files are ready!**

1. **Set up GCP account** (free)
2. **Install gcloud CLI**
3. **Run the deployment script**
4. **Get your live URLs!**

**Need help?** Check the troubleshooting section or run the automated deployment script! 🚀

---

**🚀 Deploy to Google Cloud and showcase your MLOps skills professionally!**
