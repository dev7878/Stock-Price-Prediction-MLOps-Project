# 🚀 Cloud Deployment Guide

## 🌟 **Deploy to Render (Free Tier)**

### **Step 1: Prepare Your Repository**

1. **Push all changes to GitHub main branch**

   ```bash
   git add .
   git commit -m "feat: production-ready deployment setup"
   git push origin main
   ```

2. **Ensure these files are in your repo:**
   - ✅ `render.yaml` - Render configuration
   - ✅ `requirements.txt` - Production dependencies
   - ✅ `requirements-ui.txt` - UI dependencies
   - ✅ `.dockerignore` - Docker optimization

### **Step 2: Deploy to Render**

1. **Go to [Render.com](https://render.com)**

   - Sign up with GitHub account
   - Click "New +" → "Blueprint"

2. **Connect Your Repository**

   - Select your GitHub repo
   - Render will auto-detect `render.yaml`
   - Click "Connect"

3. **Configure Services**

   - **API Service**: `stock-prediction-api`
   - **UI Service**: `stock-prediction-ui`
   - **MLflow Service**: `stock-prediction-mlflow` (optional)

4. **Set Environment Variables**

   - **Required**: None (all set in render.yaml)
   - **Optional**: Add your API keys for live data
     ```
     ALPHA_VANTAGE_API_KEY=your_key_here
     POLYGON_API_KEY=your_key_here
     ```

5. **Deploy**
   - Click "Apply"
   - Wait 5-10 minutes for build
   - Services will auto-deploy from main branch

### **Step 3: Get Your URLs**

After deployment, you'll get:

- **API**: `https://stock-prediction-api.onrender.com`
- **Dashboard**: `https://stock-prediction-ui.onrender.com`
- **MLflow**: `https://stock-prediction-mlflow.onrender.com`

## 📱 **Resume Integration**

### **Option 1: Direct Links**

```
🔗 Live Demo: https://stock-prediction-ui.onrender.com
📊 API Docs: https://stock-prediction-api.onrender.com/docs
```

### **Option 2: QR Code**

- Generate QR codes for mobile access
- Add to resume as visual element

### **Option 3: Portfolio Section**

```
🌐 Portfolio Projects
Stock Price Prediction MLOps Platform
• Live Demo: [Link]
• Tech Stack: FastAPI, Streamlit, MLflow, Docker
• Features: Real-time predictions, ML model tracking
```

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Build Fails**: Check requirements.txt compatibility
2. **Service Unhealthy**: Verify health check endpoints
3. **CORS Errors**: Check API origins in FastAPI

### **Debug Commands**

```bash
# Check service logs
curl https://your-api.onrender.com/health

# Test API endpoints
curl https://your-api.onrender.com/version
```

## 🚀 **Auto-Deployment**

- **Every push to main** triggers auto-deploy
- **Zero-downtime updates**
- **Rollback available** in Render dashboard

## 💰 **Cost Management**

- **Free Tier**: 750 hours/month
- **Services sleep** after 15 minutes of inactivity
- **Wake up** on first request (may take 30 seconds)

## 🔒 **Security Notes**

- **API keys**: Set as environment variables
- **CORS**: Configured for public access
- **Rate limiting**: Consider adding for production use

## 📈 **Monitoring**

- **Health checks**: Automatic monitoring
- **Logs**: Available in Render dashboard
- **Performance**: Built-in metrics

---

**🎯 Your project will be live and accessible to anyone with the URL!**
