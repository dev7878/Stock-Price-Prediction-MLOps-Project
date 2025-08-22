# GCP Deployment Script for Windows PowerShell
# Stock Price Prediction MLOps Project

Write-Host "🚀 Starting GCP Deployment..." -ForegroundColor Green

# Check if gcloud is installed
try {
    $gcloudVersion = gcloud --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud not found"
    }
    Write-Host "✅ Google Cloud CLI found" -ForegroundColor Green
} catch {
    Write-Host "❌ Google Cloud CLI not found. Please install it first:" -ForegroundColor Red
    Write-Host "   https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

# Check if user is authenticated
try {
    $authStatus = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
    if (-not $authStatus) {
        throw "Not authenticated"
    }
    Write-Host "✅ Authenticated with GCP as: $authStatus" -ForegroundColor Green
} catch {
    Write-Host "🔐 Please authenticate with GCP first:" -ForegroundColor Yellow
    Write-Host "   gcloud auth login" -ForegroundColor Cyan
    Write-Host "   gcloud auth application-default login" -ForegroundColor Cyan
    exit 1
}

# Get project ID from user
$PROJECT_ID = Read-Host "Enter your GCP Project ID"
if (-not $PROJECT_ID) {
    Write-Host "❌ Project ID is required" -ForegroundColor Red
    exit 1
}

$REGION = "us-central1"

Write-Host "📋 Using Project: $PROJECT_ID" -ForegroundColor Cyan
Write-Host "🌍 Using Region: $REGION" -ForegroundColor Cyan

# Set the project
Write-Host "🔧 Setting project..." -ForegroundColor Yellow
gcloud config set project $PROJECT_ID

# Enable required APIs
Write-Host "🔧 Enabling required APIs..." -ForegroundColor Yellow
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable appengine.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy API to Cloud Run
Write-Host "🏗️ Building and deploying API to Cloud Run..." -ForegroundColor Yellow

# Build the container
Write-Host "📦 Building container..." -ForegroundColor Cyan
gcloud builds submit --tag gcr.io/$PROJECT_ID/stock-prediction-api

# Deploy to Cloud Run
Write-Host "🚀 Deploying to Cloud Run..." -ForegroundColor Cyan
gcloud run deploy stock-prediction-api `
    --image gcr.io/$PROJECT_ID/stock-prediction-api `
    --platform managed `
    --region $REGION `
    --allow-unauthenticated `
    --memory 512Mi `
    --cpu 1 `
    --port 8080 `
    --set-env-vars="MLFLOW_TRACKING_URI=sqlite:///mlflow.db" `
    --set-env-vars="API_HOST=0.0.0.0"

# Get the API URL
Write-Host "🔍 Getting API URL..." -ForegroundColor Cyan
$API_URL = gcloud run services describe stock-prediction-api --platform managed --region $REGION --format="value(status.url)"

Write-Host "✅ API deployed successfully!" -ForegroundColor Green
Write-Host "🔗 API URL: $API_URL" -ForegroundColor Green

# Deploy UI to App Engine
Write-Host "🎨 Deploying UI to App Engine..." -ForegroundColor Yellow

# Update the API URL in the Streamlit app
Write-Host "🔧 Updating API URL in Streamlit app..." -ForegroundColor Cyan
$streamlitContent = Get-Content "streamlit_app.py" -Raw
$updatedContent = $streamlitContent -replace "https://stock-prediction-api.onrender.com", $API_URL
Set-Content "streamlit_app.py" $updatedContent

# Deploy to App Engine
Write-Host "🚀 Deploying to App Engine..." -ForegroundColor Cyan
gcloud app deploy app.yaml --quiet

# Get the App Engine URL
Write-Host "🔍 Getting App Engine URL..." -ForegroundColor Cyan
$APP_URL = gcloud app browse --no-launch-browser

Write-Host "✅ UI deployed successfully!" -ForegroundColor Green
Write-Host "🔗 App URL: $APP_URL" -ForegroundColor Green

# Create a summary
Write-Host ""
Write-Host "🎉 Deployment Complete!" -ForegroundColor Green
Write-Host "========================" -ForegroundColor Green
Write-Host "🔧 API (Cloud Run): $API_URL" -ForegroundColor Cyan
Write-Host "🎨 UI (App Engine): $APP_URL" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Test your deployment:" -ForegroundColor Yellow
Write-Host "   - API Health: $API_URL/health" -ForegroundColor White
Write-Host "   - UI Dashboard: $APP_URL" -ForegroundColor White
Write-Host ""
Write-Host "💡 Add these URLs to your resume and portfolio!" -ForegroundColor Green
Write-Host "🚀 Your MLOps project is now live on Google Cloud!" -ForegroundColor Green

# Test API health
Write-Host ""
Write-Host "🧪 Testing API health..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$API_URL/health" -Method Get -TimeoutSec 10
    Write-Host "✅ API Health Check: $($response.status)" -ForegroundColor Green
} catch {
    Write-Host "⚠️ API Health Check failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎯 Your deployment is complete! Share these URLs on your resume!" -ForegroundColor Green
