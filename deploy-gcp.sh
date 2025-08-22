#!/bin/bash

# GCP Deployment Script for Stock Prediction MLOps Project
# This script deploys both the API (Cloud Run) and UI (App Engine)

set -e

echo "🚀 Starting GCP Deployment..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "🔐 Please authenticate with GCP first:"
    echo "   gcloud auth login"
    exit 1
fi

# Set project ID (replace with your project ID)
PROJECT_ID="your-project-id"
REGION="us-central1"

echo "📋 Using Project: $PROJECT_ID"
echo "🌍 Using Region: $REGION"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable appengine.googleapis.com

# Build and deploy API to Cloud Run
echo "🏗️ Building and deploying API to Cloud Run..."

# Build the container
gcloud builds submit --tag gcr.io/$PROJECT_ID/stock-prediction-api

# Deploy to Cloud Run
gcloud run deploy stock-prediction-api \
    --image gcr.io/$PROJECT_ID/stock-prediction-api \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --port 8080 \
    --set-env-vars="MLFLOW_TRACKING_URI=sqlite:///mlflow.db" \
    --set-env-vars="API_HOST=0.0.0.0"

# Get the API URL
API_URL=$(gcloud run services describe stock-prediction-api --platform managed --region $REGION --format="value(status.url)")

echo "✅ API deployed successfully!"
echo "🔗 API URL: $API_URL"

# Deploy UI to App Engine
echo "🎨 Deploying UI to App Engine..."

# Copy the API URL to the Streamlit app
sed -i "s|https://stock-prediction-api.onrender.com|$API_URL|g" streamlit_app.py

# Deploy to App Engine
gcloud app deploy app.yaml --quiet

# Get the App Engine URL
APP_URL=$(gcloud app browse --no-launch-browser)

echo "✅ UI deployed successfully!"
echo "🔗 App URL: $APP_URL"

# Create a summary
echo ""
echo "🎉 Deployment Complete!"
echo "========================"
echo "🔧 API (Cloud Run): $API_URL"
echo "🎨 UI (App Engine): $APP_URL"
echo ""
echo "📊 Test your deployment:"
echo "   - API Health: $API_URL/health"
echo "   - UI Dashboard: $APP_URL"
echo ""
echo "💡 Add these URLs to your resume and portfolio!"
echo "🚀 Your MLOps project is now live on Google Cloud!"
