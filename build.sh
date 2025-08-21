#!/bin/bash

# Build script for Render deployment
set -e

echo "🚀 Starting build process..."

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

if [ -f "requirements-ui.txt" ]; then
    pip install -r requirements-ui.txt
fi

# Install additional production dependencies
echo "📦 Installing production dependencies..."
pip install gunicorn

# Verify installation
echo "✅ Verifying installations..."
python -c "import numpy, pandas, fastapi, streamlit, mlflow; print('All dependencies installed successfully!')"

echo "🎉 Build completed successfully!"
