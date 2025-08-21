#!/bin/bash

# MLflow startup script
set -e

echo "Starting MLflow server..."

# Create directories if they don't exist
mkdir -p /app/mlruns
mkdir -p /app/mlartifacts

# Set permissions
chmod 755 /app/mlruns
chmod 755 /app/mlartifacts

# Initialize database if it doesn't exist
if [ ! -f /app/mlflow.db ]; then
    echo "Initializing MLflow database..."
    sqlite3 /app/mlflow.db "VACUUM;"
fi

# Start MLflow server
exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --workers 1
