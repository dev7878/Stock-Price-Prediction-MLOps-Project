import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import mlflow
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class StockPricePredictor:
    def __init__(self, config: dict):
        self.config = config
        self.processed_data_path = Path(config["data"]["processed_data_path"])
        self.models_path = Path(config["model"]["model_path"])
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Set up MLflow
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

    def prepare_data(self, symbol: str):
        """Prepare data for training."""
        # Load processed data
        data_path = Path(self.config["data"]["processed_data_path"]) / f"{symbol}_processed.parquet"
        df = pd.read_parquet(data_path)
        
        # Drop object columns
        df = df.select_dtypes(exclude=['object'])
        
        # Create target variable (next day's closing price)
        df['target'] = df['close'].shift(-1)
        df = df.dropna()
        
        # Select features
        features = [col for col in df.columns if col not in ['target']]
        X = df[features].values
        y = df['target'].values
        
        # Split data into train, validation, and test sets
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        y_test = y[train_size + val_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Return both original and reshaped data
        return X_train, y_train, X_val, y_val, X_test, y_test

    def create_lstm_model(self, input_shape: int) -> Sequential:
        """Create and compile LSTM model."""
        units = self.config["model"]["lstm"]["units"]
        dropout = self.config["model"]["lstm"]["dropout"]
        
        model = Sequential()
        model.add(LSTM(units=units[0], return_sequences=True, input_shape=(1, input_shape)))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units[1], return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mape']
        )
        return model

    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Sequential, Dict]:
        """Train LSTM model."""
        # Reshape data for LSTM (samples, timesteps, features)
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        
        # Create and compile model
        model = self.create_lstm_model(X_train.shape[1])
        
        # Train model
        history = model.fit(
            X_train_lstm, y_train,
            epochs=self.config["model"]["lstm"]["epochs"],
            batch_size=self.config["model"]["lstm"]["batch_size"],
            validation_data=(X_val_lstm, y_val),
            verbose=1
        )
        
        return model, history.history

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model."""
        model = XGBRegressor(
            n_estimators=self.config["model"]["xgboost"]["n_estimators"],
            max_depth=self.config["model"]["xgboost"]["max_depth"],
            learning_rate=self.config["model"]["xgboost"]["learning_rate"],
            eval_metric=["rmse", "mae"]
        )
        
        eval_set = [(X_val, y_val)]
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        return model

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model."""
        model = lgb.LGBMRegressor(
            n_estimators=self.config["model"]["lightgbm"]["n_estimators"],
            max_depth=self.config["model"]["lightgbm"]["max_depth"],
            learning_rate=self.config["model"]["lightgbm"]["learning_rate"],
            metric=["rmse", "mae"]
        )
        
        eval_set = [(X_val, y_val)]
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set
        )
        
        return model

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str) -> Dict[str, float]:
        """Evaluate model performance."""
        if model_name == 'lstm':
            X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            y_pred = model.predict(X_test_reshaped)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred.flatten()))
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

    def train_models(self, symbol: str):
        """Train models for a given symbol."""
        try:
            logging.info(f"Training models for {symbol}")
            model_path = self.models_path / symbol
            model_path.mkdir(parents=True, exist_ok=True)

            # Prepare data
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(symbol)

            # Train LSTM
            lstm_model, history = self.train_lstm(X_train, y_train, X_val, y_val)
            lstm_model.save(str(model_path / 'lstm_model.keras'))

            # Train XGBoost
            xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
            xgb_model.save_model(str(model_path / 'xgboost_model.json'))

            # Train LightGBM
            lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val)
            lgb_model.booster_.save_model(str(model_path / 'lightgbm_model.txt'))

            # Get predictions
            xgb_val_pred = xgb_model.predict(X_val)
            lgb_val_pred = lgb_model.predict(X_val)

            # Log metrics
            with mlflow.start_run(run_name=f"{symbol}_training"):
                # Log LSTM metrics
                for metric, values in history.items():
                    mlflow.log_metric(f"lstm_{metric}", values[-1])

                # Log XGBoost metrics
                xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
                xgb_mae = mean_absolute_error(y_val, xgb_val_pred)
                mlflow.log_metric("xgb_rmse", xgb_rmse)
                mlflow.log_metric("xgb_mae", xgb_mae)

                # Log LightGBM metrics
                lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
                lgb_mae = mean_absolute_error(y_val, lgb_val_pred)
                mlflow.log_metric("lgb_rmse", lgb_rmse)
                mlflow.log_metric("lgb_mae", lgb_mae)

        except Exception as e:
            logging.error(f"Error training models for {symbol}: {str(e)}")
            raise

    def train_all_symbols(self):
        """Train models for all configured symbols."""
        for symbol in self.config["data"]["symbols"]:
            self.train_models(symbol)

def main():
    """Main function to orchestrate model training."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize predictor
        predictor = StockPricePredictor(config)
        
        # Train models for all symbols
        predictor.train_all_symbols()
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 