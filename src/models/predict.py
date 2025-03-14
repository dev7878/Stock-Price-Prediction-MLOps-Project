import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

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
        
    def prepare_data(self, symbol: str):
        """Prepare data for prediction."""
        # Load processed data
        data_path = self.processed_data_path / f"{symbol}_processed.parquet"
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
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, df.index, features

    def load_models(self, symbol: str):
        """Load trained models for a given symbol."""
        model_path = self.models_path / symbol
        
        # Load LSTM model
        lstm_model = tf.keras.models.load_model(str(model_path / 'lstm_model.keras'))
        
        # Load XGBoost model
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(str(model_path / 'xgboost_model.json'))
        
        # Load LightGBM model
        lgb_model = lgb.Booster(model_file=str(model_path / 'lightgbm_model.txt'))
        
        return lstm_model, xgb_model, lgb_model

    def make_predictions(self, symbol: str, last_n_days: int = 30):
        """Make predictions using all three models."""
        try:
            # Load and prepare data
            X, y_true, dates, features = self.prepare_data(symbol)
            
            # Get last n days of data
            X_recent = X[-last_n_days:]
            y_true_recent = y_true[-last_n_days:]
            dates_recent = dates[-last_n_days:]
            
            # Load models
            lstm_model, xgb_model, lgb_model = self.load_models(symbol)
            
            # Make predictions
            # LSTM predictions
            X_lstm = X_recent.reshape((X_recent.shape[0], 1, X_recent.shape[1]))
            lstm_pred = lstm_model.predict(X_lstm).flatten()
            
            # XGBoost predictions
            xgb_pred = xgb_model.predict(X_recent)
            
            # LightGBM predictions
            lgb_pred = lgb_model.predict(X_recent)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'Date': dates_recent,
                'Actual': y_true_recent,
                'LSTM_Prediction': lstm_pred,
                'XGBoost_Prediction': xgb_pred,
                'LightGBM_Prediction': lgb_pred
            })
            
            # Calculate metrics
            for model_name in ['LSTM', 'XGBoost', 'LightGBM']:
                pred_col = f'{model_name}_Prediction'
                mse = np.mean((results['Actual'] - results[pred_col]) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(results['Actual'] - results[pred_col]))
                mape = np.mean(np.abs((results['Actual'] - results[pred_col]) / results['Actual'])) * 100
                
                logger.info(f"\n{model_name} Metrics for {symbol}:")
                logger.info(f"RMSE: {rmse:.2f}")
                logger.info(f"MAE: {mae:.2f}")
                logger.info(f"MAPE: {mape:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions for {symbol}: {str(e)}")
            raise

def main():
    """Main function to test predictions."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize predictor
        predictor = StockPricePredictor(config)
        
        # Make predictions for all symbols
        for symbol in config["data"]["symbols"]:
            logger.info(f"\n{'='*50}")
            logger.info(f"Predictions for {symbol}")
            logger.info(f"{'='*50}")
            
            results = predictor.make_predictions(symbol)
            
            # Display the last 5 predictions
            logger.info(f"\nLast 5 predictions for {symbol}:")
            logger.info(results.tail().to_string())
            logger.info(f"\n{'='*50}\n")
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 