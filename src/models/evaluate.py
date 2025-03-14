import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb

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

class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.processed_data_path = Path(config["data"]["processed_data_path"])
        self.models_path = Path(config["model"]["model_path"])
        
        # Set up MLflow
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(f"{config['mlflow']['experiment_name']}_evaluation")

    def load_models(self, symbol: str) -> Tuple:
        """Load all models for a given symbol."""
        model_path = self.models_path / symbol
        
        # Load LSTM model
        lstm_model = tf.keras.models.load_model(str(model_path / 'lstm_model.keras'))
        
        # Load XGBoost model
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(str(model_path / 'xgboost_model.json'))
        
        # Load LightGBM model
        lgb_model = lgb.Booster(model_file=str(model_path / 'lightgbm_model.txt'))
        
        return lstm_model, xgb_model, lgb_model

    def prepare_evaluation_data(self, symbol: str) -> Tuple:
        """Prepare data for model evaluation."""
        # Load processed data
        data_path = self.processed_data_path / f"{symbol}_processed.parquet"
        df = pd.read_parquet(data_path)
        
        # Drop object columns and handle missing values
        df = df.select_dtypes(exclude=['object'])
        df['target'] = df['close'].shift(-1)
        df = df.dropna()
        
        # Select features
        features = [col for col in df.columns if col not in ['target']]
        X = df[features].values
        y = df['target'].values
        dates = df.index
        
        return X, y, dates, features

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': r2_score(y_true, y_pred),
            'directional_accuracy': np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
        }
        
        # Calculate Sharpe Ratio
        returns_true = np.diff(y_true) / y_true[:-1]
        returns_pred = np.diff(y_pred) / y_pred[:-1]
        metrics['sharpe_ratio'] = np.mean(returns_pred) / np.std(returns_pred) if np.std(returns_pred) != 0 else 0
        
        return metrics

    def plot_predictions(self, dates: pd.DatetimeIndex, y_true: np.ndarray, 
                        predictions: Dict[str, np.ndarray], symbol: str) -> str:
        """Create and save prediction plots."""
        plt.figure(figsize=(15, 8))
        plt.plot(dates, y_true, label='Actual', linewidth=2)
        
        colors = ['red', 'green', 'blue']
        for (model_name, y_pred), color in zip(predictions.items(), colors):
            plt.plot(dates, y_pred, label=f'{model_name} Predictions', 
                    linestyle='--', alpha=0.7, color=color)
        
        plt.title(f'Stock Price Predictions for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = f"reports/figures/{symbol}_predictions.png"
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path

    def evaluate_models(self, symbol: str):
        """Evaluate all models for a given symbol."""
        try:
            # Load data and models
            X, y, dates, features = self.prepare_evaluation_data(symbol)
            lstm_model, xgb_model, lgb_model = self.load_models(symbol)
            
            # Prepare data for LSTM
            X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
            
            # Make predictions
            predictions = {
                'LSTM': lstm_model.predict(X_lstm).flatten(),
                'XGBoost': xgb_model.predict(X),
                'LightGBM': lgb_model.predict(X)
            }
            
            # Calculate metrics for each model
            all_metrics = {}
            for model_name, y_pred in predictions.items():
                metrics = self.calculate_metrics(y, y_pred)
                all_metrics[model_name] = metrics
                
                logger.info(f"\n{model_name} Metrics for {symbol}:")
                for metric_name, value in metrics.items():
                    logger.info(f"{metric_name}: {value:.2f}")
            
            # Create and save plots
            plot_path = self.plot_predictions(dates, y, predictions, symbol)
            
            # Log to MLflow
            with mlflow.start_run(run_name=f"{symbol}_evaluation"):
                # Log metrics
                for model_name, metrics in all_metrics.items():
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{model_name.lower()}_{metric_name}", value)
                
                # Log plot
                mlflow.log_artifact(plot_path)
                
                # Log feature importance for tree-based models
                feature_imp_xgb = pd.DataFrame({
                    'feature': features,
                    'importance': xgb_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save feature importance plot
                plt.figure(figsize=(10, 6))
                sns.barplot(data=feature_imp_xgb.head(10), x='importance', y='feature')
                plt.title(f'Top 10 Feature Importance for {symbol} (XGBoost)')
                plt.tight_layout()
                feature_imp_path = f"reports/figures/{symbol}_feature_importance.png"
                plt.savefig(feature_imp_path)
                plt.close()
                
                mlflow.log_artifact(feature_imp_path)
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating models for {symbol}: {str(e)}")
            raise

    def evaluate_all_symbols(self):
        """Evaluate models for all symbols."""
        results = {}
        for symbol in self.config["data"]["symbols"]:
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating models for {symbol}")
            logger.info(f"{'='*50}")
            
            results[symbol] = self.evaluate_models(symbol)
        
        return results

def main():
    """Main function to run model evaluation."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize evaluator
        evaluator = ModelEvaluator(config)
        
        # Evaluate all models
        results = evaluator.evaluate_all_symbols()
        
        # Save overall results
        Path("reports").mkdir(exist_ok=True)
        with open("reports/evaluation_results.yaml", "w") as f:
            yaml.dump(results, f)
        
        logger.info("\nEvaluation completed successfully")
        logger.info("Results saved to reports/evaluation_results.yaml")
        logger.info("Plots saved to reports/figures/")
        logger.info("Detailed metrics available in MLflow")
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 