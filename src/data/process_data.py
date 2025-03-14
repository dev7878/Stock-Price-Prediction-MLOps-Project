import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

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

def load_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data from specified paths."""
    try:
        train_data = pd.read_csv(Path(config["data"]["raw_data_path"]) / config["data"]["train_data"])
        test_data = pd.read_csv(Path(config["data"]["raw_data_path"]) / config["data"]["test_data"])
        logger.info(f"Successfully loaded train data shape: {train_data.shape}")
        logger.info(f"Successfully loaded test data shape: {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by handling missing values and outliers."""
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle outliers (example using IQR method)
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    
    return df

def preprocess_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess the data including feature engineering and scaling."""
    # Combine train and test for consistent preprocessing
    combined_data = pd.concat([train_data, test_data], axis=0)
    
    # Feature engineering
    if config["features"]["feature_engineering"]["use_encoding"]:
        combined_data = pd.get_dummies(
            combined_data,
            columns=config["features"]["categorical_features"],
            prefix=config["features"]["categorical_features"]
        )
    
    # Split back into train and test
    train_processed = combined_data[:len(train_data)]
    test_processed = combined_data[len(train_data):]
    
    # Split train into train and validation
    train_final, val_data = train_test_split(
        train_processed,
        test_size=config["data"]["validation_split"],
        random_state=config["training"]["random_state"]
    )
    
    return train_final, val_data, test_processed

def save_processed_data(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: dict
) -> None:
    """Save processed datasets to specified paths."""
    processed_path = Path(config["data"]["processed_data_path"])
    processed_path.mkdir(parents=True, exist_ok=True)
    
    train_data.to_csv(processed_path / "train_processed.csv", index=False)
    val_data.to_csv(processed_path / "val_processed.csv", index=False)
    test_data.to_csv(processed_path / "test_processed.csv", index=False)
    
    logger.info("Successfully saved processed datasets")

def main():
    """Main function to orchestrate the data processing pipeline."""
    try:
        # Load configuration
        config = load_config()
        
        # Load raw data
        train_data, test_data = load_data(config)
        
        # Clean data
        train_clean = clean_data(train_data)
        test_clean = clean_data(test_data)
        
        # Preprocess data
        train_processed, val_processed, test_processed = preprocess_data(
            train_clean, test_clean, config
        )
        
        # Save processed data
        save_processed_data(train_processed, val_processed, test_processed, config)
        
        logger.info("Data processing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 