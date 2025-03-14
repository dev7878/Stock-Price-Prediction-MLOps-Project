import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import yaml

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

class StockDataCollector:
    def __init__(self, config: dict):
        self.config = config
        self.symbols = config["data"]["symbols"]
        self.raw_data_path = Path(config["data"]["raw_data_path"])
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

    def fetch_yahoo_finance_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(
                period=f"{self.config['data']['history_length']}d",
                interval=self.config['data']['timeframe']
            )
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Ensure we have all required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            logger.info(f"Successfully fetched Yahoo Finance data for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {str(e)}")
            raise

    def fetch_alpha_vantage_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage."""
        try:
            if not self.config["data_sources"]["alpha_vantage"]["enabled"]:
                return None

            ts = TimeSeries(key=self.config["data_sources"]["alpha_vantage"]["api_key"])
            data, _ = ts.get_daily(symbol=symbol, outputsize='full')
            df = pd.DataFrame(data).T
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            
            logger.info(f"Successfully fetched Alpha Vantage data for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
            return None

    def fetch_market_indicators(self) -> Dict[str, pd.DataFrame]:
        """Fetch market indicators (indices, VIX, etc.)."""
        indicators = {}
        try:
            # Fetch major indices
            for index in ["^GSPC", "^IXIC", "^VIX"]:
                index_data = yf.download(index, 
                    period=f"{self.config['data']['history_length']}d",
                    interval=self.config['data']['timeframe']
                )
                indicators[index] = index_data

            logger.info("Successfully fetched market indicators")
            return indicators
        except Exception as e:
            logger.error(f"Error fetching market indicators: {str(e)}")
            return indicators

    def merge_data_sources(self, symbol: str, yahoo_data: pd.DataFrame, 
                         alpha_vantage_data: pd.DataFrame) -> pd.DataFrame:
        """Merge data from different sources."""
        # Start with Yahoo Finance data as base
        merged_data = yahoo_data.copy()
        
        # Add source identifier columns
        merged_data['volume_yf'] = merged_data['volume']
        merged_data['close_yf'] = merged_data['close']
        
        # If Alpha Vantage data is available, merge relevant columns
        if alpha_vantage_data is not None:
            # Resample to match frequencies if needed
            alpha_vantage_data = alpha_vantage_data.resample(self.config['data']['timeframe']).last()
            
            # Add Alpha Vantage data
            merged_data['volume_av'] = alpha_vantage_data['volume']
            merged_data['close_av'] = alpha_vantage_data['close']
            
            # Calculate consensus values
            merged_data['volume'] = (merged_data['volume_yf'] + merged_data['volume_av']) / 2
            merged_data['close'] = (merged_data['close_yf'] + merged_data['close_av']) / 2
        else:
            # Use Yahoo Finance data when Alpha Vantage is not available
            merged_data['volume_av'] = None
            merged_data['close_av'] = None
            merged_data['volume'] = merged_data['volume_yf']
            merged_data['close'] = merged_data['close_yf']

        return merged_data

    def save_data(self, symbol: str, data: pd.DataFrame):
        """Save collected data to disk."""
        try:
            filename = self.raw_data_path / f"{symbol}_raw.parquet"
            data.to_parquet(filename)
            logger.info(f"Successfully saved data for {symbol} to {filename}")
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {str(e)}")
            raise

    def collect_all_data(self):
        """Collect data for all configured symbols."""
        for symbol in self.symbols:
            try:
                # Fetch data from different sources
                yahoo_data = self.fetch_yahoo_finance_data(symbol)
                alpha_vantage_data = self.fetch_alpha_vantage_data(symbol)
                
                # Merge data from different sources
                merged_data = self.merge_data_sources(symbol, yahoo_data, alpha_vantage_data)
                
                # Save the merged data
                self.save_data(symbol, merged_data)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue

        # Fetch and save market indicators
        market_indicators = self.fetch_market_indicators()
        for indicator_name, indicator_data in market_indicators.items():
            try:
                filename = self.raw_data_path / f"{indicator_name}_indicator.parquet"
                indicator_data.to_parquet(filename)
                logger.info(f"Successfully saved market indicator {indicator_name}")
            except Exception as e:
                logger.error(f"Error saving market indicator {indicator_name}: {str(e)}")

def main():
    """Main function to orchestrate data collection."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize data collector
        collector = StockDataCollector(config)
        
        # Collect all data
        collector.collect_all_data()
        
        logger.info("Data collection completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data collection pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 