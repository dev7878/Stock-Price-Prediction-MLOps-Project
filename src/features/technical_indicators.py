import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import ta
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

class TechnicalIndicatorCalculator:
    def __init__(self, config: dict):
        self.config = config
        self.raw_data_path = Path(config["data"]["raw_data_path"])
        self.processed_data_path = Path(config["data"]["processed_data_path"])
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Simple and Exponential Moving Averages."""
        df = data.copy()
        
        # Calculate SMAs
        for period in self.config["features"]["technical_indicators"]["sma"]["periods"]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
            
        # Calculate EMAs
        for period in self.config["features"]["technical_indicators"]["ema"]["periods"]:
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
            
        return df

    def calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        df = data.copy()
        period = self.config["features"]["technical_indicators"]["rsi"]["period"]
        df['rsi'] = ta.momentum.rsi(df['close'], window=period)
        return df

    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator."""
        df = data.copy()
        macd_config = self.config["features"]["technical_indicators"]["macd"]
        
        macd = ta.trend.MACD(
            close=df['close'],
            window_slow=macd_config["slow_period"],
            window_fast=macd_config["fast_period"],
            window_sign=macd_config["signal_period"]
        )
        
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        return df

    def calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df = data.copy()
        bb_config = self.config["features"]["technical_indicators"]["bollinger_bands"]
        
        indicator_bb = ta.volatility.BollingerBands(
            close=df["close"], 
            window=bb_config["period"], 
            window_dev=bb_config["std_dev"]
        )
        
        df['bb_high'] = indicator_bb.bollinger_hband()
        df['bb_mid'] = indicator_bb.bollinger_mavg()
        df['bb_low'] = indicator_bb.bollinger_lband()
        df['bb_width'] = indicator_bb.bollinger_wband()
        
        return df

    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        df = data.copy()
        
        # Volume Weighted Average Price (VWAP)
        df['vwap'] = ta.volume.volume_weighted_average_price(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )
        
        # On-Balance Volume (OBV)
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volume Price Trend (VPT)
        df['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])
        
        return df

    def calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators."""
        df = data.copy()
        
        # Average Directional Index (ADX)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        
        # Commodity Channel Index (CCI)
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        # Trix indicator
        df['trix'] = ta.trend.trix(df['close'])
        
        return df

    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        df = data.copy()
        
        # Stochastic RSI
        df['stoch_rsi'] = ta.momentum.stochrsi(df['close'])
        
        # Rate of Change (ROC)
        df['roc'] = ta.momentum.roc(df['close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        return df

    def calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators."""
        df = data.copy()
        
        # Average True Range (ATR)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['keltner_high'] = keltner.keltner_channel_hband()
        df['keltner_mid'] = keltner.keltner_channel_mband()
        df['keltner_low'] = keltner.keltner_channel_lband()
        
        return df

    def process_stock_data(self, symbol: str):
        """Process data for a single stock symbol."""
        try:
            # Load raw data
            raw_data = pd.read_parquet(self.raw_data_path / f"{symbol}_raw.parquet")
            
            # Calculate all technical indicators
            df = raw_data.copy()
            df = self.calculate_moving_averages(df)
            df = self.calculate_rsi(df)
            df = self.calculate_macd(df)
            df = self.calculate_bollinger_bands(df)
            df = self.calculate_volume_indicators(df)
            df = self.calculate_trend_indicators(df)
            df = self.calculate_momentum_indicators(df)
            df = self.calculate_volatility_indicators(df)
            
            # Forward fill NaN values using ffill() instead of deprecated fillna(method='ffill')
            df = df.ffill()
            
            # Save processed data
            output_file = self.processed_data_path / f"{symbol}_processed.parquet"
            df.to_parquet(output_file)
            logger.info(f"Successfully processed technical indicators for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing technical indicators for {symbol}: {str(e)}")
            raise

    def process_all_stocks(self):
        """Process technical indicators for all configured stocks."""
        for symbol in self.config["data"]["symbols"]:
            self.process_stock_data(symbol)

def main():
    """Main function to orchestrate technical indicator calculation."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize calculator
        calculator = TechnicalIndicatorCalculator(config)
        
        # Process all stocks
        calculator.process_all_stocks()
        
        logger.info("Technical indicator calculation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in technical indicator calculation pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 