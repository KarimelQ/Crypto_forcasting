import requests
import pandas as pd
import time
import numpy as np


def get_kraken_data(symbol_pair, interval=1440, start_time=None):
    """
    Fetch OHLCV data from Kraken API
    interval in minutes: 1 5 15 30 60 240 1440 10080 21600
    """
    endpoint = "https://api.kraken.com/0/public/OHLC"

    if not start_time:
        start_time = int(time.time() - 365 * 24 * 60 * 60)  # Default to 1 year ago

    params = {"pair": symbol_pair, "interval": interval, "since": start_time}

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()

        # Extract the OHLCV data
        ohlcv_data = next(iter(data["result"].values()))

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv_data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "vwap",
                "volume",
                "count",
            ],
        )

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        # Convert string values to float
        for col in ["open", "high", "low", "close", "vwap", "volume"]:
            df[col] = df[col].astype(float)

        datetime_format: str = "%Y-%m-%d %H:%M",

        metadata = {
            "symbol": symbol_pair,
            "interval": interval,
            "url": endpoint,
            "datetime_start": start_time,
            "datetime_format": datetime_format,
        }

        return df, metadata
    else:
        print(f"Error fetching data: {response.status_code}")
        return None


def process_crypto_data(df):
    """Process and engineer features from raw OHLCV data"""

    # Add technical indicators
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"]).diff()

    # Add moving averages
    df["ma7"] = df["close"].rolling(window=7).mean()
    df["ma21"] = df["close"].rolling(window=21).mean()

    # Add volatility measure
    df["volatility"] = df["returns"].rolling(window=30).std()

    # Trading volume features :  trading interest
    df["volume_ma7"] = df["volume"].rolling(window=7).mean()
    df["volume_ma21"] = df["volume"].rolling(window=21).mean()

    target = "close"
    selected_features = ["timestamp", "volume", "volume_ma7", "log_returns", "ma7"]
    processed_df = df[selected_features + [target]].fillna(method='bfill')
    return processed_df