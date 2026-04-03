"""
extractor.py
------------
Handles data extraction from market data sources.

PRODUCTION SWAP:
  Replace _simulate_ticker() with yfinance:
    import yfinance as yf
    df = yf.download(ticker, period=f"{self.lookback_days}d", auto_adjust=True)

  Or Alpha Vantage:
    from alpha_vantage.timeseries import TimeSeries
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    df, _ = ts.get_daily(symbol=ticker, outputsize='compact')
"""

import logging
import random
import math
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger("pipeline.extractor")

# Realistic base prices and volatility profiles per ticker
TICKER_PROFILES = {
    "AAPL":  {"base": 185.0,  "vol": 0.015, "drift": 0.0003},
    "MSFT":  {"base": 415.0,  "vol": 0.013, "drift": 0.0004},
    "JPM":   {"base": 198.0,  "vol": 0.016, "drift": 0.0002},
    "GS":    {"base": 480.0,  "vol": 0.018, "drift": 0.0002},
    "BAC":   {"base": 38.0,   "vol": 0.020, "drift": 0.0001},
    "MS":    {"base": 102.0,  "vol": 0.017, "drift": 0.0002},
    "BLK":   {"base": 845.0,  "vol": 0.014, "drift": 0.0003},
    "SCHW":  {"base": 72.0,   "vol": 0.019, "drift": 0.0001},
    "C":     {"base": 63.0,   "vol": 0.021, "drift": 0.0001},
    "WFC":   {"base": 57.0,   "vol": 0.018, "drift": 0.0001},
}

SECTORS = {
    "AAPL": "Technology",   "MSFT": "Technology",
    "JPM":  "Financials",   "GS":   "Financials",
    "BAC":  "Financials",   "MS":   "Financials",
    "BLK":  "Financials",   "SCHW": "Financials",
    "C":    "Financials",   "WFC":  "Financials",
}


class MarketDataExtractor:
    def __init__(self, tickers: list, lookback_days: int = 90):
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.raw_dir = Path("data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def extract(self) -> pd.DataFrame:
        """Extract OHLCV data for all tickers. Returns combined DataFrame."""
        all_frames = []
        for ticker in self.tickers:
            log.info(f"  Fetching {ticker}...")
            df = self._simulate_ticker(ticker)
            all_frames.append(df)
        combined = pd.concat(all_frames, ignore_index=True)
        combined["ingested_at"] = datetime.utcnow().isoformat()
        return combined

    def _simulate_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Simulate realistic OHLCV data via geometric Brownian motion.
        Replace this method body with a real API call in production.
        """
        profile = TICKER_PROFILES.get(ticker, {"base": 100.0, "vol": 0.015, "drift": 0.0002})
        random.seed(hash(ticker) % 9999)  # reproducible per ticker

        trading_days = self._get_trading_days(self.lookback_days)
        price = profile["base"]
        rows = []

        for date in trading_days:
            # GBM price simulation
            shock = random.gauss(0, 1)
            daily_return = profile["drift"] + profile["vol"] * shock
            price = price * math.exp(daily_return)

            # Intraday range
            daily_range = price * random.uniform(0.005, profile["vol"] * 2.5)
            open_p  = round(price * random.uniform(0.998, 1.002), 4)
            high_p  = round(max(open_p, price) + random.uniform(0, daily_range), 4)
            low_p   = round(min(open_p, price) - random.uniform(0, daily_range), 4)
            close_p = round(price, 4)

            # Volume: scale with price level and add noise
            base_vol = int(1_000_000 / (price / 50))
            volume = int(base_vol * random.uniform(0.6, 1.8))

            rows.append({
                "ticker":       ticker,
                "date":         date.strftime("%Y-%m-%d"),
                "open":         open_p,
                "high":         high_p,
                "low":          low_p,
                "close":        close_p,
                "volume":       volume,
                "sector":       SECTORS.get(ticker, "Unknown"),
                "source":       "simulated_gbm",  # swap to "yfinance" or "alpha_vantage"
            })

        return pd.DataFrame(rows)

    def _get_trading_days(self, n: int) -> list:
        """Return last N weekdays (trading days) excluding weekends."""
        days = []
        current = datetime.today()
        while len(days) < n:
            current -= timedelta(days=1)
            if current.weekday() < 5:  # Mon–Fri
                days.append(current)
        return list(reversed(days))

    def save_raw(self, df: pd.DataFrame):
        """Persist raw extract to CSV and JSON for audit trail."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.raw_dir / f"market_raw_{ts}.csv"
        json_path = self.raw_dir / f"market_raw_{ts}.json"
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        log.info(f"  Raw data saved → {csv_path}")
