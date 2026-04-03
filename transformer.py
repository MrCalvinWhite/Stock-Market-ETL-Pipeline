"""
transformer.py
--------------
Multi-stage transformation layer.

Applies:
  - Data type casting and normalization
  - Financial feature engineering (returns, moving averages,
    volatility, RSI, VWAP, Bollinger Bands)
  - Sector-level aggregations
  - Anomaly / outlier flagging
"""

import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

log = logging.getLogger("pipeline.transformer")


class MarketDataTransformer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.staging_dir = Path("data/staging")
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def transform(self) -> pd.DataFrame:
        log.info("  Casting data types...")
        df = self._cast_types(self.df)

        log.info("  Engineering price features...")
        df = self._add_price_features(df)

        log.info("  Computing moving averages & volatility...")
        df = self._add_moving_averages(df)

        log.info("  Calculating RSI (14-day)...")
        df = self._add_rsi(df)

        log.info("  Adding VWAP and Bollinger Bands...")
        df = self._add_vwap_and_bands(df)

        log.info("  Flagging anomalies...")
        df = self._flag_anomalies(df)

        log.info("  Adding metadata columns...")
        df["pipeline_version"] = "1.0.0"
        df["transformed_at"] = datetime.utcnow().isoformat()

        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── Type Casting ──────────────────────────────────────

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"]   = pd.to_datetime(df["date"])
        df["open"]   = df["open"].astype(float)
        df["high"]   = df["high"].astype(float)
        df["low"]    = df["low"].astype(float)
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(int)
        return df

    # ── Price Features ────────────────────────────────────

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ticker", "date"])

        df["daily_return"]     = df.groupby("ticker")["close"].pct_change()
        df["log_return"]       = df.groupby("ticker")["close"].transform(
                                    lambda x: x.apply(lambda p: 0).where(x.shift(1).isna(),
                                    other=(x / x.shift(1)).apply(lambda r: __import__("math").log(r) if r > 0 else 0)))
        df["daily_range"]      = df["high"] - df["low"]
        df["daily_range_pct"]  = df["daily_range"] / df["close"]
        df["dollar_volume"]    = df["close"] * df["volume"]
        df["gap_open"]         = df.groupby("ticker").apply(
                                    lambda g: g["open"] - g["close"].shift(1)
                                 ).reset_index(level=0, drop=True)
        df["is_up_day"]        = (df["close"] > df["open"]).astype(int)
        return df

    # ── Moving Averages & Volatility ──────────────────────

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        for ticker, group in df.groupby("ticker"):
            idx = group.index
            df.loc[idx, "sma_10"]   = group["close"].rolling(10).mean()
            df.loc[idx, "sma_20"]   = group["close"].rolling(20).mean()
            df.loc[idx, "sma_50"]   = group["close"].rolling(50, min_periods=30).mean()
            df.loc[idx, "ema_12"]   = group["close"].ewm(span=12, adjust=False).mean()
            df.loc[idx, "ema_26"]   = group["close"].ewm(span=26, adjust=False).mean()
            df.loc[idx, "macd"]     = df.loc[idx, "ema_12"] - df.loc[idx, "ema_26"]

            # Rolling 20-day annualized volatility
            df.loc[idx, "vol_20d"]  = (group["daily_return"]
                                       .rolling(20)
                                       .std() * (252 ** 0.5))

        # Price position relative to moving averages
        df["above_sma20"]   = (df["close"] > df["sma_20"]).astype(int)
        df["above_sma50"]   = (df["close"] > df["sma_50"]).astype(int)
        return df

    # ── RSI ───────────────────────────────────────────────

    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        for ticker, group in df.groupby("ticker"):
            idx   = group.index
            delta = group["close"].diff()
            gain  = delta.clip(lower=0)
            loss  = (-delta).clip(lower=0)
            avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
            avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
            rs   = avg_gain / avg_loss.replace(0, float("inf"))
            rsi  = 100 - (100 / (1 + rs))
            df.loc[idx, "rsi_14"] = rsi.round(2)

        df["rsi_signal"] = pd.cut(
            df["rsi_14"],
            bins=[0, 30, 45, 55, 70, 100],
            labels=["oversold", "weak", "neutral", "strong", "overbought"],
        )
        return df

    # ── VWAP and Bollinger Bands ──────────────────────────

    def _add_vwap_and_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        # VWAP (rolling 20-day)
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        for ticker, group in df.groupby("ticker"):
            idx = group.index
            tp_vol = (group["typical_price"] * group["volume"]).rolling(20).sum()
            vol_sum = group["volume"].rolling(20).sum()
            df.loc[idx, "vwap_20d"] = (tp_vol / vol_sum).round(4)

        # Bollinger Bands (20-day, 2σ)
        df["bb_mid"]   = df["sma_20"]
        df["bb_upper"] = df["sma_20"] + 2 * df.groupby("ticker")["close"].transform(
                            lambda x: x.rolling(20).std())
        df["bb_lower"] = df["sma_20"] - 2 * df.groupby("ticker")["close"].transform(
                            lambda x: x.rolling(20).std())
        df["bb_width"]     = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct_rank"]  = ((df["close"] - df["bb_lower"]) /
                              (df["bb_upper"] - df["bb_lower"])).clip(0, 1).round(4)
        return df

    # ── Anomaly Detection ─────────────────────────────────

    def _flag_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag statistically unusual days using z-score on returns and volume."""
        for ticker, group in df.groupby("ticker"):
            idx = group.index

            # Return z-score (rolling 30d)
            mu   = group["daily_return"].rolling(30).mean()
            sigma = group["daily_return"].rolling(30).std()
            df.loc[idx, "return_zscore"] = ((group["daily_return"] - mu) / sigma).round(3)

            # Volume z-score (rolling 30d)
            vol_mu    = group["volume"].rolling(30).mean()
            vol_sigma = group["volume"].rolling(30).std()
            df.loc[idx, "volume_zscore"] = ((group["volume"] - vol_mu) / vol_sigma).round(3)

        df["is_price_anomaly"]  = (df["return_zscore"].abs() > 2.5).astype(int)
        df["is_volume_anomaly"] = (df["volume_zscore"].abs() > 2.5).astype(int)
        df["is_any_anomaly"]    = ((df["is_price_anomaly"] == 1) |
                                   (df["is_volume_anomaly"] == 1)).astype(int)

        anomaly_count = df["is_any_anomaly"].sum()
        log.info(f"    Flagged {anomaly_count} anomalous trading days")
        return df

    def save_staging(self, df: pd.DataFrame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.staging_dir / f"market_staging_{ts}.csv"
        df.to_csv(path, index=False)
        log.info(f"  Staging data saved → {path}")
