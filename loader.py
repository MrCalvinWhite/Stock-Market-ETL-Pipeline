"""
loader.py
---------
Loads transformed market data into a SQLite data warehouse.

Schema (star model):
  fact_daily_prices     — Core OHLCV + all engineered features (fact table)
  dim_tickers           — Ticker dimension (sector, description)
  analytics_returns     — Pre-aggregated return analytics by ticker
  analytics_sector      — Sector-level aggregations
  analytics_signals     — Trading signal summary (RSI, MACD, Bollinger)

PRODUCTION SWAP:
  Replace sqlite3 with DuckDB:
    import duckdb
    conn = duckdb.connect("warehouse/market_warehouse.duckdb")

  Or Snowflake:
    from snowflake.connector.pandas_tools import write_pandas
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

log = logging.getLogger("pipeline.loader")

WAREHOUSE_DIR = Path("data/warehouse")
WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = WAREHOUSE_DIR / "market_warehouse.db"


TICKER_META = {
    "AAPL":  ("Apple Inc.",                "Technology",  "NASDAQ"),
    "MSFT":  ("Microsoft Corporation",     "Technology",  "NASDAQ"),
    "JPM":   ("JPMorgan Chase & Co.",      "Financials",  "NYSE"),
    "GS":    ("The Goldman Sachs Group",   "Financials",  "NYSE"),
    "BAC":   ("Bank of America Corp.",     "Financials",  "NYSE"),
    "MS":    ("Morgan Stanley",            "Financials",  "NYSE"),
    "BLK":   ("BlackRock Inc.",            "Financials",  "NYSE"),
    "SCHW":  ("Charles Schwab Corp.",      "Financials",  "NYSE"),
    "C":     ("Citigroup Inc.",            "Financials",  "NYSE"),
    "WFC":   ("Wells Fargo & Company",     "Financials",  "NYSE"),
}


class WarehouseLoader:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        log.info(f"  Connected to warehouse: {DB_PATH}")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS dim_tickers (
                ticker          TEXT PRIMARY KEY,
                company_name    TEXT,
                sector          TEXT,
                exchange        TEXT,
                loaded_at       TEXT
            );

            CREATE TABLE IF NOT EXISTS fact_daily_prices (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT,
                date            TEXT,
                open            REAL,
                high            REAL,
                low             REAL,
                close           REAL,
                volume          INTEGER,
                dollar_volume   REAL,
                daily_return    REAL,
                daily_range     REAL,
                daily_range_pct REAL,
                gap_open        REAL,
                is_up_day       INTEGER,
                sma_10          REAL,
                sma_20          REAL,
                sma_50          REAL,
                ema_12          REAL,
                ema_26          REAL,
                macd            REAL,
                vol_20d         REAL,
                rsi_14          REAL,
                rsi_signal      TEXT,
                vwap_20d        REAL,
                bb_upper        REAL,
                bb_lower        REAL,
                bb_width        REAL,
                bb_pct_rank     REAL,
                return_zscore   REAL,
                volume_zscore   REAL,
                is_price_anomaly  INTEGER,
                is_volume_anomaly INTEGER,
                is_any_anomaly    INTEGER,
                above_sma20     INTEGER,
                above_sma50     INTEGER,
                pipeline_version TEXT,
                loaded_at       TEXT,
                UNIQUE(ticker, date)
            );

            CREATE TABLE IF NOT EXISTS analytics_returns (
                ticker              TEXT PRIMARY KEY,
                avg_daily_return    REAL,
                return_std          REAL,
                sharpe_ratio        REAL,
                total_return_pct    REAL,
                max_drawdown_pct    REAL,
                annualized_vol      REAL,
                up_days             INTEGER,
                down_days           INTEGER,
                win_rate            REAL,
                avg_dollar_volume   REAL,
                loaded_at           TEXT
            );

            CREATE TABLE IF NOT EXISTS analytics_sector (
                sector              TEXT PRIMARY KEY,
                ticker_count        INTEGER,
                avg_return          REAL,
                avg_volatility      REAL,
                total_dollar_volume REAL,
                loaded_at           TEXT
            );

            CREATE TABLE IF NOT EXISTS analytics_signals (
                ticker          TEXT PRIMARY KEY,
                latest_close    REAL,
                latest_rsi      REAL,
                rsi_signal      TEXT,
                latest_macd     REAL,
                macd_signal     TEXT,
                bb_pct_rank     REAL,
                bb_signal       TEXT,
                anomaly_days    INTEGER,
                overall_signal  TEXT,
                loaded_at       TEXT
            );
        """)
        self.conn.commit()

    # ── Loaders ───────────────────────────────────────────

    def load_fact_prices(self, df: pd.DataFrame):
        cols = [
            "ticker", "date", "open", "high", "low", "close", "volume",
            "dollar_volume", "daily_return", "daily_range", "daily_range_pct",
            "gap_open", "is_up_day", "sma_10", "sma_20", "sma_50",
            "ema_12", "ema_26", "macd", "vol_20d", "rsi_14", "rsi_signal",
            "vwap_20d", "bb_upper", "bb_lower", "bb_width", "bb_pct_rank",
            "return_zscore", "volume_zscore", "is_price_anomaly",
            "is_volume_anomaly", "is_any_anomaly", "above_sma20", "above_sma50",
            "pipeline_version",
        ]
        load_df = df[[c for c in cols if c in df.columns]].copy()
        load_df["date"]      = load_df["date"].astype(str)
        load_df["loaded_at"] = datetime.utcnow().isoformat()

        load_df.to_sql("fact_daily_prices", self.conn,
                       if_exists="replace", index=False)
        log.info(f"  Loaded {len(load_df):,} rows → fact_daily_prices")

    def build_dim_tickers(self, df: pd.DataFrame):
        rows = []
        for ticker in df["ticker"].unique():
            meta = TICKER_META.get(ticker, (ticker, "Unknown", "Unknown"))
            rows.append({
                "ticker":       ticker,
                "company_name": meta[0],
                "sector":       meta[1],
                "exchange":     meta[2],
                "loaded_at":    datetime.utcnow().isoformat(),
            })
        pd.DataFrame(rows).to_sql("dim_tickers", self.conn,
                                  if_exists="replace", index=False)
        log.info(f"  Loaded {len(rows)} rows → dim_tickers")

    def build_analytics_tables(self, df: pd.DataFrame):
        self._build_analytics_returns(df)
        self._build_analytics_sector(df)
        self._build_analytics_signals(df)

    def _build_analytics_returns(self, df: pd.DataFrame):
        rows = []
        for ticker, g in df.groupby("ticker"):
            g = g.dropna(subset=["daily_return"])
            returns = g["daily_return"]
            cum = (1 + returns).cumprod()
            roll_max = cum.cummax()
            drawdowns = (cum - roll_max) / roll_max
            max_dd = drawdowns.min()

            avg_r = returns.mean()
            std_r = returns.std()
            sharpe = (avg_r / std_r * (252 ** 0.5)) if std_r > 0 else 0

            rows.append({
                "ticker":             ticker,
                "avg_daily_return":   round(avg_r, 6),
                "return_std":         round(std_r, 6),
                "sharpe_ratio":       round(sharpe, 4),
                "total_return_pct":   round((cum.iloc[-1] - 1) * 100, 4) if len(cum) > 0 else None,
                "max_drawdown_pct":   round(max_dd * 100, 4),
                "annualized_vol":     round(std_r * (252 ** 0.5) * 100, 4),
                "up_days":            int((returns > 0).sum()),
                "down_days":          int((returns < 0).sum()),
                "win_rate":           round((returns > 0).mean() * 100, 2),
                "avg_dollar_volume":  round(g["dollar_volume"].mean(), 0),
                "loaded_at":          datetime.utcnow().isoformat(),
            })
        pd.DataFrame(rows).to_sql("analytics_returns", self.conn,
                                  if_exists="replace", index=False)
        log.info(f"  Loaded {len(rows)} rows → analytics_returns")

    def _build_analytics_sector(self, df: pd.DataFrame):
        rows = []
        for sector, g in df.groupby("sector"):
            rows.append({
                "sector":              sector,
                "ticker_count":        g["ticker"].nunique(),
                "avg_return":          round(g["daily_return"].mean() * 100, 4),
                "avg_volatility":      round(g["vol_20d"].mean() * 100, 4),
                "total_dollar_volume": round(g["dollar_volume"].sum(), 0),
                "loaded_at":           datetime.utcnow().isoformat(),
            })
        pd.DataFrame(rows).to_sql("analytics_sector", self.conn,
                                  if_exists="replace", index=False)
        log.info(f"  Loaded {len(rows)} rows → analytics_sector")

    def _build_analytics_signals(self, df: pd.DataFrame):
        rows = []
        for ticker, g in df.groupby("ticker"):
            latest = g.sort_values("date").iloc[-1]
            rsi    = latest.get("rsi_14", None)
            macd   = latest.get("macd", None)
            bb     = latest.get("bb_pct_rank", None)

            rsi_sig  = latest.get("rsi_signal", "unknown")
            macd_sig = "bullish" if (macd or 0) > 0 else "bearish"
            bb_sig   = ("overbought" if (bb or 0.5) > 0.8
                        else "oversold" if (bb or 0.5) < 0.2
                        else "neutral")

            bull_signals = sum([
                rsi_sig in ("neutral", "strong"),
                macd_sig == "bullish",
                bb_sig == "neutral",
            ])
            overall = "bullish" if bull_signals >= 2 else "bearish"

            rows.append({
                "ticker":        ticker,
                "latest_close":  round(float(latest["close"]), 4),
                "latest_rsi":    round(float(rsi), 2) if rsi else None,
                "rsi_signal":    str(rsi_sig),
                "latest_macd":   round(float(macd), 4) if macd else None,
                "macd_signal":   macd_sig,
                "bb_pct_rank":   round(float(bb), 4) if bb else None,
                "bb_signal":     bb_sig,
                "anomaly_days":  int(g["is_any_anomaly"].sum()),
                "overall_signal": overall,
                "loaded_at":     datetime.utcnow().isoformat(),
            })
        pd.DataFrame(rows).to_sql("analytics_signals", self.conn,
                                  if_exists="replace", index=False)
        log.info(f"  Loaded {len(rows)} rows → analytics_signals")

    def print_warehouse_summary(self):
        log.info("  ── Warehouse Summary ──────────────────────")
        tables = ["fact_daily_prices", "dim_tickers",
                  "analytics_returns", "analytics_sector", "analytics_signals"]
        for t in tables:
            count = self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            log.info(f"    {t:<28} {count:>6,} rows")

        log.info("  ── Top 5 Tickers by Sharpe Ratio ──────────")
        rows = self.conn.execute("""
            SELECT ticker, sharpe_ratio, total_return_pct, win_rate, annualized_vol
            FROM analytics_returns
            ORDER BY sharpe_ratio DESC LIMIT 5
        """).fetchall()
        for r in rows:
            log.info(f"    {r[0]}  Sharpe={r[1]:.2f}  Return={r[2]:.2f}%  "
                     f"WinRate={r[3]:.1f}%  AnnVol={r[4]:.2f}%")

        log.info("  ── Trading Signals ─────────────────────────")
        sigs = self.conn.execute("""
            SELECT ticker, latest_close, rsi_signal, macd_signal, overall_signal
            FROM analytics_signals ORDER BY ticker
        """).fetchall()
        for s in sigs:
            log.info(f"    {s[0]:<6} ${s[1]:>8.2f}  RSI={s[2]:<10} "
                     f"MACD={s[3]:<8} → {s[4].upper()}")
