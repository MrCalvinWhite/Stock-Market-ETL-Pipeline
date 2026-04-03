# Market Data ETL Pipeline

> **A production-grade, multi-stage ETL pipeline for equity market data — built for the capital markets space.**

---

## Overview

This pipeline ingests OHLCV market data for a configurable equity universe (defaulting to 10 large-cap US financials and tech stocks), validates it through a data quality gate, enriches it with quantitative financial features, and loads analytics-ready tables into a data warehouse.

Designed to mirror real-world DE workflows at firms like Goldman Sachs, JPMorgan, and BlackRock.

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   EXTRACT   │────▶│  QC / GATE  │────▶│  TRANSFORM  │────▶│    LOAD     │
│             │     │             │     │             │     │             │
│ yfinance /  │     │ 7 automated │     │ Returns,    │     │ SQLite /    │
│ Alpha Vant. │     │ DQ checks   │     │ MAs, RSI,   │     │ DuckDB /    │
│ (simulated) │     │ Schema,     │     │ MACD, VWAP, │     │ Snowflake   │
│             │     │ Nulls, OHLC │     │ Bollinger,  │     │             │
│             │     │ integrity   │     │ Anomalies   │     │ Star Schema │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                                                             │
  data/raw/                                                  data/warehouse/
  (CSV + JSON)             data/staging/                    market_warehouse.db
```

---

## Warehouse Schema

| Table | Type | Description |
|---|---|---|
| `fact_daily_prices` | Fact | OHLCV + 30+ engineered features per ticker/day |
| `dim_tickers` | Dimension | Company metadata, sector, exchange |
| `analytics_returns` | Aggregate | Sharpe, drawdown, win rate per ticker |
| `analytics_sector` | Aggregate | Sector-level return and volume metrics |
| `analytics_signals` | Aggregate | RSI, MACD, Bollinger Band signals per ticker |

---

## Features Engineered

**Returns & Price**
- Daily return, log return, gap-open, daily range %
- Up-day flag, dollar volume

**Trend Indicators**
- SMA (10, 20, 50-day), EMA (12, 26-day)
- MACD (EMA12 − EMA26)

**Momentum**
- RSI (14-day) with signal classification (oversold / weak / neutral / strong / overbought)

**Volatility**
- 20-day rolling annualized volatility
- Bollinger Bands (20-day, 2σ): upper, lower, width, % rank

**Volume**
- 20-day VWAP
- Dollar volume

**Anomaly Detection**
- Z-score on rolling 30-day returns and volume
- Flags statistically unusual trading days (|z| > 2.5σ)

---

## Getting Started

### Requirements
```bash
pip install pandas yfinance duckdb
```

### Run
```bash
python pipeline.py
```

### Configure Universe
Edit `TICKERS` and `LOOKBACK_DAYS` at the top of `pipeline.py`.

---

## Swapping to Production Data Sources

**Yahoo Finance (yfinance):**
In `extractor.py`, replace `_simulate_ticker()` with:
```python
import yfinance as yf
df = yf.download(ticker, period=f"{self.lookback_days}d", auto_adjust=True)
```

**Alpha Vantage:**
```python
from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key="YOUR_API_KEY", output_format='pandas')
df, _ = ts.get_daily(symbol=ticker, outputsize='compact')
```

**Snowflake (instead of SQLite):**
In `loader.py`, replace the sqlite3 connection with:
```python
from snowflake.connector.pandas_tools import write_pandas
import snowflake.connector
conn = snowflake.connector.connect(user=..., password=..., account=...)
write_pandas(conn, df, table_name)
```

---

## Project Structure

```
market_etl_pipeline/
├── pipeline.py          # Orchestrator — runs all stages end-to-end
├── extractor.py         # Stage 1: Data extraction (swap API here)
├── quality.py           # Stage 2: Data quality gate (7 checks)
├── transformer.py       # Stage 3: Feature engineering & transformation
├── loader.py            # Stage 4: Warehouse load (star schema)
├── data/
│   ├── raw/             # Raw extract files (CSV + JSON)
│   ├── staging/         # Transformed staging files
│   └── warehouse/       # SQLite warehouse DB
└── logs/                # Timestamped pipeline run logs
```

---

## Skills Demonstrated

- End-to-end ETL pipeline design and orchestration
- Data quality framework (Great Expectations-style checks)
- Financial feature engineering (RSI, MACD, Bollinger, VWAP, Sharpe)
- Star schema data warehouse modeling
- Anomaly detection using rolling z-scores
- Modular, production-swap-ready architecture
- Logging, error handling, audit trail (raw file persistence)
