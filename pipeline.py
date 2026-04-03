"""
============================================================
  Market Data ETL Pipeline
  Author: Calvin White
  Description:
    End-to-end ETL pipeline that ingests OHLCV market data
    for a configurable equity universe, validates and enriches
    it through a multi-stage transformation layer, and loads
    analytics-ready tables into a SQLite data warehouse.

    Designed to swap yfinance / Alpha Vantage as the source
    with a one-line change in extractor.py.
============================================================
"""

import logging
import time
from datetime import datetime
from pathlib import Path

from extractor import MarketDataExtractor
from transformer import MarketDataTransformer
from loader import WarehouseLoader
from quality import DataQualityChecker

# ── Logging Setup ─────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("pipeline.orchestrator")

# ── Config ────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "JPM", "GS", "BAC", "MS", "BLK", "SCHW", "C", "WFC"]
LOOKBACK_DAYS = 90


def run_pipeline():
    start = time.time()
    log.info("=" * 60)
    log.info("Market Data ETL Pipeline — START")
    log.info(f"Universe: {TICKERS}")
    log.info(f"Lookback: {LOOKBACK_DAYS} trading days")
    log.info("=" * 60)

    # ── Stage 1: Extract ──────────────────────────────────
    log.info("[STAGE 1] Extracting raw market data...")
    extractor = MarketDataExtractor(tickers=TICKERS, lookback_days=LOOKBACK_DAYS)
    raw_df = extractor.extract()
    log.info(f"  Extracted {len(raw_df):,} raw records across {raw_df['ticker'].nunique()} tickers")
    extractor.save_raw(raw_df)

    # ── Stage 2: Data Quality Checks ─────────────────────
    log.info("[STAGE 2] Running data quality checks on raw layer...")
    checker = DataQualityChecker(raw_df, stage="raw")
    qc_report = checker.run_all_checks()
    checker.log_report(qc_report)
    if not qc_report["passed"]:
        log.error("  Data quality check FAILED — aborting pipeline.")
        raise RuntimeError("Data quality gate failed at raw stage.")
    log.info("  All data quality checks PASSED ✓")

    # ── Stage 3: Transform ────────────────────────────────
    log.info("[STAGE 3] Applying transformations...")
    transformer = MarketDataTransformer(raw_df)
    staging_df = transformer.transform()
    log.info(f"  Transformed {len(staging_df):,} records with {len(staging_df.columns)} feature columns")
    transformer.save_staging(staging_df)

    # ── Stage 4: Load to Warehouse ────────────────────────
    log.info("[STAGE 4] Loading to SQLite data warehouse...")
    loader = WarehouseLoader()
    loader.load_fact_prices(staging_df)
    loader.build_dim_tickers(staging_df)
    loader.build_analytics_tables(staging_df)
    loader.print_warehouse_summary()

    elapsed = round(time.time() - start, 2)
    log.info("=" * 60)
    log.info(f"Pipeline completed successfully in {elapsed}s")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
