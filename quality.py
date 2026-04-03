"""
quality.py
----------
Data quality validation layer — acts as a gate between
Extract and Transform stages. Modeled after Great Expectations
check patterns (column existence, nulls, ranges, freshness).
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

log = logging.getLogger("pipeline.quality")


class DataQualityChecker:
    def __init__(self, df: pd.DataFrame, stage: str = "raw"):
        self.df = df
        self.stage = stage
        self.results = []

    def run_all_checks(self) -> dict:
        self._check_schema()
        self._check_nulls()
        self._check_price_integrity()
        self._check_volume_integrity()
        self._check_row_counts()
        self._check_date_freshness()
        self._check_duplicates()

        passed = all(r["passed"] for r in self.results)
        return {
            "stage": self.stage,
            "timestamp": datetime.utcnow().isoformat(),
            "passed": passed,
            "total_checks": len(self.results),
            "passed_checks": sum(1 for r in self.results if r["passed"]),
            "failed_checks": sum(1 for r in self.results if not r["passed"]),
            "checks": self.results,
        }

    def log_report(self, report: dict):
        status = "PASSED ✓" if report["passed"] else "FAILED ✗"
        log.info(f"  QC Report [{self.stage}]: {status}")
        log.info(f"  {report['passed_checks']}/{report['total_checks']} checks passed")
        for check in report["checks"]:
            icon = "✓" if check["passed"] else "✗"
            log.info(f"    [{icon}] {check['name']}: {check['message']}")

    # ── Individual Checks ─────────────────────────────────

    def _check_schema(self):
        required = {"ticker", "date", "open", "high", "low", "close", "volume"}
        missing = required - set(self.df.columns)
        self.results.append({
            "name": "schema_completeness",
            "passed": len(missing) == 0,
            "message": f"All required columns present" if not missing else f"Missing columns: {missing}",
        })

    def _check_nulls(self):
        price_cols = ["open", "high", "low", "close", "volume"]
        null_counts = self.df[price_cols].isnull().sum()
        total_nulls = null_counts.sum()
        self.results.append({
            "name": "null_values",
            "passed": total_nulls == 0,
            "message": f"No nulls found in price/volume columns" if total_nulls == 0
                       else f"Found {total_nulls} nulls: {null_counts[null_counts > 0].to_dict()}",
        })

    def _check_price_integrity(self):
        violations = self.df[
            (self.df["high"] < self.df["low"]) |
            (self.df["close"] < self.df["low"]) |
            (self.df["close"] > self.df["high"]) |
            (self.df["open"] < self.df["low"]) |
            (self.df["open"] > self.df["high"]) |
            (self.df["close"] <= 0)
        ]
        self.results.append({
            "name": "price_integrity",
            "passed": len(violations) == 0,
            "message": f"All OHLC relationships valid across {len(self.df):,} records"
                       if len(violations) == 0
                       else f"Found {len(violations)} OHLC integrity violations",
        })

    def _check_volume_integrity(self):
        bad_vol = self.df[self.df["volume"] <= 0]
        self.results.append({
            "name": "volume_positive",
            "passed": len(bad_vol) == 0,
            "message": "All volume values positive",
            }) if len(bad_vol) == 0 else self.results.append({
            "name": "volume_positive",
            "passed": False,
            "message": f"{len(bad_vol)} records with non-positive volume",
        })

    def _check_row_counts(self):
        counts = self.df.groupby("ticker").size()
        low_count = counts[counts < 20]
        self.results.append({
            "name": "minimum_row_count",
            "passed": len(low_count) == 0,
            "message": f"All tickers have ≥20 trading days of data",
        }) if len(low_count) == 0 else self.results.append({
            "name": "minimum_row_count",
            "passed": False,
            "message": f"Tickers with <20 rows: {low_count.to_dict()}",
        })

    def _check_date_freshness(self):
        latest = pd.to_datetime(self.df["date"]).max()
        cutoff = datetime.today() - timedelta(days=7)
        self.results.append({
            "name": "data_freshness",
            "passed": latest >= cutoff,
            "message": f"Latest data: {latest.date()} (within 7-day freshness window)",
        })

    def _check_duplicates(self):
        dupes = self.df.duplicated(subset=["ticker", "date"]).sum()
        self.results.append({
            "name": "no_duplicates",
            "passed": dupes == 0,
            "message": f"No duplicate ticker/date combinations",
        }) if dupes == 0 else self.results.append({
            "name": "no_duplicates",
            "passed": False,
            "message": f"Found {dupes} duplicate ticker/date records",
        })
