"""
Pipeline entry point — config-driven OHLCV fetch.

Usage
-----
# Run with a config file
python run_pipeline.py --config config/daily_fetch.yaml

# Override the end date (e.g. run today's data only)
python run_pipeline.py --config config/daily_fetch.yaml --date 2026-04-10

# Use a different config for intraday data
python run_pipeline.py --config config/intraday_5min.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from auth import get_authenticated_kite
from database import ensure_table, data_exists, save_to_db, log_pipeline_error
from fetcher import fetch_historical_data, _to_datetime, INTERVAL_MAX_DAYS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _with_retry(fn, *args, max_attempts: int = 3, backoff_seconds: int = 2, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff on failure.

    Waits: 2s → 4s → 8s … then re-raises on the final attempt.
    Formula: wait = backoff_seconds * 2^(attempt - 1)
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == max_attempts:
                raise
            wait = backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %ds.",
                attempt, max_attempts, exc, wait,
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# ETL steps
# ---------------------------------------------------------------------------

def fetch(kite, instrument: str, exchange: str, from_dt: datetime,
          to_dt: datetime, interval: str,
          max_attempts: int = 3, backoff_seconds: int = 2) -> tuple:
    """Extract: pull raw OHLCV candles from Kite API (auto-chunked)."""
    chunk     = timedelta(days=INTERVAL_MAX_DAYS[interval] - 1)
    frames    = []
    token     = None
    cur_start = from_dt

    while cur_start <= to_dt:
        cur_end = min(cur_start + chunk, to_dt)
        df, instrument_token = _with_retry(
            fetch_historical_data,
            kite=kite,
            symbol=instrument,
            from_date=cur_start,
            to_date=cur_end,
            interval=interval,
            exchange=exchange,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        if not df.empty:
            frames.append(df)
            token = instrument_token
        cur_start = cur_end + timedelta(days=1)

    if not frames:
        import pandas as pd
        return pd.DataFrame(), token

    import pandas as pd
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined, token


def transform(df):
    """Transform: light cleaning — remove bad rows, ensure correct types."""
    if df.empty:
        return df

    before = len(df)

    # Drop rows where high < low (corrupt data)
    df = df[df["high"] >= df["low"]]

    # Drop rows with zero or negative close price
    df = df[df["close"] > 0]

    # Drop duplicate index entries (safety net)
    df = df[~df.index.duplicated(keep="first")]

    dropped = before - len(df)
    if dropped:
        logger.warning("transform: dropped %d bad row(s).", dropped)

    return df


def load(df, symbol: str, instrument_token: int, exchange: str,
         interval: str, from_dt: datetime, to_dt: datetime) -> int:
    """Load: insert cleaned DataFrame into MySQL, return rows saved."""
    if df.empty:
        return 0
    return save_to_db(df, symbol, instrument_token, exchange, interval, from_dt, to_dt)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run(config: dict, date_override: str | None = None, config_file: str = "") -> None:
    src       = config["data_source"]
    exchange  = src["exchange"].upper()
    interval  = src["interval"]
    instruments = src["instruments"]

    # Resolve date range — --date overrides both start and end to a single day
    if date_override:
        from_dt = _to_datetime(date_override, is_start=True)
        to_dt   = _to_datetime(date_override, is_start=False)
        logger.info("Date override: single day %s", date_override)
    else:
        from_dt = _to_datetime(config["date_range"]["start"], is_start=True)
        to_dt   = _to_datetime(config["date_range"]["end"],   is_start=False)

    logger.info(
        "Pipeline '%s' | %d instruments | %s | %s → %s",
        config["pipeline_name"], len(instruments), interval,
        from_dt.date(), to_dt.date(),
    )

    retry         = config.get("retry", {})
    max_attempts  = int(retry.get("max_attempts",  3))
    backoff_secs  = int(retry.get("backoff_seconds", 2))

    ensure_table()
    kite = get_authenticated_kite()

    total_rows  = 0
    succeeded   = []
    failed      = []

    for instrument in instruments:
        # Skip if already in DB
        if data_exists(instrument, exchange, interval, from_dt, to_dt):
            logger.info("[SKIP]  %s — already in database.", instrument)
            succeeded.append(instrument)
            continue

        try:
            raw_df, token = fetch(
                kite, instrument, exchange, from_dt, to_dt, interval,
                max_attempts=max_attempts, backoff_seconds=backoff_secs,
            )
            clean_df      = transform(raw_df)
            rows          = load(clean_df, instrument, token, exchange, interval, from_dt, to_dt)
            total_rows   += rows
            succeeded.append(instrument)
            logger.info("[OK]    %s — %d rows saved.", instrument, rows)

        except Exception as exc:
            failed.append(instrument)
            logger.error("[FAIL]  %s — %s", instrument, exc, exc_info=False)
            try:
                log_pipeline_error(
                    pipeline_name=config["pipeline_name"],
                    instrument=instrument,
                    exchange=exchange,
                    interval=interval,
                    from_dt=from_dt,
                    to_dt=to_dt,
                    error=exc,
                    attempts=max_attempts,
                    config_file=config_file,
                )
            except Exception as db_exc:
                logger.warning("Could not write to pipeline_errors: %s", db_exc)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info(
        "Pipeline complete | %d succeeded | %d failed | %d total rows saved.",
        len(succeeded), len(failed), total_rows,
    )
    if failed:
        logger.warning("Failed instruments: %s", failed)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Run a config-driven OHLCV fetch pipeline")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to a pipeline YAML config file",
    )
    parser.add_argument(
        "--date", "-d",
        metavar="YYYY-MM-DD",
        help="Override date range — fetch this single day only",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)-12s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run(config, date_override=args.date, config_file=str(config_path))
