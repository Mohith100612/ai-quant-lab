"""
Config-driven pipeline runner.

Reads a YAML config file and fetches OHLCV data for every instrument listed,
storing results in the database.  The Python code never changes — only the
YAML config changes when you need different stocks, dates, or intervals.

Usage
-----
    python pipeline.py configs/daily_nifty50_fetch.yaml
    python pipeline.py configs/intraday_5min.yaml
    python pipeline.py configs/daily_nifty50_fetch.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import yaml
import pandas as pd

from auth import get_authenticated_kite
from database import (
    ensure_table, ensure_extended_tables,
    data_exists, find_missing_date_ranges, save_to_db,
    push_to_dlq,
)
from fetcher import fetch_historical_data, _to_datetime, INTERVAL_MAX_DAYS, VALID_INTERVALS
from log_config import configure_logging, get_logger, new_correlation_id, set_correlation_id

configure_logging()
logger = get_logger(__name__)

# Limit concurrent Kite API calls across all worker threads to avoid rate-limit
# errors (Kite enforces ~3 req/s per session on historical data endpoints).
_API_SEMAPHORE = threading.Semaphore(3)


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"pipeline_name", "data_source", "date_range"}


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        sys.exit(f"ERROR: Config file not found: {path}")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg, path)
    return cfg


def _validate(cfg: dict, path: str) -> None:
    missing = REQUIRED_KEYS - set(cfg.keys())
    if missing:
        sys.exit(f"ERROR: Config '{path}' is missing required keys: {missing}")

    src = cfg["data_source"]
    if "instruments" not in src or not src["instruments"]:
        sys.exit(f"ERROR: Config '{path}' must have at least one instrument under data_source.instruments")

    dr = cfg["date_range"]
    for key in ("start", "end", "interval"):
        if key not in dr:
            sys.exit(f"ERROR: Config '{path}' is missing date_range.{key}")

    interval = dr["interval"]
    if interval not in VALID_INTERVALS:
        sys.exit(
            f"ERROR: Invalid interval '{interval}' in '{path}'. "
            f"Valid options: {sorted(VALID_INTERVALS)}"
        )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _fetch_one(
    symbol: str,
    kite,
    exchange: str,
    interval: str,
    from_dt,
    to_dt,
    chunk: timedelta,
    continuous: bool,
    oi: bool,
    max_attempts: int,
    backoff_secs: float,
    correlation_id: str = "",
) -> tuple[str, str, str]:
    """
    Fetch and save one symbol.

    Returns
    -------
    (symbol, status, error_msg) where status is 'fetched' | 'skipped' | 'failed'
    and error_msg is the last exception message on failure, else ''.
    """
    # Inherit the parent pipeline's correlation ID so every worker thread's
    # log lines share the same trace ID.
    if correlation_id:
        set_correlation_id(correlation_id)

    # Find only the sub-ranges not yet in the DB (gap detection)
    gaps = find_missing_date_ranges(symbol, exchange, interval, from_dt, to_dt)

    if not gaps:
        logger.info("Skipping %s — fully covered in DB", symbol, extra={"symbol": symbol, "status": "skipped"})
        return symbol, "skipped", ""

    logger.info(
        "%s — %d missing range(s) to fetch",
        symbol, len(gaps),
        extra={"symbol": symbol, "gaps": len(gaps)},
    )

    last_error = ""
    for gap_from, gap_to in gaps:
        for attempt in range(1, max_attempts + 1):
            try:
                all_frames, token = [], None
                cur = gap_from

                while cur <= gap_to:
                    end = min(cur + chunk, gap_to)
                    # Semaphore caps parallel API requests at 3 to stay within
                    # Kite's rate limit on the historical data endpoint.
                    with _API_SEMAPHORE:
                        df, tok = fetch_historical_data(
                            kite=kite,
                            symbol=symbol,
                            from_date=cur,
                            to_date=end,
                            interval=interval,
                            exchange=exchange,
                            continuous=continuous,
                            oi=oi,
                        )
                    if not df.empty:
                        all_frames.append(df)
                        token = tok
                    cur = end + timedelta(days=1)

                if all_frames:
                    full_df = pd.concat(all_frames)
                    full_df = full_df[~full_df.index.duplicated(keep="first")].sort_index()
                    rows = save_to_db(
                        full_df, symbol, token, exchange,
                        interval, gap_from, gap_to,
                    )
                    logger.info(
                        "%s [%s → %s] — %d rows saved",
                        symbol, gap_from.date(), gap_to.date(), rows,
                        extra={"symbol": symbol, "rows_saved": rows, "status": "ok"},
                    )
                else:
                    logger.warning(
                        "%s [%s → %s] — no data from API",
                        symbol, gap_from.date(), gap_to.date(),
                        extra={"symbol": symbol, "status": "no_data"},
                    )
                break   # gap succeeded — move to next gap

            except Exception as exc:
                last_error = str(exc)
                if attempt < max_attempts:
                    wait = min(backoff_secs * (2 ** (attempt - 1)), 60.0)
                    logger.warning(
                        "%s attempt %d/%d failed: %s — retrying in %.1fs",
                        symbol, attempt, max_attempts, exc, wait,
                        extra={"symbol": symbol, "attempt": attempt, "wait_secs": wait},
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "%s — all %d attempts exhausted: %s",
                        symbol, max_attempts, exc,
                        extra={"symbol": symbol, "attempts": max_attempts, "status": "failed"},
                    )
                    return symbol, "failed", last_error

    return symbol, "fetched", ""


def run_pipeline(cfg: dict, dry_run: bool = False, workers: int = 4) -> dict:
    cid = new_correlation_id()
    set_correlation_id(cid)

    name        = cfg.get("pipeline_name", "unnamed")
    description = cfg.get("description", "")
    src         = cfg["data_source"]
    dr          = cfg["date_range"]

    exchange   = src.get("exchange", "NSE").upper()
    interval   = dr["interval"]
    from_dt    = _to_datetime(str(dr["start"]), is_start=True)
    to_dt      = _to_datetime(str(dr["end"]),   is_start=False)
    symbols    = [s.strip().upper() for s in src["instruments"]]
    continuous = src.get("continuous", False)
    oi         = src.get("oi", False)

    retry_cfg    = cfg.get("retry", {})
    max_attempts = int(retry_cfg.get("max_attempts", 3))
    backoff_secs = float(retry_cfg.get("backoff_seconds", 2))

    effective_workers = min(workers, len(symbols)) if symbols else 1

    logger.info(
        "Pipeline '%s' starting",
        name,
        extra={
            "pipeline": name,
            "exchange": exchange,
            "interval": interval,
            "from":     str(from_dt.date()),
            "to":       str(to_dt.date()),
            "symbols":  len(symbols),
            "workers":  effective_workers,
            "dry_run":  dry_run,
        },
    )

    if dry_run:
        logger.info("dry-run mode — no data fetched or saved", extra={"symbols": symbols})
        return {}

    ensure_table()
    ensure_extended_tables()
    kite  = get_authenticated_kite()
    chunk = timedelta(days=INTERVAL_MAX_DAYS.get(interval, 60) - 1)

    results: dict[str, list]  = {"fetched": [], "skipped": [], "failed": []}
    errors:  dict[str, str]   = {}   # symbol → last error message

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(
                _fetch_one,
                symbol, kite, exchange, interval,
                from_dt, to_dt, chunk,
                continuous, oi, max_attempts, backoff_secs,
                cid,   # propagate correlation_id so worker logs share the trace
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):
            symbol, status, error_msg = future.result()
            results[status].append(symbol)
            if status == "failed":
                errors[symbol] = error_msg

    # ── Persist failed symbols to the dead-letter queue ───────────────────
    if results["failed"]:
        logger.error(
            "Pipeline '%s' — %d symbol(s) failed; writing to dead-letter queue",
            name, len(results["failed"]),
            extra={"failed_symbols": results["failed"]},
        )
        for symbol in results["failed"]:
            try:
                dlq_id = push_to_dlq(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    from_dt=from_dt,
                    to_dt=to_dt,
                    error_msg=errors.get(symbol, ""),
                    pipeline_name=name,
                )
                logger.warning(
                    "DLQ entry created for %s (id=%d)", symbol, dlq_id,
                    extra={"symbol": symbol, "dlq_id": dlq_id},
                )
            except Exception as dlq_exc:
                logger.error(
                    "Failed to write %s to DLQ: %s", symbol, dlq_exc,
                    extra={"symbol": symbol},
                )

    logger.info(
        "Pipeline '%s' complete — fetched=%d skipped=%d failed=%d",
        name,
        len(results["fetched"]),
        len(results["skipped"]),
        len(results["failed"]),
        extra={
            "pipeline":       name,
            "fetched_count":  len(results["fetched"]),
            "skipped_count":  len(results["skipped"]),
            "failed_count":   len(results["failed"]),
        },
    )

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a config-driven OHLCV fetch pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", help="Path to a YAML pipeline config file")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without hitting the API or DB",
    )
    p.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel worker threads (default: 4)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cfg  = load_config(args.config)
    run_pipeline(cfg, dry_run=args.dry_run, workers=args.workers)
