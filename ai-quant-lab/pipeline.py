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
from database import ensure_table, data_exists, find_missing_date_ranges, save_to_db
from fetcher import fetch_historical_data, _to_datetime, INTERVAL_MAX_DAYS, VALID_INTERVALS

_print_lock = threading.Lock()


def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


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
) -> tuple[str, str]:
    """
    Fetch and save one symbol. Returns (symbol, status) where status is one of
    'fetched', 'skipped', or 'failed'.
    """
    # Find only the sub-ranges not yet in the DB (gap detection)
    gaps = find_missing_date_ranges(symbol, exchange, interval, from_dt, to_dt)

    if not gaps:
        _safe_print(f"  [SKIP]  {symbol} — fully covered in DB")
        return symbol, "skipped"

    _safe_print(f"  [GAP]   {symbol} — {len(gaps)} missing range(s) to fetch")

    for gap_from, gap_to in gaps:
        for attempt in range(1, max_attempts + 1):
            try:
                all_frames, token = [], None
                cur = gap_from

                while cur <= gap_to:
                    end = min(cur + chunk, gap_to)
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
                    _safe_print(
                        f"  [OK]    {symbol} "
                        f"[{gap_from.date()} → {gap_to.date()}] — {rows} rows saved"
                    )
                else:
                    _safe_print(
                        f"  [WARN]  {symbol} "
                        f"[{gap_from.date()} → {gap_to.date()}] — no data from API"
                    )
                break   # gap succeeded — move to next gap

            except Exception as exc:
                if attempt < max_attempts:
                    wait = min(backoff_secs * (2 ** (attempt - 1)), 60.0)
                    _safe_print(
                        f"  [RETRY] {symbol} (attempt {attempt}/{max_attempts}): "
                        f"{exc} — retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)
                else:
                    _safe_print(f"  [FAIL]  {symbol} — {exc}")
                    return symbol, "failed"

    return symbol, "fetched"


def run_pipeline(cfg: dict, dry_run: bool = False, workers: int = 4) -> dict:
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

    print(f"\n{'='*60}")
    print(f"Pipeline : {name}")
    if description:
        print(f"           {description}")
    print(f"Exchange : {exchange}  |  Interval: {interval}")
    print(f"Range    : {from_dt.date()} → {to_dt.date()}")
    print(f"Symbols  : {len(symbols)} instruments  |  Workers: {effective_workers}")
    print(f"{'='*60}")

    if dry_run:
        print("[dry-run] Symbols that would be fetched:")
        for s in symbols:
            print(f"  - {s}")
        print("[dry-run] No data fetched or saved.")
        return {}

    ensure_table()
    kite  = get_authenticated_kite()
    chunk = timedelta(days=INTERVAL_MAX_DAYS.get(interval, 60) - 1)

    results: dict[str, list] = {"fetched": [], "skipped": [], "failed": []}

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(
                _fetch_one,
                symbol, kite, exchange, interval,
                from_dt, to_dt, chunk,
                continuous, oi, max_attempts, backoff_secs,
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):
            symbol, status = future.result()
            results[status].append(symbol)

    # Summary
    print(f"\n{'='*60}")
    print(f"Done: {len(results['fetched'])} fetched, "
          f"{len(results['skipped'])} skipped, "
          f"{len(results['failed'])} failed")
    if results["failed"]:
        print(f"Failed symbols: {results['failed']}")
    print(f"{'='*60}\n")

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
