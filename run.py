"""
Unified entry point for ai-quant-lab.

Dispatches to one of three modes:

  server    Start the FastAPI HTTP server via uvicorn
  pipeline  Run a config-driven YAML batch-fetch pipeline
  fetch     Fetch historical data for a single symbol (CLI)

Usage
-----
    python run.py server [--host HOST] [--port PORT] [--reload]
    python run.py pipeline configs/daily_nifty50.yaml [--dry-run] [--workers N]
    python run.py fetch --symbol RELIANCE --from 2025-01-01 --to 2025-03-31 \\
                        [--interval day] [--exchange NSE] [--output FILE.csv]
                        [--oi] [--continuous]
"""

from __future__ import annotations

import argparse
import sys

from log_config import configure_logging

configure_logging()


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _run_server(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        sys.exit("ERROR: uvicorn is not installed.  Run: pip install uvicorn")
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,   # let our JSON handler own all log output
    )


def _run_pipeline(args: argparse.Namespace) -> None:
    from pipeline import load_config, run_pipeline
    cfg = load_config(args.config)
    run_pipeline(cfg, dry_run=args.dry_run, workers=args.workers)


def _run_fetch(args: argparse.Namespace) -> None:
    # Reconstruct sys.argv so main.py's argparse sees the right flags
    argv = [
        "fetch",
        "--symbol",   args.symbol,
        "--from",     args.from_date,
        "--to",       args.to_date,
        "--interval", args.interval,
        "--exchange", args.exchange,
    ]
    if args.output:
        argv += ["--output", args.output]
    if args.oi:
        argv.append("--oi")
    if args.continuous:
        argv.append("--continuous")
    sys.argv = argv

    from main import main as fetch_main
    fetch_main()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Unified ai-quant-lab entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── server ────────────────────────────────────────────────────────────────
    sv = sub.add_parser(
        "server",
        help="Start the FastAPI HTTP server",
        description="Launch uvicorn serving api:app",
    )
    sv.add_argument("--host",   default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    sv.add_argument("--port",   type=int, default=8000, help="Bind port (default: 8000)")
    sv.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")

    # ── pipeline ──────────────────────────────────────────────────────────────
    pl = sub.add_parser(
        "pipeline",
        help="Run a config-driven YAML batch pipeline",
        description="Read a YAML config and fetch OHLCV data for all listed instruments",
    )
    pl.add_argument("config", help="Path to a YAML pipeline config file")
    pl.add_argument("--dry-run", action="store_true",
                    help="Print what would be fetched without hitting the API or DB")
    pl.add_argument("--workers", "-w", type=int, default=4, metavar="N",
                    help="Parallel worker threads (default: 4)")

    # ── fetch ─────────────────────────────────────────────────────────────────
    fe = sub.add_parser(
        "fetch",
        help="Fetch historical data for a single symbol",
        description="Single-symbol OHLCV fetch — checks DB first, calls Kite API only when needed",
    )
    fe.add_argument("--symbol",     "-s", required=True,
                    help='Trading symbol, e.g. RELIANCE, "NIFTY 50"')
    fe.add_argument("--from",       dest="from_date", required=True, metavar="YYYY-MM-DD",
                    help="Start date (inclusive)")
    fe.add_argument("--to",         dest="to_date",   required=True, metavar="YYYY-MM-DD",
                    help="End date (inclusive)")
    fe.add_argument("--interval",   "-i", default="day",
                    help="Candle interval: minute/5minute/15minute/day/… (default: day)")
    fe.add_argument("--exchange",   "-e", default="NSE",
                    help="Exchange: NSE, BSE, NFO, MCX (default: NSE)")
    fe.add_argument("--output",     "-o", metavar="FILE.csv",
                    help="Save results to a CSV file (optional)")
    fe.add_argument("--oi",         action="store_true",
                    help="Include open-interest column (F&O)")
    fe.add_argument("--continuous", action="store_true",
                    help="Fetch continuous futures/options data")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == "server":
        _run_server(args)
    elif args.command == "pipeline":
        _run_pipeline(args)
    elif args.command == "fetch":
        _run_fetch(args)
