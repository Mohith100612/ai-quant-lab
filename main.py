"""
Zerodha Kite Connect — CLI entry point.

Examples
--------
# Daily candles for RELIANCE over the last month
python main.py --symbol RELIANCE --from 2025-03-01 --to 2025-03-31 --interval day

# 15-minute candles for INFY on BSE, saved to CSV
python main.py --symbol INFY --exchange BSE --from 2025-03-01 --to 2025-03-31 \
               --interval 15minute --output infy_15min.csv

# 5-minute candles for NIFTY 50 index
python main.py --symbol "NIFTY 50" --from 2025-03-24 --to 2025-03-31 \
            --interval 5minute --output nifty_5min.csv
"""

import argparse
import os
import sys

import pandas as pd

from auth import get_authenticated_kite
from fetcher import fetch_historical_data, VALID_INTERVALS, _to_datetime
from database import ensure_table, data_exists, save_to_db


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch historical OHLCV data from Zerodha Kite Connect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help='Trading symbol, e.g. RELIANCE, INFY, "NIFTY 50"',
    )
    parser.add_argument(
        "--from", "-f",
        dest="from_date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Start date (inclusive)",
    )
    parser.add_argument(
        "--to", "-t",
        dest="to_date",
        required=True,
        metavar="YYYY-MM-DD",
        help="End date (inclusive)",
    )
    parser.add_argument(
        "--interval", "-i",
        default="day",
        choices=sorted(VALID_INTERVALS),
        help="Candle interval (default: day)",
    )
    parser.add_argument(
        "--exchange", "-e",
        default="NSE",
        help="Exchange segment: NSE, BSE, NFO, MCX (default: NSE)",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE.csv",
        help="Save results to this CSV file (optional)",
    )
    parser.add_argument(
        "--oi",
        action="store_true",
        help="Include open interest column (useful for F&O instruments)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Fetch continuous data (for futures/options)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    print("[DEBUG] Arguments parsed successfully")

    symbol   = args.symbol
    exchange = args.exchange.upper()
    interval = args.interval

    # Parse dates once so we can reuse them for both DB checks and the API call
    from_dt = _to_datetime(args.from_date, is_start=True)
    to_dt   = _to_datetime(args.to_date,   is_start=False)
    print(f"[DEBUG] Date range: {from_dt} to {to_dt}")

    # Ensure the MySQL table exists
    print("[DEBUG] Initializing database...")
    ensure_table()
    print("[DEBUG] Database table ready")

    if data_exists(symbol, exchange, interval, from_dt, to_dt):
        print("Data is already present in the database.")
        sys.exit(0)
    else:
        print("[DEBUG] Data not found in DB, fetching from Kite API...")
        kite = get_authenticated_kite()
        df, instrument_token = fetch_historical_data(
            kite=kite,
            symbol=symbol,
            from_date=from_dt,
            to_date=to_dt,
            interval=interval,
            exchange=exchange,
            continuous=args.continuous,
            oi=args.oi,
        )
        if not df.empty:
            rows_saved = save_to_db(df, symbol, instrument_token, exchange, interval, from_dt, to_dt)
            print(f"Saved {rows_saved} candle(s) to the stock_data table.")

    if df.empty:
        print("No data to display.")
        sys.exit(0)

    # Display a preview
    pd.set_option("display.max_rows", 20)
    pd.set_option("display.float_format", "{:.2f}".format)
    print("\n" + df.to_string())

    # Optionally save to CSV
    if args.output:
        out_path = os.path.join(os.path.dirname(__file__), args.output)
        df.to_csv(out_path)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
