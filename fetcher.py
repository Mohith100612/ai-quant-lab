"""
Zerodha Kite Connect — Historical data fetcher.

Given a trading symbol, exchange, date range, and candle interval this module:
  1. Looks up the instrument token for the symbol.
  2. Calls kite.historical_data() to retrieve OHLCV candles.
  3. Returns a pandas DataFrame and optionally saves it to CSV.
"""

from __future__ import annotations

import sys
import time
import logging
from datetime import datetime, date
from typing import Union

import pandas as pd
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)

# Candle intervals supported by Kite Connect
VALID_INTERVALS = {
    "minute", "3minute", "5minute", "10minute", "15minute",
    "30minute", "60minute", "day",
}

# Maximum lookback in calendar days per interval (Kite Connect limits)
INTERVAL_MAX_DAYS = {
    "minute": 60,
    "3minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
    "day": 2000,
}


def _fetch_with_retry(fn, symbol: str, max_attempts: int = 3, backoff_seconds: int = 2):
    """
    Call fn(), retrying on failure with exponential backoff.

    Wait schedule (base=2):  attempt 1 → 2s, attempt 2 → 4s, attempt 3 → give up.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Attempt %d/%d failed for %s: %s — retrying in %ds…",
                    attempt + 1, max_attempts, symbol, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "All %d attempts exhausted for %s: %s",
                    max_attempts, symbol, exc,
                )
    raise last_exc


def lookup_instrument_token(
    kite: KiteConnect,
    symbol: str,
    exchange: str = "NSE",
) -> int:
    """
    Return the instrument_token for `symbol` on `exchange`.

    Parameters
    ----------
    kite     : authenticated KiteConnect instance
    symbol   : trading symbol, e.g. "RELIANCE", "NIFTY 50"
    exchange : exchange segment, e.g. "NSE", "BSE", "NFO", "MCX"
    """
    instruments = kite.instruments(exchange=exchange)
    symbol_upper = symbol.upper().strip()

    matches = [
        inst for inst in instruments
        if inst["tradingsymbol"].upper() == symbol_upper
    ]

    if not matches:
        # Provide a helpful list of close matches
        close = [
            inst["tradingsymbol"]
            for inst in instruments
            if symbol_upper in inst["tradingsymbol"].upper()
        ][:10]
        hint = f" Possible matches: {close}" if close else ""
        raise ValueError(
            f"Symbol '{symbol}' not found on {exchange}.{hint} "
            f"Check the symbol name or try a different exchange (NSE, BSE, NFO, MCX)."
        )

    if len(matches) > 1:
        logger.warning(
            "Multiple instruments match '%s' on %s. Using first: %s (token=%s, name='%s').",
            symbol, exchange,
            matches[0]["tradingsymbol"],
            matches[0]["instrument_token"],
            matches[0]["name"],
        )

    token: int = matches[0]["instrument_token"]
    return token


def fetch_historical_data(
    kite: KiteConnect,
    symbol: str,
    from_date: Union[str, date, datetime],
    to_date: Union[str, date, datetime],
    interval: str = "day",
    exchange: str = "NSE",
    continuous: bool = False,
    oi: bool = False,
    max_attempts: int = 3,
    backoff_seconds: int = 2,
) -> pd.DataFrame:
    """
    Fetch OHLCV candle data for a symbol and return a DataFrame.

    Parameters
    ----------
    kite       : authenticated KiteConnect instance
    symbol     : trading symbol, e.g. "RELIANCE"
    from_date  : start date  — "YYYY-MM-DD" string, date, or datetime
    to_date    : end date    — "YYYY-MM-DD" string, date, or datetime
    interval   : candle interval — one of VALID_INTERVALS
    exchange   : exchange segment, default "NSE"
    continuous : True for continuous data (futures/options)
    oi         : True to include open interest column

    Returns
    -------
    pd.DataFrame with columns: date, open, high, low, close, volume[, oi]
    """
    interval = interval.lower().strip()
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. "
            f"Choose from: {sorted(VALID_INTERVALS)}"
        )

    # Normalise dates to datetime objects for the SDK
    from_dt = _to_datetime(from_date, is_start=True)
    to_dt   = _to_datetime(to_date,   is_start=False)

    if from_dt > to_dt:
        raise ValueError("from_date must be earlier than to_date.")

    delta_days = (to_dt - from_dt).days
    max_days = INTERVAL_MAX_DAYS[interval]
    if delta_days > max_days:
        logger.warning(
            "Requested range (%d days) exceeds Kite limit of %d days for '%s' candles.",
            delta_days, max_days, interval,
        )

    instrument_token = lookup_instrument_token(kite, symbol, exchange)

    logger.info(
        "Fetching %s candles for %s (%s) from %s to %s [token=%s].",
        interval, symbol, exchange, from_dt.date(), to_dt.date(), instrument_token,
    )

    records = _fetch_with_retry(
        lambda: kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_dt,
            to_date=to_dt,
            interval=interval,
            continuous=continuous,
            oi=oi,
        ),
        symbol=symbol,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )

    if not records:
        logger.warning("No data returned for %s (%s) %s — %s.", symbol, exchange, from_dt.date(), to_dt.date())
        return pd.DataFrame(), instrument_token

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    logger.info("Retrieved %d candles for %s.", len(df), symbol)
    return df, instrument_token


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_datetime(value: Union[str, date, datetime], is_start: bool) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        time_part = "00:00:00" if is_start else "23:59:59"
        return datetime.strptime(f"{value} {time_part}", "%Y-%m-%d %H:%M:%S")
    if isinstance(value, str):
        value = value.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(value, fmt)
                if fmt == "%Y-%m-%d":
                    dt = dt.replace(hour=0 if is_start else 23,
                                    minute=0 if is_start else 59,
                                    second=0 if is_start else 59)
                return dt
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date '{value}'. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.")
    raise ValueError(f"Unsupported date type: {type(value)}")
