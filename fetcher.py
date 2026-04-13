"""
Zerodha Kite Connect — Historical data fetcher.

Given a trading symbol, exchange, date range, and candle interval this module:
  1. Looks up the instrument token for the symbol.
  2. Calls kite.historical_data() to retrieve OHLCV candles.
  3. Returns a pandas DataFrame and optionally saves it to CSV.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Union

import pandas as pd
from kiteconnect import KiteConnect

from exceptions import KiteAPIError, SymbolNotFoundError, InvalidIntervalError
from log_config import get_logger

logger = get_logger(__name__)

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


def lookup_instrument_token(
    kite: KiteConnect,
    symbol: str,
    exchange: str = "NSE",
) -> int:
    """
    Return the instrument_token for `symbol` on `exchange`.

    Resolution order
    ----------------
    1. Check the local ``instrument_master`` DB cache (fast, no API call).
    2. On cache miss, fetch the full instrument list from Kite and return the
       matching token (the cache is *not* populated here — use
       ``refresh_instrument_master()`` for bulk warming).

    Raises
    ------
    SymbolNotFoundError  if the symbol does not exist on the exchange.
    KiteAPIError         if the instruments list cannot be fetched.
    """
    from database import get_instrument_token_from_db

    # ── Fast path: local DB cache ──────────────────────────────────────────
    cached = get_instrument_token_from_db(symbol, exchange)
    if cached is not None:
        return cached

    # ── Slow path: live Kite API call ──────────────────────────────────────
    logger.debug(
        "Instrument cache miss for %s:%s — fetching from Kite API",
        exchange, symbol,
    )
    try:
        instruments = kite.instruments(exchange=exchange)
    except Exception as exc:
        raise KiteAPIError(
            f"Failed to fetch instrument list for exchange '{exchange}': {exc}"
        ) from exc

    symbol_upper = symbol.upper().strip()

    try:
        matches = [
            inst for inst in instruments
            if inst["tradingsymbol"].upper() == symbol_upper
        ]
    except Exception as exc:
        raise KiteAPIError(
            f"Error while searching instrument list: {exc}"
        ) from exc

    if not matches:
        close = [
            inst["tradingsymbol"]
            for inst in instruments
            if symbol_upper in inst["tradingsymbol"].upper()
        ][:10]
        hint = f"\n  Possible matches: {close}" if close else ""
        raise SymbolNotFoundError(
            f"Symbol '{symbol}' not found on {exchange}.{hint}\n"
            f"Check the symbol name or try a different exchange (NSE, BSE, NFO, MCX)."
        )

    if len(matches) > 1:
        logger.warning(
            "Multiple instruments match '%s' on %s — using first: %s (token=%s, name='%s')",
            symbol, exchange,
            matches[0]["tradingsymbol"],
            matches[0]["instrument_token"],
            matches[0]["name"],
        )

    return int(matches[0]["instrument_token"])


def refresh_instrument_master(
    kite:          KiteConnect,
    exchanges:     list[str] | None = None,
    max_age_hours: float = 24.0,
) -> dict[str, int]:
    """
    Fetch the full instrument list for each exchange from Kite and persist it
    to the ``instrument_master`` DB table.

    Staleness check
    ---------------
    If the cached data for an exchange is younger than *max_age_hours* the
    exchange is skipped.  Pass ``max_age_hours=0`` to force a full refresh.

    Returns
    -------
    A dict mapping ``{exchange: rows_upserted}``.  Exchanges that were
    skipped because their cache was fresh have a value of ``0``.
    """
    from database import upsert_instruments, get_instruments_refreshed_at

    exchanges = exchanges or ["NSE", "BSE", "NFO", "MCX"]
    now       = datetime.utcnow()
    results: dict[str, int] = {}

    for exchange in exchanges:
        if max_age_hours > 0:
            last = get_instruments_refreshed_at(exchange)
            if last and (now - last) < timedelta(hours=max_age_hours):
                logger.info(
                    "Instrument master for %s is fresh (age < %.0fh) — skipping",
                    exchange, max_age_hours,
                )
                results[exchange] = 0
                continue

        try:
            instruments = kite.instruments(exchange=exchange)
            count       = upsert_instruments(exchange, instruments)
            logger.info(
                "Instrument master refreshed for %s: %d rows upserted",
                exchange, count,
            )
            results[exchange] = count
        except Exception as exc:
            raise KiteAPIError(
                f"Failed to refresh instrument master for {exchange}: {exc}"
            ) from exc

    return results


def fetch_historical_data(
    kite: KiteConnect,
    symbol: str,
    from_date: Union[str, date, datetime],
    to_date: Union[str, date, datetime],
    interval: str = "day",
    exchange: str = "NSE",
    continuous: bool = False,
    oi: bool = False,
) -> tuple[pd.DataFrame, int]:
    """
    Fetch OHLCV candle data for a symbol and return a (DataFrame, token) tuple.

    Raises
    ------
    InvalidIntervalError   if `interval` is not in VALID_INTERVALS.
    ValueError             if from_date > to_date.
    SymbolNotFoundError    if the symbol does not exist.
    KiteAPIError           if the historical_data API call fails.
    """
    interval = interval.lower().strip()
    if interval not in VALID_INTERVALS:
        raise InvalidIntervalError(
            f"Invalid interval '{interval}'. "
            f"Choose from: {sorted(VALID_INTERVALS)}"
        )

    try:
        from_dt = _to_datetime(from_date, is_start=True)
        to_dt   = _to_datetime(to_date,   is_start=False)
    except ValueError:
        raise

    if from_dt > to_dt:
        raise ValueError(
            f"from_date ({from_dt.date()}) must be earlier than to_date ({to_dt.date()})."
        )

    delta_days = (to_dt - from_dt).days
    max_days   = INTERVAL_MAX_DAYS[interval]
    if delta_days > max_days:
        logger.warning(
            "Requested range (%d days) exceeds Kite limit of %d days for '%s' candles — "
            "API may return partial data or raise an error.",
            delta_days, max_days, interval,
        )

    # Raises SymbolNotFoundError / KiteAPIError if lookup fails
    instrument_token = lookup_instrument_token(kite, symbol, exchange)

    logger.info(
        "Fetching %s candles for %s (%s) from %s to %s [token=%s]",
        interval, symbol, exchange, from_dt.date(), to_dt.date(), instrument_token,
    )

    try:
        records = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_dt,
            to_date=to_dt,
            interval=interval,
            continuous=continuous,
            oi=oi,
        )
    except Exception as exc:
        raise KiteAPIError(
            f"historical_data() failed for {symbol} ({exchange}) "
            f"[{from_dt.date()} → {to_dt.date()}]: {exc}"
        ) from exc

    if not records:
        logger.warning("No data returned for %s (%s) [%s → %s].", symbol, exchange, from_dt.date(), to_dt.date())
        return pd.DataFrame(), instrument_token

    try:
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
    except Exception as exc:
        raise KiteAPIError(
            f"Failed to parse historical data response for {symbol}: {exc}"
        ) from exc

    logger.info("Retrieved %d candles for %s (%s).", len(df), symbol, exchange)
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
                    dt = dt.replace(
                        hour=0 if is_start else 23,
                        minute=0 if is_start else 59,
                        second=0 if is_start else 59,
                    )
                return dt
            except ValueError:
                continue
        raise ValueError(
            f"Cannot parse date '{value}'. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."
        )
    raise TypeError(f"Unsupported date type: {type(value)}")
