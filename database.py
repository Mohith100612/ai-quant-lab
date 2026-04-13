"""
MySQL database helpers for storing and retrieving historical candle data.

Tables used:
  stock_data      — OHLCV candles keyed by (instrument_token, timestamp)
  fetch_log       — tracks which (symbol, exchange, interval, from_date, to_date)
                    queries have been fetched; used for exact-range duplicate detection
  watched_symbols — symbols the WebSocket / REST ticker subscribes to
  tick_data       — real-time tick snapshots (last_price, OHLC, volume,
                    buy/sell qty, change%) stored by both the WS and REST tickers
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, date, timedelta
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv

ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")

try:
    import pymysql
    import pymysql.cursors
except ImportError:
    print("ERROR: PyMySQL is not installed.\nRun: pip install pymysql")
    sys.exit(1)

try:
    from dbutils.pooled_db import PooledDB
except ImportError:
    print("ERROR: dbutils is not installed.\nRun: pip install dbutils")
    sys.exit(1)

_pool: PooledDB | None = None


def _get_pool() -> PooledDB:
    global _pool
    if _pool is not None:
        return _pool

    load_dotenv(dotenv_path=ENV_FILE)
    url = os.getenv("MYSQL_URL", "")
    if not url:
        sys.exit("ERROR: MYSQL_URL not set in .env file.")

    parsed   = urlparse(url)
    host     = parsed.hostname
    port     = parsed.port or 3306
    user     = parsed.username
    password = parsed.password
    database = parsed.path.lstrip("/")

    try:
        _pool = PooledDB(
            creator=pymysql,
            mincached=2,        # keep 2 connections open at idle
            maxcached=5,        # max idle connections kept in pool
            maxconnections=10,  # hard cap on total connections
            blocking=True,      # wait instead of raising when pool is full
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            connect_timeout=10,
            cursorclass=pymysql.cursors.DictCursor,
        )
        print(f"[DB] Pool created: {user}@{host}:{port}/{database}")
        return _pool
    except Exception as exc:
        sys.exit(f"ERROR: Cannot create DB pool: {exc}")


def _get_connection():
    return _get_pool().connection()


_CREATE_FETCH_LOG_SQL = """
CREATE TABLE IF NOT EXISTS fetch_log (
    symbol        VARCHAR(50)  NOT NULL,
    exchange      VARCHAR(20)  NOT NULL,
    interval_type VARCHAR(20)  NOT NULL,
    from_date     DATE         NOT NULL,
    to_date       DATE         NOT NULL,
    fetched_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, exchange, interval_type, from_date, to_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def ensure_table() -> None:
    """Verify stock_data is accessible and create fetch_log if needed."""
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM stock_data LIMIT 1")
    cursor.execute(_CREATE_FETCH_LOG_SQL)
    conn.commit()
    cursor.close()
    conn.close()


def find_missing_date_ranges(
    symbol: str,
    exchange: str,
    interval: str,
    from_dt: datetime,
    to_dt: datetime,
) -> list[tuple[datetime, datetime]]:
    """
    Return a list of (start, end) datetime pairs representing date ranges
    within [from_dt, to_dt] that have NOT yet been fetched and logged.

    Example
    -------
    Requested : 2021-01-01 → 2026-04-10
    In fetch_log: [2021-01-01 → 2022-12-31]  [2024-06-01 → 2026-04-10]
    Returns   : [(2023-01-01, 2024-05-31)]   ← the gap in 2023/early 2024
    """
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT from_date, to_date FROM fetch_log
         WHERE symbol        = %s
           AND exchange      = %s
           AND interval_type = %s
           AND from_date     <= %s
           AND to_date       >= %s
         ORDER BY from_date ASC
        """,
        (
            symbol.upper(),
            exchange.upper(),
            interval,
            to_dt.date(),
            from_dt.date(),
        ),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Convert to (date, date) tuples and merge overlapping / adjacent ranges
    fetched: list[tuple[date, date]] = [
        (row["from_date"], row["to_date"]) for row in rows
    ]

    if not fetched:
        return [(from_dt, to_dt)]

    # Merge overlapping/adjacent fetched ranges
    fetched.sort()
    merged: list[tuple[date, date]] = [fetched[0]]
    for start, end in fetched[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + timedelta(days=1):
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # Walk the timeline and collect gaps
    gaps: list[tuple[datetime, datetime]] = []
    cursor_date = from_dt.date()
    target_end  = to_dt.date()

    for seg_start, seg_end in merged:
        if cursor_date < seg_start and cursor_date <= target_end:
            gap_end = min(seg_start - timedelta(days=1), target_end)
            gaps.append((
                datetime.combine(cursor_date, datetime.min.time()),
                datetime.combine(gap_end,     datetime.max.time().replace(microsecond=0)),
            ))
        cursor_date = max(cursor_date, seg_end + timedelta(days=1))

    # Trailing gap after the last fetched segment
    if cursor_date <= target_end:
        gaps.append((
            datetime.combine(cursor_date,  datetime.min.time()),
            datetime.combine(target_end,   datetime.max.time().replace(microsecond=0)),
        ))

    return gaps


def data_exists(
    symbol: str,
    exchange: str,
    interval: str,
    from_dt: datetime,
    to_dt: datetime,
) -> bool:
    """
    Return True only if this exact query (symbol+exchange+interval+date range)
    was previously fetched and logged in fetch_log.
    """
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) AS cnt FROM fetch_log
         WHERE symbol        = %s
           AND exchange      = %s
           AND interval_type = %s
           AND from_date     <= %s
           AND to_date       >= %s
        """,
        (
            symbol.upper(),
            exchange.upper(),
            interval,
            from_dt.date(),
            to_dt.date(),
        ),
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return int(row["cnt"]) > 0


def save_to_db(
    df: pd.DataFrame,
    symbol: str,
    instrument_token: int,
    exchange: str,
    interval: str,
    from_dt: datetime,
    to_dt: datetime,
) -> int:
    """
    Insert candles into stock_data and log the query in fetch_log.
    Duplicate candle rows (instrument_token, timestamp) are silently skipped.

    Returns
    -------
    Number of candle rows actually inserted.
    """
    if df.empty:
        return 0

    conn   = _get_connection()
    cursor = conn.cursor()
    rows_inserted = 0

    for ts, row in df.iterrows():
        cursor.execute(
            """
            INSERT IGNORE INTO stock_data
                (instrument_token, symbol, timestamp, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                instrument_token,
                symbol.upper(),
                ts.to_pydatetime(),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
            ),
        )
        rows_inserted += cursor.rowcount

    # Log the fetched query range so future identical queries are detected
    cursor.execute(
        """
        INSERT IGNORE INTO fetch_log
            (symbol, exchange, interval_type, from_date, to_date)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            symbol.upper(),
            exchange.upper(),
            interval,
            from_dt.date(),
            to_dt.date(),
        ),
    )

    conn.commit()
    cursor.close()
    conn.close()
    return rows_inserted


# ---------------------------------------------------------------------------
# Real-time tick tables
# ---------------------------------------------------------------------------

_CREATE_WATCHED_SYMBOLS_SQL = """
CREATE TABLE IF NOT EXISTS watched_symbols (
    symbol    VARCHAR(50) NOT NULL,
    exchange  VARCHAR(20) NOT NULL DEFAULT 'NSE',
    added_at  DATETIME    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, exchange)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_TICK_DATA_SQL = """
CREATE TABLE IF NOT EXISTS tick_data (
    id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    instrument_token INT             NOT NULL,
    symbol           VARCHAR(50)     NOT NULL,
    exchange         VARCHAR(20)     NOT NULL DEFAULT 'NSE',
    captured_at      DATETIME(3)     NOT NULL,
    last_price       DECIMAL(14,4)   NOT NULL,
    open             DECIMAL(14,4),
    high             DECIMAL(14,4),
    low              DECIMAL(14,4),
    close            DECIMAL(14,4),
    volume           BIGINT          DEFAULT 0,
    buy_quantity     INT             DEFAULT 0,
    sell_quantity    INT             DEFAULT 0,
    change_pct       DECIMAL(10,4),
    PRIMARY KEY (id),
    INDEX idx_symbol_time (symbol, captured_at),
    INDEX idx_token_time  (instrument_token, captured_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def ensure_tick_tables() -> None:
    """Create watched_symbols and tick_data tables if they do not exist."""
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(_CREATE_WATCHED_SYMBOLS_SQL)
    cursor.execute(_CREATE_TICK_DATA_SQL)
    conn.commit()
    cursor.close()
    conn.close()


def get_watched_symbols() -> list[dict]:
    """Return all rows from watched_symbols as a list of dicts."""
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, exchange, added_at FROM watched_symbols ORDER BY added_at")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [
        {
            "symbol":   row["symbol"],
            "exchange": row["exchange"],
            "added_at": row["added_at"].isoformat() if row["added_at"] else None,
        }
        for row in rows
    ]


def add_watched_symbol(symbol: str, exchange: str = "NSE") -> bool:
    """
    Insert a symbol into watched_symbols.
    Returns True if inserted, False if it already existed.
    """
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT IGNORE INTO watched_symbols (symbol, exchange) VALUES (%s, %s)",
        (symbol.upper(), exchange.upper()),
    )
    inserted = cursor.rowcount > 0
    conn.commit()
    cursor.close()
    conn.close()
    return inserted


def remove_watched_symbol(symbol: str, exchange: str = "NSE") -> bool:
    """
    Delete a symbol from watched_symbols.
    Returns True if deleted, False if not found.
    """
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM watched_symbols WHERE symbol = %s AND exchange = %s",
        (symbol.upper(), exchange.upper()),
    )
    deleted = cursor.rowcount > 0
    conn.commit()
    cursor.close()
    conn.close()
    return deleted


def save_ticks(ticks: list[dict]) -> int:
    """
    Bulk-insert real-time tick snapshots into tick_data.

    Each dict must have: symbol, instrument_token, captured_at, last_price
    Optional keys:       exchange, open, high, low, close, volume,
                         buy_quantity, sell_quantity, change

    Returns number of rows inserted.
    """
    if not ticks:
        return 0

    conn   = _get_connection()
    cursor = conn.cursor()
    rows_inserted = 0

    for t in ticks:
        cursor.execute(
            """
            INSERT INTO tick_data
                (instrument_token, symbol, exchange, captured_at,
                 last_price, open, high, low, close,
                 volume, buy_quantity, sell_quantity, change_pct)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                int(t["instrument_token"]),
                t["symbol"].upper(),
                t.get("exchange", "NSE").upper(),
                t["captured_at"],
                float(t["last_price"]),
                float(t["open"])   if t.get("open")   is not None else None,
                float(t["high"])   if t.get("high")   is not None else None,
                float(t["low"])    if t.get("low")    is not None else None,
                float(t["close"])  if t.get("close")  is not None else None,
                int(t.get("volume") or 0),
                int(t.get("buy_quantity") or 0),
                int(t.get("sell_quantity") or 0),
                float(t["change"]) if t.get("change") is not None else None,
            ),
        )
        rows_inserted += cursor.rowcount

    conn.commit()
    cursor.close()
    conn.close()
    return rows_inserted


def query_tick_data(
    symbol: str,
    from_dt: datetime,
    to_dt: datetime,
    limit: int = 3600,
) -> list[dict]:
    """
    Return tick_data rows for a symbol within the given time range.

    Rows are ordered oldest-first and capped at `limit` (default 3600 = 1 hour
    of 1-second ticks).
    """
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT instrument_token, symbol, exchange, captured_at,
               last_price, open, high, low, close,
               volume, buy_quantity, sell_quantity, change_pct
          FROM tick_data
         WHERE symbol      = %s
           AND captured_at BETWEEN %s AND %s
         ORDER BY captured_at ASC
         LIMIT %s
        """,
        (symbol.upper(), from_dt, to_dt, limit),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return list(rows)
