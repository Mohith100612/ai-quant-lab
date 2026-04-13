"""
MySQL database helpers for storing and retrieving historical candle data.

Tables used:
  stock_data      — OHLCV candles keyed by (instrument_token, timestamp)
  fetch_log       — tracks which (symbol, exchange, interval, from_date, to_date)
                    queries have been fetched; used for exact-range duplicate detection
  watched_symbols — symbols to poll every second during market hours
  tick_data       — 1-second real-time snapshots captured during market hours

Connection pooling
  A single PooledDB pool is created on first use (lazy init, thread-safe).
  All callers still do conn = _get_connection() / conn.close() — close()
  returns the connection to the pool instead of actually closing it.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from urllib.parse import urlparse
from contextlib import contextmanager

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")

try:
    import pymysql
    import pymysql.cursors
except ImportError as exc:
    raise ImportError("PyMySQL is not installed. Run: pip install pymysql") from exc

try:
    from dbutils.pooled_db import PooledDB
except ImportError as exc:
    raise ImportError("dbutils is not installed. Run: pip install dbutils") from exc


# ---------------------------------------------------------------------------
# Connection pool — one pool per process, created on first use
# ---------------------------------------------------------------------------

_pool: PooledDB | None = None
_pool_lock = threading.Lock()

POOL_MIN_CACHED    = 2    # idle connections to keep open
POOL_MAX_CACHED    = 5    # max idle connections in pool
POOL_MAX_CONN      = 10   # hard cap on total open connections
CONNECT_TIMEOUT    = 10   # seconds


def _get_pool() -> PooledDB:
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:          # double-checked locking
            return _pool

        load_dotenv(dotenv_path=ENV_FILE)
        url = os.getenv("MYSQL_URL", "")
        if not url:
            raise RuntimeError("MYSQL_URL not set in .env file.")

        parsed = urlparse(url)

        _pool = PooledDB(
            creator=pymysql,
            mincached=POOL_MIN_CACHED,
            maxcached=POOL_MAX_CACHED,
            maxconnections=POOL_MAX_CONN,
            blocking=True,             # wait for a free connection rather than raising
            host=parsed.hostname,
            port=parsed.port or 3306,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.lstrip("/"),
            connect_timeout=CONNECT_TIMEOUT,
            cursorclass=pymysql.cursors.DictCursor,
        )
        logger.info(
            "Connection pool created — %s@%s:%s/%s (min=%d, max=%d)",
            parsed.username, parsed.hostname, parsed.port or 3306,
            parsed.path.lstrip("/"),
            POOL_MIN_CACHED, POOL_MAX_CONN,
        )

    return _pool


def _get_connection():
    """Return a connection from the pool. Call conn.close() to return it."""
    return _get_pool().connection()


@contextmanager
def get_db_connection():
    """Context manager that yields a pooled connection and returns it on exit."""
    conn = None
    try:
        conn = _get_connection()
    except Exception as exc:
        logger.error(f"DB connection error: {exc}")
    try:
        yield conn
    finally:
        if conn:
            conn.close()

# ---------------------------------------------------------------------------
# Table DDL
# ---------------------------------------------------------------------------

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

_CREATE_PIPELINE_ERRORS_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_errors (
    id            INT          AUTO_INCREMENT PRIMARY KEY,
    source        VARCHAR(200),
    symbol        VARCHAR(50),
    exchange      VARCHAR(20),
    interval_type VARCHAR(20),
    error_type    VARCHAR(100),
    error_message TEXT,
    retry_count   INT          DEFAULT 1,
    resolved      BOOLEAN      DEFAULT FALSE,
    timestamp     DATETIME     DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# Columns added in later versions — (column_name, ALTER statement)
_PIPELINE_ERRORS_EXTRA_COLS = [
    ("from_date",   "ALTER TABLE pipeline_errors ADD COLUMN from_date   DATE         DEFAULT NULL"),
    ("to_date",     "ALTER TABLE pipeline_errors ADD COLUMN to_date     DATE         DEFAULT NULL"),
    ("config_file", "ALTER TABLE pipeline_errors ADD COLUMN config_file VARCHAR(200) DEFAULT NULL"),
]


_CREATE_ENRICHED_VIEW_SQL = """
CREATE OR REPLACE VIEW ohlcv_enriched AS
SELECT
    symbol,
    timestamp,
    open,
    high,
    low,
    close,
    volume,
    ROUND(high - low, 4)                              AS candle_range,
    ROUND(ABS(close - open), 4)                       AS body_size,
    CASE WHEN close >= open THEN 1 ELSE 0 END         AS is_bullish,
    ROUND(
        100.0 * (close - LAG(close) OVER w)
              / NULLIF(LAG(close) OVER w, 0),
        4
    )                                                  AS daily_return_pct
FROM stock_data
WINDOW w AS (PARTITION BY symbol ORDER BY timestamp);
"""


def ensure_table() -> None:
    """Verify stock_data is accessible and create tables + enriched view if needed."""
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM stock_data LIMIT 1")
    cursor.execute(_CREATE_FETCH_LOG_SQL)
    cursor.execute(_CREATE_PIPELINE_ERRORS_SQL)
    # Add extra columns only if they don't exist (compatible with older MySQL)
    cursor.execute(
        "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'pipeline_errors'"
    )
    existing_cols = {r["COLUMN_NAME"] for r in cursor.fetchall()}
    for col_name, alter_sql in _PIPELINE_ERRORS_EXTRA_COLS:
        if col_name not in existing_cols:
            cursor.execute(alter_sql)
            logger.info("Added column pipeline_errors.%s", col_name)
    cursor.execute(_CREATE_ENRICHED_VIEW_SQL)
    conn.commit()
    cursor.close()
    conn.close()


def log_pipeline_error(
    pipeline_name: str,
    instrument: str,
    exchange: str,
    interval: str,
    from_dt: datetime,
    to_dt: datetime,
    error: Exception,
    attempts: int,
    config_file: str = "",
) -> None:
    """Write a failed fetch into pipeline_errors (Dead Letter Queue)."""
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO pipeline_errors
            (source, symbol, exchange, interval_type,
             from_date, to_date, error_type, error_message, retry_count, config_file)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            pipeline_name,
            instrument.upper(),
            exchange.upper(),
            interval,
            from_dt.date(),
            to_dt.date(),
            type(error).__name__,
            str(error),
            attempts,
            config_file,
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()


def data_exists(
    symbol: str,
    exchange: str,
    interval: str,
    from_dt: datetime,
    to_dt: datetime,
) -> bool:
    """Check if query is already fetched."""
    try:
        with get_db_connection() as conn:
            if not conn:
                return False
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
            return int(row["cnt"]) > 0
    except Exception as exc:
        logger.error(f"data_exists: {exc}")
        return False


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleaning before DB insert:
      - drop exact duplicate timestamps
      - drop rows where high < low (corrupt candles)
      - ensure the index is datetime
    """
    df = df[~df.index.duplicated(keep="first")]
    df = df[df["high"] >= df["low"]]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


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
    Clean, then insert candles into stock_data and log the query in fetch_log.
    Duplicate candle rows (instrument_token, timestamp) are silently skipped.

    Returns number of candle rows actually inserted.
    """
    if df.empty:
        return 0

    df = transform(df)

    rows_inserted = 0
    try:
        with get_db_connection() as conn:
            if not conn:
                return 0
            cursor = conn.cursor()
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
    except Exception as exc:
        logger.error(f"save_to_db: {exc}")
    
    return rows_inserted


# ---------------------------------------------------------------------------
# Enriched view query
# ---------------------------------------------------------------------------

def get_enriched_candles(
    symbol: str,
    from_dt: datetime,
    to_dt: datetime,
) -> list[dict]:
    """
    Query the ohlcv_enriched view — returns OHLCV plus DB-computed columns:
    candle_range, body_size, is_bullish, daily_return_pct.
    """
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT symbol, timestamp, open, high, low, close, volume,
               candle_range, body_size, is_bullish, daily_return_pct
          FROM ohlcv_enriched
         WHERE symbol    = %s
           AND timestamp BETWEEN %s AND %s
         ORDER BY timestamp ASC
        """,
        (symbol.upper(), from_dt, to_dt),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [
        {
            "timestamp":        row["timestamp"].isoformat(),
            "open":             float(row["open"]),
            "high":             float(row["high"]),
            "low":              float(row["low"]),
            "close":            float(row["close"]),
            "volume":           int(row["volume"]),
            "candle_range":     float(row["candle_range"])     if row["candle_range"]     is not None else None,
            "body_size":        float(row["body_size"])        if row["body_size"]        is not None else None,
            "is_bullish":       bool(row["is_bullish"]),
            "daily_return_pct": float(row["daily_return_pct"]) if row["daily_return_pct"] is not None else None,
        }
        for row in rows
    ]


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


def ensure_tick_tables() -> None:
    """Create watched_symbols table if it does not exist."""
    try:
        with get_db_connection() as conn:
            if not conn:
                return
            cursor = conn.cursor()
            cursor.execute(_CREATE_WATCHED_SYMBOLS_SQL)
            conn.commit()
            cursor.close()
    except Exception as exc:
        logger.error(f"ensure_tick_tables: {exc}")


def get_watched_symbols() -> list[dict]:
    """Return all rows from watched_symbols as a list of dicts."""
    try:
        with get_db_connection() as conn:
            if not conn:
                return []
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, exchange, added_at FROM watched_symbols ORDER BY added_at")
            rows = cursor.fetchall()
            cursor.close()
            return [
                {
                    "symbol":   row["symbol"],
                    "exchange": row["exchange"],
                    "added_at": row["added_at"].isoformat() if row["added_at"] else None,
                }
                for row in rows
            ]
    except Exception as exc:
        logger.error(f"get_watched_symbols: {exc}")
        return []


def add_watched_symbol(symbol: str, exchange: str = "NSE") -> bool:
    """Insert a symbol into watched_symbols."""
    try:
        with get_db_connection() as conn:
            if not conn:
                return False
            cursor = conn.cursor()
            cursor.execute(
                "INSERT IGNORE INTO watched_symbols (symbol, exchange) VALUES (%s, %s)",
                (symbol.upper(), exchange.upper()),
            )
            inserted = cursor.rowcount > 0
            conn.commit()
            cursor.close()
            return inserted
    except Exception as exc:
        logger.error(f"add_watched_symbol: {exc}")
        return False


def remove_watched_symbol(symbol: str, exchange: str = "NSE") -> bool:
    """Delete a symbol from watched_symbols."""
    try:
        with get_db_connection() as conn:
            if not conn:
                return False
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM watched_symbols WHERE symbol = %s AND exchange = %s",
                (symbol.upper(), exchange.upper()),
            )
            deleted = cursor.rowcount > 0
            conn.commit()
            cursor.close()
            return deleted
    except Exception as exc:
        logger.error(f"remove_watched_symbol: {exc}")
        return False


def save_ticks(ticks: list[dict]) -> int:
    """
    Bulk-insert real-time 1-second tick snapshots into stock_data.
    Returns number of rows inserted.
    """
    if not ticks:
        return 0

    rows_inserted = 0
    try:
        with get_db_connection() as conn:
            if not conn:
                return 0
            cursor = conn.cursor()
            for t in ticks:
                cursor.execute(
                    """
                    INSERT IGNORE INTO stock_data
                        (instrument_token, symbol, timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        int(t["instrument_token"]),
                        t["symbol"].upper(),
                        t["captured_at"],
                        float(t["open"])  if t.get("open")  is not None else float(t["last_price"]),
                        float(t["high"])  if t.get("high")  is not None else float(t["last_price"]),
                        float(t["low"])   if t.get("low")   is not None else float(t["last_price"]),
                        float(t["last_price"]),
                        int(t.get("volume") or 0),
                    ),
                )
                rows_inserted += cursor.rowcount

            conn.commit()
            cursor.close()
    except Exception as exc:
        logger.error(f"save_ticks: {exc}")

    return rows_inserted
