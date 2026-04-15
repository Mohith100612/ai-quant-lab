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
from datetime import datetime, date, timedelta
from urllib.parse import urlparse
from contextlib import contextmanager

import json

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
    """Verify stock_data is accessible and create supporting tables/views if needed."""
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


def ensure_extended_tables() -> None:
    """Alias kept for pipeline.py compatibility — delegates to ensure_table()."""
    ensure_table()


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


def push_to_dlq(
    symbol: str,
    exchange: str,
    interval: str,
    from_dt: datetime,
    to_dt: datetime,
    error_msg: str,
    pipeline_name: str,
) -> int:
    """
    Insert a failed fetch into pipeline_errors (dead-letter queue).
    Returns the inserted row id, or -1 on failure.
    """
    try:
        with get_db_connection() as conn:
            if not conn:
                return -1
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO pipeline_errors
                    (source, symbol, exchange, interval_type,
                     from_date, to_date, error_type, error_message, config_file)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    pipeline_name,
                    symbol.upper(),
                    exchange.upper(),
                    interval,
                    from_dt.date(),
                    to_dt.date(),
                    "FetchError",
                    error_msg,
                    pipeline_name,
                ),
            )
            row_id = cursor.lastrowid
            conn.commit()
            cursor.close()
            return row_id
    except Exception as exc:
        logger.error("push_to_dlq error for %s: %s", symbol, exc)
        return -1


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


def _date_to_dt(d: date, is_start: bool) -> datetime:
    """Convert a date to start-of-day or end-of-day datetime."""
    if is_start:
        return datetime(d.year, d.month, d.day, 0, 0, 0)
    return datetime(d.year, d.month, d.day, 23, 59, 59)


def find_missing_date_ranges(
    symbol: str,
    exchange: str,
    interval: str,
    from_dt: datetime,
    to_dt: datetime,
) -> list[tuple[datetime, datetime]]:
    """
    Return (start, end) datetime pairs not yet covered in fetch_log for this
    symbol/exchange/interval within [from_dt, to_dt].

    Example: if 2025-01-01→2025-06-30 is already logged and the requested
    range is 2025-01-01→2026-04-10, only [(2025-07-01, 2026-04-10)] is returned.
    """
    try:
        with get_db_connection() as conn:
            if not conn:
                return [(from_dt, to_dt)]
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT from_date, to_date FROM fetch_log
                 WHERE symbol        = %s
                   AND exchange      = %s
                   AND interval_type = %s
                   AND from_date     <= %s
                   AND to_date       >= %s
                 ORDER BY from_date
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
    except Exception as exc:
        logger.error("find_missing_date_ranges error for %s: %s", symbol, exc)
        return [(from_dt, to_dt)]

    if not rows:
        return [(from_dt, to_dt)]

    covered = sorted((r["from_date"], r["to_date"]) for r in rows)

    gaps: list[tuple[datetime, datetime]] = []
    current = from_dt.date()
    end_date = to_dt.date()

    for cov_start, cov_end in covered:
        if current < cov_start:
            gap_end = min(cov_start - timedelta(days=1), end_date)
            gaps.append((_date_to_dt(current, True), _date_to_dt(gap_end, False)))
        current = max(current, cov_end + timedelta(days=1))
        if current > end_date:
            break

    if current <= end_date:
        gaps.append((_date_to_dt(current, True), _date_to_dt(end_date, False)))

    return gaps


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

_CREATE_TICK_DATA_SQL = """
CREATE TABLE IF NOT EXISTS tick_data (
    id                   BIGINT UNSIGNED  AUTO_INCREMENT PRIMARY KEY,
    instrument_token     INT              NOT NULL,
    symbol               VARCHAR(50)      NOT NULL,
    exchange             VARCHAR(20)      NOT NULL DEFAULT '',
    captured_at          DATETIME(3)      NOT NULL,
    last_price           DECIMAL(14, 4)   NOT NULL,
    open                 DECIMAL(14, 4),
    high                 DECIMAL(14, 4),
    low                  DECIMAL(14, 4),
    close                DECIMAL(14, 4),
    volume               BIGINT           DEFAULT 0,
    buy_quantity         INT              DEFAULT 0,
    sell_quantity        INT              DEFAULT 0,
    change_pct           DECIMAL(10, 4),
    last_traded_quantity INT              DEFAULT 0,
    avg_traded_price     DECIMAL(14, 4),
    oi                   BIGINT           DEFAULT 0,
    oi_day_high          BIGINT           DEFAULT 0,
    oi_day_low           BIGINT           DEFAULT 0,
    last_trade_time      DATETIME(3),
    exchange_timestamp   DATETIME(3),
    depth                JSON,
    UNIQUE KEY uq_tick (instrument_token, captured_at, last_price),
    INDEX idx_symbol_captured (symbol, captured_at),
    INDEX idx_token_captured  (instrument_token, captured_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


_CREATE_INSTRUMENT_MASTER_SQL = """
CREATE TABLE IF NOT EXISTS instrument_master (
    instrument_token INT            NOT NULL,
    exchange         VARCHAR(20)    NOT NULL,
    tradingsymbol    VARCHAR(50)    NOT NULL,
    name             VARCHAR(100),
    expiry           DATE,
    strike           DECIMAL(14, 4),
    tick_size        DECIMAL(10, 4),
    lot_size         INT,
    instrument_type  VARCHAR(20),
    segment          VARCHAR(20),
    exchange_token   INT,
    last_price       DECIMAL(14, 4),
    updated_at       DATETIME       DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (instrument_token, exchange),
    INDEX idx_exchange_symbol (exchange, tradingsymbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_ORDER_UPDATES_SQL = """
CREATE TABLE IF NOT EXISTS order_updates (
    id            BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    order_id      VARCHAR(50),
    status        VARCHAR(50),
    tradingsymbol VARCHAR(50),
    exchange      VARCHAR(20),
    raw_data      JSON,
    received_at   DATETIME        DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_order_id (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def ensure_tick_tables() -> None:
    """Create watched_symbols, tick_data, instrument_master, order_updates if they do not exist."""
    try:
        with get_db_connection() as conn:
            if not conn:
                return
            cursor = conn.cursor()
            cursor.execute(_CREATE_WATCHED_SYMBOLS_SQL)
            cursor.execute(_CREATE_TICK_DATA_SQL)
            cursor.execute(_CREATE_INSTRUMENT_MASTER_SQL)
            cursor.execute(_CREATE_ORDER_UPDATES_SQL)
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
    Bulk-insert real-time tick snapshots into tick_data.
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
                depth = t.get("depth")
                cursor.execute(
                    """
                    INSERT IGNORE INTO tick_data
                        (instrument_token, symbol, exchange, captured_at,
                         last_price, open, high, low, close,
                         volume, buy_quantity, sell_quantity, change_pct,
                         last_traded_quantity, avg_traded_price,
                         oi, oi_day_high, oi_day_low,
                         last_trade_time, exchange_timestamp, depth)
                    VALUES
                        (%s, %s, %s, %s,
                         %s, %s, %s, %s, %s,
                         %s, %s, %s, %s,
                         %s, %s,
                         %s, %s, %s,
                         %s, %s, %s)
                    """,
                    (
                        int(t["instrument_token"]),
                        t["symbol"].upper(),
                        t.get("exchange", ""),
                        t["captured_at"],
                        float(t["last_price"]),
                        float(t["open"])             if t.get("open")             is not None else None,
                        float(t["high"])             if t.get("high")             is not None else None,
                        float(t["low"])              if t.get("low")              is not None else None,
                        float(t["close"])            if t.get("close")            is not None else None,
                        int(t.get("volume") or 0),
                        int(t.get("buy_quantity") or 0),
                        int(t.get("sell_quantity") or 0),
                        float(t["change"])           if t.get("change")           is not None else None,
                        int(t.get("last_traded_quantity") or 0),
                        float(t["avg_traded_price"]) if t.get("avg_traded_price") is not None else None,
                        int(t.get("oi") or 0),
                        int(t.get("oi_day_high") or 0),
                        int(t.get("oi_day_low") or 0),
                        t.get("last_trade_time"),
                        t.get("exchange_timestamp"),
                        json.dumps(depth) if depth is not None else None,
                    ),
                )
                rows_inserted += cursor.rowcount

            conn.commit()
            cursor.close()
    except Exception as exc:
        logger.error(f"save_ticks: {exc}")

    return rows_inserted


def get_instruments_for_exchange(exchange: str) -> list[dict]:
    """Return cached instruments for an exchange from instrument_master."""
    try:
        with get_db_connection() as conn:
            if not conn:
                return []
            cursor = conn.cursor()
            cursor.execute(
                "SELECT instrument_token, tradingsymbol FROM instrument_master WHERE exchange = %s",
                (exchange.upper(),),
            )
            rows = cursor.fetchall()
            cursor.close()
            return rows
    except Exception as exc:
        logger.error(f"get_instruments_for_exchange: {exc}")
        return []


def upsert_instruments(exchange: str, instruments: list[dict]) -> int:
    """Upsert a list of instruments from kite.instruments() into instrument_master."""
    if not instruments:
        return 0
    count = 0
    try:
        with get_db_connection() as conn:
            if not conn:
                return 0
            cursor = conn.cursor()
            for inst in instruments:
                expiry = inst.get("expiry")
                if expiry and str(expiry) in ("0000-00-00", ""):
                    expiry = None
                cursor.execute(
                    """
                    INSERT INTO instrument_master
                        (instrument_token, exchange, tradingsymbol, name,
                         expiry, strike, tick_size, lot_size,
                         instrument_type, segment, exchange_token, last_price)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        tradingsymbol   = VALUES(tradingsymbol),
                        name            = VALUES(name),
                        expiry          = VALUES(expiry),
                        strike          = VALUES(strike),
                        tick_size       = VALUES(tick_size),
                        lot_size        = VALUES(lot_size),
                        instrument_type = VALUES(instrument_type),
                        segment         = VALUES(segment),
                        exchange_token  = VALUES(exchange_token),
                        last_price      = VALUES(last_price),
                        updated_at      = CURRENT_TIMESTAMP
                    """,
                    (
                        int(inst.get("instrument_token", 0)),
                        exchange.upper(),
                        inst.get("tradingsymbol", ""),
                        inst.get("name", ""),
                        expiry or None,
                        float(inst["strike"])       if inst.get("strike")       else None,
                        float(inst["tick_size"])    if inst.get("tick_size")    else None,
                        int(inst["lot_size"])       if inst.get("lot_size")     else None,
                        inst.get("instrument_type", ""),
                        inst.get("segment", ""),
                        int(inst["exchange_token"]) if inst.get("exchange_token") else None,
                        float(inst["last_price"])   if inst.get("last_price")   else None,
                    ),
                )
                count += 1
            conn.commit()
            cursor.close()
    except Exception as exc:
        logger.error(f"upsert_instruments: {exc}")
    return count


def save_order_update(data: dict) -> int:
    """Persist a Kite WebSocket order-update event to order_updates. Returns row id."""
    try:
        with get_db_connection() as conn:
            if not conn:
                return -1
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO order_updates (order_id, status, tradingsymbol, exchange, raw_data)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    data.get("order_id", ""),
                    data.get("status", ""),
                    data.get("tradingsymbol", ""),
                    data.get("exchange", ""),
                    json.dumps(data),
                ),
            )
            row_id = cursor.lastrowid
            conn.commit()
            cursor.close()
            return row_id
    except Exception as exc:
        logger.error(f"save_order_update: {exc}")
        return -1
