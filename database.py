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

import json
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

    # Build all rows up-front so the DB round-trip is a single executemany call
    rows = [
        (
            instrument_token,
            symbol.upper(),
            ts.to_pydatetime(),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            int(row["volume"]),
        )
        for ts, row in df.iterrows()
    ]

    conn   = _get_connection()
    cursor = conn.cursor()

    cursor.executemany(
        """
        INSERT IGNORE INTO stock_data
            (instrument_token, symbol, timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        rows,
    )
    rows_inserted = cursor.rowcount

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
    id                   BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    instrument_token     INT             NOT NULL,
    symbol               VARCHAR(50)     NOT NULL,
    exchange             VARCHAR(20)     NOT NULL DEFAULT 'NSE',
    captured_at          DATETIME(3)     NOT NULL,
    last_price           DECIMAL(14,4)   NOT NULL,
    open                 DECIMAL(14,4),
    high                 DECIMAL(14,4),
    low                  DECIMAL(14,4),
    close                DECIMAL(14,4),
    volume               BIGINT          DEFAULT 0,
    buy_quantity         INT             DEFAULT 0,
    sell_quantity        INT             DEFAULT 0,
    change_pct           DECIMAL(10,4),
    last_traded_quantity INT             DEFAULT 0,
    avg_traded_price     DECIMAL(14,4),
    oi                   BIGINT          DEFAULT 0,
    oi_day_high          BIGINT          DEFAULT 0,
    oi_day_low           BIGINT          DEFAULT 0,
    last_trade_time      DATETIME(3),
    exchange_timestamp   DATETIME(3),
    depth                JSON,
    PRIMARY KEY (id),
    INDEX idx_symbol_time (symbol, captured_at),
    INDEX idx_token_time  (instrument_token, captured_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# New columns added after initial release — migrated onto existing tables.
_TICK_DATA_V2_COLUMNS = [
    ("last_traded_quantity", "INT DEFAULT 0"),
    ("avg_traded_price",     "DECIMAL(14,4) DEFAULT NULL"),
    ("oi",                   "BIGINT DEFAULT 0"),
    ("oi_day_high",          "BIGINT DEFAULT 0"),
    ("oi_day_low",           "BIGINT DEFAULT 0"),
    ("last_trade_time",      "DATETIME(3) DEFAULT NULL"),
    ("exchange_timestamp",   "DATETIME(3) DEFAULT NULL"),
    ("depth",                "JSON DEFAULT NULL"),
]


def _add_column_if_missing(cursor, table: str, column: str, definition: str) -> bool:
    """
    Add *column* to *table* with *definition* only if it does not already exist.
    Uses INFORMATION_SCHEMA so it is safe to call on every startup.
    Returns True if the column was added, False if it already existed.
    """
    cursor.execute(
        """
        SELECT COUNT(*) AS cnt
          FROM INFORMATION_SCHEMA.COLUMNS
         WHERE TABLE_SCHEMA = DATABASE()
           AND TABLE_NAME   = %s
           AND COLUMN_NAME  = %s
        """,
        (table, column),
    )
    if cursor.fetchone()["cnt"] == 0:
        cursor.execute(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {definition}")
        return True
    return False


def ensure_tick_tables() -> None:
    """Create watched_symbols and tick_data tables; migrate any missing v2 columns."""
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(_CREATE_WATCHED_SYMBOLS_SQL)
    cursor.execute(_CREATE_TICK_DATA_SQL)
    # Idempotent migration — adds v2 columns to tables that pre-date them
    for col, defn in _TICK_DATA_V2_COLUMNS:
        _add_column_if_missing(cursor, "tick_data", col, defn)
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
                         buy_quantity, sell_quantity, change,
                         last_traded_quantity, avg_traded_price,
                         oi, oi_day_high, oi_day_low,
                         last_trade_time, exchange_timestamp, depth

    Returns number of rows inserted.
    """
    if not ticks:
        return 0

    rows = [
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
            int(t.get("last_traded_quantity") or 0),
            float(t["avg_traded_price"]) if t.get("avg_traded_price") is not None else None,
            int(t.get("oi") or 0),
            int(t.get("oi_day_high") or 0),
            int(t.get("oi_day_low") or 0),
            t.get("last_trade_time"),
            t.get("exchange_timestamp"),
            json.dumps(t["depth"]) if t.get("depth") is not None else None,
        )
        for t in ticks
    ]

    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT INTO tick_data
            (instrument_token, symbol, exchange, captured_at,
             last_price, open, high, low, close,
             volume, buy_quantity, sell_quantity, change_pct,
             last_traded_quantity, avg_traded_price,
             oi, oi_day_high, oi_day_low,
             last_trade_time, exchange_timestamp, depth)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        rows,
    )
    rows_inserted = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    return rows_inserted


# ---------------------------------------------------------------------------
# Order updates — streaming order status changes from KiteTicker
# ---------------------------------------------------------------------------

_CREATE_ORDER_UPDATES_SQL = """
CREATE TABLE IF NOT EXISTS order_updates (
    id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    order_id            VARCHAR(100)    NOT NULL,
    exchange_order_id   VARCHAR(100),
    status              VARCHAR(50)     NOT NULL,
    status_message      TEXT,
    tradingsymbol       VARCHAR(50),
    exchange            VARCHAR(20),
    instrument_token    INT,
    order_type          VARCHAR(20),
    transaction_type    VARCHAR(10),
    product             VARCHAR(20),
    quantity            INT,
    average_price       DECIMAL(14,4),
    filled_quantity     INT,
    pending_quantity    INT,
    cancelled_quantity  INT,
    order_timestamp     DATETIME,
    exchange_timestamp  DATETIME,
    raw_data            JSON,
    received_at         DATETIME(3)     DEFAULT CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    INDEX idx_order_id  (order_id),
    INDEX idx_status    (status),
    INDEX idx_symbol    (tradingsymbol, exchange)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


# ---------------------------------------------------------------------------
# Dead-letter queue — failed pipeline symbols awaiting retry
# ---------------------------------------------------------------------------

_CREATE_DEAD_LETTER_SQL = """
CREATE TABLE IF NOT EXISTS dead_letter (
    id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    pipeline_name VARCHAR(100)    NOT NULL DEFAULT '',
    symbol        VARCHAR(50)     NOT NULL,
    exchange      VARCHAR(20)     NOT NULL,
    interval_type VARCHAR(20)     NOT NULL,
    from_date     DATE            NOT NULL,
    to_date       DATE            NOT NULL,
    error_msg     TEXT,
    attempts      INT             NOT NULL DEFAULT 1,
    status        ENUM('pending','resolved') NOT NULL DEFAULT 'pending',
    created_at    DATETIME        DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME        DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_dlq_status (status),
    INDEX idx_dlq_symbol (symbol, exchange)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


# ---------------------------------------------------------------------------
# Instrument master — local token-lookup cache (refreshed daily)
# ---------------------------------------------------------------------------

_CREATE_INSTRUMENT_MASTER_SQL = """
CREATE TABLE IF NOT EXISTS instrument_master (
    instrument_token INT            NOT NULL,
    exchange         VARCHAR(20)    NOT NULL,
    tradingsymbol    VARCHAR(50)    NOT NULL,
    name             VARCHAR(255),
    segment          VARCHAR(50),
    sector           VARCHAR(100),
    instrument_type  VARCHAR(50),
    expiry           DATE,
    strike           DECIMAL(10,2),
    lot_size         INT,
    tick_size        DECIMAL(10,4),
    last_updated     DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (instrument_token),
    INDEX idx_im_symbol_exchange (tradingsymbol, exchange),
    INDEX idx_im_segment (segment)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def ensure_extended_tables() -> None:
    """Create order_updates, dead_letter, and instrument_master tables if they do not exist."""
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(_CREATE_ORDER_UPDATES_SQL)
    cursor.execute(_CREATE_DEAD_LETTER_SQL)
    cursor.execute(_CREATE_INSTRUMENT_MASTER_SQL)
    conn.commit()
    cursor.close()
    conn.close()


def save_order_update(data: dict) -> int:
    """
    Persist a single order-update event received via KiteTicker's
    ``on_order_update`` callback.

    Stores key fields in typed columns plus the full payload in ``raw_data``
    for complete auditability.  Returns the new row id.
    """
    import json as _json

    def _ts(val) -> "datetime | None":
        """Coerce a string or datetime to a plain datetime (no tzinfo)."""
        if val is None:
            return None
        if isinstance(val, datetime):
            return val.replace(tzinfo=None)
        try:
            return datetime.fromisoformat(str(val))
        except Exception:
            return None

    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO order_updates
            (order_id, exchange_order_id, status, status_message,
             tradingsymbol, exchange, instrument_token,
             order_type, transaction_type, product,
             quantity, average_price, filled_quantity,
             pending_quantity, cancelled_quantity,
             order_timestamp, exchange_timestamp, raw_data)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            str(data.get("order_id", "")),
            str(data.get("exchange_order_id", "") or ""),
            str(data.get("status", "")),
            str(data.get("status_message", "") or ""),
            str(data.get("tradingsymbol", "") or ""),
            str(data.get("exchange", "") or ""),
            int(data["instrument_token"]) if data.get("instrument_token") else None,
            str(data.get("order_type", "") or ""),
            str(data.get("transaction_type", "") or ""),
            str(data.get("product", "") or ""),
            int(data["quantity"]) if data.get("quantity") is not None else None,
            float(data["average_price"]) if data.get("average_price") else None,
            int(data.get("filled_quantity") or 0),
            int(data.get("pending_quantity") or 0),
            int(data.get("cancelled_quantity") or 0),
            _ts(data.get("order_timestamp")),
            _ts(data.get("exchange_timestamp")),
            _json.dumps(data, default=str),
        ),
    )
    row_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    conn.close()
    return row_id


# ── Dead-letter helpers ────────────────────────────────────────────────────

def push_to_dlq(
    symbol:        str,
    exchange:      str,
    interval:      str,
    from_dt:       datetime,
    to_dt:         datetime,
    error_msg:     str,
    pipeline_name: str = "",
) -> int:
    """
    Insert a failed fetch job into dead_letter.
    Returns the new row id.
    """
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO dead_letter
            (pipeline_name, symbol, exchange, interval_type,
             from_date, to_date, error_msg)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            pipeline_name,
            symbol.upper(),
            exchange.upper(),
            interval,
            from_dt.date() if hasattr(from_dt, "date") else from_dt,
            to_dt.date()   if hasattr(to_dt,   "date") else to_dt,
            error_msg,
        ),
    )
    row_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    conn.close()
    return row_id


def get_dlq_pending(limit: int = 100) -> list[dict]:
    """Return up to *limit* pending dead-letter items, oldest first."""
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, pipeline_name, symbol, exchange, interval_type,
               from_date, to_date, error_msg, attempts, created_at
          FROM dead_letter
         WHERE status = 'pending'
         ORDER BY created_at ASC
         LIMIT %s
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return list(rows)


def mark_dlq_resolved(dlq_id: int) -> None:
    """Mark a dead-letter item as resolved (no longer needs retry)."""
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE dead_letter SET status = 'resolved' WHERE id = %s",
        (dlq_id,),
    )
    conn.commit()
    cursor.close()
    conn.close()


# ── Instrument-master helpers ──────────────────────────────────────────────

def upsert_instruments(exchange: str, instruments: list[dict]) -> int:
    """
    Bulk-upsert instrument records fetched from ``kite.instruments()``.

    Existing rows are updated in place (ON DUPLICATE KEY UPDATE) so stale
    expiry / tick_size / lot_size values are always replaced with the latest
    from the Kite API.

    Returns the number of rows inserted or updated.
    """
    if not instruments:
        return 0

    from datetime import date as _date

    now  = datetime.utcnow()
    conn = _get_connection()
    cursor = conn.cursor()
    count  = 0

    for inst in instruments:
        # Normalise expiry: Kite returns a date object, empty string, or None
        expiry = inst.get("expiry") or None
        if isinstance(expiry, str):
            try:
                expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
            except ValueError:
                expiry = None
        elif isinstance(expiry, datetime):
            expiry = expiry.date()
        # Already a date object → leave as-is

        strike = inst.get("strike") or None
        if strike is not None:
            try:
                strike = float(strike)
            except (TypeError, ValueError):
                strike = None

        cursor.execute(
            """
            INSERT INTO instrument_master
                (instrument_token, exchange, tradingsymbol, name,
                 segment, instrument_type, expiry, strike,
                 lot_size, tick_size, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                exchange        = VALUES(exchange),
                tradingsymbol   = VALUES(tradingsymbol),
                name            = VALUES(name),
                segment         = VALUES(segment),
                instrument_type = VALUES(instrument_type),
                expiry          = VALUES(expiry),
                strike          = VALUES(strike),
                lot_size        = VALUES(lot_size),
                tick_size       = VALUES(tick_size),
                last_updated    = VALUES(last_updated)
            """,
            (
                int(inst.get("instrument_token", 0)),
                exchange.upper(),
                str(inst.get("tradingsymbol", "")).upper(),
                str(inst.get("name", "")) or None,
                str(inst.get("segment", "")) or None,
                str(inst.get("instrument_type", "")) or None,
                expiry,
                strike,
                int(inst.get("lot_size") or 1),
                float(inst.get("tick_size") or 0.05),
                now,
            ),
        )
        count += cursor.rowcount

    conn.commit()
    cursor.close()
    conn.close()
    return count


def get_instrument_token_from_db(symbol: str, exchange: str) -> "int | None":
    """
    Look up ``instrument_token`` for *symbol* on *exchange* from the local
    cache.  Returns ``None`` on cache miss or if the table does not exist yet.
    """
    try:
        conn   = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT instrument_token FROM instrument_master
             WHERE tradingsymbol = %s AND exchange = %s
             LIMIT 1
            """,
            (symbol.upper(), exchange.upper()),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return int(row["instrument_token"]) if row else None
    except Exception:
        return None


def get_instruments_for_exchange(exchange: str) -> list[dict]:
    """
    Return all cached instrument records for *exchange*.
    Returns an empty list if the table is empty or does not exist.
    """
    try:
        conn   = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT instrument_token, tradingsymbol, name,
                   instrument_type, expiry, lot_size, tick_size
              FROM instrument_master
             WHERE exchange = %s
             ORDER BY tradingsymbol
            """,
            (exchange.upper(),),
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return list(rows)
    except Exception:
        return []


def get_instruments_refreshed_at(exchange: str) -> "datetime | None":
    """
    Return the most recent ``last_updated`` timestamp for *exchange*, or
    ``None`` if no rows exist (i.e. the cache has never been populated).
    """
    try:
        conn   = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(last_updated) AS last_refresh FROM instrument_master WHERE exchange = %s",
            (exchange.upper(),),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row["last_refresh"] if row else None
    except Exception:
        return None


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
               volume, buy_quantity, sell_quantity, change_pct,
               last_traded_quantity, avg_traded_price,
               oi, oi_day_high, oi_day_low,
               last_trade_time, exchange_timestamp, depth
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
