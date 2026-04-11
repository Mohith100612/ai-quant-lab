"""
Zerodha Kite Connect — FastAPI server.

Endpoints
---------
GET  /auth/login-url        → Kite login URL to open in browser
POST /auth/session          → Exchange request_token for access_token

GET  /candles               → Check DB first; if missing fetch from Kite, save, return

POST /ticker/watch          → Add a symbol to the real-time watch list
DEL  /ticker/watch/{symbol} → Remove a symbol from the watch list
GET  /ticker/watch          → List all watched symbols
POST /ticker/start          → Manually start real-time polling
POST /ticker/stop           → Manually stop real-time polling
GET  /ticker/status         → Is the ticker running?

GET  /ticks                 → Query stored 1-second tick snapshots

Run
---
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from auth import get_login_url, generate_session, get_authenticated_kite
from fetcher import fetch_historical_data, _to_datetime
from database import (
    ensure_table, data_exists, save_to_db, get_db_connection,
    ensure_tick_tables,
    get_watched_symbols, add_watched_symbol, remove_watched_symbol,
    get_enriched_candles,
)
from ticker import ticker_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI lifespan — runs on startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    try:
        ensure_tick_tables()
        logger.info("Tick tables ready.")
    except Exception as exc:
        logger.error("Could not create tick tables: %s", exc)

    ticker_manager.start_scheduler()   # registers 09:15 / 15:30 cron jobs

    yield   # ← application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    ticker_manager.stop_scheduler()


app = FastAPI(
    title="Zerodha Kite Connect API",
    description="REST wrapper around Kite Connect with real-time tick collection",
    version="2.0.0",
    lifespan=lifespan,
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


IntervalLiteral = Literal[
    "minute", "3minute", "5minute", "10minute",
    "15minute", "30minute", "60minute", "day",
]

INTERVAL_MAX_DAYS = {
    "minute": 60, "3minute": 100, "5minute": 100, "10minute": 100,
    "15minute": 200, "30minute": 200, "60minute": 400, "day": 2000,
}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.get("/auth/login-url", tags=["Auth"])
def login_url():
    """Return the Kite Connect login URL to open in a browser."""
    try:
        return {"login_url": get_login_url()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class SessionRequest(BaseModel):
    request_token: str


@app.post("/auth/session", tags=["Auth"])
def create_session(body: SessionRequest):
    """Exchange a request_token for an access_token and save it to .env."""
    try:
        return {"access_token": generate_session(body.request_token)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Candles — single endpoint (DB-first, then Kite API with auto-chunking)
# ---------------------------------------------------------------------------

@app.get("/candles", tags=["Candles"])
def get_candles(
    symbol:     str             = Query(...,    description="Trading symbol e.g. MRF, RELIANCE, M&M"),
    from_date:  str             = Query(...,    alias="from",  description="Start date YYYY-MM-DD"),
    to_date:    str             = Query(...,    alias="to",    description="End date   YYYY-MM-DD"),
    interval:   IntervalLiteral = Query("day",  description="Candle interval (default: day)"),
    exchange:   str             = Query("NSE",  description="Exchange: NSE, BSE, NFO, MCX"),
    continuous: bool            = Query(False,  description="Continuous data for futures/options"),
    oi:         bool            = Query(False,  description="Include open interest column"),
):
    """
    1. Validate inputs.
    2. Check the database — if data exists, return it immediately.
    3. If not, call Kite Connect API (auto-chunked for large ranges), save to DB, then return.
    """
    try:
        from_dt = _to_datetime(from_date, is_start=True)
        to_dt   = _to_datetime(to_date,   is_start=False)
    except (Exception, SystemExit) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if from_dt > to_dt:
        raise HTTPException(status_code=422, detail="from_date must be earlier than to_date")

    try:
        ensure_table()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")

    try:
        in_db = data_exists(symbol, exchange, interval, from_dt, to_dt)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database check failed: {exc}")

    if in_db:
        candles = _fetch_candles_from_db(symbol, from_dt, to_dt)
        return {
            "source": "database", "symbol": symbol.upper(), "exchange": exchange.upper(),
            "interval": interval, "from": from_dt.date().isoformat(),
            "to": to_dt.date().isoformat(), "total": len(candles), "candles": candles,
        }

    try:
        kite = get_authenticated_kite()
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Kite authentication failed: {exc}")

    chunk     = timedelta(days=INTERVAL_MAX_DAYS[interval] - 1)
    all_frames, instrument_token = [], None
    cur_start = from_dt

    try:
        while cur_start <= to_dt:
            cur_end = min(cur_start + chunk, to_dt)
            df, token = fetch_historical_data(
                kite=kite, symbol=symbol,
                from_date=cur_start, to_date=cur_end,
                interval=interval, exchange=exchange,
                continuous=continuous, oi=oi,
            )
            if not df.empty:
                all_frames.append(df)
                instrument_token = token
            cur_start = cur_end + timedelta(days=1)
    except Exception as exc:
        if "api_key" in str(exc) or "access_token" in str(exc):
            raise HTTPException(
                status_code=404, 
                detail="No historical data available for this date range (Kite Connect subscription limits reached)."
            )
        raise HTTPException(status_code=400, detail=f"Kite API error: {exc}")

    if not all_frames:
        return {
            "source": "kite_api", "symbol": symbol.upper(), "exchange": exchange.upper(),
            "interval": interval, "from": from_dt.date().isoformat(),
            "to": to_dt.date().isoformat(), "total": 0, "candles": [],
        }

    full_df = pd.concat(all_frames)
    full_df = full_df[~full_df.index.duplicated(keep="first")].sort_index()

    try:
        rows_saved = save_to_db(full_df, symbol, instrument_token, exchange, interval, from_dt, to_dt)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database save failed: {exc}")

    candles = _df_to_candles(full_df)
    return {
        "source": "kite_api", "symbol": symbol.upper(), "exchange": exchange.upper(),
        "interval": interval, "from": from_dt.date().isoformat(),
        "to": to_dt.date().isoformat(), "rows_saved": rows_saved,
        "total": len(candles), "candles": candles,
    }


# ---------------------------------------------------------------------------
# Enriched candles — DB-computed derived columns
# ---------------------------------------------------------------------------

@app.get("/candles/enriched", tags=["Candles"])
def get_candles_enriched(
    symbol:    str = Query(...,   description="Trading symbol e.g. RELIANCE"),
    from_date: str = Query(...,   alias="from", description="Start date YYYY-MM-DD"),
    to_date:   str = Query(...,   alias="to",   description="End date YYYY-MM-DD"),
):
    """
    Return OHLCV candles with DB-computed columns from the ohlcv_enriched view:
    candle_range, body_size, is_bullish, daily_return_pct.

    Data must already be loaded via GET /candles first.
    """
    try:
        from_dt = _to_datetime(from_date, is_start=True)
        to_dt   = _to_datetime(to_date,   is_start=False)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if from_dt > to_dt:
        raise HTTPException(status_code=422, detail="from_date must be earlier than to_date")

    try:
        candles = get_enriched_candles(symbol, from_dt, to_dt)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")

    return {
        "symbol": symbol.upper(),
        "from":   from_dt.date().isoformat(),
        "to":     to_dt.date().isoformat(),
        "total":  len(candles),
        "candles": candles,
    }


# ---------------------------------------------------------------------------
# Ticker — watch list management
# ---------------------------------------------------------------------------

class WatchRequest(BaseModel):
    symbol:   str
    exchange: str = "NSE"


@app.post("/ticker/watch", tags=["Ticker"])
def add_symbol_to_watch(body: WatchRequest):
    """Add a symbol to the real-time watch list (starts collecting at next 09:15 IST)."""
    try:
        inserted = add_watched_symbol(body.symbol, body.exchange)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")
    return {
        "symbol":   body.symbol.upper(),
        "exchange": body.exchange.upper(),
        "status":   "added" if inserted else "already_watching",
    }


@app.delete("/ticker/watch/{symbol}", tags=["Ticker"])
def remove_symbol_from_watch(
    symbol:   str,
    exchange: str = Query("NSE", description="Exchange: NSE, BSE, NFO, MCX"),
):
    """Remove a symbol from the real-time watch list."""
    try:
        deleted = remove_watched_symbol(symbol, exchange)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()}:{exchange.upper()} is not in the watch list.",
        )
    return {"symbol": symbol.upper(), "exchange": exchange.upper(), "status": "removed"}


@app.get("/ticker/watch", tags=["Ticker"])
def list_watched_symbols():
    """List all symbols currently in the real-time watch list."""
    try:
        symbols = get_watched_symbols()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")
    return {"watched": symbols, "count": len(symbols)}


# ---------------------------------------------------------------------------
# Ticker — manual start / stop / status
# ---------------------------------------------------------------------------

@app.post("/ticker/start", tags=["Ticker"])
def start_ticker():
    """Manually start real-time polling (without waiting for 09:15 IST)."""
    try:
        status = ticker_manager.start_polling()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"status": status}


@app.post("/ticker/stop", tags=["Ticker"])
def stop_ticker():
    """Manually stop real-time polling."""
    return {"status": ticker_manager.stop_polling()}


@app.get("/ticker/status", tags=["Ticker"])
def ticker_status():
    """Return whether the ticker is running and how many symbols are watched."""
    try:
        count = len(get_watched_symbols())
    except Exception:
        count = 0
    return {
        "polling_active":   ticker_manager.is_running,
        "scheduler_active": ticker_manager.scheduler_running,
        "watched_count":    count,
        "auto_start":       "09:15 IST (Mon–Fri)",
        "auto_stop":        "15:30 IST (Mon–Fri)",
    }


# ---------------------------------------------------------------------------
# Ticks — query stored 1-second snapshots
# ---------------------------------------------------------------------------

@app.get("/ticks", tags=["Ticks"])
def get_ticks(
    symbol:  str = Query(...,   description="Trading symbol e.g. MRF"),
    from_dt: str = Query(...,   alias="from", description="Start: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"),
    to_dt:   str = Query(...,   alias="to",   description="End:   YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"),
    limit:   int = Query(3600,  description="Max rows (default 3600 = 1 hour, max 86400)", ge=1, le=86400),
):
    """Query real-time 1-second tick snapshots from stock_data for a symbol within a time range."""
    try:
        from_parsed = _to_datetime(from_dt, is_start=True)
        to_parsed   = _to_datetime(to_dt,   is_start=False)
    except (Exception, SystemExit) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if from_parsed > to_parsed:
        raise HTTPException(status_code=422, detail="from must be earlier than to")

    try:
        with get_db_connection() as conn:
            if not conn:
                raise HTTPException(status_code=503, detail="Database connection failed")
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT instrument_token, symbol, timestamp,
                       open, high, low, close, volume
                  FROM stock_data
                 WHERE symbol    = %s
                   AND timestamp BETWEEN %s AND %s
                 ORDER BY timestamp ASC
                 LIMIT %s
                """,
                (symbol.upper(), from_parsed, to_parsed, limit),
            )
            rows = cursor.fetchall()
            cursor.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database query failed: {exc}")

    ticks = [
        {
            "timestamp":        row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
            "open":             float(row["open"])   if row["open"]   is not None else None,
            "high":             float(row["high"])   if row["high"]   is not None else None,
            "low":              float(row["low"])    if row["low"]    is not None else None,
            "close":            float(row["close"])  if row["close"]  is not None else None,
            "volume":           int(row["volume"])   if row["volume"] is not None else 0,
            "instrument_token": int(row["instrument_token"]),
        }
        for row in rows
    ]

    return {
        "symbol":  symbol.upper(),
        "from":    from_parsed.isoformat(),
        "to":      to_parsed.isoformat(),
        "total":   len(ticks),
        "ticks":   ticks,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_candles_from_db(symbol: str, from_dt: datetime, to_dt: datetime) -> list[dict]:
    try:
        with get_db_connection() as conn:
            if not conn:
                raise HTTPException(status_code=503, detail="Database connection failed")
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                  FROM stock_data
                 WHERE symbol    = %s
                   AND timestamp BETWEEN %s AND %s
                 ORDER BY timestamp
                """,
                (symbol.upper(), from_dt, to_dt),
            )
            rows = cursor.fetchall()
            cursor.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database query failed: {exc}")

    return [
        {
            "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
            "open":   float(row["open"]),
            "high":   float(row["high"]),
            "low":    float(row["low"]),
            "close":  float(row["close"]),
            "volume": int(row["volume"]),
        }
        for row in rows
    ]


def _df_to_candles(df) -> list[dict]:
    result = []
    for ts, row in df.iterrows():
        entry = {
            "timestamp": ts.isoformat(),
            "open":   float(row["open"]),
            "high":   float(row["high"]),
            "low":    float(row["low"]),
            "close":  float(row["close"]),
            "volume": int(row["volume"]),
        }
        if "oi" in row:
            entry["oi"] = int(row["oi"])
        result.append(entry)
    return result
