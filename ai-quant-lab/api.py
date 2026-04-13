"""
Zerodha Kite Connect — FastAPI server.

Endpoints
---------
GET  /auth/login-url               → Kite login URL to open in browser
POST /auth/session                 → Exchange request_token for access_token

GET  /candles                      → Check DB first; if missing fetch from Kite, save, return

POST /ws-ticker/watch              → Add a symbol to the WebSocket watch list
DEL  /ws-ticker/watch/{symbol}     → Remove a symbol from the watch list
GET  /ws-ticker/watch              → List all watched symbols
POST /ws-ticker/start              → Manually start WebSocket streaming
POST /ws-ticker/stop               → Manually stop WebSocket streaming
GET  /ws-ticker/status             → Streaming state, subscribed symbols, scheduler info

GET  /ticks                        → Query stored real-time tick snapshots

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
    ensure_table, data_exists, save_to_db, _get_connection,
    ensure_tick_tables,
    get_watched_symbols, add_watched_symbol, remove_watched_symbol,
    query_tick_data,
)
from ws_ticker import ws_ticker_manager

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

    ws_ticker_manager.start_scheduler()    # 09:15 / 15:30 IST (Mon–Fri)

    yield   # ← application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    ws_ticker_manager.stop_scheduler()


app = FastAPI(
    title="Zerodha Kite Connect API",
    description="REST wrapper around Kite Connect with WebSocket real-time tick streaming",
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
# Candles
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
    except Exception as exc:
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
# WebSocket Ticker — watch list
# ---------------------------------------------------------------------------

class WatchRequest(BaseModel):
    symbol:   str
    exchange: str = "NSE"


@app.post("/ws-ticker/watch", tags=["WebSocket Ticker"])
def add_symbol_to_watch(body: WatchRequest):
    """Add a symbol to the WebSocket watch list. Takes effect on the next connect or reconnect."""
    try:
        inserted = add_watched_symbol(body.symbol, body.exchange)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")
    return {
        "symbol":   body.symbol.upper(),
        "exchange": body.exchange.upper(),
        "status":   "added" if inserted else "already_watching",
    }


@app.delete("/ws-ticker/watch/{symbol}", tags=["WebSocket Ticker"])
def remove_symbol_from_watch(
    symbol:   str,
    exchange: str = Query("NSE", description="Exchange: NSE, BSE, NFO, MCX"),
):
    """Remove a symbol from the WebSocket watch list."""
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


@app.get("/ws-ticker/watch", tags=["WebSocket Ticker"])
def list_watched_symbols():
    """List all symbols currently in the WebSocket watch list."""
    try:
        symbols = get_watched_symbols()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")
    return {"watched": symbols, "count": len(symbols)}


# ---------------------------------------------------------------------------
# WebSocket Ticker — control & status
# ---------------------------------------------------------------------------

@app.post("/ws-ticker/start", tags=["WebSocket Ticker"])
def start_ws_ticker():
    """
    Manually start KiteTicker WebSocket streaming.
    Resolves all watched symbols to instrument tokens and subscribes.
    Ticks are persisted in real time via the producer-consumer queue.
    """
    try:
        status = ws_ticker_manager.start()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"status": status}


@app.post("/ws-ticker/stop", tags=["WebSocket Ticker"])
def stop_ws_ticker():
    """Manually stop KiteTicker WebSocket streaming and drain the store queue."""
    return {"status": ws_ticker_manager.stop()}


@app.get("/ws-ticker/status", tags=["WebSocket Ticker"])
def ws_ticker_status():
    """Return streaming state, subscribed symbols, queue depth, and scheduler info."""
    try:
        count = len(get_watched_symbols())
    except Exception:
        count = 0
    return {
        "streaming_active":   ws_ticker_manager.is_running,
        "scheduler_active":   ws_ticker_manager.scheduler_running,
        "subscribed_symbols": ws_ticker_manager.subscribed_symbols,
        "watched_count":      count,
        "queue_depth":        ws_ticker_manager.queue_depth,
        "last_tick_at":       ws_ticker_manager.last_tick_at,
        "auto_start":         "09:15 IST (Mon–Fri)",
        "auto_stop":          "15:30 IST (Mon–Fri)",
    }


# ---------------------------------------------------------------------------
# Ticks — query stored real-time snapshots
# ---------------------------------------------------------------------------

@app.get("/ticks", tags=["Ticks"])
def get_ticks(
    symbol:  str = Query(...,   description="Trading symbol e.g. MRF"),
    from_dt: str = Query(...,   alias="from", description="Start: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"),
    to_dt:   str = Query(...,   alias="to",   description="End:   YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"),
    limit:   int = Query(3600,  description="Max rows (default 3600 = 1 hour, max 86400)", ge=1, le=86400),
):
    """Query real-time tick snapshots from tick_data for a symbol within a time range."""
    try:
        from_parsed = _to_datetime(from_dt, is_start=True)
        to_parsed   = _to_datetime(to_dt,   is_start=False)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if from_parsed > to_parsed:
        raise HTTPException(status_code=422, detail="from must be earlier than to")

    try:
        rows = query_tick_data(symbol, from_parsed, to_parsed, limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database query failed: {exc}")

    ticks = [
        {
            "timestamp":        row["captured_at"].isoformat() if hasattr(row["captured_at"], "isoformat") else str(row["captured_at"]),
            "instrument_token": int(row["instrument_token"]),
            "last_price":       float(row["last_price"]),
            "open":             float(row["open"])        if row["open"]         is not None else None,
            "high":             float(row["high"])        if row["high"]         is not None else None,
            "low":              float(row["low"])         if row["low"]          is not None else None,
            "close":            float(row["close"])       if row["close"]        is not None else None,
            "volume":           int(row["volume"])        if row["volume"]       is not None else 0,
            "buy_quantity":     int(row["buy_quantity"])  if row["buy_quantity"] is not None else 0,
            "sell_quantity":    int(row["sell_quantity"]) if row["sell_quantity"] is not None else 0,
            "change_pct":       float(row["change_pct"]) if row["change_pct"]   is not None else None,
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
        conn   = _get_connection()
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
        conn.close()
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
