"""
Zerodha Kite Connect — WebSocket tick streamer.

Uses KiteTicker (WebSocket) for true real-time streaming tick data instead
of REST polling.  Ticks arrive as they happen at the exchange level with
sub-second precision and full market depth (bid/ask, last traded price, OI).

Thread design (producer-consumer)
----------------------------------
  KiteTicker internal thread         queue.Queue        ws-store thread
  (WebSocket callbacks)              (buffer)           (consumer)
  on_ticks() → _tick_queue.put()  ──────────────────→  _flush() → MySQL

The on_ticks() callback is called by KiteTicker's own internal thread.
It puts the raw batch into the Queue immediately and returns — the WebSocket
thread is never blocked by a slow DB write.

The ws-store thread drains the Queue, accumulates multiple batches into one
bulk write, then inserts into the tick_data table.

Scheduler
---------
WsTickerManager registers two APScheduler cron jobs:
    09:15 IST Mon–Fri  → start()   (WebSocket connect + subscription)
    15:30 IST Mon–Fri  → stop()    (disconnect + drain queue)

Call start_scheduler() at application startup and stop_scheduler() at shutdown.

Usage
-----
    from ws_ticker import ws_ticker_manager
    ws_ticker_manager.start_scheduler()   # called at FastAPI startup
    ws_ticker_manager.stop_scheduler()    # called at FastAPI shutdown
    ws_ticker_manager.start()             # manual start (skips scheduler)
    ws_ticker_manager.stop()              # manual stop

Subscription modes
------------------
    TICK_MODE_LTP   = "ltp"    — last traded price only
    TICK_MODE_QUOTE = "quote"  — LTP + OHLC + volume (no depth)
    TICK_MODE_FULL  = "full"   — LTP + OHLC + volume + market depth + OI

    Default: TICK_MODE_FULL.  Pass mode= to WsTickerManager() to change.

Instrument tokens
-----------------
KiteTicker subscribes by integer instrument_token, not by symbol string.
On every connect / reconnect, _on_connect() resolves the watch list symbols
to their tokens via kite.instruments() and subscribes automatically.
"""

from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime
from typing import Optional

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from kiteconnect import KiteTicker

from auth import get_authenticated_kite
from database import get_watched_symbols, save_ticks

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

_STOP_SENTINEL = None   # sentinel value that tells the store thread to exit

MARKET_OPEN_HOUR,  MARKET_OPEN_MIN  = 9,  15   # 09:15 IST
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 15, 30   # 15:30 IST

# Subscription modes — passed directly to KiteTicker.set_mode()
TICK_MODE_LTP   = "ltp"    # last traded price only
TICK_MODE_QUOTE = "quote"  # LTP + OHLC + volume (no market depth)
TICK_MODE_FULL  = "full"   # LTP + OHLC + volume + market depth + OI

_VALID_MODES = {TICK_MODE_LTP, TICK_MODE_QUOTE, TICK_MODE_FULL}


class WsTickerManager:
    """
    WebSocket-based real-time tick streamer using KiteTicker.

    start()            → authenticate, connect WebSocket, start store thread
    stop()             → close WebSocket, drain store thread
    start_scheduler()  → register 09:15 / 15:30 IST cron jobs (Mon–Fri)
    stop_scheduler()   → unregister jobs and shut down APScheduler
    """

    def __init__(self, mode: str = TICK_MODE_FULL) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose from: {sorted(_VALID_MODES)}"
            )
        self._mode            = mode
        self._tick_queue: queue.Queue           = queue.Queue()
        self._kws:        Optional[KiteTicker]  = None
        self._store_thread: Optional[threading.Thread] = None
        self._running         = threading.Event()
        self._last_tick_at:   Optional[datetime] = None
        # instrument_token → {"symbol": str, "exchange": str}
        self._token_meta: dict[int, dict] = {}
        self._scheduler = BackgroundScheduler(timezone=IST)

    # ------------------------------------------------------------------
    # Scheduler — market-hours auto-start / auto-stop
    # ------------------------------------------------------------------

    def start_scheduler(self) -> None:
        """Register market-open / market-close cron jobs and start APScheduler."""
        self._scheduler.add_job(
            self._on_market_open,
            CronTrigger(
                day_of_week="mon-fri",
                hour=MARKET_OPEN_HOUR,
                minute=MARKET_OPEN_MIN,
                timezone=IST,
            ),
            id="ws_market_open",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._on_market_close,
            CronTrigger(
                day_of_week="mon-fri",
                hour=MARKET_CLOSE_HOUR,
                minute=MARKET_CLOSE_MIN,
                timezone=IST,
            ),
            id="ws_market_close",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "WS scheduler started — market open %02d:%02d IST, "
            "close %02d:%02d IST (Mon–Fri).",
            MARKET_OPEN_HOUR, MARKET_OPEN_MIN,
            MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN,
        )

    def stop_scheduler(self) -> None:
        """Stop WebSocket (if running) and shut down APScheduler."""
        self.stop()
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("WS scheduler shut down.")

    # ------------------------------------------------------------------
    # APScheduler callbacks
    # ------------------------------------------------------------------

    def _on_market_open(self) -> None:
        logger.info(
            "Market open at %s IST — starting WS ticker.",
            datetime.now(IST).strftime("%H:%M:%S"),
        )
        try:
            self.start()
        except Exception as exc:
            logger.error("Failed to start WS ticker at market open: %s", exc)

    def _on_market_close(self) -> None:
        logger.info(
            "Market close at %s IST — stopping WS ticker.",
            datetime.now(IST).strftime("%H:%M:%S"),
        )
        self.stop()

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def start(self) -> str:
        """
        Authenticate, create KiteTicker, start the store thread, and connect.
        Returns 'started' | 'already_running'.
        """
        if self._running.is_set():
            return "already_running"

        kite = get_authenticated_kite()
        self._kws = KiteTicker(kite.api_key, kite.access_token)

        # Wire up all callbacks
        self._kws.on_ticks       = self._on_ticks
        self._kws.on_connect     = self._on_connect
        self._kws.on_close       = self._on_close
        self._kws.on_error       = self._on_error
        self._kws.on_reconnect   = self._on_reconnect
        self._kws.on_noreconnect = self._on_noreconnect

        # Flush any leftover items from a previous run
        while not self._tick_queue.empty():
            try:
                self._tick_queue.get_nowait()
            except queue.Empty:
                break

        self._running.set()

        # Consumer thread — started before connect so it is ready immediately
        self._store_thread = threading.Thread(
            target=self._store_loop,
            name="ws-store",
            daemon=True,
        )
        self._store_thread.start()

        # KiteTicker manages its own internal WebSocket thread
        self._kws.connect(threaded=True)

        logger.info("WebSocket ticker started (mode=%s).", self._mode.upper())
        return "started"

    def stop(self) -> str:
        """
        Close the WebSocket and drain the store thread cleanly.
        Returns 'stopped' | 'not_running'.
        """
        if not self._running.is_set():
            return "not_running"

        self._running.clear()

        if self._kws:
            try:
                self._kws.close()
            except Exception:
                pass
            self._kws = None

        # Signal the store thread to finish draining and exit
        self._tick_queue.put(_STOP_SENTINEL)
        if self._store_thread and self._store_thread.is_alive():
            self._store_thread.join(timeout=10)

        logger.info("WebSocket ticker stopped.")
        return "stopped"

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def scheduler_running(self) -> bool:
        return self._scheduler.running

    @property
    def queue_depth(self) -> int:
        """Number of tick batches waiting in the queue (useful for monitoring)."""
        return self._tick_queue.qsize()

    @property
    def subscribed_symbols(self) -> list[str]:
        return [m["symbol"] for m in self._token_meta.values()]

    @property
    def last_tick_at(self) -> Optional[str]:
        """ISO-8601 timestamp of the most recently received tick, or None."""
        return self._last_tick_at.isoformat() if self._last_tick_at else None

    # ------------------------------------------------------------------
    # KiteTicker callbacks — producer side (run in KiteTicker's thread)
    # ------------------------------------------------------------------

    def _on_connect(self, ws, response) -> None:
        """
        Called once after the WebSocket handshake succeeds.
        Resolves symbol strings → instrument tokens and subscribes.
        KiteTicker also calls this automatically on every reconnect.
        """
        logger.info("WebSocket connected.")
        watched = get_watched_symbols()
        if not watched:
            logger.info("No symbols in watch list — nothing to subscribe.")
            return

        kite = get_authenticated_kite()
        self._token_meta.clear()
        tokens_to_subscribe: list[int] = []

        # Group by exchange so we call kite.instruments() once per exchange
        by_exchange: dict[str, list[str]] = {}
        for entry in watched:
            by_exchange.setdefault(entry["exchange"], []).append(entry["symbol"])

        for exchange, symbols in by_exchange.items():
            try:
                instruments = kite.instruments(exchange=exchange)
                token_map = {
                    inst["tradingsymbol"].upper(): inst["instrument_token"]
                    for inst in instruments
                }
                for symbol in symbols:
                    token = token_map.get(symbol.upper())
                    if token:
                        self._token_meta[token] = {
                            "symbol":   symbol.upper(),
                            "exchange": exchange.upper(),
                        }
                        tokens_to_subscribe.append(token)
                    else:
                        logger.warning(
                            "No instrument token found for %s:%s", exchange, symbol
                        )
            except Exception as exc:
                logger.error(
                    "Instrument lookup failed for exchange %s: %s", exchange, exc
                )

        if tokens_to_subscribe:
            ws.subscribe(tokens_to_subscribe)
            ws.set_mode(self._mode, tokens_to_subscribe)
            logger.info(
                "Subscribed to %d instrument(s) in %s mode: %s",
                len(tokens_to_subscribe),
                self._mode.upper(),
                [self._token_meta[t]["symbol"] for t in tokens_to_subscribe],
            )

    def _on_ticks(self, ws, ticks: list[dict]) -> None:
        """
        Called by KiteTicker's thread on every tick batch pushed by the exchange.
        Puts the batch into the queue and records the arrival time.
        Must return immediately — never block here.
        """
        if ticks:
            self._tick_queue.put(ticks)
            self._last_tick_at = datetime.now(IST)

    def _on_close(self, ws, code, reason) -> None:
        logger.warning("WebSocket closed: code=%s  reason=%s", code, reason)

    def _on_error(self, ws, code, reason) -> None:
        logger.error("WebSocket error: code=%s  reason=%s", code, reason)

    def _on_reconnect(self, ws, attempts_count) -> None:
        logger.info("WebSocket reconnecting (attempt %d)...", attempts_count)

    def _on_noreconnect(self, ws) -> None:
        """Max reconnect attempts exhausted — shut down gracefully."""
        logger.error("WebSocket max reconnect attempts reached — stopping.")
        self._running.clear()
        self._tick_queue.put(_STOP_SENTINEL)

    # ------------------------------------------------------------------
    # Store loop — consumer side (runs in ws-store thread)
    # ------------------------------------------------------------------

    def _store_loop(self) -> None:
        """
        Drains tick batches from the queue and bulk-inserts into MySQL.

        Accumulation strategy: after receiving the first batch from a blocking
        get(), immediately drain all additional batches that have piled up
        via non-blocking get_nowait() calls.  This turns many small inserts
        into one larger bulk insert when the exchange pushes ticks faster than
        the DB can absorb them.

        Exits when it receives the _STOP_SENTINEL value.
        """
        logger.info("WS store loop started.")
        while True:
            # Block until at least one batch arrives
            first = self._tick_queue.get()
            if first is _STOP_SENTINEL:
                logger.info("WS store loop received sentinel — exiting.")
                break

            # Drain any additional batches that queued up while we waited
            accumulated: list[dict] = list(first)
            while True:
                try:
                    more = self._tick_queue.get_nowait()
                except queue.Empty:
                    break
                if more is _STOP_SENTINEL:
                    # Flush what we have, then exit
                    self._flush(accumulated)
                    logger.info("WS store loop draining on sentinel — exiting.")
                    return
                accumulated.extend(more)

            self._flush(accumulated)
        logger.info("WS store loop exited.")

    def _flush(self, raw_ticks: list[dict]) -> None:
        """Normalize and persist a batch of raw KiteTicker ticks to tick_data."""
        try:
            normalized = _normalize_ticks(raw_ticks, self._token_meta)
            if normalized:
                saved = save_ticks(normalized)
                logger.debug("WS stored %d tick(s).", saved)
        except Exception as exc:
            logger.error("WS DB store error: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Tick normalizer — maps KiteTicker wire format → save_ticks() format
# ---------------------------------------------------------------------------

def _normalize_ticks(
    raw_ticks: list[dict],
    token_meta: dict[int, dict],
) -> list[dict]:
    """
    Map KiteTicker raw tick fields → the dict format expected by save_ticks().

    KiteTicker field       save_ticks() field
    ─────────────────────  ──────────────────
    instrument_token    →  instrument_token
    last_price          →  last_price
    volume_traded       →  volume
    buy_quantity        →  buy_quantity
    sell_quantity       →  sell_quantity
    ohlc.open/high/low/close → open / high / low / close
    change              →  change
    (token_meta lookup) →  symbol, exchange
    datetime.now(IST)   →  captured_at  (naive IST for MySQL storage)
    """
    captured_at = datetime.now(IST).replace(tzinfo=None)
    result      = []

    for t in raw_ticks:
        token = t.get("instrument_token", 0)
        meta  = token_meta.get(token, {})
        ohlc  = t.get("ohlc", {})
        result.append({
            "symbol":           meta.get("symbol", str(token)),
            "exchange":         meta.get("exchange", ""),
            "instrument_token": token,
            "last_price":       t.get("last_price", 0.0),
            "volume":           t.get("volume_traded", 0),
            "buy_quantity":     t.get("buy_quantity", 0),
            "sell_quantity":    t.get("sell_quantity", 0),
            "open":             ohlc.get("open"),
            "high":             ohlc.get("high"),
            "low":              ohlc.get("low"),
            "close":            ohlc.get("close"),
            "change":           t.get("change"),
            "captured_at":      captured_at,
        })

    return result


# Module-level singleton — imported by api.py
ws_ticker_manager = WsTickerManager()
