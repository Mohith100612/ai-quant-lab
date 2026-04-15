"""
Zerodha Kite Connect — WebSocket tick streamer.

Uses KiteTicker (WebSocket) for true real-time streaming tick data instead
of REST polling.  Ticks arrive as they happen at the exchange level with
sub-second precision and full market depth (bid/ask, last traded price, OI).

Optimised producer-consumer design
------------------------------------
                                       bounded
  KiteTicker thread                  queue.Queue            N ws-store-* threads
  (WebSocket callbacks)              (max 10 000)           (configurable, default 2)
  on_ticks()                         ┌─────────┐            _store_loop()
    └─ put_nowait() ────────────────►│  batch  │──────────► _flush() → executemany → MySQL
       (drops + counts if full)      │  batch  │
                                     │   ...   │
                                     └─────────┘

Why multiple consumers?
  A single consumer means slow DB writes back-pressure the queue.
  Two consumers interleave their writes — each independently accumulates
  a mini-batch, inserts it, then goes back to draining.

Batch-size cap (BATCH_MAX = 500)
  Prevents a single giant transaction from locking the tick_data table.

Flush timeout (FLUSH_TIMEOUT = 0.5 s)
  In quiet periods (pre-market, low-traffic symbols) ticks arrive slowly.
  Without a timeout the last partial batch would sit in memory until the
  next tick arrives.  The timeout flushes it after half a second.

Back-pressure drop
  If the DB is so slow that the queue fills to QUEUE_MAX_SIZE, new tick
  batches are dropped (not the WebSocket thread blocked).  The count is
  tracked in ticks_dropped for monitoring.

Flush retry
  Transient MySQL errors (connection reset, brief lock) trigger up to
  FLUSH_RETRY_MAX retries with exponential backoff before the batch is
  logged as lost.

Scheduler
---------
WsTickerManager registers three APScheduler cron jobs:
    08:30 IST Mon–Fri  → _on_instrument_refresh()  (warm DB cache)
    09:15 IST Mon–Fri  → start()                   (WebSocket connect)
    15:30 IST Mon–Fri  → stop()                    (disconnect + drain)
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from typing import Optional

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from kiteconnect import KiteTicker

from auth import get_authenticated_kite
from database import (
    get_watched_symbols, save_ticks, save_order_update,
    remove_watched_symbol,
    get_instruments_for_exchange, upsert_instruments,
)
from log_config import get_logger, get_correlation_id, set_correlation_id

logger = get_logger(__name__)

IST = pytz.timezone("Asia/Kolkata")

_STOP_SENTINEL = None   # sentinel that tells each store thread to exit

MARKET_OPEN_HOUR,  MARKET_OPEN_MIN  = 9,  15   # 09:15 IST
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 15, 30   # 15:30 IST

# ── Subscription modes ───────────────────────────────────────────────────────
TICK_MODE_LTP   = "ltp"    # last traded price only
TICK_MODE_QUOTE = "quote"  # LTP + OHLC + volume (no market depth)
TICK_MODE_FULL  = "full"   # LTP + OHLC + volume + market depth + OI

_VALID_MODES = {TICK_MODE_LTP, TICK_MODE_QUOTE, TICK_MODE_FULL}

# ── Tuning knobs (override via subclass or constructor kwargs) ────────────────
QUEUE_MAX_SIZE  = 10_000   # hard cap — put_nowait drops when full
BATCH_MAX       = 500      # max ticks per flush  — caps transaction size
FLUSH_TIMEOUT   = 0.5      # seconds — flush partial batch even at low traffic
FLUSH_RETRY_MAX = 3        # flush retry attempts before logging ticks as lost


class WsTickerManager:
    """
    WebSocket-based real-time tick streamer using KiteTicker.

    start()            → authenticate, connect WebSocket, start N store threads
    stop()             → close WebSocket, send N sentinels, drain all threads
    start_scheduler()  → register 08:30 / 09:15 / 15:30 IST cron jobs (Mon–Fri)
    stop_scheduler()   → unregister jobs and shut down APScheduler

    Parameters
    ----------
    mode        Tick subscription mode (default: TICK_MODE_FULL)
    num_workers Number of parallel DB-writer threads (default: 2)
    """

    def __init__(
        self,
        mode:                str  = TICK_MODE_FULL,
        num_workers:         int  = 2,
        reconnect_max_tries: int  = 50,
        reconnect_max_delay: int  = 60,
        connect_timeout:     int  = 30,
        debug:               bool = False,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {sorted(_VALID_MODES)}")

        self._mode                = mode
        self._num_workers         = max(1, num_workers)
        self._reconnect_max_tries = reconnect_max_tries
        self._reconnect_max_delay = reconnect_max_delay
        self._connect_timeout     = connect_timeout
        self._debug               = debug

        # Bounded queue — producer never blocks; drops on overflow
        self._tick_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)

        self._kws: Optional[KiteTicker]      = None
        self._store_threads: list[threading.Thread] = []
        self._running         = threading.Event()
        self._last_tick_at:   Optional[datetime] = None

        # instrument_token → {"symbol": str, "exchange": str}
        self._token_meta: dict[int, dict] = {}

        # Per-symbol mode overrides: symbol.upper() → mode string
        # Symbols not listed here use self._mode as the default.
        self._symbol_modes: dict[str, str] = {}

        # Observability counters (thread-safe via GIL for int += 1)
        self._ticks_dropped = 0
        self._ticks_stored  = 0

        # 1-second throttle: instrument_token → last queued epoch timestamp
        # Ensures at most one tick per symbol per second is stored.
        self._last_tick_time: dict[int, float] = {}

        self._scheduler = BackgroundScheduler(timezone=IST)

    # -------------------------------------------------------------------------
    # Scheduler — market-hours auto-start / auto-stop
    # -------------------------------------------------------------------------

    def start_scheduler(self) -> None:
        """Register instrument-refresh / market-open / market-close cron jobs."""
        self._scheduler.add_job(
            self._on_instrument_refresh,
            CronTrigger(day_of_week="mon-fri", hour=8, minute=30, timezone=IST),
            id="instrument_refresh",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._on_market_open,
            CronTrigger(
                day_of_week="mon-fri",
                hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN,
                timezone=IST,
            ),
            id="ws_market_open",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._on_market_close,
            CronTrigger(
                day_of_week="mon-fri",
                hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN,
                timezone=IST,
            ),
            id="ws_market_close",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "WS scheduler started — instrument refresh 08:30 IST, "
            "market open %02d:%02d IST, close %02d:%02d IST (Mon–Fri).",
            MARKET_OPEN_HOUR, MARKET_OPEN_MIN,
            MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN,
        )

    def stop_scheduler(self) -> None:
        """Stop WebSocket (if running) and shut down APScheduler."""
        self.stop()
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("WS scheduler shut down.")

    # -------------------------------------------------------------------------
    # APScheduler callbacks
    # -------------------------------------------------------------------------

    def _on_market_open(self) -> None:
        logger.info("Market open at %s IST — starting WS ticker.", datetime.now(IST).strftime("%H:%M:%S"))
        try:
            self.start()
        except Exception as exc:
            logger.error("Failed to start WS ticker at market open: %s", exc)

    def _on_market_close(self) -> None:
        logger.info("Market close at %s IST — stopping WS ticker.", datetime.now(IST).strftime("%H:%M:%S"))
        self.stop()

    def _on_instrument_refresh(self) -> None:
        """
        Refresh instrument_master for watched exchanges at 08:30 IST so
        _on_connect uses the local DB cache instead of a live API call.
        """
        logger.info("Scheduled instrument master refresh starting.")
        try:
            kite = get_authenticated_kite()
        except Exception as exc:
            logger.error("Instrument refresh aborted — auth failed: %s", exc)
            return

        try:
            watched   = get_watched_symbols()
            exchanges = list({e["exchange"] for e in watched}) or ["NSE", "BSE"]
        except Exception:
            exchanges = ["NSE", "BSE"]

        for exchange in exchanges:
            try:
                instruments = kite.instruments(exchange=exchange)
                count       = upsert_instruments(exchange, instruments)
                logger.info("Instrument master refreshed for %s: %d rows upserted", exchange, count)
            except Exception as exc:
                logger.error("Instrument master refresh failed for %s: %s", exchange, exc)

    # -------------------------------------------------------------------------
    # Public control
    # -------------------------------------------------------------------------

    def start(self) -> str:
        """
        Authenticate, create KiteTicker, launch N consumer threads, and connect.
        Returns 'started' | 'already_running'.
        """
        if self._running.is_set():
            return "already_running"

        kite      = get_authenticated_kite()
        self._kws = KiteTicker(
            kite.api_key,
            kite.access_token,
            debug=self._debug,
            reconnect=True,
            reconnect_max_tries=self._reconnect_max_tries,
            reconnect_max_delay=self._reconnect_max_delay,
            connect_timeout=self._connect_timeout,
        )

        self._kws.on_ticks        = self._on_ticks
        self._kws.on_open         = self._on_open
        self._kws.on_connect      = self._on_connect
        self._kws.on_close        = self._on_close
        self._kws.on_error        = self._on_error
        self._kws.on_reconnect    = self._on_reconnect
        self._kws.on_noreconnect  = self._on_noreconnect
        self._kws.on_order_update = self._on_order_update

        # Drain leftover items from a previous run
        while not self._tick_queue.empty():
            try:
                self._tick_queue.get_nowait()
            except queue.Empty:
                break

        self._running.set()

        # Capture the current correlation ID so workers can log under the same ID
        parent_cid = get_correlation_id()

        # Start N consumer threads before connecting — they must be ready when
        # the first ticks arrive
        self._store_threads = [
            threading.Thread(
                target=self._store_loop,
                args=(parent_cid,),
                name=f"ws-store-{i}",
                daemon=True,
            )
            for i in range(self._num_workers)
        ]
        for t in self._store_threads:
            t.start()

        self._kws.connect(threaded=True)

        logger.info(
            "WebSocket ticker started (mode=%s, workers=%d).",
            self._mode.upper(), self._num_workers,
        )
        return "started"

    def stop(self) -> str:
        """
        Close the WebSocket, send one sentinel per consumer thread, and join all.
        Returns 'stopped' | 'not_running'.
        """
        if not self._running.is_set():
            return "not_running"

        self._running.clear()

        if self._kws:
            try:
                # stop_retry() halts the ReconnectingClientFactory so the
                # reactor won't attempt further reconnects after close().
                self._kws.stop_retry()
            except Exception:
                pass
            try:
                self._kws.close()
            except Exception:
                pass
            self._kws = None

        # Send exactly one sentinel per worker so every thread exits cleanly
        for _ in range(self._num_workers):
            self._tick_queue.put(_STOP_SENTINEL)

        for t in self._store_threads:
            if t.is_alive():
                t.join(timeout=10)
        self._store_threads = []

        logger.info(
            "WebSocket ticker stopped. Stored=%d  Dropped=%d",
            self._ticks_stored, self._ticks_dropped,
        )
        return "stopped"

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def scheduler_running(self) -> bool:
        return self._scheduler.running

    @property
    def queue_depth(self) -> int:
        """Tick batches waiting in the bounded queue."""
        return self._tick_queue.qsize()

    @property
    def queue_capacity(self) -> int:
        """Hard cap on queue size (QUEUE_MAX_SIZE)."""
        return QUEUE_MAX_SIZE

    @property
    def ticks_dropped(self) -> int:
        """Ticks dropped due to queue overflow since last start()."""
        return self._ticks_dropped

    @property
    def ticks_stored(self) -> int:
        """Ticks successfully written to MySQL since last start()."""
        return self._ticks_stored

    @property
    def subscribed_symbols(self) -> list[str]:
        return [m["symbol"] for m in self._token_meta.values()]

    @property
    def last_tick_at(self) -> Optional[str]:
        """ISO-8601 timestamp of the most recently received tick, or None."""
        return self._last_tick_at.isoformat() if self._last_tick_at else None

    @property
    def is_connected(self) -> bool:
        """True if the underlying WebSocket TCP connection is currently open."""
        return bool(self._kws and self._kws.is_connected())

    # -------------------------------------------------------------------------
    # Dynamic symbol management
    # -------------------------------------------------------------------------

    def remove_symbol(self, symbol: str, exchange: str = "NSE") -> bool:
        """
        Remove a symbol from the persistent watch list **and** immediately
        unsubscribe its instrument token if streaming is active.

        Returns True if the symbol was found and removed from the DB.
        """
        removed = remove_watched_symbol(symbol.upper(), exchange.upper())

        if removed and self._running.is_set() and self._kws:
            token = next(
                (tok for tok, meta in self._token_meta.items()
                 if meta["symbol"] == symbol.upper()
                 and meta["exchange"] == exchange.upper()),
                None,
            )
            if token:
                try:
                    self._kws.unsubscribe([token])
                    del self._token_meta[token]
                    logger.info(
                        "Unsubscribed %s:%s (token=%d) — live effect.",
                        symbol.upper(), exchange.upper(), token,
                    )
                except Exception as exc:
                    logger.warning("Unsubscribe failed for %s: %s", symbol.upper(), exc)

        return removed

    def set_symbol_modes(self, modes: dict[str, str]) -> None:
        """
        Override the subscription mode for specific symbols.

        Example
        -------
        ``{"RELIANCE": "full", "INFY": "quote", "TCS": "ltp"}``

        Symbols not listed continue to use the default ``mode`` passed
        to the constructor.  Takes effect on the **next** connect / reconnect.
        """
        for sym, m in modes.items():
            if m not in _VALID_MODES:
                raise ValueError(f"Invalid mode '{m}' for {sym}. Choose from: {sorted(_VALID_MODES)}")
            self._symbol_modes[sym.upper()] = m

    # -------------------------------------------------------------------------
    # KiteTicker callbacks — producer side  (KiteTicker's internal thread)
    # -------------------------------------------------------------------------

    def _on_open(self, ws) -> None:
        """Fires after the WebSocket TCP handshake, before on_connect."""
        logger.debug("WebSocket handshake complete (on_open).")

    def _on_connect(self, ws, response) -> None:
        """
        Resolve watch-list symbols → instrument tokens and subscribe.

        Resolution order
        ----------------
        1. Local instrument_master DB cache  (populated by 08:30 IST job)
        2. Live kite.instruments() API call  (on cache miss; also warms cache)
        """
        logger.info("WebSocket connected.")
        watched = get_watched_symbols()
        if not watched:
            logger.info("No symbols in watch list — nothing to subscribe.")
            return

        self._token_meta.clear()
        tokens_to_subscribe: list[int] = []

        by_exchange: dict[str, list[str]] = {}
        for entry in watched:
            by_exchange.setdefault(entry["exchange"], []).append(entry["symbol"])

        for exchange, symbols in by_exchange.items():
            try:
                token_map = _build_token_map_from_db(exchange)

                if not token_map:
                    logger.info("Instrument cache empty for %s — fetching from Kite API", exchange)
                    kite      = get_authenticated_kite()
                    live_insts = kite.instruments(exchange=exchange)
                    upsert_instruments(exchange, live_insts)
                    token_map = {
                        inst["tradingsymbol"].upper(): inst["instrument_token"]
                        for inst in live_insts
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
                        logger.warning("No instrument token for %s:%s", exchange, symbol)
            except Exception as exc:
                logger.error("Instrument lookup failed for %s: %s", exchange, exc)

        if tokens_to_subscribe:
            ws.subscribe(tokens_to_subscribe)

            # Apply per-symbol mode overrides; fall back to self._mode for
            # any symbol not in self._symbol_modes.
            modes_to_tokens: dict[str, list[int]] = {}
            for token in tokens_to_subscribe:
                sym  = self._token_meta[token]["symbol"]
                mode = self._symbol_modes.get(sym, self._mode)
                modes_to_tokens.setdefault(mode, []).append(token)

            for mode, tokens_for_mode in modes_to_tokens.items():
                ws.set_mode(mode, tokens_for_mode)

            mode_summary = {
                m: [self._token_meta[t]["symbol"] for t in toks]
                for m, toks in modes_to_tokens.items()
            }
            logger.info(
                "Subscribed to %d instrument(s) — modes: %s",
                len(tokens_to_subscribe), mode_summary,
            )

    def _on_ticks(self, ws, ticks: list[dict]) -> None:
        """
        Producer callback — must return immediately, never block.

        Applies a 1-second per-symbol throttle: only the first tick received
        for each instrument token within any 1-second window is queued; the
        rest are silently discarded.  This satisfies the requirement to store
        stock data at a 1-second interval.

        Uses put_nowait() so the WebSocket thread is never stalled by a
        full queue.  Dropped ticks are counted for monitoring.
        """
        if not ticks:
            return

        now = time.time()
        throttled = [
            tick for tick in ticks
            if now - self._last_tick_time.get(tick.get("instrument_token", 0), 0) >= 1.0
        ]
        for tick in throttled:
            self._last_tick_time[tick.get("instrument_token", 0)] = now

        if not throttled:
            return

        try:
            self._tick_queue.put_nowait(throttled)
            self._last_tick_at = datetime.now(IST)
        except queue.Full:
            self._ticks_dropped += len(throttled)
            logger.warning(
                "Tick queue full (%d/%d) — dropped %d tick(s). "
                "Total dropped this session: %d",
                self._tick_queue.qsize(), QUEUE_MAX_SIZE,
                len(throttled), self._ticks_dropped,
            )

    def _on_close(self, ws, code, reason) -> None:
        logger.warning("WebSocket closed: code=%s  reason=%s", code, reason)

    def _on_error(self, ws, code, reason) -> None:
        logger.error("WebSocket error: code=%s  reason=%s", code, reason)

    def _on_reconnect(self, ws, attempts_count) -> None:
        logger.info("WebSocket reconnecting (attempt %d)...", attempts_count)

    def _on_noreconnect(self, ws) -> None:
        """Max reconnects exhausted — signal ALL consumer threads to exit."""
        logger.error("WebSocket max reconnect attempts reached — stopping.")
        self._running.clear()
        # One sentinel per worker so every thread wakes up and exits
        for _ in range(self._num_workers):
            self._tick_queue.put(_STOP_SENTINEL)

    def _on_order_update(self, ws, data: dict) -> None:
        """
        Fires when an order placed via Kite changes state (filled, rejected,
        cancelled, etc.).  Logs the event and persists it to ``order_updates``.
        """
        order_id = data.get("order_id", "?")
        status   = data.get("status", "?")
        symbol   = data.get("tradingsymbol", "?")
        logger.info(
            "Order update: %s  %s  status=%s",
            order_id, symbol, status,
            extra={"order_id": order_id, "status": status, "symbol": symbol},
        )
        try:
            row_id = save_order_update(data)
            logger.debug("Order update persisted (id=%d).", row_id)
        except Exception as exc:
            logger.error("Failed to save order update %s: %s", order_id, exc)

    # -------------------------------------------------------------------------
    # Store loop — consumer side  (runs in each ws-store-N thread)
    # -------------------------------------------------------------------------

    def _store_loop(self, correlation_id: str = "") -> None:
        """
        Each consumer thread independently drains the shared queue and flushes
        to MySQL.

        Accumulation strategy
        ---------------------
        1. Block on queue.get(timeout=FLUSH_TIMEOUT).
        2. On arrival, drain all immediately-available additional batches via
           get_nowait() until either the queue is empty or BATCH_MAX is reached.
        3. Flush the accumulated batch to MySQL.
        4. On timeout (queue empty for FLUSH_TIMEOUT seconds), flush whatever
           partial batch has accumulated — ensures low-traffic periods don't
           leave ticks stranded in memory.
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        logger.info("WS store loop started.")

        accumulated: list[dict] = []

        while True:
            # ── wait for first item (with timeout for periodic partial flush) ──
            try:
                first = self._tick_queue.get(timeout=FLUSH_TIMEOUT)
            except queue.Empty:
                # Timeout — flush whatever we have accumulated so far
                if accumulated:
                    self._flush(accumulated)
                    accumulated = []
                # If we were told to stop between ticks, exit now
                if not self._running.is_set():
                    break
                continue

            if first is _STOP_SENTINEL:
                # Flush the last partial batch before exiting
                if accumulated:
                    self._flush(accumulated)
                logger.info("WS store loop received sentinel — exiting.")
                break

            accumulated.extend(first)

            # ── drain additional batches without blocking ──────────────────────
            while len(accumulated) < BATCH_MAX:
                try:
                    more = self._tick_queue.get_nowait()
                except queue.Empty:
                    break
                if more is _STOP_SENTINEL:
                    self._flush(accumulated)
                    logger.info("WS store loop draining on sentinel — exiting.")
                    return
                accumulated.extend(more)

            # ── flush when batch is at or above the cap ────────────────────────
            if len(accumulated) >= BATCH_MAX:
                self._flush(accumulated)
                accumulated = []

        logger.info("WS store loop exited.")

    def _flush(self, raw_ticks: list[dict]) -> None:
        """
        Normalise and persist a batch of raw KiteTicker ticks.

        Retries up to FLUSH_RETRY_MAX times with exponential backoff on
        transient MySQL errors.  Ticks are logged as lost only after all
        retries are exhausted.
        """
        try:
            normalized = _normalize_ticks(raw_ticks, self._token_meta)
            if not normalized:
                return
        except Exception as exc:
            logger.error("Tick normalisation failed: %s", exc, exc_info=True)
            return

        last_exc: Exception | None = None
        for attempt in range(1, FLUSH_RETRY_MAX + 1):
            try:
                saved = save_ticks(normalized)
                self._ticks_stored += saved
                logger.debug("WS stored %d tick(s) (attempt %d).", saved, attempt)
                return
            except Exception as exc:
                last_exc = exc
                if attempt < FLUSH_RETRY_MAX:
                    wait = 0.1 * (2 ** (attempt - 1))   # 0.1 s, 0.2 s
                    logger.warning(
                        "Tick flush attempt %d/%d failed: %s — retrying in %.2fs",
                        attempt, FLUSH_RETRY_MAX, exc, wait,
                    )
                    time.sleep(wait)

        logger.error(
            "Tick flush failed after %d attempts — %d tick(s) lost: %s",
            FLUSH_RETRY_MAX, len(normalized), last_exc,
        )


# ---------------------------------------------------------------------------
# Tick normalizer
# ---------------------------------------------------------------------------

def _normalize_ticks(
    raw_ticks: list[dict],
    token_meta: dict[int, dict],
) -> list[dict]:
    """Map KiteTicker wire format → save_ticks() dict format."""
    fallback_captured_at = datetime.now(IST).replace(tzinfo=None)
    result               = []

    for t in raw_ticks:
        token = t.get("instrument_token", 0)
        meta  = token_meta.get(token, {})
        ohlc  = t.get("ohlc", {})

        # exchange_timestamp / last_trade_time arrive as tz-aware datetimes
        # from KiteTicker; strip tzinfo so MySQL DATETIME(3) accepts them.
        def _strip_tz(dt):
            if dt is None:
                return None
            return dt.replace(tzinfo=None) if hasattr(dt, "tzinfo") else dt

        # Use exchange_timestamp per tick so each row gets its true event time.
        # Fall back to last_trade_time, then to a single wall-clock snapshot if
        # neither is present (e.g. LTP-mode ticks that omit full metadata).
        exchange_ts   = _strip_tz(t.get("exchange_timestamp"))
        last_trade_ts = _strip_tz(t.get("last_trade_time"))
        captured_at   = exchange_ts or last_trade_ts or fallback_captured_at

        result.append({
            "symbol":                meta.get("symbol", str(token)),
            "exchange":              meta.get("exchange", ""),
            "instrument_token":      token,
            "last_price":            t.get("last_price", 0.0),
            "volume":                t.get("volume_traded", 0),
            "buy_quantity":          t.get("total_buy_quantity", 0),
            "sell_quantity":         t.get("total_sell_quantity", 0),
            "open":                  ohlc.get("open"),
            "high":                  ohlc.get("high"),
            "low":                   ohlc.get("low"),
            "close":                 ohlc.get("close"),
            "change":                t.get("change"),
            "last_traded_quantity":  t.get("last_traded_quantity", 0),
            "avg_traded_price":      t.get("average_traded_price"),
            "oi":                    t.get("oi", 0),
            "oi_day_high":           t.get("oi_day_high", 0),
            "oi_day_low":            t.get("oi_day_low", 0),
            "last_trade_time":       last_trade_ts,
            "exchange_timestamp":    exchange_ts,
            "depth":                 t.get("depth"),
            "captured_at":           captured_at,
        })

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_token_map_from_db(exchange: str) -> dict[str, int]:
    """
    Return ``{tradingsymbol.upper(): instrument_token}`` from the local cache.
    Returns an empty dict when the cache is empty or unavailable.
    """
    rows = get_instruments_for_exchange(exchange)
    return {row["tradingsymbol"].upper(): int(row["instrument_token"]) for row in rows}


# Module-level singleton — imported by api.py
ws_ticker_manager = WsTickerManager()
