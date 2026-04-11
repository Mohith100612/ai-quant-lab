"""
Zerodha Kite Connect — Real-time stock quote poller.

How it works
------------
  - Calls kite.quote() every 1 second for all symbols in the watched_symbols table.
  - Stores each snapshot into tick_data via save_ticks().
  - APScheduler fires two cron jobs daily (Mon–Fri, IST):
      09:15 → start_polling()
      15:30 → stop_polling()
  - The watch list is re-read from the DB on every poll cycle, so
    adding/removing symbols via the API takes effect within 1 second.

Usage
-----
  from ticker import ticker_manager

  # Call once at FastAPI startup:
  ticker_manager.start_scheduler()

  # Call once at FastAPI shutdown:
  ticker_manager.stop_scheduler()

  # Manual control (also available via API endpoints):
  ticker_manager.start_polling()
  ticker_manager.stop_polling()
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from auth import get_authenticated_kite
from database import get_watched_symbols, save_ticks

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

POLL_INTERVAL_SEC: float = 1.0   # seconds between each kite.quote() call

MARKET_OPEN_HOUR,  MARKET_OPEN_MIN  = 9,  15   # 09:15 IST
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 15, 30   # 15:30 IST


class TickerManager:
    """
    Owns the 1-second polling thread and the APScheduler that starts/stops it.
    Instantiated once at module level as `ticker_manager`.
    """

    def __init__(self) -> None:
        self._running   = threading.Event()          # set = polling is active
        self._thread:   threading.Thread | None = None
        self._kite      = None                       # authenticated KiteConnect instance
        self._lock      = threading.Lock()           # guards _kite access
        self._scheduler = BackgroundScheduler(timezone=IST)

    # ------------------------------------------------------------------
    # Scheduler (call at FastAPI startup / shutdown)
    # ------------------------------------------------------------------

    def start_scheduler(self) -> None:
        """Register market-open / market-close cron jobs and start APScheduler."""
        self._scheduler.add_job(
            self._on_market_open,
            CronTrigger(day_of_week="mon-fri", hour=MARKET_OPEN_HOUR,
                        minute=MARKET_OPEN_MIN, timezone=IST),
            id="market_open",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._on_market_close,
            CronTrigger(day_of_week="mon-fri", hour=MARKET_CLOSE_HOUR,
                        minute=MARKET_CLOSE_MIN, timezone=IST),
            id="market_close",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "Scheduler started — market open %02d:%02d IST, close %02d:%02d IST (Mon-Fri).",
            MARKET_OPEN_HOUR, MARKET_OPEN_MIN,
            MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN,
        )

    def stop_scheduler(self) -> None:
        """Stop polling (if running) and shut down APScheduler."""
        self.stop_polling()
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler shut down.")

    # ------------------------------------------------------------------
    # Polling control (also called by API endpoints)
    # ------------------------------------------------------------------

    def start_polling(self) -> str:
        """
        Authenticate with Kite and launch the 1-second polling thread.
        Returns a status string: 'started' | 'already_running' | raises RuntimeError.
        """
        if self._running.is_set():
            return "already_running"

        with self._lock:
            try:
                self._kite = get_authenticated_kite()
                logger.info("Kite authenticated — starting real-time polling.")
            except SystemExit as exc:
                raise RuntimeError(
                    f"Kite authentication failed: {exc}\n"
                    "Run POST /auth/session to refresh your access token."
                ) from exc

        self._running.set()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="ticker-poll",
            daemon=True,   # dies automatically when the main process exits
        )
        self._thread.start()
        logger.info("Polling thread started.")
        return "started"

    def stop_polling(self) -> str:
        """Signal the polling thread to exit and wait for it to finish."""
        if not self._running.is_set():
            return "not_running"

        self._running.clear()           # wakes the sleeping thread immediately
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("Polling thread stopped.")
        return "stopped"

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def scheduler_running(self) -> bool:
        return self._scheduler.running

    # ------------------------------------------------------------------
    # APScheduler callbacks
    # ------------------------------------------------------------------

    def _on_market_open(self) -> None:
        logger.info("Market open at %s IST — starting ticker.", datetime.now(IST).strftime("%H:%M:%S"))
        try:
            self.start_polling()
        except Exception as exc:
            logger.error("Failed to start polling at market open: %s", exc)

    def _on_market_close(self) -> None:
        logger.info("Market close at %s IST — stopping ticker.", datetime.now(IST).strftime("%H:%M:%S"))
        self.stop_polling()

    # ------------------------------------------------------------------
    # Core 1-second polling loop (runs inside _thread)
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        logger.info("Poll loop started.")
        while self._running.is_set():
            cycle_start = time.monotonic()
            try:
                self._poll_once()
            except Exception as exc:
                logger.error("Poll cycle error: %s", exc, exc_info=True)

            # Sleep for the remainder of the 1-second window.
            # Event.wait() returns early if stop_polling() clears _running.
            elapsed    = time.monotonic() - cycle_start
            sleep_time = max(0.0, POLL_INTERVAL_SEC - elapsed)
            self._running.wait(timeout=sleep_time)

        logger.info("Poll loop exited.")

    def _poll_once(self) -> None:
        """
        One poll cycle:
          1. Read watch list from DB (reflects real-time add/remove).
          2. Call kite.quote() for all symbols in one HTTP request.
          3. Bulk-insert the snapshots into tick_data.
        """
        watched = get_watched_symbols()
        if not watched:
            return   # nothing to watch — idle

        # kite.quote() expects "EXCHANGE:SYMBOL" strings
        instrument_keys = [f"{w['exchange']}:{w['symbol']}" for w in watched]
        symbol_meta     = {f"{w['exchange']}:{w['symbol']}": w for w in watched}

        with self._lock:
            kite = self._kite

        try:
            quotes = kite.quote(instrument_keys)
        except Exception as exc:
            logger.error("kite.quote() failed: %s", exc)
            return

        captured_at = datetime.now(IST).replace(tzinfo=None)   # naive IST datetime for DB
        ticks = []

        for key, q in quotes.items():
            meta = symbol_meta.get(key)
            if not meta:
                continue
            ohlc = q.get("ohlc", {})
            ticks.append({
                "symbol":           meta["symbol"],
                "exchange":         meta["exchange"],
                "instrument_token": q.get("instrument_token", 0),
                "last_price":       q.get("last_price", 0.0),
                "volume":           q.get("volume_traded", 0),
                "buy_quantity":     q.get("buy_quantity", 0),
                "sell_quantity":    q.get("sell_quantity", 0),
                "open":             ohlc.get("open"),
                "high":             ohlc.get("high"),
                "low":              ohlc.get("low"),
                "close":            ohlc.get("close"),
                "change":           q.get("change"),
                "captured_at":      captured_at,
            })

        if ticks:
            saved = save_ticks(ticks)
            logger.debug("Saved %d tick(s) at %s.", saved, captured_at)


# Module-level singleton — imported by api.py
ticker_manager = TickerManager()
