"""
Structured JSON logging for ai-quant-lab.

Every log record is emitted as a single-line JSON object so it can be
ingested by any log aggregator (Datadog, CloudWatch, Loki, etc.).

A thread-local correlation ID is attached to every record, making it
easy to trace all log lines that belong to one pipeline run or HTTP
request across threads.

Public API
----------
    configure_logging(level=None)   — wire JSON handler to root logger
    get_logger(name)                — logging.getLogger wrapper
    set_correlation_id(cid)         — bind ID to current thread
    get_correlation_id()            — read back current thread's ID
    new_correlation_id()            — generate a fresh 12-char hex ID

Usage
-----
    from log_config import configure_logging, get_logger, new_correlation_id, set_correlation_id

    configure_logging()                         # once at process startup
    set_correlation_id(new_correlation_id())    # once per pipeline run / request
    logger = get_logger(__name__)
    logger.info("pipeline started", extra={"pipeline": "nifty50_daily"})
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Thread-local correlation ID
# ---------------------------------------------------------------------------

_ctx = threading.local()

# Standard LogRecord attributes — we skip these when forwarding extras to JSON
_LOGRECORD_FIELDS = frozenset({
    "name", "msg", "args", "created", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message",
    "pathname", "process", "processName", "relativeCreated",
    "stack_info", "thread", "threadName", "exc_info", "exc_text",
    "taskName",
})


def new_correlation_id() -> str:
    """Return a fresh 12-character hex UUID fragment."""
    return uuid.uuid4().hex[:12]


def set_correlation_id(cid: str) -> None:
    """Bind *cid* to the current thread so it appears in every log line."""
    _ctx.correlation_id = cid


def get_correlation_id() -> str:
    """Return the current thread's correlation ID, or ``'-'`` if unset."""
    return getattr(_ctx, "correlation_id", "-")


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """
    Emit each log record as a compact single-line JSON object.

    Mandatory JSON fields
    ---------------------
    ts              ISO-8601 UTC timestamp with millisecond precision
    level           DEBUG / INFO / WARNING / ERROR / CRITICAL
    logger          Logger name (usually the module __name__)
    msg             Formatted log message
    correlation_id  Thread-local ID set via set_correlation_id()

    Optional fields
    ---------------
    exc_info        Formatted traceback string (only present when there is one)
    <extras>        Any key=value pairs passed via extra={...} that are
                    JSON-serialisable are forwarded verbatim.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":             datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level":          record.levelname,
            "logger":         record.name,
            "msg":            record.getMessage(),
            "correlation_id": get_correlation_id(),
        }

        # Forward caller-supplied extra= fields that are JSON-serialisable
        for key, val in record.__dict__.items():
            if key in _LOGRECORD_FIELDS or key.startswith("_"):
                continue
            try:
                json.dumps(val)
                payload[key] = val
            except (TypeError, ValueError):
                payload[key] = str(val)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Root-logger configuration
# ---------------------------------------------------------------------------

def configure_logging(level: str | None = None) -> None:
    """
    Attach a JSON stream handler to the root logger.

    Level precedence
    ----------------
    1. *level* argument
    2. ``LOG_LEVEL`` environment variable
    3. ``INFO`` (default)

    Safe to call multiple times — the handler is installed only once.
    Subsequent calls are no-ops so library code cannot hijack the config.
    """
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric  = getattr(logging, resolved, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        return  # already configured — skip

    root.setLevel(numeric)
    handler = logging.StreamHandler()
    handler.setLevel(numeric)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Thin wrapper around ``logging.getLogger`` for uniform imports."""
    return logging.getLogger(name)
