"""
Custom exception hierarchy for the Zerodha Kite Connect project.

Raising specific exception types (instead of sys.exit or bare Exception)
lets every caller decide independently how to handle each failure:
  - CLI entry points print a clean message and exit with a non-zero code.
  - FastAPI endpoints map each type to the right HTTP status code.
  - Background threads log the error and keep running.
"""


class KiteBaseError(Exception):
    """Root class for all project exceptions."""


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class KiteAuthError(KiteBaseError):
    """
    Raised when Kite Connect authentication fails.
    Examples: missing API key, expired access token, bad request token.
    """


# ---------------------------------------------------------------------------
# API / data fetching
# ---------------------------------------------------------------------------

class KiteAPIError(KiteBaseError):
    """
    Raised when a Kite Connect API call fails at the HTTP/SDK level.
    Examples: network timeout, rate-limit (429), server error (5xx).
    """


class SymbolNotFoundError(KiteAPIError):
    """
    Raised when a trading symbol cannot be found on the given exchange.
    Includes a list of close matches when available.
    """


class InvalidIntervalError(KiteAPIError):
    """Raised when an unsupported candle interval is requested."""


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class DatabaseError(KiteBaseError):
    """
    Raised when a MySQL operation fails (connection, query, insert, etc.).
    Wraps the underlying pymysql / dbutils exception as __cause__.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigError(KiteBaseError):
    """
    Raised when a pipeline YAML config is missing required keys or has
    invalid values.
    """
