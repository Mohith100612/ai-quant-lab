"""
Zerodha Kite Connect — Authentication helpers.

Usage
-----
Step 1 — Get the login URL and open it in your browser:
    python auth.py --login-url

Step 2 — After Kite redirects you, copy the `request_token` from the URL and run:
    python auth.py --generate-session <request_token>

This prints the access token and writes it to .env automatically.
"""

import os
import sys
import argparse
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

from exceptions import KiteAuthError

ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
ENV_EXAMPLE = os.path.join(os.path.dirname(__file__), ".env.example")


def _load_kite() -> KiteConnect:
    """Load credentials and return a KiteConnect instance."""
    try:
        load_dotenv(dotenv_path=ENV_FILE)
        api_key = os.getenv("KITE_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            raise KiteAuthError(
                "KITE_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        return KiteConnect(api_key=api_key)
    except KiteAuthError:
        raise
    except Exception as exc:
        raise KiteAuthError(f"Failed to initialise KiteConnect: {exc}") from exc


def get_login_url() -> str:
    try:
        kite = _load_kite()
        return kite.login_url()
    except KiteAuthError:
        raise
    except Exception as exc:
        raise KiteAuthError(f"Failed to generate login URL: {exc}") from exc


def generate_session(request_token: str) -> str:
    """Exchange a request_token for an access_token and persist it to .env."""
    try:
        load_dotenv(dotenv_path=ENV_FILE)
        api_secret = os.getenv("KITE_API_SECRET")
        if not api_secret or api_secret == "your_api_secret_here":
            raise KiteAuthError(
                "KITE_API_SECRET not set. Copy .env.example to .env and fill it in."
            )
        kite = _load_kite()
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token: str = data["access_token"]
    except KiteAuthError:
        raise
    except Exception as exc:
        raise KiteAuthError(
            f"Failed to generate session for request_token '{request_token}': {exc}"
        ) from exc

    try:
        if os.path.exists(ENV_FILE):
            set_key(ENV_FILE, "KITE_ACCESS_TOKEN", access_token)
            print(f"Access token saved to {ENV_FILE}")
        else:
            print("WARNING: .env file not found — could not save access token.")
    except Exception as exc:
        # Non-fatal: token was obtained; just warn if we can't persist it
        print(f"WARNING: Could not write access token to .env: {exc}")

    print(f"Access token: {access_token}")
    return access_token


def get_authenticated_kite() -> KiteConnect:
    """Return a fully authenticated KiteConnect instance ready to make API calls."""
    try:
        load_dotenv(dotenv_path=ENV_FILE)
        kite = _load_kite()
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        if not access_token:
            raise KiteAuthError(
                "KITE_ACCESS_TOKEN not set.\n"
                "Run `python auth.py --login-url`, complete the browser login,\n"
                "then run `python auth.py --generate-session <request_token>`."
            )
        kite.set_access_token(access_token)
        return kite
    except KiteAuthError:
        raise
    except Exception as exc:
        raise KiteAuthError(f"Authentication failed: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Kite Connect authentication helper")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--login-url",
        action="store_true",
        help="Print the Kite login URL to open in your browser",
    )
    group.add_argument(
        "--generate-session",
        metavar="REQUEST_TOKEN",
        help="Exchange a request_token (from the redirect URL) for an access_token",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        if args.login_url:
            url = get_login_url()
            print(f"\nOpen this URL in your browser to log in:\n\n  {url}\n")
            print(
                "After login, Kite redirects to your configured redirect URL.\n"
                "Copy the `request_token` query parameter from that URL, then run:\n\n"
                "  python auth.py --generate-session <request_token>\n"
            )
        elif args.generate_session:
            generate_session(args.generate_session)
    except KiteAuthError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
