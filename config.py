"""Centralised configuration for the auto trader.

The defaults are intentionally conservative so the module can be imported in
unit tests without environment variables or network credentials. Everything
can still be overridden via the environment if required in production.
"""
from __future__ import annotations

import os
from typing import Iterable, Tuple

try:  # Optional dependency when running locally
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - fallback when dotenv is unavailable
    load_dotenv = None

if load_dotenv:
    load_dotenv()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_float_tuple(value: str | None, default: Tuple[float, float]) -> Tuple[float, float]:
    if not value:
        return default
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        return default
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return default


def _as_list(value: str | None, default: Iterable[str]) -> list[str]:
    if not value:
        return list(default)
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts if parts else list(default)


def _as_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _as_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# ----------------------------------------------------------------------------
# Core Alpaca credentials & runtime flags
# ----------------------------------------------------------------------------

ALPACA_KEY_ID = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID") or "paper-key"
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or "paper-secret"
ALPACA_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
IS_PAPER = "paper" in ALPACA_BASE_URL.lower()

# Backwards compatible aliases used across the codebase
API_KEY = ALPACA_KEY_ID
API_SECRET = ALPACA_SECRET_KEY

MAX_RISK_PER_TRADE = _as_float("MAX_RISK_PER_TRADE", 0.05)  # 5% default

DAY_RUN = _as_bool(os.getenv("DAY_RUN"), True)
WATCHLIST = _as_list(os.getenv("WATCHLIST"), [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "NFLX",
    "TSLA", "AMD", "CRM", "AVGO", "ORCL", "NOW", "SHOP",
    "JPM", "BAC", "C", "MA", "V",
    "HD", "LOW", "MCD", "SBUX",
    "CAT", "BA", "LMT",
    "XOM", "CVX", "COP",
    "JNJ", "PFE", "MRK", "MRNA",
    "ASML", "QCOM", "LRCX",
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE",
])

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")


# ----------------------------------------------------------------------------
# Options configuration
# ----------------------------------------------------------------------------

ENABLE_OPTIONS = _as_bool(os.getenv("ENABLE_OPTIONS"), True)
OPTIONS_UNDERLYINGS = _as_list(os.getenv("OPTIONS_UNDERLYINGS"), ["SPY", "AAPL"])
OPTIONS_TARGET_DELTA = _as_float_tuple(os.getenv("OPTIONS_TARGET_DELTA"), (0.20, 0.30))
OPTIONS_MIN_OI = _as_int("OPTIONS_MIN_OI", 100)
OPTIONS_MAX_REL_SPREAD = _as_float("OPTIONS_MAX_REL_SPREAD", 0.15)
OPTIONS_MIN_DTE = _as_int("OPTIONS_MIN_DTE", 1)
OPTIONS_MAX_DTE = _as_int("OPTIONS_MAX_DTE", 90)
MAX_OPEN_CSP_POSITIONS = _as_int("MAX_OPEN_CSP_POSITIONS", 8)
MAX_BP_AT_RISK = _as_float("MAX_BP_AT_RISK", 0.25)
TAKE_PROFIT_PCT = _as_float("TAKE_PROFIT_PCT", 0.50)
STOP_LOSS_MULT = _as_float("STOP_LOSS_MULT", 2.0)
OPTIONS_MAX_CANDIDATES_PER_TICK = _as_int("OPTIONS_MAX_CANDIDATES_PER_TICK", 8)
OPTIONS_USE_WATCHLIST = _as_bool(os.getenv("OPTIONS_USE_WATCHLIST"), True)


# ----------------------------------------------------------------------------
# Spread strategies
# ----------------------------------------------------------------------------

SPREADS_MIN_DTE = _as_int("SPREADS_MIN_DTE", 7)
SPREADS_MAX_DTE = _as_int("SPREADS_MAX_DTE", 30)
SPREADS_MIN_OI = _as_int("SPREADS_MIN_OI", 300)
SPREADS_MAX_REL_SPREAD = _as_float("SPREADS_MAX_REL_SPREAD", 0.20)
SPREADS_TARGET_DELTA = _as_float_tuple(os.getenv("SPREADS_TARGET_DELTA"), (0.20, 0.35))
SPREADS_MAX_RISK_PER_TRADE = _as_float("SPREADS_MAX_RISK_PER_TRADE", 150.0)
SPREADS_MAX_OPEN = _as_int("SPREADS_MAX_OPEN", 2)
SPREADS_MAX_CANDIDATES_PER_TICK = _as_int("SPREADS_MAX_CANDIDATES_PER_TICK", 8)
