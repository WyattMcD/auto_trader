"""Decision helpers for routing equity signals into options structures.

These heuristics determine when an equity signal should be executed with
options instead of shares based on account constraints and configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

from config import (
    ENABLE_OPTIONS,
    OPTIONS_ROUTER_ALLOW_CSP,
    OPTIONS_ROUTER_ALLOW_SPREADS,
    OPTIONS_ROUTER_MAX_SHARE_PRICE,
    OPTIONS_ROUTER_MIN_SHARE_QTY,
    OPTIONS_UNDERLYINGS,
    SPREADS_MAX_OPEN,
    SPREADS_MAX_RISK_PER_TRADE,
)
from strategies.options_csp import pick_csp_intent
from strategies.spreads import pick_bull_put_spread


@dataclass(frozen=True)
class RouterContext:
    """Inputs required to decide between equity and options execution."""

    ticker: str
    price: float
    equity_qty: float
    state: Dict[str, Any]
    open_spreads: int
    open_short_puts: int
    account: Any | None = None


def _occ_underlying(symbol: str) -> str:
    """Best-effort extraction of the OCC underlying root."""
    if not symbol:
        return ""
    symbol = symbol.strip().upper()
    # OCC contracts use ROOT + YYMMDD + C/P + strike*1000 (8 digits)
    if len(symbol) <= 15:
        return symbol
    return symbol[:-15]


def _has_open_option_position(state: Dict[str, Any], ticker: str) -> bool:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return False

    singles = state.get("options_positions", {}) or {}
    for meta in singles.values():
        underlying = str(meta.get("underlying") or "").upper()
        if underlying == ticker:
            return True
        symbol = meta.get("symbol") or meta.get("occ_symbol")
        if symbol and _occ_underlying(symbol) == ticker:
            return True

    spreads = state.get("option_spreads", {}) or {}
    for meta in spreads.values():
        underlying = str(meta.get("underlying") or "").upper()
        if underlying == ticker:
            return True
        for leg in meta.get("legs", []):
            leg_sym = leg.get("symbol")
            if leg_sym and _occ_underlying(leg_sym) == ticker:
                return True

    return False


def plan_option_intent_for_signal(
    ctx: RouterContext,
    *,
    options_data,
    allow_spreads: bool = OPTIONS_ROUTER_ALLOW_SPREADS,
    allow_csp: bool = OPTIONS_ROUTER_ALLOW_CSP,
    min_equity_qty: float = OPTIONS_ROUTER_MIN_SHARE_QTY,
    max_equity_price: float = OPTIONS_ROUTER_MAX_SHARE_PRICE,
    spreads_max_open: int = SPREADS_MAX_OPEN,
    spreads_max_risk: float = SPREADS_MAX_RISK_PER_TRADE,
    underlyings: Iterable[str] = OPTIONS_UNDERLYINGS,
    pick_spread: Callable[[Any, str], Optional[Dict[str, Any]]] = pick_bull_put_spread,
    pick_csp: Callable[[Any, str], Optional[Dict[str, Any]]] = pick_csp_intent,
    approve_csp: Optional[Callable[[Dict[str, Any], Any, int, float], bool]] = None,
) -> Optional[Dict[str, Any]]:
    """Return an option intent when the signal should use derivatives.

    Parameters mirror configuration to ease unit testing.  The router only
    considers option execution when:
    * options trading is enabled globally,
    * the ticker is part of the configured options universe,
    * the equity sizing either fails (qty <= 0) or is capital-inefficient
      (price too high or qty below a minimum threshold), and
    * there is no existing open option position for the ticker.

    The router prefers defined-risk spreads and falls back to cash-secured
    puts when allowed and risk checks pass.
    """

    if not ENABLE_OPTIONS:
        return None

    ticker = (ctx.ticker or "").strip().upper()
    if not ticker:
        return None

    allowed_underlyings = {str(sym).strip().upper() for sym in underlyings}
    if ticker not in allowed_underlyings:
        return None

    if _has_open_option_position(ctx.state, ticker):
        return None

    # Only route to options if equity sizing is impractical
    qty = float(ctx.equity_qty or 0.0)
    price = float(ctx.price or 0.0)
    if qty > 0 and qty >= min_equity_qty and 0 < price <= max_equity_price:
        return None

    # Avoid exceeding spread concurrency limits
    if allow_spreads and ctx.open_spreads < spreads_max_open:
        intent = pick_spread(options_data, ticker)
        risk = (intent or {}).get("risk", {}) if intent else {}
        max_loss = risk.get("max_loss")
        if intent and (max_loss is None or max_loss <= spreads_max_risk):
            intent.setdefault("meta", {})["underlying"] = ticker
            intent["underlying"] = ticker
            return intent

    if not allow_csp:
        return None

    if approve_csp is None or ctx.account is None:
        return None

    intent = pick_csp(options_data, ticker)
    if not intent:
        return None

    symbol = intent.get("symbol")
    strike = None
    try:
        strike = int(str(symbol)[-8:]) / 1000.0
    except Exception:
        strike = None

    if strike is None:
        return None

    est_bpr = strike * 100.0
    if approve_csp(intent, ctx.account, ctx.open_short_puts, est_bpr):
        meta = intent.setdefault("meta", {})
        meta.setdefault("underlying", ticker)
        return intent

    return None
