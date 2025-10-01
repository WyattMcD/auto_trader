"""Risk management helpers for the auto trader."""
from __future__ import annotations

from typing import Optional

from config import MAX_OPEN_CSP_POSITIONS, MAX_BP_AT_RISK, MAX_RISK_PER_TRADE


# ===== Existing CSP guard you already had =====
def _safe_float(value, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _account_equity(account) -> float:
    """Best-effort equity lookup from the Alpaca account object."""
    for attr in ("equity", "last_equity", "cash", "buying_power"):
        val = getattr(account, attr, None)
        if val is None:
            continue
        num = _safe_float(val, default=None)
        if num is not None and num > 0:
            return num
    return 0.0


def approve_csp_intent(intent, account, open_csp_count: int, est_bpr: float,
                       risk_pct: float = MAX_RISK_PER_TRADE) -> bool:
    """Approve CSP orders only when they fit within position limits.

    The guard checks three independent limits:
    1. Maximum number of concurrent CSP positions.
    2. Buying-power-at-risk guardrail that already existed in the project.
    3. A 5% default cap of total account equity (configurable via
       ``MAX_RISK_PER_TRADE``).
    """
    if open_csp_count >= MAX_OPEN_CSP_POSITIONS:
        return False

    bp = _safe_float(getattr(account, "buying_power", 0.0)) or 0.0
    if bp and est_bpr > bp * MAX_BP_AT_RISK:
        return False

    equity = _account_equity(account)
    if risk_pct and equity and est_bpr > equity * risk_pct:
        return False

    return True


# ===== New: 5% position cap for equities/ETFs (e.g., XLF) =====
# If you want this configurable, also add MAX_POS_PCT=0.05 into config.py and import it here.
MAX_POS_PCT = 0.05  # 5% of current account equity per single symbol


def get_equity(trading_client) -> float:
    """Pull current account equity from Alpaca."""
    acct = trading_client.get_account()
    return float(_safe_float(getattr(acct, "equity", 0.0), 0.0) or 0.0)


def get_pos_market_value(trading_client, symbol: str) -> float:
    """Return market value of an open position in dollars."""
    try:
        pos = trading_client.get_open_position(symbol)
        return float(_safe_float(getattr(pos, "market_value", 0.0), 0.0) or 0.0)
    except Exception:
        return 0.0  # no open position or API raised error


def get_pending_notional_for_symbol(trading_client, symbol: str) -> float:
    """Add up notional value of open BUY orders for the symbol."""
    pending = 0.0
    try:
        open_orders = trading_client.get_orders(status="open")
        for o in open_orders:
            if getattr(o, "symbol", "") != symbol:
                continue
            side = str(getattr(o, "side", "")).lower()
            if side != "buy":
                continue
            qty = float(_safe_float(getattr(o, "qty", 0.0), 0.0) or 0.0)
            px = None
            for attr in ("limit_price", "hwm", "trail_price", "stop_price", "notional"):
                val = getattr(o, attr, None)
                if val is not None:
                    px = _safe_float(val)
                    break
            if px is None or px == 0.0:
                continue
            pending += qty * px
    except Exception:
        pass
    return pending


def max_dollars_for_symbol(trading_client, cap_pct: float = MAX_POS_PCT) -> float:
    return get_equity(trading_client) * cap_pct


def capped_qty_to_buy(trading_client, symbol: str, price: float, intended_qty: int,
                      cap_pct: float = MAX_POS_PCT) -> int:
    """Shrink intended_qty so exposure never exceeds cap_pct of equity."""
    price = float(_safe_float(price, 0.0) or 0.0)
    if price <= 0.0:
        return 0

    current_val = get_pos_market_value(trading_client, symbol)
    pending_val = get_pending_notional_for_symbol(trading_client, symbol)
    allowed = max_dollars_for_symbol(trading_client, cap_pct)

    dollars_left = max(0.0, allowed - (current_val + pending_val))
    max_additional_qty = int(dollars_left // price)
    return max(0, min(int(intended_qty), max_additional_qty))


# Optional: convenience order placer (market buy) using the cap.
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def place_capped_buy(trading_client, symbol: str, price: float, intended_qty: int,
                     tif: str = "day", cap_pct: float = MAX_POS_PCT):
    """Place a capped MARKET BUY order respecting the per-symbol exposure cap."""
    qty = capped_qty_to_buy(trading_client, symbol, price, intended_qty, cap_pct=cap_pct)
    if qty <= 0:
        return None

    tif_enum = TimeInForce.DAY if str(tif).lower() == "day" else TimeInForce.GTC
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=tif_enum,
    )
    return trading_client.submit_order(order)
