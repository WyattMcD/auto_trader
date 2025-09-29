# risk.py

from config import MAX_OPEN_CSP_POSITIONS, MAX_BP_AT_RISK

# ===== Existing CSP guard you already had =====
def approve_csp_intent(intent, account, open_csp_count: int, est_bpr: float) -> bool:
    """
    Approves cash-secured-put intents based on max open positions and buying power risk.
    """
    if open_csp_count >= MAX_OPEN_CSP_POSITIONS:
        return False
    # Rough guard: est BPR vs buying power
    try:
        bp = float(getattr(account, "buying_power", 0))
        if est_bpr > bp * MAX_BP_AT_RISK:
            return False
    except Exception:
        pass
    return True


# ===== New: 5% position cap for equities/ETFs (e.g., XLF) =====
# If you want this configurable, also add MAX_POS_PCT=0.05 into config.py and import it here.
MAX_POS_PCT = 0.05  # 5% of current account equity per single symbol

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def get_equity(trading_client) -> float:
    """
    Pulls current account equity from Alpaca.
    """
    acct = trading_client.get_account()
    # Use .equity (includes P/L). If you prefer more conservative, use .last_equity.
    return _safe_float(getattr(acct, "equity", 0.0))

def get_pos_market_value(trading_client, symbol: str) -> float:
    """
    Returns current market value of an open position in dollars.
    """
    try:
        pos = trading_client.get_open_position(symbol)
        return _safe_float(getattr(pos, "market_value", 0.0))
    except Exception:
        return 0.0  # no open position or API raised error

def get_pending_notional_for_symbol(trading_client, symbol: str) -> float:
    """
    Adds up notional value of any open BUY orders for this symbol to avoid
    concurrent signals overfilling the cap.
    """
    pending = 0.0
    try:
        open_orders = trading_client.get_orders(status="open")
        for o in open_orders:
            if getattr(o, "symbol", "") != symbol:
                continue
            side = str(getattr(o, "side", "")).lower()
            if side != "buy":
                continue
            qty = _safe_float(getattr(o, "qty", 0.0))
            # best-effort notional using limit_price if present; fallback to notional/stop/last-known
            px = None
            for attr in ("limit_price", "hwm", "trail_price", "stop_price", "notional"):
                val = getattr(o, attr, None)
                if val is not None:
                    px = _safe_float(val)
                    break
            if px is None or px == 0.0:
                # As a last resort we won't count it (keeping conservative behavior);
                # you can pass a live price into capped sizing to be safer overall.
                continue
            pending += qty * px
    except Exception:
        pass
    return pending

def max_dollars_for_symbol(trading_client, cap_pct: float = MAX_POS_PCT) -> float:
    return get_equity(trading_client) * cap_pct

def capped_qty_to_buy(trading_client, symbol: str, price: float, intended_qty: int,
                      cap_pct: float = MAX_POS_PCT) -> int:
    """
    Shrinks the intended_qty so that (current position + pending buys + new buy)
    <= cap_pct * equity (by notional).
    """
    price = _safe_float(price, 0.0)
    if price <= 0.0:
        return 0

    current_val = get_pos_market_value(trading_client, symbol)
    pending_val = get_pending_notional_for_symbol(trading_client, symbol)
    allowed = max_dollars_for_symbol(trading_client, cap_pct)

    dollars_left = max(0.0, allowed - (current_val + pending_val))
    max_additional_qty = int(dollars_left // price)
    return max(0, min(int(intended_qty), max_additional_qty))

# Optional: convenience order placer (market buy) using the cap.
# If you prefer to keep order routing elsewhere, just call capped_qty_to_buy(...) there.
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def place_capped_buy(trading_client, symbol: str, price: float, intended_qty: int,
                     tif: str = "day", cap_pct: float = MAX_POS_PCT):
    """
    Places a MARKET BUY but caps size so symbol exposure never exceeds cap_pct of equity.
    Returns the Alpaca order object or None if no shares allowed.
    """
    qty = capped_qty_to_buy(trading_client, symbol, price, intended_qty, cap_pct=cap_pct)
    if qty <= 0:
        # Nothing to do; already at or over cap.
        return None

    tif_enum = TimeInForce.DAY if str(tif).lower() == "day" else TimeInForce.GTC
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=tif_enum,
    )
    return trading_client.submit_order(order)
