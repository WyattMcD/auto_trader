# strategies/options_csp.py
from datetime import date, timedelta
from config import (OPTIONS_TARGET_DELTA, OPTIONS_MIN_OI, OPTIONS_MAX_REL_SPREAD,
                    OPTIONS_MIN_DTE, OPTIONS_MAX_DTE)
import yfinance as yf

def _parse_occ(sym: str):
    # OCC: ROOT + YYMMDD + C/P + strike*1000 (8 digits)
    try:
        right = sym[-9]
        yy, mm, dd = sym[-15:-13], sym[-13:-11], sym[-11:-9]
        strike = int(sym[-8:]) / 1000.0
        exp = date(int("20"+yy), int(mm), int(dd))
        return right, exp, strike
    except Exception:
        return None, None, None

def _good_snapshot(snap):
    q = getattr(snap, "latest_quote", None)
    return bool(q and q.ask_price and q.bid_price and q.ask_price > 0)

def _rel_spread(snap):
    q = snap.latest_quote
    return (q.ask_price - q.bid_price) / q.ask_price

def pick_csp_intent(od, underlying: str):
    snaps = od.chain_snapshots(underlying)  # snapshots with greeks, quotes, OI
    if not snaps:
        return None

    lo, hi = OPTIONS_TARGET_DELTA
    cands = []
    for s in snaps:
        sym = getattr(s, "symbol", "")
        r, exp, strike = _parse_occ(sym)
        if r != "P" or not exp:
            continue
        dte = (exp - date.today()).days
        if dte < OPTIONS_MIN_DTE or dte > OPTIONS_MAX_DTE:
            continue
        if not _good_snapshot(s):
            continue

        greeks = getattr(s, "greeks", None)
        delta  = abs(getattr(greeks, "delta", 0) or 0)
        if not (lo <= delta <= hi):
            continue

        oi = getattr(s, "open_interest", 0) or 0
        if oi < OPTIONS_MIN_OI:
            continue

        if _rel_spread(s) > OPTIONS_MAX_REL_SPREAD:
            continue

        # prefer nearer expiry, then higher premium (ask)
        cands.append((dte, s.latest_quote.ask_price, s))

    if not cands:
        return None

    cands.sort(key=lambda t: (t[0], -t[1]))  # nearest expiry, highest ask
    pick = cands[0][2]

    # price near mid; small nudge for fill probability
    q = pick.latest_quote
    mid = (q.bid_price + q.ask_price) / 2
    limit = round(max(mid * 0.98, 0.05), 2)  # SELL credit: slightly below mid is safer

    return {
        "asset_class": "option",
        "symbol": pick.symbol,
        "side": "sell",
        "type": "limit",
        "qty": 1,
        "limit_price": limit,
        "tif": "day",
        "meta": {
            "why": f"CSP ~{int(abs(getattr(getattr(pick,'greeks',None),'delta',0))*100)}Δ,"
                   f" DTE={(date.fromisoformat(str(_parse_occ(pick.symbol)[1]))-date.today()).days}"
        }
    }

def exit_rules_for_csp(snapshot_entry_price: float, current_snap):
    """
    Given the entry credit and current snapshot, decide if we BTC.
    TP: 50% of credit
    SL: 2x credit (loss)
    """
    q = getattr(current_snap, "latest_quote", None)
    if not q or not q.ask_price or not q.bid_price:
        return None
    mid = (q.bid_price + q.ask_price) / 2
    credit = snapshot_entry_price
    # If current mid <= 0.5*credit -> take profit
    if mid <= credit * (1 - 0.50):
        return ("buy", "tp_50")
    # If current mid >= credit * 2.0 -> stop loss
    if mid >= credit * 2.0:
        return ("buy", "sl_2x")
    return None
