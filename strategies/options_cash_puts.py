# strategies/options_cash_puts.py
from datetime import date, timedelta
from options.data import OptionsData
from options.filters import within_delta, good_liquidity
from config import (OPTIONS_MIN_OI, OPTIONS_MAX_SPREAD,
                    OPTIONS_MIN_DTE, OPTIONS_MAX_DTE, OPTIONS_TARGET_DELTA)

def screen_cash_secured_put(od: OptionsData, underlying: str):
    # 1) Pull chain snapshots (quotes + greeks) for underlying
    snaps = od.chain_snapshot(underlying)  # list of OptionSnapshot
    lo, hi = OPTIONS_TARGET_DELTA

    # 2) Keep PUTs with decent liquidity and delta in range
    cands = []
    for s in snaps:
        if getattr(s, "symbol", "").endswith("P"):  # PUTs
            greeks = getattr(s, "greeks", None)
            if greeks and within_delta(greeks, lo, hi) and good_liquidity(
                s, OPTIONS_MIN_OI, OPTIONS_MAX_SPREAD
            ):
                cands.append(s)

    # 3) Prefer nearer expiries within DTE window (fallback to quote mid)
    def parse_exp(sym):
        # sym like AAPL251018P00190000 -> YYMMDD at idx 5..11
        return sym[5:11]  # '251018'
    def dte(sym):
        y, m, d = int('20'+sym[0:2]), int(sym[2:4]), int(sym[4:6])
        return (date(y,m,d) - date.today()).days

    cands = [s for s in cands if dte(parse_exp(s.symbol)) >= OPTIONS_MIN_DTE
                           and (OPTIONS_MAX_DTE is None or dte(parse_exp(s.symbol)) <= OPTIONS_MAX_DTE)]
    if not cands:
        return None

    cands.sort(key=lambda s: dte(parse_exp(s.symbol)))
    pick = cands[0]
    q = pick.latest_quote
    mid = (q.bid_price + q.ask_price) / 2 if q and q.ask_price else None

    # Return a normalized order intent (SELL 1 put = short premium)
    return {
        "asset_class": "option",
        "symbol": pick.symbol,
        "side": "sell",
        "type": "limit" if mid else "market",
        "qty": 1,
        "limit_price": round(mid, 2) if mid else None,
        "tif": "day",
        "meta": {"underlying": underlying, "why": "cash-secured put near target delta"}
    }
