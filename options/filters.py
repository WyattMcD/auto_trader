# options/filters.py
def within_delta(greeks, lo, hi):
    d = getattr(greeks, "delta", None)
    return d is not None and lo <= abs(d) <= hi

def good_liquidity(snapshot, min_oi: int, max_rel_spread: float):
    oi = getattr(snapshot, "open_interest", None)
    bid = getattr(snapshot.latest_quote, "bid_price", None)
    ask = getattr(snapshot.latest_quote, "ask_price", None)
    if oi is None or bid is None or ask is None or ask <= 0:
        return False
    rel_spread = (ask - bid) / ask
    return oi >= min_oi and rel_spread <= max_rel_spread

def dte_ok(contract, min_dte: int, max_dte: int | None):
    dte = (contract.expiration_date - contract.created_at.date()).days \
          if hasattr(contract, "created_at") else None
    # If created_at is missing, skip DTE check (fallback handled in strategy)
    if dte is None:
        return True
    return dte >= min_dte and (max_dte is None or dte <= max_dte)
