# strategies/spread_exits.py
# strategies/spread_exits.py
def current_mid(quote):
    return (quote.bid_price + quote.ask_price)/2 if quote and quote.bid_price and quote.ask_price else None

def bps_decision(entry_credit, short_snap, long_snap):
    ms = current_mid(getattr(short_snap, "latest_quote", None))
    ml = current_mid(getattr(long_snap, "latest_quote", None))
    if ms is None or ml is None:
        return None
    curr_credit = ms - ml
    if curr_credit <= entry_credit * 0.50:
        return ("buy", "tp_50")   # buy-to-close both legs
    if curr_credit >= entry_credit * 2.0:
        return ("buy", "sl_2x")
    return None

def call_debit_decision(entry_debit, long_snap, short_snap, width_dollars):
    ms = current_mid(getattr(short_snap, "latest_quote", None))
    ml = current_mid(getattr(long_snap, "latest_quote", None))
    if ms is None or ml is None:
        return None
    curr_debit = ml - ms
    max_gain = width_dollars - entry_debit
    if (width_dollars - curr_debit) >= 0.50 * max_gain:
        return ("sell", "tp_50")  # sell the spread (reverse legs)
    if curr_debit <= 0.50 * entry_debit:
        return ("sell", "sl_half")
    return None

def bps_should_exit(entry_net_credit, short_snap, long_snap):
    qS, qL = getattr(short_snap, "latest_quote", None), getattr(long_snap, "latest_quote", None)
    if not (qS and qL and qS.bid_price and qS.ask_price and qL.bid_price and qL.ask_price):
        return None
    # current net credit ≈ short mid - long mid
    midS = (qS.bid_price + qS.ask_price)/2
    midL = (qL.bid_price + qL.ask_price)/2
    curr_credit = midS - midL
    # TP: captured ≥50% of entry credit
    if curr_credit <= entry_net_credit * 0.50:
        return ("buy", "tp_50")
    # SL: loss ~≥ 2x credit (i.e., curr_credit >= 2*entry)
    if curr_credit >= entry_net_credit * 2.0:
        return ("buy", "sl_2x")
    return None
