# strategies/spread_exits.py
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
