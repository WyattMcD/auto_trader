# scripts/test_options_smoke.py
from datetime import date, timedelta, datetime
from config import API_KEY, API_SECRET, IS_PAPER
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from options.orders import OptionsTrader
from executer import execute_intent
def parse_from_symbol(sym: str):
    """
    Parse OCC option symbol (e.g. AAPL251017C00190000).
    Returns (right, expiry_date, strike).
    """
    try:
        right = sym[-9]  # 'C' or 'P'
        yy, mm, dd = sym[-15:-13], sym[-13:-11], sym[-11:-9]
        strike = int(sym[-8:]) / 1000.0
        from datetime import date
        expiry = date(int("20"+yy), int(mm), int(dd))
        return right, expiry, strike
    except Exception:
        return None, None, None

def is_call_symbol(sym: str) -> bool:
    r, _, _ = parse_from_symbol(sym)
    return r == "C"


UNDERLYING = "AAPL"  # try "SPY" if needed

def normalize_contracts(resp):
    """Return a list of contract objects from any alpaca response shape."""
    if isinstance(resp, list):
        return resp
    for attr in ("contracts", "option_contracts", "items", "results", "data"):
        lst = getattr(resp, attr, None)
        if lst is not None:
            try:
                return list(lst)
            except TypeError:
                pass
    try:
        return list(resp)  # if the response is iterable
    except TypeError:
        return []

def to_date(d):
    if isinstance(d, date):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(d, fmt).date()
            except Exception:
                pass
    return None

def right_is_call(c):
    r = getattr(c, "right", None) or getattr(c, "option_type", None) or getattr(c, "type", None)
    if r is None:
        return False
    return str(r).lower() in ("call", "c")

def get_spot_px(ticker: str) -> float | None:
    try:
        import yfinance as yf
        return float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])
    except Exception:
        return None

def main():
    if not API_KEY or not API_SECRET:
        print("[diag] Alpaca API credentials missing; skipping options smoke test.")
        print("       -> Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the environment.")
        print("       -> Paper accounts also require APCA_API_BASE_URL=https://paper-api.alpaca.markets.")
        return

    trading = TradingClient(API_KEY, API_SECRET, paper=IS_PAPER)
    opt_trader = OptionsTrader(API_KEY, API_SECRET, paper=IS_PAPER)

    start = date.today()
    end = start + timedelta(days=120)  # widen to 120d just in case
    req = GetOptionContractsRequest(
        underlying_symbol=UNDERLYING,
        expiration_date_gte=start,
        expiration_date_lte=end,
        status="active"
        # some SDK versions ignore right=...; we'll filter client-side anyway
    )

    resp = trading.get_option_contracts(req)
    contracts = normalize_contracts(resp)

    if not contracts:
        print(f"[diag] No contracts returned for {UNDERLYING}.")
        print("       -> Check Alpaca options approval (even for paper).")
        print("       -> Ensure APCA_API_BASE_URL is the paper URL.")
        return

    # client-side filter for CALLs with <= 60 DTE
    calls = []
    for c in contracts:
        sym = getattr(c, "symbol", None)
        if not sym:
            continue
        if not is_call_symbol(sym):
            continue
        r, exp, strike = parse_from_symbol(sym)
        if not exp:
            continue
        dte = (exp - date.today()).days
        if 0 < dte <= 60:
            calls.append((c, dte, strike, exp))

    if not calls:
        print("[diag] Got contracts, but none matched (call + <=60 DTE). Showing up to 10 samples:")
        sample = contracts[:10] if isinstance(contracts, list) else contracts
        shown = 0
        for c in sample:
            if shown >= 10:
                break
            print({
                "symbol": getattr(c, "symbol", None),
                "right": getattr(c, "right", None),
                "option_type": getattr(c, "option_type", None),
                "expiration_date_raw": getattr(c, "expiration_date", None),
                "expiration_date_parsed": str(to_date(getattr(c, "expiration_date", None))),
                "strike": str(getattr(c, "strike_price", None)),
                "status": getattr(c, "status", None),
            })
            shown += 1
        print("Tips:")
        print(" - If right/expiry look empty, your account may not be options-enabled yet.")
        print(" - Try UNDERLYING='SPY' during market hours.")
        return

    # choose nearest expiry, then strike closest to spot
    spot = get_spot_px(UNDERLYING)
    if spot is None:
        print(f"[diag] Could not fetch spot for {UNDERLYING}. Install yfinance or set a manual price.")
        return

    calls.sort(key=lambda t: (t[1], abs(float(t[2]) - spot)))
    chosen = calls[0][0]
    contract_symbol = getattr(chosen, "symbol", None)
    if not contract_symbol:
        print("[diag] Chosen contract missing symbol; cannot submit.")
        return

    # ---- build a LIMIT order so it works outside market hours ----
    # try to set a sane limit; if you don't have options quotes yet, use a small fallback
    limit_price = 0.10  # fallback 10¢ to validate submission; adjust once quotes are wired

    intent = {
        "asset_class": "option",
        "symbol": contract_symbol,
        "side": "buy",
        "type": "limit",  # <-- was "market"
        "qty": 1,
        "limit_price": limit_price,  # <-- required for limit
        "tif": "day",
        "meta": {"why": "smoke test: buy 1 near-ATM call (limit)"},
    }

    res = execute_intent(intent, equity_trader=None, options_trader=opt_trader)
    print("Submitted:", res)

if __name__ == "__main__":
    main()
