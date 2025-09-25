#!/usr/bin/env python3
# dry_run_positions.py - dry-run inspector: prints account equity and suggested sells to enforce MAX_NOTIONAL_PCT
from dotenv import load_dotenv
import os, pprint, math

# Try container path first, then project .env
for path in ('/app/.env', '.env'):
    if os.path.exists(path):
        load_dotenv(path)
        break

from alpaca_trade_api.rest import REST

KEY = os.getenv("APCA_API_KEY_ID")
SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

if not KEY or not SECRET:
    raise SystemExit("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in environment (.env)")

api = REST(KEY, SECRET, base_url=BASE, api_version='v2')

acct = api.get_account()
equity = float(getattr(acct, "equity", getattr(acct, "cash", 0) or 0))
print(f"Account equity: ${equity:,.2f}")

try:
    max_pct = float(os.getenv("MAX_NOTIONAL_PCT", "0.05"))
except Exception:
    print("Bad MAX_NOTIONAL_PCT in .env; defaulting to 0.05")
    max_pct = 0.05

cap = equity * max_pct
print(f"Per-position cap (equity * MAX_NOTIONAL_PCT): ${cap:,.2f} (MAX_NOTIONAL_PCT={max_pct})")
print()

positions = api.list_positions()
print(f"Open positions: {len(positions)}")
print()

suggestions = []
for p in positions:
    sym = p.symbol
    qty = float(p.qty)
    try:
        price = float(p.current_price)
    except Exception:
        price = float(getattr(p, "avg_entry_price", 0) or 0)
    mv = float(p.market_value)
    print(f"{sym:6s} qty={int(qty):6d} price=${price:8.2f} market_value=${mv:,.2f}")
    if mv > cap:
        excess = mv - cap
        # compute ceil(excess / price)
        qty_to_sell = int(math.ceil(excess / price)) if price > 0 else 0
        suggestions.append((sym, int(qty), price, mv, qty_to_sell))
print()

if suggestions:
    print("Suggested sells to respect MAX_NOTIONAL_PCT (dry-run):")
    for sym, cur_qty, price, mv, qty_to_sell in suggestions:
        print(f" - {sym}: current ${mv:,.2f}, sell {qty_to_sell} shares (current qty {cur_qty})")
else:
    print("No positions exceed the per-position cap. Nothing to suggest.")
