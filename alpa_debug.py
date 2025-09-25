#!/usr/bin/env python3
# alpa_debug.py - list recent orders for CRM (and print some env info)
from dotenv import load_dotenv
import os, pprint, sys

# try to load .env from /app first (works when run inside container), else from CWD
for path in ('/app/.env', '.env'):
    if os.path.exists(path):
        load_dotenv(path)
        break

print("APCA_API_KEY_ID present:", bool(os.getenv("APCA_API_KEY_ID")))
print("APCA_API_BASE_URL:", os.getenv("APCA_API_BASE_URL"))

try:
    from alpaca_trade_api.rest import REST
except Exception as e:
    print("Missing alpaca client import:", e, file=sys.stderr)
    raise

key = os.getenv("APCA_API_KEY_ID")
secret = os.getenv("APCA_API_SECRET_KEY")
base = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

if not key or not secret:
    print("API keys missing in environment. Exiting.", file=sys.stderr)
    sys.exit(1)

api = REST(key, secret, base_url=base, api_version='v2')

symbol = "CRM"
orders = api.list_orders(status='all', limit=500, nested=True)
crm_orders = [o for o in orders if getattr(o, "symbol", "") == symbol]
crm_orders_sorted = sorted(crm_orders, key=lambda o: getattr(o, "created_at", None) or "")

pprint.pprint([{
    "id": o.id,
    "client_order_id": getattr(o, "client_order_id", None),
    "created_at": getattr(o, "created_at", None),
    "updated_at": getattr(o, "updated_at", None),
    "side": o.side,
    "type": o.type,
    "time_in_force": o.time_in_force,
    "limit_price": getattr(o, "limit_price", None),
    "stop_price": getattr(o, "stop_price", None),
    "filled_qty": o.filled_qty,
    "filled_avg_price": getattr(o, "filled_avg_price", None),
    "status": o.status
} for o in crm_orders_sorted], width=160)

