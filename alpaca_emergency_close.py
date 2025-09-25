# alpaca_emergency_close.py
from dotenv import load_dotenv
import os, time
from alpaca_trade_api.rest import REST

load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
API_BASE = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
api = REST(API_KEY, API_SECRET, API_BASE, api_version='v2')

print("Cancelling open orders...")
orders = api.list_orders(status='open')
for o in orders:
    try:
        api.cancel_order(o.id)
        print("Cancelled", o.id, o.symbol)
    except Exception as e:
        print("Cancel error", o.id, e)

time.sleep(1)
print("Closing positions (submitting market sells)...")
positions = api.list_positions()
for p in positions:
    try:
        qty = p.qty
        print("Closing", p.symbol, qty)
        api.submit_order(symbol=p.symbol, qty=qty, side='sell', type='market', time_in_force='day')
    except Exception as e:
        print("Close error", p.symbol, e)

print("Done. Check your Alpaca dashboard to confirm fills.")
