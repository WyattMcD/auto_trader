# webhook_server.py
APCA_API_KEY_ID= "PKLN6KEQA849SPCA792X"
APCA_API_SECRET_KEY= "JJZvZ5y8vErdsWleZvy1JwPEwMTxSlxUBpfS0uSe"
APCA_API_BASE_URL= "https://paper-api.alpaca.markets/PA3D4OQR9V0Z"

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("PKLN6KEQA849SPCA792X")
API_SECRET = os.getenv("JJZvZ5y8vErdsWleZvy1JwPEwMTxSlxUBpfS0uSe")
API_BASE = os.getenv("https://paper-api.alpaca.markets/PA3D4OQR9V0Z")

import os, json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame

load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
API_BASE = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets/PA3D4OQR9V0Z")

app = Flask(__name__)
import alpaca_trade_api as tradeapi

API_KEY = "PKLN6KEQA849SPCA792X"
API_SECRET = "JJZvZ5y8vErdsWleZvy1JwPEwMTxSlxUBpfS0uSe"
API_BASE = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, API_SECRET, base_url=API_BASE, api_version='v2')

account = api.get_account()
print("Account status:", account.status)
print("Cash balance:", account.cash)

MAX_RISK_PCT = 0.02  # 2% per trade
ACCOUNT_CACHE = {}

def get_account_equity():
    acct = api.get_account()
    return float(acct.equity)

def calc_position_notional(equity, entry_price, stop_pct=0.06):
    # Max dollar risk = MAX_RISK_PCT * equity
    max_risk = MAX_RISK_PCT * equity
    # If stop loss is stop_pct distance away, notional = max_risk / stop_pct
    notional = max_risk / stop_pct
    return notional

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    # expected: {"action":"BUY","symbol":"AAPL","price":175.23}
    action = data.get("action")
    symbol = data.get("symbol")
    price = float(data.get("price", 0))
    if action not in ["BUY","SELL"]:
        return jsonify({"status":"ignored","reason":"unknown action"}), 400

    equity = get_account_equity()
    notional = calc_position_notional(equity, price, stop_pct=0.06)  # 6% stop
    qty = round(notional / price, 6)  # fractional allowed

    if action == "BUY":
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='limit',
            time_in_force='day',
            limit_price=str(price)
        )
    else:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='day'
        )
    return jsonify({"status":"order_sent","order_id":order.id}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
