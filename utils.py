from alpaca_trade_api.rest import REST
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, MAX_RISK_PER_TRADE

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

def calculate_position_size(account, price):
    cash = float(account.cash)
    risk_amount = cash * MAX_RISK_PER_TRADE
    qty = max(int(risk_amount / price), 1)
    return qty

def submit_order(symbol, qty, side):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )
        return order
    except Exception as e:
        print(f"Order failed: {e}")
        return None
