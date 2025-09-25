import alpaca_trade_api as tradeapi
API_KEY = "PASTE_YOUR_PAPER_KEY"
API_SECRET = "PASTE_YOUR_PAPER_SECRET"
API_BASE = "https://paper-api.alpaca.markets"
api = tradeapi.REST(API_KEY, API_SECRET, base_url=API_BASE, api_version='v2')
print("OK? account:", api.get_account().status, "cash:", api.get_account().cash)
