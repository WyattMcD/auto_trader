import requests
import time

# --- Configuration ---
WEBHOOK_URL = "http://127.0.0.1:5001/webhook"  # Flask server must be running
WEBHOOK_SECRET = "74f8158e46f0ae3371295b3e705946b23920677532b85128"  # your secret
WATCHLIST = ["AAPL", "MSFT", "TSLA", "NVDA"]  # subset of symbols for testing
POLL_DELAY = 2  # seconds between sending each webhook

# --- Loop through symbols and send test signals ---
for symbol in WATCHLIST:
    payload = {
        "symbol": symbol,
        "action": "buy",
        "qty": 1
    }
    headers = {
        "Content-Type": "application/json",
        "X-WEBHOOK-SECRET": WEBHOOK_SECRET
    }

    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers)
        print(f"Sent webhook for {symbol}, response: {response.json()}")
    except Exception as e:
        print(f"Error sending webhook for {symbol}: {e}")

    time.sleep(POLL_DELAY)
