#!/usr/bin/env python3
import os
import logging
import signal
import threading
import time
from queue import Queue
from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

# --- Config ---
load_dotenv()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change_this")
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
API_BASE = os.getenv("APCA_API_BASE", "https://paper-api.alpaca.markets")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5001"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes")
LOG_FILE = os.getenv("LOG_FILE", "webhook_server.log")
WATCHLIST = os.getenv("WATCHLIST", None)
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))

if not API_KEY or not API_SECRET:
    raise RuntimeError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set.")

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Alpaca Client ---
api = tradeapi.REST(API_KEY, API_SECRET, base_url=API_BASE, api_version='v2')

# --- Flask App ---
app = Flask(__name__)

# --- Queue & Worker ---
signal_queue = Queue()
worker_thread = None
worker_stop_event = threading.Event()

# --- Helper Functions ---

def download_data(symbol, period="60d", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        if df.empty or len(df) < 2:
            logger.warning(f"{symbol} returned insufficient data")
            return None
        df = df.reset_index()
        df.rename(columns={"Close": "close"}, inplace=True)
        return df[['close']]
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {e}")
        return None

def check_signal(df, short=10, long=50):
    if df is None or len(df) < long:
        return None
    df['fast_sma'] = df['close'].rolling(window=short).mean()
    df['slow_sma'] = df['close'].rolling(window=long).mean()
    if df['fast_sma'].iloc[-2] < df['slow_sma'].iloc[-2] and df['fast_sma'].iloc[-1] > df['slow_sma'].iloc[-1]:
        return "buy"
    elif df['fast_sma'].iloc[-2] > df['slow_sma'].iloc[-2] and df['fast_sma'].iloc[-1] < df['slow_sma'].iloc[-1]:
        return "sell"
    return None

def place_order(symbol, side, qty=1, notional=None, order_type="market", time_in_force="day"):
    logger.info(f"Placing order: {symbol} {side} qty={qty} notional={notional}")
    if DRY_RUN:
        logger.info("DRY_RUN enabled. Order not sent.")
        return {"status": "dry_run"}
    kwargs = {"symbol": symbol, "side": side, "type": order_type, "time_in_force": time_in_force}
    if qty:
        kwargs["qty"] = qty
    elif notional:
        kwargs["notional"] = str(notional)
    try:
        order = api.submit_order(**kwargs)
        return {"status": "submitted", "order_id": getattr(order, "id", None)}
    except Exception as e:
        logger.error(f"Order error: {e}")
        return {"status": "error", "error": str(e)}

def process_signal(signal_payload):
    symbol = signal_payload.get("symbol")
    action = signal_payload.get("action")
    if not symbol or action not in ("buy", "sell"):
        return {"status": "invalid"}
    return place_order(symbol=symbol, side=action, qty=signal_payload.get("qty", 1))

# --- Worker Loop ---
def worker_loop():
    while not worker_stop_event.is_set():
        try:
            while not signal_queue.empty():
                signal_payload = signal_queue.get()
                process_signal(signal_payload)
            for symbol in WATCHLIST or []:
                df = download_data(symbol)
                action = check_signal(df)
                if action:
                    signal_queue.put({"symbol": symbol, "action": action, "qty": 1})
        except Exception as e:
            logger.error(f"Worker error: {e}")
        time.sleep(POLL_INTERVAL)

def start_worker():
    global worker_thread, worker_stop_event
    if worker_thread and worker_thread.is_alive():
        return False
    worker_stop_event.clear()
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()
    return True

def stop_worker():
    global worker_thread, worker_stop_event
    if worker_thread:
        worker_stop_event.set()
        worker_thread.join(timeout=5)
        return True
    return False

# --- Flask Routes ---
@app.route("/", methods=["GET"])
def root():
    try:
        account = api.get_account()
        return jsonify({
            "account_status": account.status,
            "cash": account.cash,
            "portfolio_value": account.portfolio_value,
            "dry_run": DRY_RUN
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# near top of webhook_server2.py
from auto_trader import handle_signal   # import the handler
from notifier import send_slack         # optional: keep for server-level notifications

@app.route("/webhook", methods=["POST"])
def webhook():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"status":"error","reason":"no_json"}), 400
    # optionally validate secrets, signature, etc.
    # Forward payload to executor
    result = handle_signal(payload)
    # return the result to TradingView / caller
    return jsonify(result), 200


@app.route("/start_auto", methods=["POST"])
def start_auto():
    return jsonify({"started": start_worker()})

@app.route("/stop_auto", methods=["POST"])
def stop_auto():
    return jsonify({"stopped": stop_worker()})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "worker_running": bool(worker_thread and worker_thread.is_alive())})

# --- Graceful Shutdown ---
def handle_shutdown(signum, frame):
    logger.info("Shutting down...")
    stop_worker()
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# --- Run Server ---
if __name__ == "__main__":
    start_worker()
    app.run(host="0.0.0.0", port=FLASK_PORT)

