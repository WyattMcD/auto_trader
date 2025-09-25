#!/usr/bin/env python3
"""
auto_trader.py

Automated SMA watcher + Alpaca order executor (paper mode by default).

Drop into your project root. Configure via the top constants or .env variables.

Run:
    source .venv/bin/activate
    python auto_trader.py

Stop with Ctrl-C. To run 24/7 use screen/systemd/docker.
"""

# use external watchlist if present
#!/usr/bin/env python3
"""
auto_trader.py

Automated SMA watcher + Alpaca order executor (paper mode by default).
"""

# ----------------------------
# CONFIG LOAD, IMPORTS, SANITY
# ----------------------------
import os
import time
import math
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf

# Safety / sizing overrides (add near other config constants)
MIN_PRICE = float(os.getenv("MIN_PRICE", "5.0"))            # skip tickers cheaper than this
MAX_NOTIONAL_PCT = float(os.getenv("MAX_NOTIONAL_PCT", "0.10"))  # max notional per trade as pct of account
MAX_SHARES = int(os.getenv("MAX_SHARES", "1000"))          # hard cap on share count
FLOOR_SHARES = os.getenv("FLOOR_SHARES", "int")            # "int" or "fractional"


# load .env (if exists)
load_dotenv()

# --- sanitize API base and load keys ---
API_BASE_RAW = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets").strip()
if "/v2" in API_BASE_RAW:
    API_BASE = API_BASE_RAW.split("/v2")[0].rstrip("/")
else:
    API_BASE = API_BASE_RAW.rstrip("/")

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

# logging basic config
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info(f"Using Alpaca API base: {API_BASE}")
logging.info(f"API key present: {bool(API_KEY)}")

# Try to import WATCHLIST from config.py (your file). Fallback to env string.
try:
    from config import WATCHLIST as WATCHLIST
    logging.info("Loaded WATCHLIST from config.py")
except Exception as e:
    logging.warning("config.py not found or import failed; falling back to WATCHLIST env. Error: %s", e)
    raw = os.getenv("WATCHLIST", "AAPL,MSFT,AMZN,NVDA,AMD,TSLA,GOOGL,META")
    WATCHLIST = [s.strip().upper() for s in raw.split(",") if s.strip()]

# Ensure WATCHLIST is a list
if isinstance(WATCHLIST, str):
    WATCHLIST = [s.strip().upper() for s in WATCHLIST.split(",") if s.strip()]
elif not isinstance(WATCHLIST, (list, tuple)):
    raise SystemExit("WATCHLIST must be a list in config.py or a comma-separated env var")

logging.info("Final WATCHLIST ({} symbols): {}".format(len(WATCHLIST), WATCHLIST))

# Alpaca client import (with helpful error if missing)
try:
    from alpaca_trade_api.rest import REST, APIError
except ImportError:
    raise SystemExit("Missing dependency: alpaca-trade-api. Activate your .venv and run: pip install alpaca-trade-api")

# Strategy / risk settings (these will be used below)
SMA_LEN = int(os.getenv("SMA_LEN", "20"))
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))   # how often to scan
MAX_RISK_PCT = float(os.getenv("MAX_RISK_PCT", "0.05"))  # 5% per-trade by default
STOP_PCT = float(os.getenv("STOP_PCT", "0.06"))          # stop distance for sizing calc
ACCOUNT_DD_LIMIT = float(os.getenv("ACCOUNT_DD_LIMIT", "0.20"))  # pause if drawdown reached
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "6"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "60"))  # don't re-enter same ticker within this window
ORDER_TYPE = os.getenv("ORDER_TYPE", "market")  # 'market' or 'limit'

LOG_CSV = os.getenv("TRADE_LOG_CSV", "auto_trade_log.csv")
STATE_FILE = os.getenv("STATE_FILE", "auto_state.json")

# quick validation
if not API_KEY or not API_SECRET:
    raise SystemExit("Missing Alpaca keys in .env (APCA_API_KEY_ID, APCA_API_SECRET_KEY)")

# Setup Alpaca client object
api = REST(API_KEY, API_SECRET, API_BASE, api_version='v2')

SMA_LEN = int(os.getenv("SMA_LEN", "20"))
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))   # how often to scan
MAX_RISK_PCT = float(os.getenv("MAX_RISK_PCT", "0.05"))  # 5% per-trade by default
STOP_PCT = float(os.getenv("STOP_PCT", "0.06"))          # stop distance for sizing calc
ACCOUNT_DD_LIMIT = float(os.getenv("ACCOUNT_DD_LIMIT", "0.20"))  # pause if drawdown reached
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "6"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "60"))  # don't re-enter same ticker within this window
ORDER_TYPE = os.getenv("ORDER_TYPE", "market")  # 'market' or 'limit'

LOG_CSV = os.getenv("TRADE_LOG_CSV", "auto_trade_log.csv")
STATE_FILE = os.getenv("STATE_FILE", "auto_state.json")

# quick validation
if not API_KEY or not API_SECRET:
    raise SystemExit("Missing Alpaca keys in .env (APCA_API_KEY_ID, APCA_API_SECRET_KEY)")

# ----------------------------
# Setup
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
api = REST(API_KEY, API_SECRET, API_BASE, api_version='v2')

# load or init state
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "r") as fh:
        state = json.load(fh)
else:
    state = {"last_signal": {}, "positions": {}, "peak_equity": None, "last_scan": None}

# ensure log CSV
if not os.path.exists(LOG_CSV):
    pd.DataFrame(columns=["timestamp","ticker","action","signal_price","qty","order_id","status","notes","equity","cash"]).to_csv(LOG_CSV, index=False)

# ----------------------------
# Utility functions
# ----------------------------
def save_state():
    with open(STATE_FILE, "w") as fh:
        json.dump(state, fh, indent=2)

def get_account_info():
    acct = api.get_account()
    return {"equity": float(acct.equity), "cash": float(acct.cash)}

def update_peak_equity(equity):
    peak = state.get("peak_equity") or equity
    if equity > peak:
        state["peak_equity"] = equity
    return state["peak_equity"]

def account_paused():
    acct = get_account_info()
    equity = acct["equity"]
    peak = state.get("peak_equity") or equity
    if equity > peak:
        state["peak_equity"] = equity
        save_state()
        return False, 0.0
    drawdown = (peak - equity) / peak if peak > 0 else 0.0
    if drawdown >= ACCOUNT_DD_LIMIT:
        return True, drawdown
    return False, drawdown

def fetch_history_yf(ticker, lookback_days=60, interval="1d"):
    """
    Robust wrapper around yfinance.download.
    Returns a DataFrame with standard columns (Open, High, Low, Close, Volume, Adj Close)
    or None if download fails or no data.
    """
    try:
        # yfinance can sometimes accept a list or a string; make sure we pass a string
        symbol = str(ticker).strip().upper()
        # request slightly more days to ensure we have SMA window
        df = yf.download(symbol, period=f"{lookback_days}d", interval=interval, progress=False)
        if df is None or df.empty:
            return None

        # If columns are MultiIndex (happens when yf.download is called with lists),
        # try to extract the per-symbol columns by taking xs on the second level if ticker present.
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            # Try to get columns for this ticker
            try:
                if symbol in df.columns.get_level_values(1):
                    df = df.xs(symbol, axis=1, level=1)
                else:
                    # fallback: collapse multiindex by taking first element of tuple
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            except Exception:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # Ensure we have a 'Close' column
        if "Close" not in df.columns:
            # try alternative names
            possible = [c for c in df.columns if "close" in str(c).lower()]
            if possible:
                df = df.rename(columns={possible[0]: "Close"})
            else:
                return None

        return df
    except Exception as e:
        logging.exception("YFinance download error for %s: %s", ticker, e)
        return None


def compute_sma_signal(ticker):
    """
    Compute SMA crossover signal robustly and return dict or None.
    Uses numeric scalars for comparisons to avoid Series-vs-Series problems.
    """
    try:
        df = fetch_history_yf(ticker, lookback_days=max(90, SMA_LEN*3), interval="1d")
        if df is None or len(df) < SMA_LEN + 2:
            return None

        # compute SMA on Close
        df = df.copy()
        df["sma"] = df["Close"].rolling(SMA_LEN).mean()

        # Ensure we have enough non-NaN SMA values at the tail
        if df["sma"].iloc[-1] is None or pd.isna(df["sma"].iloc[-1]):
            return None

        # Extract scalar values (floats) for comparison (avoids Series comparisons)
        try:
            prev_close = float(df["Close"].iat[-2])
            last_close = float(df["Close"].iat[-1])
            prev_sma = float(df["sma"].iat[-2])
            last_sma = float(df["sma"].iat[-1])
        except Exception as e:
            logging.exception("Failed to extract scalar values for %s: %s", ticker, e)
            return None

        signal = None
        # Long signal: prev <= prev_sma and last > last_sma
        if prev_close <= prev_sma and last_close > last_sma:
            signal = "buy"
        # Exit/short signal
        elif prev_close >= prev_sma and last_close < last_sma:
            signal = "sell"

        return {
            "ticker": str(ticker).upper(),
            "signal": signal,
            "price": float(last_close),
            "sma": float(last_sma),
            "time": str(df.index[-1].date())
        }
    except Exception as e:
        logging.exception("compute_sma_signal error for %s: %s", ticker, e)
        return None

def calc_shares_for_risk(equity, risk_pct, entry_price, stop_pct):
    """
    Safer sizing:
     - skip very low-priced tickers (MIN_PRICE)
     - cap by both risk-dollar and a max-notional-per-trade percent
     - hard cap by MAX_SHARES
     - returns 0.0 if trade should be skipped
    """
    # sanitiy
    if entry_price <= 0 or stop_pct <= 0 or equity <= 0:
        return 0.0

    # skip very cheap tickers (avoid insane share counts)
    if entry_price < MIN_PRICE:
        logging.info("Entry price %.4f < MIN_PRICE %.2f -> skipping", entry_price, MIN_PRICE)
        return 0.0

    max_dollars_by_risk = equity * risk_pct
    # shares by risk constraint
    shares_by_risk = max_dollars_by_risk / (entry_price * stop_pct)

    # shares by notional cap
    shares_by_notional = (equity * MAX_NOTIONAL_PCT) / entry_price

    # pick the more conservative (min)
    shares = min(shares_by_risk, shares_by_notional, MAX_SHARES)

    # round shares conservatively:
    # if you want integer shares for >= $1 tickers, floor; else allow 3-decimal fractional shares
    if FLOOR_SHARES == "int" or entry_price >= 1.0:
        shares = math.floor(shares)
    else:
        # allow limited fractional precision
        shares = math.floor(shares * 1000) / 1000.0

    if shares <= 0:
        return 0.0
    return shares


def place_order_market_buy(ticker, qty):
    try:
        order = api.submit_order(symbol=ticker, qty=qty, side='buy', type='market', time_in_force='day')
        return order
    except APIError as e:
        logging.exception("Alpaca APIError placing market buy for %s: %s", ticker, e)
        return None

def place_order_limit_buy(ticker, qty, limit_price):
    try:
        order = api.submit_order(symbol=ticker, qty=qty, side='buy', type='limit', time_in_force='day', limit_price=str(limit_price))
        return order
    except APIError as e:
        logging.exception("Alpaca APIError placing limit buy for %s: %s", ticker, e)
        return None

def log_trade_row(row):
    df = pd.read_csv(LOG_CSV)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_CSV, index=False)

# ----------------------------
# Main scanning / execution loop
# ----------------------------
def run_scan_once():
    # check account pause
    paused, drawdown = account_paused()
    if paused:
        logging.error("Account paused due to drawdown %.2f%%", drawdown*100)
        return

    acct = get_account_info()
    equity = acct["equity"]

    # update peak
    update_peak_equity(equity)
    save_state()

    # get existing positions count
    open_positions = api.list_positions()
    open_symbols = set([p.symbol for p in open_positions])
    if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
        logging.info("Max concurrent positions reached (%d). Skipping new entries.", MAX_CONCURRENT_POSITIONS)
        return

    for ticker in WATCHLIST:
        try:
            ticker = ticker.strip().upper()
            # cooldown check
            last = state["last_signal"].get(ticker)
            if last:
                last_time = datetime.fromisoformat(last.get("time")) if last.get("time") else None
                if last_time and datetime.utcnow() - last_time < timedelta(minutes=COOLDOWN_MINUTES):
                    # skip until cooldown passes
                    continue

            res = compute_sma_signal(ticker)
            if res is None:
                logging.debug("No data for %s", ticker)
                continue
            sig = res["signal"]
            price = res["price"]
            logging.debug("Ticker %s signal=%s price=%.2f sma=%.2f", ticker, sig, price, res["sma"])

            # if buy signal and we don't already have a position in ticker -> consider entry
            if sig == "buy" and ticker not in open_symbols:
                # sizing
                qty = calc_shares_for_risk(equity, MAX_RISK_PCT, price, STOP_PCT)
                # sanity: if qty < tiny threshold (e.g., $1 of position) skip
                if qty <= 0 or qty * price < 1.0:
                    logging.info("Sizing for %s returned qty=%s (insufficient). Skipping.", ticker, qty)
                    continue

                # place order
                if ORDER_TYPE == "market":
                    order = place_order_market_buy(ticker, qty)
                else:
                    order = place_order_limit_buy(ticker, qty, price)
                order_id = getattr(order, "id", None) if order else None
                status = getattr(order, "status", None) if order else "failed"
                notes = ""
                logging.info("Placed order for %s qty=%s id=%s status=%s", ticker, qty, order_id, status)

                # record state and log
                state["last_signal"][ticker] = {"time": datetime.utcnow().isoformat(), "signal": "buy", "price": price}
                state["positions"][ticker] = {"entry_time": datetime.utcnow().isoformat(), "qty": qty, "order_id": order_id}
                save_state()

                log_trade_row({
                    "timestamp": datetime.utcnow().isoformat(),
                    "ticker": ticker,
                    "action": "BUY",
                    "signal_price": price,
                    "qty": qty,
                    "order_id": order_id,
                    "status": status,
                    "notes": notes,
                    "equity": equity,
                    "cash": acct["cash"]
                })
                # refresh positions and potentially throttle more entries
                open_positions = api.list_positions()
                open_symbols = set([p.symbol for p in open_positions])
                if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
                    logging.info("Reached max concurrent positions after entry.")
                    break

            # if sell/exit signal and we already have position -> exit (market sell)
            elif sig == "sell" and ticker in open_symbols:
                # place market sell for quantity we hold
                pos = next((p for p in open_positions if p.symbol == ticker), None)
                if not pos:
                    continue
                qty = float(pos.qty)
                try:
                    order = api.submit_order(symbol=ticker, qty=qty, side='sell', type='market', time_in_force='day')
                    order_id = getattr(order, "id", None)
                    status = getattr(order, "status", None)
                    notes = "exit signal"
                    logging.info("Placed exit for %s qty=%s id=%s", ticker, qty, order_id)
                    # clear state
                    state["last_signal"][ticker] = {"time": datetime.utcnow().isoformat(), "signal": "sell", "price": price}
                    state["positions"].pop(ticker, None)
                    save_state()
                    log_trade_row({
                        "timestamp": datetime.utcnow().isoformat(),
                        "ticker": ticker,
                        "action": "SELL",
                        "signal_price": price,
                        "qty": qty,
                        "order_id": order_id,
                        "status": status,
                        "notes": notes,
                        "equity": equity,
                        "cash": acct["cash"]
                    })
                except Exception as e:
                    logging.exception("Failed to place exit order %s: %s", ticker, e)
                    continue

        except Exception as e:
            logging.exception("Error scanning %s: %s", ticker, e)
            continue

def main_loop():
    logging.info("Starting auto_trader main loop. Watchlist: %s", WATCHLIST)
    while True:
        try:
            run_scan_once()
            state["last_scan"] = datetime.utcnow().isoformat()
            save_state()
        except Exception as e:
            logging.exception("Top-level error in main loop: %s", e)
        time.sleep(SCAN_INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    main_loop()
