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
# auto_trader.py (very top)
import os, atexit, signal, errno, uuid
from typing import Optional

LOCK_PATH = "/app/state/auto_trader.lock"
LOCK_ENV_KEY = "AUTO_TRADER_LOCK_ID"

def _pid_running(pid: int, *, lock_id: Optional[str] = None) -> bool:
    """Return True if *pid* looks like a live auto_trader process."""

    if pid <= 0:
        return False

    try:
        os.kill(pid, 0)
    except OSError as exc:  # process definitely gone
        if exc.errno in {errno.ESRCH, errno.ENOENT}:
            return False
        if exc.errno != errno.EPERM:  # treat unknown errors as not running
            return False

    cmdline_path = f"/proc/{pid}/cmdline"
    try:
        with open(cmdline_path, "rb") as fh:
            raw = fh.read().decode("utf-8", "ignore")
    except FileNotFoundError:
        return False
    except Exception:
        # If we cannot read the cmdline we conservatively assume the process is
        # unrelated so that a stale lock does not block startup forever.
        return False

    if not raw:
        return False

    argv = [part for part in raw.split("\0") if part]
    if not argv:
        return False

    # The dev autoreloader executes ``python scripts/run_with_reloader.py -- ...`` and
    # therefore has ``auto_trader.py`` in its argument list even though it is not the
    # process we want to guard against.  To avoid treating the reloader itself as a
    # live trading instance we only consider a PID active when ``auto_trader.py`` is
    # the script being executed directly (the second argument to the Python
    # interpreter) and the reloader helper is absent from the command line.
    try:
        script_arg = argv[1]
    except IndexError:
        script_arg = ""

    script_name = os.path.basename(script_arg)
    if not script_name.endswith("auto_trader.py"):
        return False

    if any("run_with_reloader.py" in os.path.basename(arg) for arg in argv):
        return False

    if lock_id:
        env_path = f"/proc/{pid}/environ"
        expected = f"{LOCK_ENV_KEY}={lock_id}"
        try:
            with open(env_path, "rb") as fh:
                env_raw = fh.read().decode("utf-8", "ignore")
        except FileNotFoundError:
            return False
        except Exception:
            return False

        if not env_raw:
            return False

        entries = [part for part in env_raw.split("\0") if part]
        if expected not in entries:
            return False

    return True
  
    if script_arg.endswith("auto_trader.py") and "scripts/run_with_reloader.py" not in argv:
        return True

    return False

def acquire_lock():
    # make sure state dir exists
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)

    # check existing lock
    if os.path.exists(LOCK_PATH):
        try:
            with open(LOCK_PATH, "r") as f:
                raw = (f.read() or "").strip()
        except Exception:
            old_pid = 0
            old_lock_id = None
        else:
            old_lock_id = None
            if ":" in raw:
                pid_part, lock_part = raw.split(":", 1)
            else:
                pid_part, lock_part = raw, ""
            try:
                old_pid = int((pid_part or "0").strip())
            except Exception:
                old_pid = 0
            lock_part = lock_part.strip()
            old_lock_id = lock_part or None

        # active process? -> exit
        if old_pid and _pid_running(old_pid, lock_id=old_lock_id):
            raise SystemExit(f"Another instance is running (pid {old_pid}). Exiting.")

        # stale -> remove
        try:
            os.remove(LOCK_PATH)
        except FileNotFoundError:
            pass

    # atomically create lock & write our PID
    lock_id = uuid.uuid4().hex
    os.environ[LOCK_ENV_KEY] = lock_id

    fd = os.open(LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as f:
        f.write(f"{os.getpid()}:{lock_id}")

    def _cleanup(*_):
        try:
            os.remove(LOCK_PATH)
        except FileNotFoundError:
            pass

    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (_cleanup(), os._exit(0)))

acquire_lock()

import os
import time
import math
import json
import logging
import tempfile
from datetime import datetime, timedelta, timezone, timezone
now = datetime.now(timezone.utc)
from dotenv import load_dotenv
import pandas as pd
from alpaca.trading.client import TradingClient
from config import API_KEY, API_SECRET, IS_PAPER
from executer import execute_intent
from options.orders import OptionsTrader
import yfinance as yf
from risk import approve_csp_intent
from strategies.options_csp import pick_csp_intent, exit_rules_for_csp
from config import API_KEY, API_SECRET, IS_PAPER, ENABLE_OPTIONS, OPTIONS_UNDERLYINGS
from alpaca.trading.client import TradingClient
from executer import execute_intent
from options.orders import OptionsTrader
from options.data import OptionsData
from notifier import send_slack
from strategies.spreads import pick_bull_put_spread, pick_call_debit_spread
from strategies.spread_exits import bps_should_exit, call_debit_decision
from config import (
    SPREADS_MAX_CANDIDATES_PER_TICK, SPREADS_MAX_RISK_PER_TRADE, SPREADS_MAX_OPEN
)
from datetime import datetime, timedelta, timezone
# init clients (global singletons ok)
equity_trader = TradingClient(API_KEY, API_SECRET, paper=IS_PAPER)
opt_trader    = OptionsTrader(API_KEY, API_SECRET, paper=IS_PAPER)
od            = OptionsData(API_KEY, API_SECRET, paper=IS_PAPER)
from config import (
    API_KEY, API_SECRET, IS_PAPER,
    ENABLE_OPTIONS,
    WATCHLIST,                 # ← use your existing watchlist
    OPTIONS_MAX_CANDIDATES_PER_TICK,
)
# If you use pandas timestamps anywhere:
try:
    import pandas as pd
except Exception:
    pd = None

try:
    from strategies.rsi_momentum import compute_rsi_sma_signal
except Exception:
    compute_rsi_sma_signal = None

# ---- safety helpers: MAX_NOTIONAL enforcement ----
def _safe_float_env(name: str, default: float):
    """
    Read an env var robustly: strip whitespace, remove trailing inline comments,
    and convert to float. Returns default on failure.
    """
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    # remove inline comments after a # (if someone left them)
    raw = raw.split('#', 1)[0].strip()
    raw = raw.replace('"', '').replace("'", "").strip()
    try:
        return float(raw)
    except Exception:
        logging.warning("Invalid float for env %s: %r -> using default %s", name, raw, default)
        return float(default)


def _int_from_env(name: str, default: int) -> int:
    """Read an integer environment variable with graceful fallback."""
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        logging.warning("Invalid int for env %s: %r -> using default %s", name, raw, default)
        return int(default)

def _get_account_equity(api):
    """
    Return account equity (float). If API call fails, return None.
    """
    try:
        acct = api.get_account()
        # some Alpaca clients use .equity, others expose .cash; handle both
        val = getattr(acct, "equity", None) or getattr(acct, "cash", None)
        if val is None:
            return None
        return float(val)
    except Exception:
        logging.exception("Failed to read account equity from Alpaca")
        return None

def compute_allowed_qty_for_buy(api, ticker: str, price: float) -> int:
    """
    Compute the maximum integer quantity we are allowed to buy for `ticker`
    at `price` such that existing_position_value + (qty*price) <= equity * MAX_NOTIONAL_PCT.

    Returns an integer quantity (>=0). Conservative: if equity cannot be read, returns 0.
    """
    MAX_NOTIONAL_PCT = _safe_float_env("MAX_NOTIONAL_PCT", 0.05)
    equity = _get_account_equity(api)
    if equity is None or price is None or price <= 0:
        # be conservative if we can't determine price/equity
        logging.warning("compute_allowed_qty_for_buy: missing equity or price (equity=%s price=%s) -> 0", equity, price)
        return 0
    per_position_cap = equity * float(MAX_NOTIONAL_PCT)

    # current market value for ticker (if position exists)
    try:
        pos = api.get_position(ticker)
        current_value = float(getattr(pos, "market_value", 0) or 0)
    except Exception:
        current_value = 0.0

    allowed_notional = per_position_cap - current_value
    if allowed_notional <= 0:
        return 0
    # floor allowed qty: we must not exceed allowed_notional
    allowed_qty = int(math.floor(allowed_notional / float(price)))
    return max(0, allowed_qty)

# Safety / sizing overrides (add near other config constants)
MIN_PRICE = float(os.getenv("MIN_PRICE", "5.0"))            # skip tickers cheaper than this
MAX_NOTIONAL_PCT = float(os.getenv("MAX_NOTIONAL_PCT", "0.05"))  # max notional per trade as pct of account
MAX_SHARES = int(os.getenv("MAX_SHARES", "1000"))          # hard cap on share count
FLOOR_SHARES = os.getenv("FLOOR_SHARES", "int")            # "int" or "fractional"

# Automatic bracket order + EOD close settings
USE_BRACKET_ORDERS = os.getenv("USE_BRACKET_ORDERS", "True").lower() in ("true","1","yes")
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.30"))   # default 30% take profit
STOP_PCT = float(os.getenv("STOP_PCT", "0.15"))                 # default 15% stop (you already have)


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
CONFIG_MAX_CONCURRENT_POSITIONS = None
try:
    from config import WATCHLIST as WATCHLIST, IS_PAPER
    try:
        from config import MAX_CONCURRENT_POSITIONS as CONFIG_MAX_CONCURRENT_POSITIONS
    except Exception:
        CONFIG_MAX_CONCURRENT_POSITIONS = None

    logging.info("Loaded WATCHLIST from config.py")
except Exception as e:
    logging.warning("config.py not found or import failed; falling back to WATCHLIST env. Error: %s", e)
    raw = os.getenv("WATCHLIST", "AAPL,MSFT,AMZN,NVDA,AMD,TSLA,GOOGL,META")
    WATCHLIST = [s.strip().upper() for s in raw.split(",") if s.strip()]
    CONFIG_MAX_CONCURRENT_POSITIONS = None

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

config_max_positions = None
if CONFIG_MAX_CONCURRENT_POSITIONS is not None:
    try:
        config_max_positions = int(str(CONFIG_MAX_CONCURRENT_POSITIONS).strip())
    except Exception:
        logging.warning(
            "Invalid MAX_CONCURRENT_POSITIONS in config.py: %r — ignoring.",
            CONFIG_MAX_CONCURRENT_POSITIONS,
        )

DEFAULT_MAX_CONCURRENT_POSITIONS = config_max_positions if config_max_positions is not None else 20
raw_env_max_positions = os.getenv("MAX_CONCURRENT_POSITIONS")
MAX_CONCURRENT_POSITIONS = _int_from_env(
    "MAX_CONCURRENT_POSITIONS",
    DEFAULT_MAX_CONCURRENT_POSITIONS,
)

COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "60"))  # don't re-enter same ticker within this window
ORDER_TYPE = os.getenv("ORDER_TYPE", "market")  # 'market' or 'limit'

COUNT_OPTION_POSITIONS_TOWARD_MAX = (
    os.getenv("COUNT_OPTION_POSITIONS_TOWARD_MAX", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)


def _position_counts_toward_limit(position) -> bool:
    """Return True if the position should count toward the concurrency ceiling."""
    asset_class = str(getattr(position, "asset_class", "") or "").lower()
    if not COUNT_OPTION_POSITIONS_TOWARD_MAX and asset_class in {"option", "us_option"}:
        return False
    return True

LOG_CSV = os.getenv("TRADE_LOG_CSV", "auto_trade_log.csv")
STATE_FILE = os.getenv("STATE_FILE", "auto_state.json")
ROTATION_PERSIST_INTERVAL = float(os.getenv("ROTATION_PERSIST_INTERVAL", "300"))
_LAST_SAVED_STATE = None
_LAST_ROTATION_SAVE = 0.0

# quick validation
if not API_KEY or not API_SECRET:
    raise SystemExit("Missing Alpaca keys in .env (APCA_API_KEY_ID, APCA_API_SECRET_KEY)")

# Setup Alpaca client object
api = REST(API_KEY, API_SECRET, API_BASE, api_version='v2')

if raw_env_max_positions is not None:
    logging.info(
        "Max concurrent equity positions limit set to %d (from MAX_CONCURRENT_POSITIONS env)",
        MAX_CONCURRENT_POSITIONS,
    )
elif config_max_positions is not None:
    logging.info(
        "Max concurrent equity positions limit set to %d (from config.py)",
        MAX_CONCURRENT_POSITIONS,
    )
else:
    logging.info(
        "Max concurrent equity positions limit set to %d (internal default)",
        MAX_CONCURRENT_POSITIONS,
    )

if COUNT_OPTION_POSITIONS_TOWARD_MAX:
    logging.info("Option positions will count toward the max concurrent positions limit.")
else:
    logging.info(
        "Option positions will be ignored when enforcing the max concurrent positions limit."
    )

# load or init state
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "r") as fh:
        state = json.load(fh)
else:
    state = {"last_signal": {}, "positions": {}, "peak_equity": None, "last_scan": None}

# ensure new option-tracking containers exist
state.setdefault("options_positions", {})   # single-leg option entries keyed by contract symbol
state.setdefault("option_spreads", {})      # multi-leg spreads keyed by combined leg symbols
state.setdefault("rotation", {})            # round-robin cursors (e.g. options watchlist scans)

try:
    _LAST_SAVED_STATE = json.loads(json.dumps(state, sort_keys=True))
except Exception:
    _LAST_SAVED_STATE = None
else:
    _LAST_ROTATION_SAVE = time.monotonic()

atexit.register(lambda: save_state(force=True))

# ensure log CSV
if not os.path.exists(LOG_CSV):
    pd.DataFrame(columns=["timestamp","ticker","action","signal_price","qty","order_id","status","notes","equity","cash"]).to_csv(LOG_CSV, index=False)

def to_aware_datetime(val):
    """
    Convert val (None | datetime | str | pandas.Timestamp) -> timezone-aware datetime in UTC.
    Returns None if val is falsy.
    """
    if not val:
        return None

    # If pandas is available and it's a pandas Timestamp
    if pd is not None and isinstance(val, pd.Timestamp):
        dt = val.to_pydatetime()
    elif isinstance(val, datetime):
        dt = val
    else:
        # try parsing ISO formats first (handles most `.isoformat()` strings)
        try:
            dt = datetime.fromisoformat(val)
        except Exception:
            # fallback: try common formats (adjust if you use different formats)
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                try:
                    dt = datetime.strptime(val, fmt)
                    break
                except Exception:
                    dt = None
            if dt is None:
                # last-resort: raise so you know there is an unexpected format
                raise

    # If it's naive, assume UTC (because earlier code used utcnow())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

# ----------------------------
# Utility functions
# ----------------------------
def _snapshot_state(obj):
    try:
        return json.loads(json.dumps(obj, sort_keys=True))
    except Exception:
        logging.exception("Unable to snapshot state for persistence")
        return None


def save_state(*, force: bool = False):
    """Persist state to disk while avoiding noisy writes."""

    global _LAST_SAVED_STATE, _LAST_ROTATION_SAVE

    snapshot = _snapshot_state(state)
    if snapshot is None:
        return

    prior = _LAST_SAVED_STATE
    rotation_changed = True
    if prior is not None:
        rotation_changed = snapshot.get("rotation") != prior.get("rotation")

    if not force and prior is not None:
        if snapshot == prior:
            return

        snap_no_rot = dict(snapshot)
        snap_no_rot.pop("rotation", None)
        prior_no_rot = dict(prior)
        prior_no_rot.pop("rotation", None)

        if snap_no_rot == prior_no_rot:
            now = time.monotonic()
            if now - _LAST_ROTATION_SAVE < ROTATION_PERSIST_INTERVAL:
                return

    state_dir = os.path.dirname(os.path.abspath(STATE_FILE)) or "."
    try:
        os.makedirs(state_dir, exist_ok=True)
    except Exception:
        logging.exception("Failed to ensure state directory %s exists", state_dir)
        return

    try:
        fd, tmp_path = tempfile.mkstemp(dir=state_dir, prefix=".auto_state.", suffix=".tmp")
    except Exception:
        logging.exception("Failed to create temporary file for state save")
        return

    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(snapshot, fh, indent=2)
        os.replace(tmp_path, STATE_FILE)
    except Exception:
        logging.exception("Failed writing state to %s", STATE_FILE)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return

    _LAST_SAVED_STATE = snapshot
    if rotation_changed or force:
        _LAST_ROTATION_SAVE = time.monotonic()

def get_account_info():
    """Return basic account metrics while preferring cash over margin."""
    acct = api.get_account()

    def _to_float(val, fallback=0.0):
        try:
            return float(val)
        except Exception:
            return float(fallback)

    equity = _to_float(getattr(acct, "equity", 0.0))
    cash = _to_float(getattr(acct, "cash", 0.0))
    buying_power = _to_float(getattr(acct, "buying_power", cash), fallback=cash)

    # Do not allow sizing logic to lean on margin. Once cash is exhausted we
    # return 0 so new positions are skipped.  `buying_power` can be higher than
    # cash for margin accounts, so clamp to the smaller positive amount.
    available_funds = max(0.0, min(cash, buying_power))

    return {
        "equity": equity,
        "cash": cash,
        "buying_power": buying_power,
        "available_funds": available_funds,
    }

def update_peak_equity(equity):
    peak = state.get("peak_equity")
    if peak is None or equity > peak:
        state["peak_equity"] = equity
        return True, equity
    return False, peak

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

# put these config defaults near the top if not already present
# SAFER SIZING - drop-in replacement
MIN_PRICE = float(os.getenv("MIN_PRICE", "5.0"))
MAX_NOTIONAL_PCT = float(os.getenv("MAX_NOTIONAL_PCT", "0.05"))
MAX_SHARES = int(os.getenv("MAX_SHARES", "1000"))
FLOOR_SHARES = os.getenv("FLOOR_SHARES", "int")
DEFAULT_MAX_RISK_PCT = float(os.getenv("MAX_RISK_PCT", "0.005"))

def calc_shares_for_risk(equity, available_funds, risk_pct, entry_price, stop_pct):
    try:
        equity = float(equity)
        entry_price = float(entry_price)
        stop_pct = float(stop_pct)
        risk_pct = float(risk_pct or DEFAULT_MAX_RISK_PCT)
        available_funds = float(available_funds) if available_funds is not None else equity
    except Exception:
        return 0.0

    if equity <= 0 or entry_price <= 0 or stop_pct <= 0:
        return 0.0

    # Do not size new trades if we do not have un-margined cash available.
    available_funds = max(0.0, min(equity, available_funds))
    if available_funds <= 0:
        logging.info(
            "Skipping entry: no available cash/buying power without using margin (equity=%.2f, available=%.2f)",
            equity,
            available_funds,
        )
        return 0.0

    if entry_price < MIN_PRICE:
        logging.info("Skipping ticker: price %.2f < MIN_PRICE %.2f", entry_price, MIN_PRICE)
        return 0.0

    risk_dollars = equity * risk_pct
    shares_by_risk = risk_dollars / (entry_price * stop_pct)

    per_trade_dollar_cap = min(equity * MAX_NOTIONAL_PCT, available_funds)
    shares_by_notional = per_trade_dollar_cap / entry_price
    shares = min(shares_by_risk, shares_by_notional, MAX_SHARES)

    if FLOOR_SHARES == "int" or entry_price >= 1.0:
        shares = math.floor(shares)
    else:
        shares = math.floor(shares * 1000) / 1000.0

    if shares <= 0:
        return 0.0
    return shares

def compute_atr_stop_pct(ticker, atr_period=14, atr_multiplier=None):
    """
    Robust ATR-based stop percentage.
    Returns a float stop_pct (like 0.06) or None on failure.
    Handles yfinance MultiIndex columns, NaNs, and ensures scalar outputs.
    """
    try:
        atr_multiplier = float(atr_multiplier or os.getenv("STOP_ATR_MULTIPLIER", "1.5"))

        symbol = str(ticker).strip().upper()
        df = yf.download(symbol, period="60d", interval="1d", progress=False)
        if df is None or df.empty:
            logging.debug("compute_atr_stop_pct: no data for %s", symbol)
            return None

        # Normalize multiindex columns if present
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            try:
                if symbol in df.columns.get_level_values(1):
                    df = df.xs(symbol, axis=1, level=1)
                else:
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            except Exception:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # Ensure required columns are present
        if not all(col in df.columns for col in ("High", "Low", "Close")):
            logging.debug("compute_atr_stop_pct: required columns missing for %s", symbol)
            return None

        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)

        # True Range calculation
        prev_close = close.shift(1)
        tr_df = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1)
        tr = tr_df.max(axis=1)

        atr_series = tr.rolling(atr_period).mean()
        if atr_series.empty or atr_series.isna().all():
            logging.debug("compute_atr_stop_pct: ATR series empty or all NaN for %s", symbol)
            return None

        # get scalar ATR (last non-NA)
        atr = atr_series.dropna().iloc[-1]
        # defensive: if atr is not scalar (rare), convert
        if isinstance(atr, (pd.Series, pd.DataFrame)):
            try:
                atr = float(atr.values[-1])
            except Exception:
                atr = float(atr.item())

        last_close = close.dropna().iloc[-1]
        # make sure last_close is scalar
        if isinstance(last_close, (pd.Series, pd.DataFrame)):
            try:
                last_close = float(last_close.values[-1])
            except Exception:
                last_close = float(last_close.item())

        # sanity checks
        if pd.isna(atr) or pd.isna(last_close) or last_close == 0:
            logging.debug("compute_atr_stop_pct: invalid atr/close for %s atr=%s close=%s", symbol, atr, last_close)
            return None

        stop_pct = (atr * atr_multiplier) / float(last_close)
        # clamp to reasonable range
        stop_pct = max(0.01, min(0.20, float(stop_pct)))
        return stop_pct

    except Exception:
        logging.exception("compute_atr_stop_pct failed for %s", ticker)
        return None


def place_market_buy_and_attach_stops(ticker, qty, entry_price_hint=None,
                                     stop_pct=None, take_profit_pct=None,
                                     use_bracket=True, use_trailing=False, fill_timeout=60, **kwargs):
    # ---------- enforce per-position MAX_NOTIONAL_PCT ----------
    # determine a conservative price to compute allowed qty:
    if entry_price_hint:
        price_for_sizing = float(entry_price_hint)
    else:
        try:
            recent = yf.download(ticker, period="2d", interval="1d", progress=False)
            price_for_sizing = float(recent["Close"].iloc[-1]) if recent is not None and not recent.empty else None
        except Exception:
            price_for_sizing = None

    # if price_for_sizing couldn't be determined, attempt to use last_quote via Alpaca (best-effort)
    if (price_for_sizing is None or price_for_sizing <= 0) and hasattr(api, "get_last_trade"):
        try:
            lt = api.get_last_trade(ticker)
            price_for_sizing = float(getattr(lt, "price", None) or getattr(lt, "c", None) or price_for_sizing)
        except Exception:
            pass

    allowed_qty = compute_allowed_qty_for_buy(api, ticker, price_for_sizing)
    if allowed_qty <= 0:
        logging.info("Skipping buy for %s: allowed_qty=%s <= 0 under MAX_NOTIONAL_PCT constraint", ticker, allowed_qty)
        return None, None, None

    # cap the requested qty to allowed_qty
    if float(qty) > allowed_qty:
        logging.info("Capping buy qty for %s from %s to allowed %s (price_est=%s)", ticker, qty, allowed_qty,
                     price_for_sizing)
    qty = int(allowed_qty if int(allowed_qty) < int(qty) else int(qty))
    # ---------- end enforcement ----------
    # now continue to submit buy_order as before - e.g. buy_order = api.submit_order(...)
    # Accept legacy/extra kwargs gracefully
    if not use_trailing and "use_trailing" in kwargs:
        try:
            use_trailing = bool(kwargs.pop("use_trailing"))
        except Exception:
            pass

    """
    Submit market buy and attach stop/tp.
    - If use_bracket=True: submit as bracket order (atomic).
    - Else: submit market buy, poll until filled (timeout fill_timeout seconds),
            then place stop and TP for filled quantity (handles partial fills).
    Returns tuple (buy_order, stop_order, tp_order). stop_order/tp_order may be None.
    """

    try:
        qty = float(qty)
        if qty <= 0:
            logging.warning("place_market_buy_and_attach_stops: qty <= 0 for %s", ticker)
            return None, None, None

        # Default stop/take-profit
        if stop_pct is None:
            stop_pct = compute_atr_stop_pct(ticker) or float(os.getenv("STOP_PCT", "0.06"))
        if take_profit_pct is None:
            take_profit_pct = float(os.getenv("TAKE_PROFIT_PCT", "0.12"))

        # --- Option A: submit bracket order atomically (recommended) ---
        if use_bracket:
            try:
                limit_price = None  # None -> market-like behaviour not supported for bracket; use a market buy then attach if you need market fill
                # Alpaca supports submitting a LIMIT order with order_class=bracket; for market+bracket, we use a limit slightly above market as fallback.
                # We'll try submitting a MARKET buy without bracket first — many accounts require bracket on limit orders only.
                # Recommended approach: submit limit bracket using a small slippage buffer from recent price.
                recent = yf.download(ticker, period="2d", interval="1d", progress=False)
                last_close = float(recent["Close"].iloc[-1]) if recent is not None and not recent.empty else entry_price_hint or None
                # compute a small limit, e.g., +0.5% above last close to approximate market but allow bracket
                if last_close:
                    limit_price = round(last_close * 1.005, 2)

                tp_price = round((1.0 + float(take_profit_pct)) * (limit_price or last_close), 2)
                stop_price = round((1.0 - float(stop_pct)) * (limit_price or last_close), 2)

                params = {
                    "symbol": ticker,
                    "qty": qty,
                    "side": "buy",
                    "type": "limit" if limit_price else "market",
                    "time_in_force": "day",
                    "order_class": "bracket",
                }
                if limit_price:
                    params["limit_price"] = str(limit_price)

                params["take_profit"] = {"limit_price": str(tp_price)}
                params["stop_loss"] = {"stop_price": str(stop_price)}

                buy_order = place_order_with_reason(api, ticker, qty, side="buy", ord_type="market",
                                                    reason="signal", reason_detail="mean_reversion",
                                                    extra_meta={"entry_hint": entry_price_hint})
                logging.info("Bracket order submitted for %s qty=%s id=%s", ticker, qty, getattr(buy_order, "id", None))
                # For bracket orders, Alpaca returns one order object referencing the parent with children visible via api.list_orders or get_order.
                # We won't attempt to separately place stop/tp here because they are attached atomically.
                return buy_order, None, None
            except Exception as e:
                logging.exception("Bracket order submit failed for %s: %s (falling back to market+attach)", ticker, e)
                # fall through to market + attach

        # --- Option B: market buy then attach stops after confirmed fill ---
        try:
            buy_order = place_order_with_reason(
                api,
                ticker,
                qty,
                side='buy',
                ord_type='market',
                tif='day',
                reason='signal',
                reason_detail='entry_meanrev'
            )
        except Exception as e:
            logging.exception("Failed initial market buy for %s: %s", ticker, e)
            return None, None, None

        order_id = getattr(buy_order, "id", None)
        # poll until filled (timeout)
        filled_qty = 0.0
        fill_price = None
        start = time.time()
        while time.time() - start < fill_timeout:
            try:
                o = api.get_order(order_id)
                status = getattr(o, "status", "")
                filled_qty = float(getattr(o, "filled_qty", getattr(o, "filled_qty", 0) or 0) or 0)
                # sometimes filled_avg_price is a string
                fprice = getattr(o, "filled_avg_price", None) or getattr(o, "filled_avg", None)
                if fprice:
                    try:
                        fill_price = float(fprice)
                    except Exception:
                        fill_price = None
                if status == "filled" and filled_qty > 0:
                    break
            except Exception:
                # ignore transient errors
                pass
            time.sleep(0.5)

        if filled_qty <= 0:
            # nothing filled — warn and return the buy_order. caller can decide to cancel or retry.
            logging.warning("Buy order %s for %s not filled within timeout (status=%s filled=%s).", order_id, ticker, getattr(buy_order,"status",None), filled_qty)
            return buy_order, None, None

        # if partial fill, only attach stops for the filled amount
        qty_to_protect = int(math.floor(filled_qty)) if float(filled_qty).is_integer() else float(filled_qty)

        # compute stop/tp using actual fill price if available
        if not fill_price:
            # try to read position avg entry as fallback
            try:
                pos = api.get_position(ticker)
                fill_price = float(getattr(pos, "avg_entry_price", None) or getattr(pos, "avg_entry", None) or 0)
            except Exception:
                fill_price = None

        if not fill_price:
            logging.info("Proceeding without precise fill_price; using last close fallback.")
            recent = yf.download(ticker, period="2d", interval="1d", progress=False)
            fill_price = float(recent["Close"].iloc[-1]) if recent is not None and not recent.empty else entry_price_hint or fill_price or 0

        stop_price = round(fill_price * (1.0 - float(stop_pct)), 2)
        tp_price = round(fill_price * (1.0 + float(take_profit_pct)), 2)

        # --- submit fixed stop and take-profit (as before) ---
        stop_order = None
        tp_order = None
        trailing_order = None

        try:
            stop_order = api.submit_order(
                symbol=ticker,
                qty=qty_to_protect,
                side='sell',
                type='stop',
                stop_price=str(stop_price),
                reason='signal',
                reason_detail='limit_entry',
                time_in_force='day'
            )
            logging.info("Fixed stop submitted for %s qty=%s id=%s", ticker, qty_to_protect,
                         getattr(stop_order, "id", None))
        except Exception:
            logging.exception("Failed to submit fixed stop order for %s qty=%s", ticker, qty_to_protect)

        try:
            tp_order = api.submit_order(
                symbol=ticker,
                qty=qty_to_protect,
                side='sell',
                type='limit',
                limit_price=str(tp_price),
                reason='tp',
                reason_detail='take_profit',
                time_in_force='day'
            )
            logging.info("TP submitted for %s qty=%s id=%s", ticker, qty_to_protect, getattr(tp_order, "id", None))
        except Exception:
            logging.exception("Failed to submit take-profit order for %s qty=%s", ticker, qty_to_protect)

        # --- optional trailing-stop logic (create trailing stop once price has advanced) ---
        # Config:
        trail_start_pct = float(
            os.getenv("TRAIL_START_PCT", state.get("positions", {}).get(ticker, {}).get("trail_start_pct", 0.03)))
        trail_pct = float(
            os.getenv("TRAIL_STEP_PCT", state.get("positions", {}).get(ticker, {}).get("trail_step_pct", 0.02)))

        try:
            # fetch a recent price to decide whether to enable trailing immediately
            recent = None
            try:
                recent = yf.download(ticker, period="1d", interval="1m", progress=False)
            except Exception:
                # fallback to daily close if intraday fetch fails
                recent = yf.download(ticker, period="2d", interval="1d", progress=False)

            current_price = None
            if recent is not None and not recent.empty:
                current_price = float(recent["Close"].iloc[-1])
            else:
                current_price = fill_price or (limit_price if 'limit_price' in locals() else None) or 0.0

            # Only attempt trailing if the environment / caller wants it
            want_trailing = False
            if use_trailing:
                want_trailing = True
            elif os.getenv("USE_TRAILING_STOP", "False").lower() in ("1", "true", "yes"):
                want_trailing = True

            if want_trailing and fill_price and current_price >= fill_price * (1.0 + float(trail_start_pct)):
                # create trailing stop immediately and cancel the fixed stop to avoid double-fill
                try:
                    trailing_order = api.submit_order(
                        symbol=ticker,
                        qty=qty_to_protect,
                        side='sell',
                        type='trailing_stop',
                        trail_percent=float(trail_pct),
                        time_in_force='day'
                    )
                    logging.info("Trailing stop submitted for %s qty=%s id=%s trail_pct=%s", ticker, qty_to_protect,
                                 getattr(trailing_order, "id", None), trail_pct)

                    # cancel the fixed stop if it exists (to prevent duplicate sells)
                    if stop_order:
                        try:
                            api.cancel_order(getattr(stop_order, "id", None))
                            logging.info("Cancelled fixed stop %s for %s after creating trailing stop.",
                                         getattr(stop_order, "id", None), ticker)
                            stop_order = None
                        except Exception:
                            logging.exception("Failed to cancel fixed stop %s for %s after creating trailing stop.",
                                              getattr(stop_order, "id", None), ticker)
                except Exception:
                    logging.exception("Failed to submit trailing stop for %s qty=%s", ticker, qty_to_protect)
            else:
                # We did not create a trailing order now. Record that trailing should be activated later.
                # state["positions"] is used elsewhere in your code to manage trailing; keep the metadata.
                logging.debug("Trailing not activated now for %s: current_price=%s fill_price=%s trail_start=%s",
                              ticker, current_price, fill_price, trail_start_pct)

        except Exception:
            logging.exception("Trailing-stop decision block failed for %s", ticker)

        # notify
        try:
            send_slack(f":white_check_mark: BUY FILLED {ticker} qty={qty_to_protect} fill=${fill_price:.2f} stop=${stop_price:.2f} tp=${tp_price:.2f}")
        except Exception:
            pass

        # record trailing metadata if needed (unchanged from previous)
        if use_bracket is False and os.getenv("USE_TRAILING_STOP", "True").lower() in ("1","true","yes"):
            state["positions"][ticker] = state.get("positions", {}).get(ticker, {})
            state["positions"][ticker].update({
                "entry_price": fill_price,
                "qty": qty_to_protect,
                "stop_order_id": getattr(stop_order, "id", None),
                "tp_order_id": getattr(tp_order, "id", None),
                "trail_active": True,
                "trail_last_price": fill_price,
                "trail_anchor_price": fill_price,
                "trail_start_pct": float(os.getenv("TRAIL_START_PCT", "0.03")),
                "trail_step_pct": float(os.getenv("TRAIL_STEP_PCT", "0.02")),
                "trail_max_step": float(os.getenv("TRAIL_MAX_STEP", "0.20")),
                "last_trail_stop": stop_price
            })
            save_state()

        return buy_order, stop_order, tp_order

    except Exception:
        logging.exception("place_market_buy_and_attach_stops: unexpected failure for %s", ticker)
        return None, None, None

import time
import logging
import json
import os
import uuid

def _make_client_tag(reason_short: str, ticker: str = "", extra: str = "") -> str:
    """
    Build a short client_order_id tag that fits Alpine limits.
    Keep it compact: e.g. bot:rs=signal_mm:t=CRM:ex=fast
    """
    ts = int(time.time())
    # sanitize simple fields (no spaces/commas)
    reason = reason_short.replace(" ", "_")[:24]
    ticker = ticker.upper()[:6]
    extra = extra.replace(" ", "_")[:20] if extra else ""
    # short uuid to ensure uniqueness if needed
    shortid = uuid.uuid4().hex[:6]
    tag = f"bot:rs={reason}:t={ticker}:ts={ts}:{shortid}"
    if extra:
        tag += f":{extra}"
    # ensure not excessively long
    return tag[:64]


def record_order_reason(order_obj, reason: str, ticker: str, meta: dict = None):
    """
    Persist order reason into in-memory state and disk so you can audit later.
    Expects `state` and `save_state()` to exist in your module.
    """
    try:
        oid = getattr(order_obj, "id", None) or getattr(order_obj, "client_order_id", None)
        state.setdefault("orders", {})
        state["orders"][oid] = {
            "ticker": ticker,
            "reason": reason,
            "client_order_id": getattr(order_obj, "client_order_id", None),
            "submitted_at": time.time(),
            "meta": meta or {}
        }
        save_state()
    except Exception:
        logging.exception("Failed to record order reason for %s", ticker)


def place_order_with_reason(api, symbol, qty, side, ord_type="market", tif="day",
                            reason="signal", reason_detail="", extra_meta=None,
                            **kwargs):
    """
    Unified wrapper to submit an Alpaca order but tag/log/persist the reason.
    - api: alpaca REST API object
    - symbol, qty, side: as usual
    - reason: short category (e.g. 'signal', 'stop', 'risk', 'manual', 'trailing')
    - reason_detail: human-friendly short detail (e.g. 'mean_reversion', 'RSI_drop')
    - extra_meta: dict stored to state with extra context
    - kwargs forwarded to api.submit_order (like limit_price, stop_price)
    Returns the submitted order object (or raises).
    """
    # build tag
    reason_short = f"{reason[:10]}_{reason_detail[:12]}" if reason_detail else reason[:12]
    tag = _make_client_tag(reason_short, ticker=symbol)
    try:
        params = dict(symbol=symbol, qty=qty, side=side, type=ord_type, time_in_force=tif, **kwargs)
        params["client_order_id"] = tag
        logging.info("Submitting %s order for %s qty=%s reason=%s tag=%s", side.upper(), symbol, qty, reason, tag)
        order = api.submit_order(**params)
        # persist reason for future audit
        record_order_reason(order, f"{reason}:{reason_detail}", symbol, meta=extra_meta)
        # send slack if you have send_slack helper (optional)
        try:
            send_slack(f":memo: ORDER SUBMITTED {symbol} {side.upper()} qty={qty} reason={reason}:{reason_detail} tag={tag}")
        except Exception:
            pass
        return order
    except Exception:
        logging.exception("Order submission failed for %s reason=%s", symbol, reason)
        raise

# Compatibility wrapper so existing code calling place_order_market_buy() still works.
def place_order_market_buy(ticker, qty):
    """
    Backwards-compatible wrapper for old callers.
    Calls the new place_market_buy_and_attach_stops and returns the buy_order object (first element).
    """
    try:
        use_trailing = os.getenv("USE_TRAILING_STOP", "True").lower() in ("1", "true", "yes")
        buy_order, stop_order, tp_order = place_market_buy_and_attach_stops(
            ticker, qty, entry_price_hint=None, stop_pct=None, take_profit_pct=None, use_trailing=use_trailing
        )
        return buy_order
    except Exception:
        logging.exception("Compatibility wrapper place_order_market_buy failed for %s", ticker)
        return None


def place_order_limit_buy(ticker, qty, limit_price, stop_pct=STOP_PCT, take_profit_pct=TAKE_PROFIT_PCT, use_bracket=USE_BRACKET_ORDERS):
    """
    Submits a limit buy. Optionally attaches bracket stop/take-profit.
    """
    try:
        if use_bracket and stop_pct:
            stop_price = round(limit_price * (1.0 - float(stop_pct)), 2)
            take_profit = None
            if take_profit_pct:
                tp_price = round(limit_price * (1.0 + float(take_profit_pct)), 2)
                take_profit = {"limit_price": str(tp_price)}
            stop_loss = {"stop_price": str(stop_price)}

            params = {
                "symbol": ticker,
                "qty": qty,
                "side": "buy",
                "type": "limit",
                "limit_price": str(limit_price),
                "time_in_force": "day",
                "order_class": "bracket",
            }
            if take_profit:
                params["take_profit"] = take_profit
            if stop_loss:
                params["stop_loss"] = stop_loss

            order = api.submit_order(**params)
            return order
        # fallback: plain limit buy
        order = place_order_with_reason(
            api,
            ticker,
            qty,
            side='buy',
            ord_type='limit',
            tif='gtc',
            limit_price=str(limit_price),
            reason='signal',
            reason_detail='limit_entry'
        )
        return order
    except Exception as e:
        logging.exception("Alpaca APIError placing limit buy for %s: %s", ticker, e)
        return None


def log_trade_row(row):
    df = pd.read_csv(LOG_CSV)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_CSV, index=False)


def _occ_strike(sym: str):
    """Extract strike price from OCC option symbol (returns float or None)."""
    try:
        return int(sym[-8:]) / 1000.0
    except Exception:
        return None


def _spread_state_key(legs):
    symbols = sorted([leg.get("symbol", "") for leg in legs if leg.get("symbol")])
    return "|".join(symbols)


def _record_option_entry(intent):
    """Persist metadata about opened option positions for exit monitoring."""
    now_iso = datetime.now(timezone.utc).isoformat()
    if intent.get("asset_class") == "option":
        entry_price = None
        if str(intent.get("type", "")).lower() == "limit":
            try:
                entry_price = float(intent.get("limit_price") or 0.0)
            except Exception:
                entry_price = None
        state["options_positions"][intent["symbol"]] = {
            "side": intent.get("side"),
            "qty": int(intent.get("qty", 1) or 1),
            "entry_price": entry_price,
            "opened": now_iso,
            "strategy": intent.get("strategy") or intent.get("meta", {}).get("why", "option")
        }
        return True

    if intent.get("asset_class") != "option_spread":
        return False

    legs = [
        {
            "symbol": leg.get("symbol"),
            "side": leg.get("side"),
            "type": leg.get("type", "limit"),
            "qty": int(leg.get("qty", 1) or 1),
            "limit_price": leg.get("limit_price")
        }
        for leg in intent.get("legs", [])
        if leg.get("symbol")
    ]
    if not legs:
        return False

    strategy = intent.get("strategy", "spread")
    short_leg = next((leg for leg in legs if str(leg.get("side", "")).lower() == "sell"), None)
    long_leg = next((leg for leg in legs if str(leg.get("side", "")).lower() == "buy"), None)

    entry_credit = None
    entry_debit = None
    width = None

    if short_leg and long_leg:
        try:
            short_price = float(short_leg.get("limit_price") or 0.0)
        except Exception:
            short_price = 0.0
        try:
            long_price = float(long_leg.get("limit_price") or 0.0)
        except Exception:
            long_price = 0.0
        short_strike = _occ_strike(short_leg["symbol"])
        long_strike = _occ_strike(long_leg["symbol"])
        if short_strike is not None and long_strike is not None:
            width = abs(short_strike - long_strike)

        if strategy == "bull_put_spread":
            entry_credit = max(0.0, short_price - long_price)
        elif strategy == "call_debit_spread":
            entry_debit = max(0.0, long_price - short_price)

    spread_key = _spread_state_key(legs)
    state["option_spreads"][spread_key] = {
        "strategy": strategy,
        "legs": legs,
        "entry_credit": entry_credit,
        "entry_debit": entry_debit,
        "width": width,
        "opened": now_iso
    }
    return True


def _submit_option_intent(intent):
    try:
        resp = execute_intent(intent, equity_trader, opt_trader)
    except Exception:
        logging.exception("Failed to submit option intent: %s", intent)
        return False

    # Only track if order submission did not raise.
    if intent.get("asset_class") in {"option", "option_spread"}:
        if _record_option_entry(intent):
            save_state()
    return True


def _looks_optionable(sym: str) -> bool:
    sym = (sym or "").strip().upper()
    return sym.isalnum() and 1 < len(sym) <= 5


def _count_open_short_puts():
    try:
        positions = opt_trader.client.get_all_positions()
    except Exception:
        return 0
    count = 0
    for pos in positions:
        symbol = getattr(pos, "symbol", "")
        qty = float(getattr(pos, "qty", 0) or 0)
        if symbol.endswith("P") and qty < 0:
            count += 1
    return count


def _infer_option_entry_price(symbol: str):
    try:
        pos = opt_trader.client.get_open_position(symbol)
    except Exception:
        return None
    try:
        price = float(getattr(pos, "avg_entry_price", 0) or 0)
    except Exception:
        price = 0.0
    return abs(price) or None


def run_option_entry_cycle():
    if not ENABLE_OPTIONS:
        return

    def _unique_watch_symbols():
        """Return de-duplicated uppercase symbols from option universe + equity watchlist."""
        unique = []
        seen = set()
        for sym in OPTIONS_UNDERLYINGS + WATCHLIST:
            if not sym:
                continue
            cleaned = sym.strip().upper()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                unique.append(cleaned)
        return unique

    def _round_robin_slice(symbols, limit, state_key):
        """Return `limit` symbols using a persistent round-robin cursor."""
        if not symbols:
            return []
        try:
            limit = int(limit) if limit else len(symbols)
        except Exception:
            limit = len(symbols)
        limit = max(1, min(limit, len(symbols)))

        cursor_map = state.setdefault("rotation", {})
        cursor = int(cursor_map.get(state_key, 0) or 0)

        selection = []
        for i in range(limit):
            selection.append(symbols[(cursor + i) % len(symbols)])

        cursor = (cursor + limit) % len(symbols)
        cursor_map[state_key] = cursor
        state["rotation"] = cursor_map
        return selection

    watch_syms = _unique_watch_symbols()

    if not watch_syms:
        return

    open_spreads = len(state.get("option_spreads", {}))
    allow_new_spread = open_spreads < SPREADS_MAX_OPEN

    # ---- Try spreads first (credit, then debit) ----
    spread_candidates = _round_robin_slice(
        watch_syms,
        SPREADS_MAX_CANDIDATES_PER_TICK,
        state_key="spread_candidates",
    )
    if allow_new_spread:
        for sym in spread_candidates:
            try:
                bps = pick_bull_put_spread(od, sym)
            except Exception:
                logging.exception("bull_put_spread scan failed for %s", sym)
                bps = None
            if not bps:
                continue
            if bps.get("risk", {}).get("max_loss", float("inf")) <= SPREADS_MAX_RISK_PER_TRADE:
                logging.info("Submitting bull put spread intent for %s", sym)
                _submit_option_intent(bps)
                return

        for sym in spread_candidates:
            try:
                cds = pick_call_debit_spread(od, sym)
            except Exception:
                logging.exception("call_debit_spread scan failed for %s", sym)
                cds = None
            if not cds:
                continue
            if cds.get("risk", {}).get("net_debit", float("inf")) <= SPREADS_MAX_RISK_PER_TRADE:
                logging.info("Submitting call debit spread intent for %s", sym)
                _submit_option_intent(cds)
                return

    # ---- Fallback: single-leg CSPs ----
    optionable = [sym for sym in watch_syms if _looks_optionable(sym)]
    optionable = _round_robin_slice(
        optionable,
        OPTIONS_MAX_CANDIDATES_PER_TICK or len(optionable),
        state_key="csp_candidates",
    )
    try:
        account = equity_trader.get_account()
    except Exception:
        logging.exception("Failed to pull account for CSP approval")
        account = None
    open_csp_count = _count_open_short_puts()

    for sym in optionable:
        try:
            intent = pick_csp_intent(od, sym)
        except Exception:
            logging.exception("pick_csp_intent failed for %s", sym)
            intent = None
        if not intent or account is None:
            continue
        strike = _occ_strike(intent.get("symbol", ""))
        if strike is None:
            continue
        est_bpr = strike * 100.0
        if approve_csp_intent(intent, account, open_csp_count, est_bpr):
            logging.info("Submitting CSP intent for %s", sym)
            _submit_option_intent(intent)
            return

    # Last resort: try SPY once
    try:
        fallback = pick_csp_intent(od, "SPY")
    except Exception:
        fallback = None
    if fallback and account is not None:
        strike = _occ_strike(fallback.get("symbol", ""))
        if strike is not None:
            est_bpr = strike * 100.0
            if approve_csp_intent(fallback, account, open_csp_count, est_bpr):
                logging.info("Submitting fallback CSP intent for SPY")
                _submit_option_intent(fallback)


def monitor_option_exits():
    if not ENABLE_OPTIONS:
        return

    # ---- Single-leg CSP exits ----
    singles = state.get("options_positions", {})
    single_symbols = list(singles.keys())
    snap_map = {}
    if single_symbols:
        try:
            snaps = od.snapshots_for(single_symbols)
            snap_map = {getattr(s, "symbol", ""): s for s in snaps}
        except Exception:
            logging.exception("Failed to load option snapshots for exits")
            snap_map = {}

    for sym, meta in list(singles.items()):
        entry_price = meta.get("entry_price")
        if not entry_price:
            inferred = _infer_option_entry_price(sym)
            if inferred:
                meta["entry_price"] = inferred
                state["options_positions"][sym] = meta
                save_state()
                entry_price = inferred
        snap = snap_map.get(sym)
        if not entry_price or not snap:
            continue
        decision = exit_rules_for_csp(entry_price, snap)
        if not decision:
            continue
        action, reason = decision
        close_intent = {
            "asset_class": "option",
            "symbol": sym,
            "side": action,
            "type": "market",
            "qty": meta.get("qty", 1),
            "tif": "day",
            "meta": {"reason": reason, "strategy": meta.get("strategy", "csp")}
        }
        try:
            execute_intent(close_intent, equity_trader, opt_trader)
            state["options_positions"].pop(sym, None)
            save_state()
            logging.info("Closed option %s via %s", sym, reason)
        except Exception:
            logging.exception("Failed to close option %s", sym)

    # ---- Spread exits ----
    spreads = state.get("option_spreads", {})
    for key, meta in list(spreads.items()):
        legs = meta.get("legs", [])
        symbols = [leg.get("symbol") for leg in legs if leg.get("symbol")]
        if not symbols:
            continue
        try:
            snaps = od.snapshots_for(symbols)
        except Exception:
            logging.exception("Failed to load spread snapshots for %s", key)
            continue
        snap_lookup = {getattr(s, "symbol", ""): s for s in snaps}
        strategy = meta.get("strategy")
        decision = None

        if strategy == "bull_put_spread":
            short_leg = next((leg for leg in legs if str(leg.get("side", "")).lower() == "sell"), None)
            long_leg = next((leg for leg in legs if str(leg.get("side", "")).lower() == "buy"), None)
            entry_credit = meta.get("entry_credit")
            if short_leg and long_leg and entry_credit:
                short_snap = snap_lookup.get(short_leg["symbol"])
                long_snap = snap_lookup.get(long_leg["symbol"])
                if short_snap and long_snap:
                    decision = bps_should_exit(entry_credit, short_snap, long_snap)
        elif strategy == "call_debit_spread":
            long_leg = next((leg for leg in legs if str(leg.get("side", "")).lower() == "buy"), None)
            short_leg = next((leg for leg in legs if str(leg.get("side", "")).lower() == "sell"), None)
            entry_debit = meta.get("entry_debit")
            width = meta.get("width")
            if long_leg and short_leg and entry_debit and width:
                long_snap = snap_lookup.get(long_leg["symbol"])
                short_snap = snap_lookup.get(short_leg["symbol"])
                if long_snap and short_snap:
                    decision = call_debit_decision(entry_debit, long_snap, short_snap, width)

        if not decision:
            continue

        _, reason = decision
        exit_legs = [
            {
                "symbol": leg["symbol"],
                "side": "buy" if str(leg.get("side", "")).lower() == "sell" else "sell",
                "type": "market",
                "qty": leg.get("qty", 1)
            }
            for leg in legs
        ]
        exit_intent = {
            "asset_class": "option_spread",
            "strategy": strategy,
            "legs": exit_legs,
            "tif": "day",
            "meta": {"reason": reason}
        }
        try:
            execute_intent(exit_intent, equity_trader, opt_trader)
            state["option_spreads"].pop(key, None)
            save_state()
            logging.info("Closed %s via %s", strategy, reason)
        except Exception:
            logging.exception("Failed to close spread %s", key)

# ----------------------------
# Main scanning / execution loop
# ----------------------------
def run_scan_once():
    def update_trailing_stops():
        """
        Called each scan: for each state position with trail_active True,
        check whether we can raise the stop based on new high price.
        """
        try:
            for ticker, pdata in list(state.get("positions", {}).items()):
                if not pdata.get("trail_active"):
                    continue
                qty = pdata.get("qty")
                entry_price = pdata.get("entry_price")
                if not qty or not entry_price:
                    continue

                # fetch recent minute bars to find the high since entry
                try:
                    df = yf.download(ticker, period="5d", interval="1m", progress=False)
                    if df is None or df.empty:
                        continue
                    # only consider bars since entry_time if available
                    # entry_time could be in pdata; fall back to last 2 days
                    high_since = df["High"].max()
                except Exception:
                    logging.exception("update_trailing_stops: yfinance fail for %s", ticker)
                    continue

                trail_anchor = pdata.get("trail_anchor_price", entry_price)
                trail_step = pdata.get("trail_step_pct", float(os.getenv("TRAIL_STEP_PCT", "0.02")))
                last_trail_stop = pdata.get("last_trail_stop", None)

                # Compute new candidate stop: we move stop to (high_since * (1 - trail_step))
                candidate_stop = round(high_since * (1.0 - float(trail_step)), 2)

                # Only raise the stop; never reduce
                if last_trail_stop is None or candidate_stop > last_trail_stop + 0.01:
                    # cancel existing stop order if present
                    old_stop_id = pdata.get("stop_order_id")
                    if old_stop_id:
                        try:
                            api.cancel_order(old_stop_id)
                        except Exception:
                            pass
                    # submit new stop order
                    try:
                        new_stop = api.submit_order(symbol=ticker, qty=qty, side='sell', type='stop',
                                                    stop_price=str(candidate_stop), time_in_force='day')
                        # update state
                        pdata["stop_order_id"] = getattr(new_stop, "id", None)
                        pdata["last_trail_stop"] = candidate_stop
                        state["positions"][ticker] = pdata
                        save_state()
                        send_slack(
                            f":arrow_up: Raised stop for {ticker} to ${candidate_stop:.2f} (based on high ${high_since:.2f})")
                    except Exception:
                        logging.exception("Failed to submit trailing stop for %s", ticker)
        except Exception:
            logging.exception("update_trailing_stops: general failure")

    # Run options lifecycle on each tick (exits before new entries)
    try:
        monitor_option_exits()
    except Exception:
        logging.exception("Option exit monitor failed")

    try:
        run_option_entry_cycle()
    except Exception:
        logging.exception("Option entry cycle failed")
    # check account pause
    paused, drawdown = account_paused()
    if paused:
        logging.error("Account paused due to drawdown %.2f%%", drawdown*100)
        if paused:
            logging.error("Account paused due to drawdown %.2f%%", drawdown * 100)
            send_slack(f":warning: TRADING PAUSED — drawdown {drawdown:.2%}. Manual review required.")
            return
        return

    acct = get_account_info()
    equity = acct["equity"]
    available_funds = acct.get("available_funds", equity)

    # update peak
    peak_updated, _ = update_peak_equity(equity)
    if peak_updated:
        save_state()

    # get existing positions count
    open_positions = api.list_positions()

    counted_positions = [p for p in open_positions if _position_counts_toward_limit(p)]
    open_symbols = set([p.symbol for p in counted_positions])
    can_open_new_positions = len(counted_positions) < MAX_CONCURRENT_POSITIONS
    if not can_open_new_positions:
        logging.info(
            "Max concurrent positions reached (%d). Counting %d qualifying positions (total open positions: %d). Suppressing new entries but continuing exit checks.",
            MAX_CONCURRENT_POSITIONS,
            len(counted_positions),
            len(open_positions),
        )

    def pick_top_n_signals(candidates, n=3):
        """
        candidates: list of dicts { 'ticker':..., 'score':..., 'price':... }
        Sorts by score desc and returns top n tickers.
        """
        if not candidates:
            return []
        sorted_c = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_c[:n]

    for ticker in WATCHLIST:
        try:
            ticker = ticker.strip().upper()
            # cooldown check
            last = state["last_signal"].get(ticker)
            if last:
                last_time = to_aware_datetime(last.get("time"))
                if last_time:
                    now = datetime.now(timezone.utc)
                    if (now - last_time) < timedelta(minutes=COOLDOWN_MINUTES):
                        logging.debug(
                            "Skipping %s — still within %s minute cooldown window",
                            ticker,
                            COOLDOWN_MINUTES,
                        )
                        continue

            # --- Multi-strategy evaluation (prefers RSI+SMA if it has a signal) ---
            sma_res = compute_sma_signal(ticker)
            rsi_res = None
            if compute_rsi_sma_signal:
                try:
                    rsi_res = compute_rsi_sma_signal(ticker, sma_len=SMA_LEN)
                except Exception:
                    rsi_res = None

            # pick chosen result: prefer RSI+SMA when it gives a non-None signal
            # auto_trader.py
            from options.data import OptionsData
            from options.orders import OptionsTrader
            from strategies.options_cash_puts import screen_cash_secured_put
            from config import ENABLE_OPTIONS

            # init once
            od = OptionsData(API_KEY, API_SECRET, paper=IS_PAPER)
            opt_trader = OptionsTrader(API_KEY, API_SECRET, paper=IS_PAPER)

            def run_strategies():
                intents = []

                # --- your existing EQUITY strategies here ---
                # intents.extend(run_equity_strats(...))

                # --- OPTIONS: pick CSPs from your WATCHLIST ---
                if ENABLE_OPTIONS:
                    # 1) Build a clean candidate list from your WATCHLIST
                    #    - uppercase symbols
                    #    - drop obvious non-option underlyings (crypto tickers, 1-char oddballs except 'A', etc.)
                    wl = [s.strip().upper() for s in WATCHLIST if isinstance(s, str)]

                    if ENABLE_OPTIONS:
                        # Try a few watchlist names for a Bull Put Spread (credit, defined risk)
                        candidates = [s.strip().upper() for s in WATCHLIST if isinstance(s, str)]
                        candidates = candidates[:SPREADS_MAX_CANDIDATES_PER_TICK]
                    submitted = False
                    for u in candidates:
                        # First, try BPS (credit spread)
                        bps = pick_bull_put_spread(od, u)
                        if bps and bps["risk"]["max_loss"] <= SPREADS_MAX_RISK_PER_TRADE:
                            # TODO: you can add an account-wide open-spreads counter here if you track it
                            intents.append(bps)
                            submitted = True
                            break

                    if not submitted:
                        # If no BPS opportunity, optionally try a small call debit spread
                        for u in candidates:
                            cds = pick_call_debit_spread(od, u)
                            if cds and cds["risk"]["net_debit"] <= SPREADS_MAX_RISK_PER_TRADE:
                                intents.append(cds)
                                submitted = True
                                break

                    if submitted:
                        return intents
                    def looks_optionable(sym: str) -> bool:
                        # quick guard: real US equities/ETFs are usually letters & digits, not crypto pairs
                        return sym.isalnum() and len(sym) <= 5  # lets SPY, AAPL, NVDA, etc. through

                    candidates = [s for s in wl if looks_optionable(s)]
                    # Keep it tame each tick (avoid hammering APIs)
                    candidates = candidates[
                        :OPTIONS_MAX_CANDIDATES_PER_TICK] if "OPTIONS_MAX_CANDIDATES_PER_TICK" in globals() else candidates[
                        :8]

                    # 2) Try each until we get a valid CSP intent (liquidity/delta/DTE filters inside pick_csp_intent)
                    for u in candidates:
                        intent = pick_csp_intent(od, u)
                        if not intent:
                            continue

                        # 3) Risk check (rough CSP collateral from OCC strike)
                        sym = intent["symbol"]
                        strike = int(sym[-8:]) / 1000.0
                        est_bpr = strike * 100.0
                        acct = equity_trader.get_account()
                        open_csp_count = 0  # TODO: replace with your real count if you track it

                        if approve_csp_intent(intent, acct, open_csp_count, est_bpr):
                            intents.append(intent)
                            break  # submit at most one options order per tick

                    # 4) Fallback (optional): if nothing in watchlist qualifies, try SPY
                    if not intents:
                        fallback = pick_csp_intent(od, "SPY")
                        if fallback:
                            sym = fallback["symbol"]
                            strike = int(sym[-8:]) / 1000.0
                            est_bpr = strike * 100.0
                            acct = equity_trader.get_account()
                            if approve_csp_intent(fallback, acct, 0, est_bpr):
                                intents.append(fallback)

                return intents

            # Global, singletons
            equity_trader = TradingClient(API_KEY, API_SECRET, paper=IS_PAPER)
            opt_trader = OptionsTrader(API_KEY, API_SECRET, paper=IS_PAPER)

            def main_tick():
                intents = run_strategies()
                for intent in intents:
                    execute_intent(intent, equity_trader, opt_trader)

            chosen = None
            strategy_name = "sma"  # default label
            if rsi_res and rsi_res.get("signal"):
                chosen = rsi_res
                strategy_name = "rsi_sma"
            elif sma_res and sma_res.get("signal"):
                chosen = sma_res
                strategy_name = "sma"
            else:
                chosen = {"signal": None}

            sig = chosen.get("signal")
            price = chosen.get("price") or (sma_res.get("price") if sma_res else None)
            score = chosen.get("score", 0.0)

            logging.debug("Ticker %s chosen_strategy=%s signal=%s price=%s score=%s", ticker, strategy_name, sig, price,
                          score)

            # if buy signal and we don't already have a position in ticker -> consider entry
            if sig == "buy" and ticker not in open_symbols:
                if not can_open_new_positions:
                    logging.debug("Skipping %s buy — concurrency ceiling reached", ticker)
                    continue
                # sizing
                qty = calc_shares_for_risk(equity, available_funds, MAX_RISK_PCT, price, STOP_PCT)
                # sanity: if qty < tiny threshold (e.g., $1 of position) skip
                if qty <= 0 or qty * price < 1.0:
                    logging.info("Sizing for %s returned qty=%s (insufficient). Skipping.", ticker, qty)
                    send_slack(f":information_source: SKIPPED {ticker} — qty={qty} price=${price:.2f} (insufficient size or below MIN_PRICE).")
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

                # ---- Slack alert for buy (safe) ----
                try:
                    notional = float(qty) * float(price)
                    send_slack(
                        f":rocket: Placed BUY {ticker} — qty={qty} @ ${price:.2f}  notional=${notional:,.2f}  status={status}  id={order_id}")
                except Exception:
                    # fallback short message if formatting or notifier fails
                    try:
                        send_slack(
                            f":rocket: Placed BUY {ticker} — qty={qty} price={price} status={status} id={order_id}")
                    except Exception:
                        logging.debug("send_slack failed for BUY alert (swallowed).")
                # Reduce the running cash/buying-power budget for this scan so
                # subsequent tickers cannot immediately reuse the same funds.
                try:
                    notional = float(qty) * float(price)
                except Exception:
                    notional = 0.0
                available_funds = max(0.0, available_funds - notional)
                # -------------------------------------

                # record state and log
                # record state + metadata for auditing
                state["last_signal"][ticker] = {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "signal": "buy",
                    "price": price,
                    "strategy": strategy_name,  # e.g. "rsi_sma" or "sma"
                    "score": float(score or 0.0)
                }
                state["positions"][ticker] = {
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "qty": qty,
                    "order_id": order_id,
                    "strategy": strategy_name,
                    "score": float(score or 0.0),
                    # we optionally include entry_price once filled in place_market wrapper
                }
                save_state()

                # extend CSV log row to include strategy and score (keeps previous fields)
                log_trade_row({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ticker": ticker,
                    "action": "BUY",
                    "signal_price": price,
                    "qty": qty,
                    "order_id": order_id,
                    "status": status,
                    "notes": notes,
                    "equity": equity,
                    "cash": acct["cash"],
                    "strategy": strategy_name,
                    "score": float(score or 0.0)
                })

                log_trade_row({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
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
                counted_positions = [p for p in open_positions if _position_counts_toward_limit(p)]
                open_symbols = set([p.symbol for p in counted_positions])
                if len(counted_positions) >= MAX_CONCURRENT_POSITIONS:
                    logging.info("Reached max concurrent positions after entry. Further buys disabled until positions close.")
                    can_open_new_positions = False

            # if sell/exit signal and we already have position -> exit (market sell)
            elif sig == "sell" and ticker in open_symbols:
                # place market sell for quantity we hold
                pos = next((p for p in counted_positions if p.symbol == ticker), None)
                if pos is None:
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
                    logging.info("Placed exit for %s qty=%s id=%s", ticker, qty, order_id)
                    # Slack alert: exit placed
                    try:
                        send_slack(f":white_check_mark: EXIT {ticker} — qty={qty} @ ${price:.2f}  id={order_id}  status={status}")
                    except Exception:
                        send_slack(f":white_check_mark: EXIT {ticker} — qty={qty} id={order_id}")
                    # clear state
                    state["last_signal"][ticker] = {"time": datetime.now(timezone.utc).isoformat(), "signal": "sell", "price": price}
                    state["positions"].pop(ticker, None)
                    save_state()
                    log_trade_row({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
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
            send_slack(f":x: Error scanning {ticker}: {e}")
            continue

    # After processing every ticker, refresh any trailing-stop adjustments.
    try:
        update_trailing_stops()
    except Exception:
        logging.exception("update_trailing_stops failed at end of scan")

def main_loop():
    logging.info("Starting auto_trader main loop. Watchlist: %s", WATCHLIST)
    while True:
        try:
            run_scan_once()
            state["last_scan"] = datetime.now(timezone.utc).isoformat()

            save_state()
        except Exception as e:
            logging.exception("Top-level error in main loop: %s", e)
            send_slack(f":bangbang: Top-level error in auto_trader: {e}")
        time.sleep(SCAN_INTERVAL_MINUTES * 60)
if __name__ == "__main__":
    main_loop()
