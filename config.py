ENABLE_OPTIONS = True
import os
try:
    # optional if you use .env locally
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
ALPACA_KEY_ID = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY = ALPACA_KEY_ID
API_SECRET = ALPACA_SECRET_KEY
# Paper/live detection (keeps your code simple)
IS_PAPER = "paper" in ALPACA_BASE_URL
from dotenv import load_dotenv
from config import API_KEY, API_SECRET, IS_PAPER

# --- Strategy universe (liquid, tight spreads) ---
OPTIONS_UNDERLYINGS = ["SPY", "AAPL"]   # tickers you want to trade options on

# --- Liquidity & quality filters ---
OPTIONS_TARGET_DELTA = (0.20, 0.30)    # short ~20–30Δ puts
OPTIONS_MIN_OI       = 100             # min open interest
OPTIONS_MAX_REL_SPREAD = 0.15          # (ask-bid)/ask <= 15%
OPTIONS_MIN_DTE      = 1               # ≥ 7 days to expiry
OPTIONS_MAX_DTE      = 90              # ≤ 30 days (short dated)
# --- Risk & exposure ---
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "20"))  # equity slots
MAX_OPEN_CSP_POSITIONS = 8             # total short puts at once
MAX_BP_AT_RISK         = 0.25          # 25% of buying power at risk for CSPs
TAKE_PROFIT_PCT        = 0.50          # buy-to-close when 50% of credit captured
STOP_LOSS_MULT         = 2.0

# Load .env file
load_dotenv()

IS_PAPER = True   # True = paper trading, False = live
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("APCA_API_BASE_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", 0.05))
DAY_RUN = os.getenv("DAY_RUN", "True").lower() in ("1", "true", "yes")
WATCHLIST = os.getenv("WATCHLIST", "").split(",")

# === Options toggles ===
ENABLE_OPTIONS = True            # master kill switch
OPTIONS_MIN_OI = 100             # min open interest
OPTIONS_MAX_SPREAD = 0.15        # max relative bid/ask spread (15%)
OPTIONS_MIN_DTE = 1              # min days to expiry
OPTIONS_MAX_DTE = 120             # optional: keep trades short-dated
OPTIONS_TARGET_DELTA = (0.20, 0.35)  # screen for ~cash-secured puts/covered calls

# --- Options strategy settings ---
OPTIONS_USE_WATCHLIST = True
OPTIONS_MAX_CANDIDATES_PER_TICK = 8  # how many tickers from watchlist to scan each tick

# --- Signal routing (when to prefer options over shares) ---
OPTIONS_ROUTER_MIN_SHARE_QTY = int(os.getenv("OPTIONS_ROUTER_MIN_SHARE_QTY", "5"))
OPTIONS_ROUTER_MAX_SHARE_PRICE = float(os.getenv("OPTIONS_ROUTER_MAX_SHARE_PRICE", "175"))
OPTIONS_ROUTER_ALLOW_SPREADS = os.getenv("OPTIONS_ROUTER_ALLOW_SPREADS", "true").lower() in ("1", "true", "yes")
OPTIONS_ROUTER_ALLOW_CSP = os.getenv("OPTIONS_ROUTER_ALLOW_CSP", "false").lower() in ("1", "true", "yes")

# --- Small-account options spread settings ---
SPREADS_MIN_DTE = 7
SPREADS_MAX_DTE = 30
SPREADS_MIN_OI = 300               # lower than CSP; spreads can trade with less OI
SPREADS_MAX_REL_SPREAD = 0.20      # allow up to 20% (tight is better)
SPREADS_TARGET_DELTA = (0.20, 0.35)  # short leg delta target for BPS; long call near ATM for call debit

# Risk & sizing for ~$1k account
SPREADS_MAX_RISK_PER_TRADE = 150    # cap max loss per spread in dollars
SPREADS_MAX_OPEN = 2                # total open spreads at once
SPREADS_MAX_CANDIDATES_PER_TICK = 8 # how many tickers we scan per tick


WATCHLIST = [
 "AAPL","MSFT","NVDA","AMZN","GOOGL","META","NFLX",
 "TSLA","AMD","CRM","AVGO","ORCL","NOW","SHOP",
 "JPM","BAC","C","MA","V",
 "HD","LOW","MCD","SBUX",
 "CAT","BA","LMT",
 "XOM","CVX","COP",
 "JNJ","PFE","MRK","MRNA",
 "ASML","QCOM","LRCX",
 "SPY","QQQ","IWM","XLK","XLF","XLE"
]
