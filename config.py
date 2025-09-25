import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("APCA_API_BASE_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", 0.05))
DAY_RUN = os.getenv("DAY_RUN", "True").lower() in ("1", "true", "yes")
WATCHLIST = os.getenv("WATCHLIST", "").split(",")


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
