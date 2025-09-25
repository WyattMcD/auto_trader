# strategies/rsi_momentum.py
import pandas as pd
import numpy as np
from auto_trader import fetch_history_yf   # relative import; if it fails, import fetch_history_yf from auto_trader directly

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder smoothing (EWMA with alpha=1/period)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral where unknown

def compute_rsi_sma_signal(ticker: str,
                           sma_len: int = 20,
                           rsi_period: int = 14,
                           rsi_buy_thresh: float = 52.0,
                           rsi_sell_thresh: float = 45.0):
    """
    Returns dict like:
    { 'ticker','signal','price','sma','time','score' }
    signal: 'buy' / 'sell' / None
    score: float (higher = stronger buy)
    """
    try:
        df = fetch_history_yf(ticker, lookback_days=max(90, sma_len*4), interval="1d")
        if df is None or len(df) < sma_len + rsi_period + 2:
            return None

        df = df.copy()
        df['sma'] = df['Close'].rolling(sma_len).mean()
        df['rsi'] = compute_rsi(df['Close'], period=rsi_period)

        # Last two rows for crossover detection
        prev_close = float(df['Close'].iat[-2])
        last_close = float(df['Close'].iat[-1])
        prev_sma = float(df['sma'].iat[-2])
        last_sma = float(df['sma'].iat[-1])
        last_rsi = float(df['rsi'].iat[-1])

        signal = None
        score = 0.0

        # Buy: SMA cross up AND RSI above threshold (momentum confirmation)
        if prev_close <= prev_sma and last_close > last_sma and last_rsi >= rsi_buy_thresh:
            signal = 'buy'
            # Score = normalized RSI above 50 + %distance over SMA
            rsi_component = (last_rsi - 50) / 50.0
            price_component = (last_close - last_sma) / last_sma
            score = float(rsi_component * 0.7 + price_component * 0.3)

        # Sell/exit: SMA cross down OR RSI falls below sell thresh
        elif (prev_close >= prev_sma and last_close < last_sma) or last_rsi <= rsi_sell_thresh:
            signal = 'sell'
            rsi_component = (50 - last_rsi) / 50.0
            price_component = (last_sma - last_close) / last_sma
            score = float(rsi_component * 0.6 + price_component * 0.4)

        return {
            'ticker': ticker.upper(),
            'signal': signal,
            'price': last_close,
            'sma': last_sma,
            'time': str(df.index[-1].date()),
            'rsi': last_rsi,
            'score': score
        }
    except Exception as e:
        # keep silent for production; logging in auto_trader will capture
        return None
