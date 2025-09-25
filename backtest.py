# backtest.py
# Defensive daily backtester for SMA crossover strategy with ATR stops.
# Standalone: does NOT import auto_trader.py, so safe to run anytime.

import math
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ------- USER SETTINGS -------
TICKER = "AAPL"               # ticker to backtest
START = "2018-01-01"
END = "2024-12-31"
SMA_LEN = 20
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5
INITIAL_CAPITAL = 100000.0
RISK_PCT = 0.005              # 0.5% risk per trade
SLIPPAGE_PCT = 0.0005         # 0.05% per fill
COMMISSION = 1.0              # per trade flat commission
MIN_PRICE = 1.0
# -----------------------------

def fetch_df(sym):
    symbol = str(sym).strip().upper()
    logging.info("Downloading %s data %s -> %s", symbol, START, END)
    df = yf.download(symbol, start=START, end=END, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No price data returned from yfinance for {symbol}. Try a different ticker or check network.")
    # if multiindex columns, try to collapse (happens rarely for some yf calls)
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        try:
            # if second level contains the ticker (unlikely in single-ticker call), pick it
            levels = df.columns.get_level_values(1)
            if symbol in levels:
                df = df.xs(symbol, axis=1, level=1)
            else:
                # collapse to first element (Open, High, Low, Close, Adj Close, Volume)
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        except Exception:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Ensure required columns exist
    expected = {"Open","High","Low","Close","Volume"}
    if "Close" not in df.columns:
        raise RuntimeError(f"Downloaded data for {symbol} missing 'Close' column. Columns present: {list(df.columns)}")

    return df

def compute_atr_series(df, period=ATR_PERIOD):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def simulate(ticker):
    df = fetch_df(ticker).copy()
    # compute sma safely
    df["sma"] = df["Close"].rolling(SMA_LEN).mean()
    # drop rows without sma
    if "sma" not in df.columns:
        raise RuntimeError("Failed to compute SMA column.")
    df = df.dropna(subset=["sma"]).copy()
    if df.empty:
        raise RuntimeError("Not enough data after computing SMA — increase lookback or reduce SMA_LEN.")

    atr = compute_atr_series(df)
    if atr.isna().all():
        logging.warning("ATR series is all NaN. Stop sizing will use fallback.")

    cash = INITIAL_CAPITAL
    position = 0.0
    entry_price = 0.0
    equity_series = []
    entry_count = 0
    exit_count = 0

    # iterate by integer index and use scalar floats for comparisons
    for i in range(1, len(df)):
        prev_close = float(df["Close"].iat[i-1])
        prev_sma   = float(df["sma"].iat[i-1])
        close      = float(df["Close"].iat[i])
        cur_sma    = float(df["sma"].iat[i])
        date       = df.index[i]

        buy_signal = (prev_close <= prev_sma) and (close > cur_sma)
        sell_signal = (prev_close >= prev_sma) and (close < cur_sma)

        # mark before action
        equity = cash + position * close

        if buy_signal and position == 0:
            # ATR from previous bar if available
            try:
                atr_val = float(atr.dropna().iat[i-1])
            except Exception:
                atr_val = None
            stop_pct = (atr_val * ATR_MULTIPLIER) / close if atr_val is not None and close > 0 else 0.06
            stop_pct = max(0.01, min(0.20, stop_pct))

            risk_dollars = (cash + position*close) * RISK_PCT
            if stop_pct * close <= 0:
                shares = 0
            else:
                shares = math.floor(risk_dollars / (stop_pct * close))

            if shares > 0 and close >= MIN_PRICE:
                executed_price = close * (1.0 + SLIPPAGE_PCT)
                cost = shares * executed_price + COMMISSION
                if cost <= cash:
                    cash -= cost
                    position = shares
                    entry_price = executed_price
                    entry_count += 1

        elif sell_signal and position > 0:
            executed_price = close * (1.0 - SLIPPAGE_PCT)
            proceeds = position * executed_price - COMMISSION
            cash += proceeds
            position = 0
            entry_price = 0
            exit_count += 1

        equity = cash + position * close
        equity_series.append({"date": date, "equity": equity})

    ec = pd.DataFrame(equity_series).set_index("date")
    ec["returns"] = ec["equity"].pct_change().fillna(0)
    total_return = ec["equity"].iloc[-1] / INITIAL_CAPITAL - 1.0
    mean_daily = ec["returns"].mean()
    std_daily = ec["returns"].std(ddof=0) if ec["returns"].size > 1 else 0.0
    ann_return = (1 + mean_daily) ** 252 - 1 if std_daily != 0 else 0.0
    sharpe = (mean_daily / std_daily * (252**0.5)) if std_daily > 0 else float("nan")
    peak = ec["equity"].cummax()
    drawdown = (peak - ec["equity"]) / peak
    max_dd = drawdown.max()

    stats = {
        "ticker": ticker,
        "final_equity": float(ec["equity"].iloc[-1]),
        "total_return": float(total_return),
        "annual_return": float(ann_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "trades_entered": int(entry_count),
        "trades_exited": int(exit_count),
        "equity_series": ec
    }
    return stats

def benchmark_spy():
    spy = fetch_df("SPY")
    spy = spy.loc[START:END]
    spy["ret"] = spy["Close"].pct_change().fillna(0)
    mean = spy["ret"].mean()
    std = spy["ret"].std(ddof=0)
    ann = (1+mean)**252 - 1 if std != 0 else 0.0
    sharpe = (mean / std * (252**0.5)) if std > 0 else float("nan")
    total = spy["Close"].iloc[-1] / spy["Close"].iloc[0] - 1.0
    return {"total_return": total, "annual_return": ann, "sharpe": sharpe}

if __name__ == "__main__":
    start = datetime.strptime(START, "%Y-%m-%d").date()
    end = datetime.strptime(END, "%Y-%m-%d").date()
    logging.info("Backtest %s from %s to %s | SMA_LEN=%d | ATR_mult=%.2f", TICKER, start, end, SMA_LEN, ATR_MULTIPLIER)
    stats = simulate(TICKER)
    spy_stats = benchmark_spy()

    print("\nRESULTS:")
    print(f" Final equity: ${stats['final_equity']:,.2f}")
    print(f" Total return: {stats['total_return']*100:.2f}%")
    print(f" Annualized return (approx): {stats['annual_return']*100:.2f}%")
    print(f" Sharpe (approx): {stats['sharpe']:.3f}")
    print(f" Max drawdown: {stats['max_drawdown']*100:.2f}%")
    print(f" Trades entered: {stats['trades_entered']}, exited: {stats['trades_exited']}")
    print("\nBenchmark SPY:")
    print(f" SPY total return: {spy_stats['total_return']*100:.2f}%")
    print(f" SPY annualized (approx): {spy_stats['annual_return']*100:.2f}%")
    print(f" SPY Sharpe (approx): {spy_stats['sharpe']:.3f}")
