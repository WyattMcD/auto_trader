"""Mean reversion strategy that fades stretched moves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .base import Signal, Strategy, StrategyContext


@dataclass
class MeanReversionConfig:
    rsi_window: int = 2
    rsi_buy: float = 10.0
    rsi_sell: float = 90.0
    atr_window: int = 14
    max_distance_atr: float = 2.5


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


class MeanReversionStrategy(Strategy):
    name = "Mean Reversion Fade"
    short_code = "MRF"

    def __init__(self, symbols: List[str], config: MeanReversionConfig | None = None):
        self.symbols = symbols
        self.config = config or MeanReversionConfig()

    def _evaluate_impl(self, context: StrategyContext) -> List[Signal]:
        signals: List[Signal] = []
        for symbol in self.symbols:
            data = context.market_data.get(symbol)
            if data is None or len(data) < 50:
                continue
            closes = data["close"].astype(float)
            rsi = _rsi(closes, self.config.rsi_window)
            atr = _atr(data, self.config.atr_window)
            if rsi.iloc[-1] < self.config.rsi_buy:
                distance = (data["close"].iloc[-1] - data["close"].rolling(20).mean().iloc[-1]) / (atr.iloc[-1] + 1e-9)
                if np.abs(distance) <= self.config.max_distance_atr:
                    signals.append(
                        Signal(
                            symbol=symbol,
                            action="buy",
                            confidence=0.6,
                            notes="RSI capitulation fade",
                            metadata={"rsi": float(rsi.iloc[-1])},
                        )
                    )
            elif rsi.iloc[-1] > self.config.rsi_sell:
                signals.append(
                    Signal(
                        symbol=symbol,
                        action="sell",
                        confidence=0.6,
                        notes="RSI exhaustion",
                        metadata={"rsi": float(rsi.iloc[-1])},
                    )
                )
        return signals


__all__ = ["MeanReversionConfig", "MeanReversionStrategy"]
