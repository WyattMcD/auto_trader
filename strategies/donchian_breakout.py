"""Donchian breakout implementation for the Turtle list."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .base import Signal, Strategy, StrategyContext


@dataclass
class DonchianBreakoutConfig:
    breakout_window: int = 20
    exit_window: int = 10
    min_atr: float = 0.5


class DonchianBreakoutStrategy(Strategy):
    name = "Donchian Breakout"
    short_code = "DBO"

    def __init__(self, symbols: List[str], config: DonchianBreakoutConfig | None = None):
        self.symbols = symbols
        self.config = config or DonchianBreakoutConfig()

    def _evaluate_impl(self, context: StrategyContext) -> List[Signal]:
        signals: List[Signal] = []
        for symbol in self.symbols:
            df = context.market_data.get(symbol)
            if df is None or len(df) < self.config.breakout_window + 5:
                continue

            highs = df["high"].astype(float)
            lows = df["low"].astype(float)
            closes = df["close"].astype(float)
            breakout_high = highs.rolling(self.config.breakout_window).max()
            breakout_low = lows.rolling(self.config.breakout_window).min()
            exit_low = lows.rolling(self.config.exit_window).min()
            exit_high = highs.rolling(self.config.exit_window).max()

            if closes.iloc[-1] > breakout_high.iloc[-2]:
                signals.append(
                    Signal(
                        symbol=symbol,
                        action="buy",
                        confidence=0.7,
                        notes="Donchian breakout",
                        metadata={"breakout_high": float(breakout_high.iloc[-2])},
                    )
                )
            elif closes.iloc[-1] < breakout_low.iloc[-2]:
                signals.append(
                    Signal(
                        symbol=symbol,
                        action="sell",
                        confidence=0.7,
                        notes="Donchian breakdown",
                        metadata={"breakout_low": float(breakout_low.iloc[-2])},
                    )
                )
            elif closes.iloc[-1] < exit_low.iloc[-2]:
                signals.append(
                    Signal(
                        symbol=symbol,
                        action="sell",
                        confidence=0.5,
                        notes="Breakout failure exit",
                        metadata={"exit_low": float(exit_low.iloc[-2])},
                    )
                )
            elif closes.iloc[-1] > exit_high.iloc[-2]:
                signals.append(
                    Signal(
                        symbol=symbol,
                        action="buy",
                        confidence=0.5,
                        notes="Short squeeze exit",
                        metadata={"exit_high": float(exit_high.iloc[-2])},
                    )
                )
        return signals


__all__ = ["DonchianBreakoutConfig", "DonchianBreakoutStrategy"]
