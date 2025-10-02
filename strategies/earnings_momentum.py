"""Earnings gap-and-go strategy for post-earnings momentum."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping

import pandas as pd

from .base import Signal, Strategy, StrategyContext


@dataclass
class EarningsMomentumConfig:
    min_gap: float = 0.04
    min_volume_surprise: float = 1.5
    hold_days: int = 5


class EarningsCalendar:
    """Light-weight store of earnings dates and surprises."""

    def __init__(self, surprises: Mapping[str, float] | None = None):
        self.surprises = dict(surprises or {})

    def get_surprise(self, symbol: str) -> float:
        return float(self.surprises.get(symbol, 0.0))


class EarningsMomentumStrategy(Strategy):
    name = "Earnings Momentum"
    short_code = "EMO"

    def __init__(
        self,
        symbols: Iterable[str],
        calendar: EarningsCalendar,
        config: EarningsMomentumConfig | None = None,
    ) -> None:
        self.symbols = list(symbols)
        self.calendar = calendar
        self.config = config or EarningsMomentumConfig()

    def _evaluate_impl(self, context: StrategyContext) -> List[Signal]:
        signals: List[Signal] = []
        for symbol in self.symbols:
            df = context.market_data.get(symbol)
            if df is None or len(df) < 10:
                continue
            gap = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]
            surprise = self.calendar.get_surprise(symbol)
            if gap >= self.config.min_gap and surprise >= self.config.min_volume_surprise:
                signals.append(
                    Signal(
                        symbol=symbol,
                        action="buy",
                        confidence=0.7,
                        notes="Earnings gap-and-go",
                        metadata={"gap": gap, "surprise": surprise},
                    )
                )
            elif gap <= -self.config.min_gap and surprise <= -self.config.min_volume_surprise:
                signals.append(
                    Signal(
                        symbol=symbol,
                        action="sell",
                        confidence=0.7,
                        notes="Earnings gap-down continuation",
                        metadata={"gap": gap, "surprise": surprise},
                    )
                )
        return signals


__all__ = [
    "EarningsCalendar",
    "EarningsMomentumConfig",
    "EarningsMomentumStrategy",
]
