"""Natural trend upgrade strategy based on dual moving averages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .base import Signal, Strategy, StrategyContext


@dataclass
class NaturalTrendConfig:
    fast_window: int = 21
    slow_window: int = 55
    slope_window: int = 5
    min_slope: float = 0.0
    min_volume_avg: int = 500_000


class NaturalTrendUpgradeStrategy(Strategy):
    name = "Natural Trend Upgrade"
    short_code = "NTU"

    def __init__(self, symbols: List[str], config: NaturalTrendConfig | None = None):
        self.symbols = symbols
        self.config = config or NaturalTrendConfig()

    def _evaluate_impl(self, context: StrategyContext) -> List[Signal]:
        signals: List[Signal] = []
        for symbol in self.symbols:
            data = context.market_data.get(symbol)
            if data is None or len(data) < self.config.slow_window + 2:
                continue

            closes = data["close"].astype(float)
            volume = data.get("volume")
            if volume is not None and volume.rolling(20).mean().iloc[-1] < self.config.min_volume_avg:
                continue

            fast = closes.rolling(self.config.fast_window).mean()
            slow = closes.rolling(self.config.slow_window).mean()
            if np.isnan(fast.iloc[-2]) or np.isnan(slow.iloc[-2]):
                continue

            slope = fast.diff(self.config.slope_window).iloc[-1]
            crossed_up = fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
            crossed_down = fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]

            if crossed_up and slope >= self.config.min_slope:
                signals.append(
                    Signal(
                        symbol=symbol,
                        action="buy",
                        confidence=0.75,
                        notes="Fast MA reclaimed slow trend",
                        metadata={"fast": float(fast.iloc[-1]), "slow": float(slow.iloc[-1])},
                    )
                )
            elif crossed_down:
                signals.append(
                    Signal(
                        symbol=symbol,
                        action="sell",
                        confidence=0.65,
                        notes="Trend breakdown",
                        metadata={"fast": float(fast.iloc[-1]), "slow": float(slow.iloc[-1])},
                    )
                )
        return signals


__all__ = ["NaturalTrendConfig", "NaturalTrendUpgradeStrategy"]
