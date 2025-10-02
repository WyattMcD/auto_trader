"""Overnight gap edge that fades SPY/QQQ/IWM gaps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .base import Signal, Strategy, StrategyContext


@dataclass
class OvernightGapConfig:
    gap_threshold: float = 0.005
    close_time: str = "15:55"
    open_time: str = "09:35"
    hold_period: int = 1


class OvernightGapStrategy(Strategy):
    name = "Overnight Gap Edge"
    short_code = "ONG"

    def __init__(self, symbols: List[str], config: OvernightGapConfig | None = None):
        self.symbols = symbols
        self.config = config or OvernightGapConfig()

    def _evaluate_impl(self, context: StrategyContext) -> List[Signal]:
        signals: List[Signal] = []
        for symbol in self.symbols:
            df = context.market_data.get(symbol)
            if df is None or len(df) < 3 or "open" not in df.columns:
                continue
            prev_close = float(df["close"].iloc[-2])
            today_open = float(df["open"].iloc[-1])
            gap = (today_open - prev_close) / prev_close
            if abs(gap) < self.config.gap_threshold:
                continue
            action = "sell" if gap > 0 else "buy"
            confidence = min(0.6 + abs(gap) * 10, 0.85)
            signals.append(
                Signal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    notes=f"Overnight gap {gap:.2%}",
                    metadata={"gap": gap},
                )
            )
        return signals


__all__ = ["OvernightGapConfig", "OvernightGapStrategy"]
