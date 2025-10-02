"""Strategy registry that wires blueprint toggles to implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

from blueprints.wyatt_v1 import Blueprint
from strategies.base import Signal, Strategy, StrategyContext
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.earnings_momentum import EarningsCalendar, EarningsMomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.natural_trend import NaturalTrendUpgradeStrategy
from strategies.overnight_gap import OvernightGapStrategy


@dataclass
class StrategyRegistry:
    blueprint: Blueprint

    def build(self, calendar: EarningsCalendar | None = None) -> Dict[str, Strategy]:
        watchlists = self.blueprint.watchlists
        calendar = calendar or EarningsCalendar()
        strategies: Dict[str, Strategy] = {
            "NTU": NaturalTrendUpgradeStrategy(symbols=watchlists.core_equities),
            "MRF": MeanReversionStrategy(symbols=watchlists.core_equities),
            "DBO": DonchianBreakoutStrategy(symbols=watchlists.turtle),
            "ONG": OvernightGapStrategy(symbols=watchlists.etfs),
            "EMO": EarningsMomentumStrategy(symbols=watchlists.core_equities, calendar=calendar),
        }
        # Filter disabled strategies early.
        enabled = {}
        for code, strategy in strategies.items():
            strategy.enabled = code in self.blueprint.enabled_strategies()
            enabled[code] = strategy
        return enabled

    def evaluate(self, context: StrategyContext, calendar: EarningsCalendar | None = None) -> List[Signal]:
        strategies = self.build(calendar=calendar)
        signals: List[Signal] = []
        for strategy in strategies.values():
            signals.extend(strategy.evaluate(context))
        return signals


__all__ = ["StrategyRegistry"]
