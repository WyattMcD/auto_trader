"""High level orchestration for the Wyatt Auto Trader architecture."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import pandas as pd

from blueprints.wyatt_v1 import Blueprint, load_blueprint
from engine.registry import StrategyRegistry
from engine.risk import RiskEnvelope
from strategies.base import Signal, StrategyContext
from strategies.earnings_momentum import EarningsCalendar


@dataclass
class WyattEngine:
    blueprint: Blueprint
    registry: StrategyRegistry

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WyattEngine":
        blueprint = load_blueprint(path)
        registry = StrategyRegistry(blueprint)
        return cls(blueprint=blueprint, registry=registry)

    @property
    def risk(self) -> RiskEnvelope:
        return self.blueprint.risk

    def evaluate(self, market_data: Mapping[str, pd.DataFrame], regime: str = "neutral",
                 risk_budget: float = 1.0, calendar: EarningsCalendar | None = None) -> Dict[str, Signal]:
        context = StrategyContext(
            market_data=market_data,
            regime=regime,
            risk_budget=risk_budget,
        )
        signals = self.registry.evaluate(context=context, calendar=calendar)
        # Deduplicate by keeping strongest per symbol
        result: Dict[str, Signal] = {}
        for signal in signals:
            existing = result.get(signal.symbol)
            if existing is None or signal.confidence > existing.confidence:
                result[signal.symbol] = signal
        return result


__all__ = ["WyattEngine"]
