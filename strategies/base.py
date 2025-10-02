"""Common strategy interfaces for the Wyatt auto trader architecture."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd


@dataclass
class Signal:
    """Normalized signal emitted by a strategy."""

    symbol: str
    action: str
    confidence: float
    notes: str = ""
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyContext:
    """Runtime data passed to strategies during evaluation."""

    market_data: Mapping[str, pd.DataFrame]
    regime: str = "neutral"
    risk_budget: float = 1.0
    account_equity: Optional[float] = None
    open_positions: Iterable[str] = ()


class Strategy:
    """Base class for all modular strategies."""

    name: str = ""
    short_code: str = ""
    enabled: bool = True

    def evaluate(self, context: StrategyContext) -> List[Signal]:
        """Return zero or more signals for the provided context."""

        if not self.enabled:
            return []
        return self._evaluate_impl(context)

    # -- internals -------------------------------------------------
    def _evaluate_impl(self, context: StrategyContext) -> List[Signal]:
        raise NotImplementedError


__all__ = ["Signal", "Strategy", "StrategyContext"]
