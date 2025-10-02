"""Risk and allocation helpers aligned with the Wyatt blueprint."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PositionSizingRules:
    equity_risk_pct: float = 0.05
    options_notional_cap: float = 0.25
    max_positions: int = 25


@dataclass
class RiskEnvelope:
    """Runtime guardrails used by the engine."""

    sizing: PositionSizingRules
    max_drawdown_pct: float = 0.15
    circuit_breaker_down_pct: float = 0.07
    circuit_breaker_up_pct: float = 0.07

    def slot_size(self, equity: float) -> float:
        return equity * self.sizing.equity_risk_pct


DEFAULT_RISK_ENVELOPE = RiskEnvelope(sizing=PositionSizingRules())


def build_risk_envelope(config: Dict) -> RiskEnvelope:
    sizing_cfg = config.get("sizing", {})
    sizing = PositionSizingRules(
        equity_risk_pct=float(sizing_cfg.get("equity_risk_pct", 0.05)),
        options_notional_cap=float(sizing_cfg.get("options_notional_cap", 0.25)),
        max_positions=int(sizing_cfg.get("max_positions", 25)),
    )
    return RiskEnvelope(
        sizing=sizing,
        max_drawdown_pct=float(config.get("max_drawdown_pct", 0.15)),
        circuit_breaker_down_pct=float(config.get("circuit_breaker_down_pct", 0.07)),
        circuit_breaker_up_pct=float(config.get("circuit_breaker_up_pct", 0.07)),
    )


__all__ = ["RiskEnvelope", "PositionSizingRules", "build_risk_envelope", "DEFAULT_RISK_ENVELOPE"]
