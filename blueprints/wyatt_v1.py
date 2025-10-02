"""Structured representation of the Wyatt Auto Trader outline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import yaml

from engine.risk import RiskEnvelope, build_risk_envelope
from strategies.donchian_breakout import DonchianBreakoutConfig
from strategies.earnings_momentum import EarningsMomentumConfig
from strategies.mean_reversion import MeanReversionConfig
from strategies.natural_trend import NaturalTrendConfig
from strategies.overnight_gap import OvernightGapConfig


@dataclass
class WatchlistUniverse:
    core_equities: List[str]
    options: List[str]
    etfs: List[str]
    turtle: List[str]


@dataclass
class StrategyToggle:
    name: str
    code: str
    enabled: bool


@dataclass
class StrategySettings:
    natural_trend: NaturalTrendConfig = field(default_factory=NaturalTrendConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    donchian: DonchianBreakoutConfig = field(default_factory=DonchianBreakoutConfig)
    overnight_gap: OvernightGapConfig = field(default_factory=OvernightGapConfig)
    earnings: EarningsMomentumConfig = field(default_factory=EarningsMomentumConfig)


@dataclass
class Blueprint:
    """High-level architecture description used by the engine."""

    watchlists: WatchlistUniverse
    risk: RiskEnvelope
    toggles: List[StrategyToggle]
    sizing_policy: Dict[str, float]
    metadata: Dict[str, str] = field(default_factory=dict)

    def enabled_strategies(self) -> List[str]:
        return [toggle.code for toggle in self.toggles if toggle.enabled]


DEFAULT_METADATA = {
    "version": "1.0",
    "purpose": (
        "Multi-strategy swing & income system: trend, mean reversion, options income, "
        "overnight edges, and event-driven plays."
    ),
}


def _load_watchlists(config: Mapping[str, Sequence[str]]) -> WatchlistUniverse:
    return WatchlistUniverse(
        core_equities=list(config.get("core_equities", [])),
        options=list(config.get("options", [])),
        etfs=list(config.get("etfs", [])),
        turtle=list(config.get("turtle", [])),
    )


def _load_toggles(entries: Sequence[Mapping[str, str]]) -> List[StrategyToggle]:
    toggles: List[StrategyToggle] = []
    for entry in entries:
        toggles.append(
            StrategyToggle(
                name=str(entry.get("name", "")),
                code=str(entry.get("code", "")).upper(),
                enabled=bool(entry.get("enabled", True)),
            )
        )
    return toggles


def load_blueprint(path: str | Path) -> Blueprint:
    raw = yaml.safe_load(Path(path).read_text())
    watchlists = _load_watchlists(raw.get("watchlists", {}))
    risk = build_risk_envelope(raw.get("risk", {}))
    toggles = _load_toggles(raw.get("strategy_toggles", []))
    sizing_policy = dict(raw.get("sizing_policy", {}))
    metadata = {**DEFAULT_METADATA, **raw.get("metadata", {})}
    return Blueprint(
        watchlists=watchlists,
        risk=risk,
        toggles=toggles,
        sizing_policy=sizing_policy,
        metadata=metadata,
    )


__all__ = ["Blueprint", "StrategySettings", "StrategyToggle", "WatchlistUniverse", "load_blueprint"]
