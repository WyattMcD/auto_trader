from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from blueprints.wyatt_v1 import load_blueprint
from engine.registry import StrategyRegistry
from engine.wyatt import WyattEngine
from strategies.earnings_momentum import EarningsCalendar


def _mock_df() -> pd.DataFrame:
    base = [100.0] * 70
    ramp = [100 + i * 0.8 for i in range(70)]
    closes = base + ramp
    opens = closes
    highs = [c * 1.005 for c in closes]
    lows = [c * 0.995 for c in closes]
    volume = [1_200_000] * len(closes)
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    })


CONFIG_PATH = REPO_ROOT / "configs" / "wyatt_v1.yaml"


def test_blueprint_loads(tmp_path):
    cfg_path = tmp_path / "wyatt.yaml"
    cfg_path.write_text(CONFIG_PATH.read_text())
    blueprint = load_blueprint(cfg_path)
    assert blueprint.watchlists.core_equities
    assert blueprint.risk.sizing.equity_risk_pct == 0.05


def test_engine_generates_signals(monkeypatch):
    blueprint = load_blueprint(CONFIG_PATH)
    engine = WyattEngine(blueprint=blueprint, registry=StrategyRegistry(blueprint))
    market_data = {
        symbol: _mock_df()
        for symbol in set(
            blueprint.watchlists.core_equities
            + blueprint.watchlists.etfs
            + blueprint.watchlists.turtle
        )
    }
    signals = engine.evaluate(market_data=market_data, calendar=EarningsCalendar())
    assert isinstance(signals, dict)
    assert signals
