import os
import sys
import importlib.util
import types
from pathlib import Path

import pytest

# Ensure dummy credentials exist before loading the trading script
os.environ.setdefault("APCA_API_KEY_ID", "test-key")
os.environ.setdefault("APCA_API_SECRET_KEY", "test-secret")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "auto_trader.py"
LOCK_PATH = Path("/app/state/auto_trader.lock")

# Make sure the project root is importable for config/strategy modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_auto_trader():
    if LOCK_PATH.exists():
        LOCK_PATH.unlink()

    original_exists = os.path.exists

    def fake_exists(path):
        if str(path) == str(LOCK_PATH):
            return False
        return original_exists(path)

    os.path.exists = fake_exists
    try:
        spec = importlib.util.spec_from_file_location("auto_trader_module", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.path.exists = original_exists

    if LOCK_PATH.exists():
        LOCK_PATH.unlink()

    return module


@pytest.fixture
def trader(monkeypatch, tmp_path):
    """Load auto_trader with isolated state for each test."""
    mod = load_auto_trader()

    # Point persistence artifacts at a temp directory and neutralize disk writes
    mod.STATE_FILE = str(tmp_path / "state.json")
    mod.LOG_CSV = str(tmp_path / "trades.csv")
    monkeypatch.setattr(mod, "save_state", lambda: None)

    # Fresh in-memory state containers
    mod.state = {
        "positions": {},
        "options_positions": {},
        "option_spreads": {},
        "strategy_state": {},
    }

    # Stubs for broker clients the lifecycle touches
    mod.equity_trader = types.SimpleNamespace(
        get_account=lambda: types.SimpleNamespace()
    )

    mod.opt_trader = types.SimpleNamespace(
        client=types.SimpleNamespace(
            get_all_positions=lambda: [],
            get_open_position=lambda symbol: None,
        )
    )

    # Default options data stub (tests override behaviour as needed)
    mod.od = types.SimpleNamespace(
        snapshots_for=lambda symbols: []
    )

    # Keep options features enabled during tests
    mod.ENABLE_OPTIONS = True

    return mod


def test_run_option_entry_cycle_submits_spread(monkeypatch, trader):
    submitted = []

    def fake_execute_intent(intent, *_):
        submitted.append(intent)
        return {"id": "mock"}

    monkeypatch.setattr(trader, "execute_intent", fake_execute_intent)
    monkeypatch.setattr(trader, "_count_open_short_puts", lambda: 0)
    monkeypatch.setattr(trader, "approve_csp_intent", lambda *args, **kwargs: True)

    trader.WATCHLIST = ["AAPL"]
    trader.OPTIONS_UNDERLYINGS = ["SPY"]
    trader.SPREADS_MAX_OPEN = 5
    trader.SPREADS_MAX_CANDIDATES_PER_TICK = 5
    trader.SPREADS_MAX_RISK_PER_TRADE = 200
    trader.OPTIONS_MAX_CANDIDATES_PER_TICK = 5

    spread_intent = {
        "asset_class": "option_spread",
        "strategy": "bull_put_spread",
        "legs": [
            {"symbol": "SPY250118P00450000", "side": "sell", "limit_price": 1.50, "qty": 1},
            {"symbol": "SPY250118P00440000", "side": "buy", "limit_price": 0.80, "qty": 1},
        ],
        "risk": {"max_loss": 140},
    }

    monkeypatch.setattr(trader, "pick_bull_put_spread", lambda *_: spread_intent)
    monkeypatch.setattr(trader, "pick_call_debit_spread", lambda *_: None)
    monkeypatch.setattr(trader, "pick_csp_intent", lambda *_: None)

    trader.run_option_entry_cycle()

    assert submitted, "Expected the entry cycle to submit at least one intent"
    intent = submitted[0]
    assert intent["strategy"] == "bull_put_spread"
    assert trader.state["option_spreads"], "Spread metadata should be persisted for exit monitoring"


def test_monitor_option_exits_closes_positions(monkeypatch, trader):
    submitted = []

    def fake_execute_intent(intent, *_):
        submitted.append(intent)
        return {"id": f"mock-{len(submitted)}"}

    monkeypatch.setattr(trader, "execute_intent", fake_execute_intent)

    # Seed one CSP and one spread position in state
    trader.state["options_positions"] = {
        "SPY250118P00450000": {
            "entry_price": 1.00,
            "qty": 1,
            "strategy": "csp",
            "strategy_ref": {"ticker": "SPY", "strategy": "csp"},
            "underlying": "SPY",
        }
    }
    trader.state["option_spreads"] = {
        "SPY250118P00440000|SPY250118P00450000": {
            "strategy": "bull_put_spread",
            "legs": [
                {"symbol": "SPY250118P00450000", "side": "sell", "qty": 1},
                {"symbol": "SPY250118P00440000", "side": "buy", "qty": 1},
            ],
            "entry_credit": 1.20,
            "strategy_ref": {"ticker": "SPY", "strategy": "bps"},
        }
    }

    trader.state["strategy_state"] = {
        "SPY": {
            "csp": {"position_active": True, "via": "option"},
            "bps": {"position_active": True, "via": "option"},
        }
    }

    class Snap:
        def __init__(self, symbol):
            self.symbol = symbol

    monkeypatch.setattr(trader.od, "snapshots_for", lambda symbols: [Snap(s) for s in symbols])

    def fake_exit_rules(entry_price, snap):
        return ("buy", "take-profit") if snap.symbol.endswith("450000") else None

    monkeypatch.setattr(trader, "exit_rules_for_csp", fake_exit_rules)
    monkeypatch.setattr(trader, "bps_should_exit", lambda *args, **kwargs: ("buy", "spread-target"))
    monkeypatch.setattr(trader, "call_debit_decision", lambda *args, **kwargs: None)

    trader.monitor_option_exits()

    assert len(submitted) == 2, "Expected exits for the single-leg and spread positions"
    assert submitted[0]["asset_class"] == "option"
    assert submitted[1]["asset_class"] == "option_spread"
    assert trader.state["options_positions"] == {}
    assert trader.state["option_spreads"] == {}
    assert trader.state["strategy_state"]["SPY"]["csp"]["position_active"] is False
    assert trader.state["strategy_state"]["SPY"]["bps"]["position_active"] is False
