from types import SimpleNamespace

import pytest

from config import MAX_RISK_PER_TRADE
from risk import approve_csp_intent, capped_qty_to_buy


@pytest.fixture
def account():
    return SimpleNamespace(equity=10000, buying_power=20000, cash=8000)


def test_approve_csp_intent_rejects_when_risk_exceeds(account):
    high_bpr = account.equity * MAX_RISK_PER_TRADE + 1
    assert not approve_csp_intent({}, account, open_csp_count=0, est_bpr=high_bpr)


def test_approve_csp_intent_allows_within_risk(account):
    safe_bpr = account.equity * MAX_RISK_PER_TRADE * 0.5
    assert approve_csp_intent({}, account, open_csp_count=0, est_bpr=safe_bpr)


def test_capped_qty_to_buy_respects_five_percent_cap(monkeypatch):
    """Ensure we never size above ~5% of equity when buying shares."""

    class Client:
        def __init__(self):
            self._orders = []

        def get_account(self):
            return SimpleNamespace(equity=10000)

        def get_open_position(self, symbol):
            raise Exception("no position")

        def get_orders(self, status="open"):
            return self._orders

    client = Client()

    qty = capped_qty_to_buy(client, symbol="SPY", price=100, intended_qty=10)
    assert qty == 5  # 5% of 10k equity = $500 => 5 shares at $100


def test_capped_qty_to_buy_accounts_for_pending_orders(monkeypatch):
    class Order(SimpleNamespace):
        pass

    class Client:
        def __init__(self):
            self._orders = [
                Order(symbol="SPY", side="buy", qty=2, limit_price=100),
            ]

        def get_account(self):
            return SimpleNamespace(equity=10000)

        def get_open_position(self, symbol):
            return SimpleNamespace(market_value=200)  # already holding $200 of SPY

        def get_orders(self, status="open"):
            return self._orders

    client = Client()

    qty = capped_qty_to_buy(client, symbol="SPY", price=100, intended_qty=10)
    # Equity cap: $500, minus $200 existing MV and $200 pending = $100 left => 1 share
    assert qty == 1
