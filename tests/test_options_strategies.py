from datetime import date, timedelta

from options.router import RouterContext, plan_option_intent_for_signal
from strategies.options_csp import pick_csp_intent
from strategies.spreads import pick_bull_put_spread


class DummyQuote:
    def __init__(self, bid, ask):
        self.bid_price = bid
        self.ask_price = ask


class DummyGreeks:
    def __init__(self, delta):
        self.delta = delta


class DummySnapshot:
    def __init__(self, symbol, bid, ask, delta, oi):
        self.symbol = symbol
        self.latest_quote = DummyQuote(bid, ask)
        self.greeks = DummyGreeks(delta)
        self.open_interest = oi


class DummyOptionsData:
    def __init__(self, snapshots):
        self._snapshots = snapshots

    def chain_snapshots(self, underlying):
        return list(self._snapshots)


def occ_symbol(root: str, exp: date, right: str, strike: float) -> str:
    return f"{root}{exp:%y%m%d}{right}{int(strike * 1000):08d}"


def make_snapshot(root: str, days: int, right: str, strike: float, bid: float, ask: float, delta: float, oi: int):
    exp = date.today() + timedelta(days=days)
    symbol = occ_symbol(root, exp, right, strike)
    return DummySnapshot(symbol, bid, ask, delta, oi)


def test_pick_csp_intent_returns_viable_cash_secured_put():
    snaps = [
        make_snapshot("SPY", 21, "P", 390, 1.25, 1.40, 0.22, 500),
        make_snapshot("SPY", 14, "P", 395, 1.35, 1.50, 0.27, 750),
    ]
    od = DummyOptionsData(snaps)

    intent = pick_csp_intent(od, "SPY")

    assert intent is not None
    assert intent["asset_class"] == "option"
    assert intent["symbol"].endswith("P00395000")
    mid = (1.35 + 1.50) / 2
    expected_limit = round(max(mid * 0.98, 0.05), 2)
    assert intent["limit_price"] == expected_limit
    meta = intent["meta"]
    assert meta["strategy"] == "cash_secured_put"
    assert meta["underlying"] == "SPY"


def test_pick_bull_put_spread_allows_lower_delta_long_leg():
    snaps = [
        make_snapshot("SPY", 21, "P", 100, 5.0, 5.6, 0.28, 600),
        make_snapshot("SPY", 21, "P", 95, 1.0, 1.2, 0.12, 600),
        make_snapshot("SPY", 28, "P", 100, 4.5, 5.1, 0.25, 400),
        make_snapshot("SPY", 28, "P", 95, 0.9, 1.1, 0.10, 400),
    ]
    od = DummyOptionsData(snaps)

    spread = pick_bull_put_spread(od, "SPY")

    assert spread is not None
    assert spread["asset_class"] == "option_spread"
    assert spread["strategy"] == "bull_put_spread"
    legs = spread["legs"]
    assert legs[0]["side"] == "sell"
    assert legs[1]["side"] == "buy"
    assert legs[0]["symbol"].endswith("P00100000")
    assert legs[1]["symbol"].endswith("P00095000")
    risk = spread["risk"]
    assert risk["max_loss"] > 0
    assert risk["max_loss"] <= 150


def test_router_can_route_to_cash_secured_put():
    context = RouterContext(
        ticker="SPY",
        price=250.0,
        equity_qty=0,
        state={"options_positions": {}, "option_spreads": {}},
        open_spreads=0,
        open_short_puts=0,
        account=object(),
    )

    called = {}

    def fake_pick_csp(data, ticker):
        called["ticker"] = ticker
        return {"asset_class": "option", "symbol": "SPY250101P00380000", "meta": {}}

    def fake_approve(intent, account, open_short_puts, est_bpr):
        called["approved"] = (intent, est_bpr)
        return True

    intent = plan_option_intent_for_signal(
        context,
        options_data=DummyOptionsData([]),
        allow_spreads=False,
        allow_csp=True,
        pick_csp=fake_pick_csp,
        approve_csp=fake_approve,
    )

    assert intent is not None
    assert intent["asset_class"] == "option"
    assert called["ticker"] == "SPY"
    assert "approved" in called
