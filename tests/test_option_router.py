from options.router import RouterContext, plan_option_intent_for_signal


class DummyOptionsData:
    pass


def test_router_prefers_spread_when_equity_sizing_too_small():
    context = RouterContext(
        ticker="SPY",
        price=200.0,
        equity_qty=0.0,
        state={"options_positions": {}, "option_spreads": {}},
        open_spreads=0,
        open_short_puts=0,
        account=None,
    )

    called = {}

    def fake_pick_spread(data, ticker):
        called["spread"] = ticker
        return {
            "asset_class": "option_spread",
            "legs": [],
            "meta": {},
            "risk": {"max_loss": 100},
        }

    intent = plan_option_intent_for_signal(
        context,
        options_data=DummyOptionsData(),
        min_equity_qty=5,
        max_equity_price=150.0,
        pick_spread=fake_pick_spread,
        allow_spreads=True,
    )

    assert intent is not None
    assert intent["asset_class"] == "option_spread"
    assert intent.get("meta", {}).get("underlying") == "SPY"
    assert called["spread"] == "SPY"


def test_router_skips_when_underlying_already_has_option_position():
    state = {
        "options_positions": {
            "SPY250118P00450000": {"underlying": "SPY"}
        },
        "option_spreads": {},
    }
    context = RouterContext(
        ticker="SPY",
        price=220.0,
        equity_qty=0.0,
        state=state,
        open_spreads=0,
        open_short_puts=0,
        account=None,
    )

    intent = plan_option_intent_for_signal(
        context,
        options_data=DummyOptionsData(),
        min_equity_qty=5,
        max_equity_price=150.0,
        allow_spreads=True,
        pick_spread=lambda *_: {"asset_class": "option_spread", "risk": {"max_loss": 50}},
    )

    assert intent is None


def test_router_leaves_equity_path_when_thresholds_met():
    context = RouterContext(
        ticker="SPY",
        price=120.0,
        equity_qty=10,
        state={"options_positions": {}, "option_spreads": {}},
        open_spreads=0,
        open_short_puts=0,
        account=None,
    )

    def fail_pick(*_):
        raise AssertionError("Should not query option chain when equity sizing is adequate")

    intent = plan_option_intent_for_signal(
        context,
        options_data=DummyOptionsData(),
        min_equity_qty=5,
        max_equity_price=150.0,
        allow_spreads=True,
        pick_spread=fail_pick,
    )

    assert intent is None
