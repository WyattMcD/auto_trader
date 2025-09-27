# executer.py
from typing import Any, Dict
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

def execute_intent(intent: Dict[str, Any],
                   equity_trader: TradingClient,
                   options_trader) -> Any:
    """
    Routes a normalized trading intent to the correct submitter.
    Supports: equity (single-leg), option (single-leg), option_spread (2-leg).
    """
    otype = str(intent.get("type", "market")).lower()
    tif   = TimeInForce(str(intent.get("tif", "day")).lower())

    # ---- Options: 2-leg spread ----
    if intent.get("asset_class") == "option_spread":
        legs = intent["legs"]  # [{symbol, side, type, qty, limit_price}, {...}]
        return options_trader.submit_spread_paired(legs=legs, tif=str(tif.value))

    # ---- Single-leg option ----
    if intent.get("asset_class") == "option":
        return options_trader.submit_single_leg(
            contract_symbol=intent["symbol"],
            side=intent["side"],
            qty=intent["qty"],
            order_type=otype,
            limit_price=intent.get("limit_price"),
            tif=str(tif.value),
        )

    # ---- Equity path (single-leg) ----
    side_enum = OrderSide.BUY if str(intent["side"]).lower() == "buy" else OrderSide.SELL

    if otype == "limit":
        req = LimitOrderRequest(
            symbol=intent["symbol"],
            qty=str(intent["qty"]),
            side=side_enum,
            time_in_force=tif,
            limit_price=float(intent["limit_price"]),
        )
    else:
        req = MarketOrderRequest(
            symbol=intent["symbol"],
            qty=str(intent["qty"]),
            side=side_enum,
            time_in_force=tif,
        )

    return equity_trader.submit_order(req)
