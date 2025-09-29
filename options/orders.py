# options/orders.py
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import TimeInForce, OrderSide, AssetClass
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from typing import List, Dict, Any


class OptionsTrader:
    def __init__(self, api_key, api_secret, paper=True):
        self.client = TradingClient(api_key, api_secret, paper=paper)

    def submit_single_leg(self, contract_symbol: str, side: str, qty: int,
                          order_type: str = "market", limit_price: float | None = None,
                          tif: str = "day"):
        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif_enum = TimeInForce(str(tif).lower())
        qty_str = str(int(qty))

        if order_type.lower() == "limit":
            if limit_price is None or float(limit_price) <= 0:
                raise ValueError("limit_price must be > 0 for limit orders")
            req = LimitOrderRequest(
                symbol=contract_symbol,
                qty=qty_str,
                side=side_enum,
                time_in_force=tif_enum,
                limit_price=float(limit_price),
                asset_class=AssetClass.US_OPTION,
            )
        else:
            # market
            req = MarketOrderRequest(
                symbol=contract_symbol,
                qty=qty_str,
                side=side_enum,
                time_in_force=tif_enum,
                asset_class=AssetClass.US_OPTION,
            )

        return self.client.submit_order(req)

    def submit_spread_paired(self, legs: List[Dict[str, Any]], tif: str = "day"):
        """
        Submit two single-leg limit orders back-to-back.
        If the second fails to submit, we cancel the first.
        NOTE: This is a pragmatic way to place spreads when a single 'multi-leg' ticket
        is not available via SDK. It does NOT guarantee simultaneous fills.
        """
        assert len(legs) == 2, "Only 2-leg spreads supported in this helper."
        tif_val = str(tif).lower()

        # Place the 'buy' leg first for debit spreads; for credit spreads, place the 'sell' first.
        # We'll infer from net (sell has positive credit at higher price).
        first, second = legs[0], legs[1]

        # Heuristic: for credit strategies (e.g., BPS), place SELL first.
        # Reorder so SELL leg is first if present.
        if second["side"].lower() == "sell" and first["side"].lower() == "buy":
            first, second = second, first

        # Submit first
        o1 = self.submit_single_leg(
            contract_symbol=first["symbol"],
            side=first["side"],
            qty=first["qty"],
            order_type=first.get("type","limit"),
            limit_price=first.get("limit_price"),
            tif=tif_val,
        )

        try:
            # Submit second
            o2 = self.submit_single_leg(
                contract_symbol=second["symbol"],
                side=second["side"],
                qty=second["qty"],
                order_type=second.get("type","limit"),
                limit_price=second.get("limit_price"),
                tif=tif_val,
            )
            return o1, o2
        except Exception as e:
            # Try to cancel first if second fails to submit
            try:
                self.client.cancel_order(o1.id)
            except Exception:
                pass
            raise
