# options/data.py
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest

# ✅ Correct imports for options market data
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionSnapshotRequest

class OptionsData:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        # Trading client (contracts, orders, etc.)
        self.trading = TradingClient(api_key, api_secret, paper=paper)
        # Historical options data client (snapshots/chain/quotes/trades)
        self.opt = OptionHistoricalDataClient(api_key, api_secret)

    def list_contracts(self, underlying: str, exp_lte: str | None = None):
        """Light wrapper over /v2/options/contracts (useful if you want contract metadata)."""
        req = GetOptionContractsRequest(underlying_symbol=underlying)
        if exp_lte:
            req.expiration_date_lte = exp_lte
        return self.trading.get_option_contracts(req)

    def chain_snapshot(self, underlying: str):
        """
        Returns the option 'chain' snapshots for an underlying:
        latest trade, latest quote, IV, and greeks for every contract.
        """
        req = OptionChainRequest(underlying_symbol=underlying)
        resp = self.opt.get_option_chain(req)   # Dict[contract_symbol -> OptionsSnapshot]
        # Your callers expect a list of snapshots
        return list(resp.values())

    # Backwards-compat alias (older code in the repo expected the pluralized name)
    def chain_snapshots(self, underlying: str):
        return self.chain_snapshot(underlying)

    def snapshots_for(self, symbols: list[str]):
        """
        Get snapshots for specific contract symbols (if you already picked contracts).
        """
        req = OptionSnapshotRequest(symbol_or_symbols=symbols)
        resp = self.opt.get_option_snapshot(req)  # Dict[symbol -> OptionsSnapshot]
        return [resp[s] for s in symbols if s in resp]
