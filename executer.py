# at top of auto_trader.py (or executor.py)
import os, logging
from notifier import send_slack
# existing imports: REST, calc_shares_for_risk, place_order_market_buy, etc.

def handle_signal(payload):
    """
    Generic handler for incoming webhook payloads.
    Expected payload keys: 'symbol'/'ticker', 'action' ('buy'/'sell'), 'price' (optional)
    Returns a dict with result status and details.
    """
    try:
        # normalize
        ticker = (payload.get("symbol") or payload.get("ticker") or payload.get("s") or "").upper()
        action = (payload.get("action") or payload.get("trade_action") or payload.get("type") or "buy").lower()
        price = float(payload.get("price") or 0.0)

        if not ticker:
            return {"status":"error","reason":"missing_ticker"}

        # fetch fresh account info
        acct = get_account_info()   # implement or reuse from auto_trader
        equity = acct["equity"]

        # simple sanity (account pause)
        paused, dd = account_paused()   # reuse from auto_trader
        if paused:
            send_slack(f":warning: Trading paused due to drawdown {dd:.2%}. Signal for {ticker} ignored.")
            return {"status":"paused","drawdown":dd}

        if action not in ("buy","long","sell","short"):
            return {"status":"error","reason":"unsupported_action", "action": action}

        # if price missing, fetch last price via yfinance or Alpaca
        if price <= 0:
            try:
                import yfinance as yf
                price = float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])
            except Exception:
                # fallback to Alpaca last trade
                try:
                    bar = api.get_last_trade(ticker)
                    price = float(bar.price)
                except Exception:
                    return {"status":"error","reason":"no_price_available"}

        # sizing (only implement buy in MVP; short/sell can be added)
        if action in ("buy","long"):
            qty = calc_shares_for_risk(equity, MAX_RISK_PCT, price, STOP_PCT)   # reuse function
            if qty <= 0 or qty * price < 1.0:
                send_slack(f":information_source: Signal {action} {ticker} skipped (insufficient size).")
                return {"status":"skipped","reason":"insufficient_size","qty":qty}
            # place order (market or limit)
            if ORDER_TYPE == "market":
                order = place_order_market_buy(ticker, qty)
            else:
                order = place_order_limit_buy(ticker, qty, price)
            order_id = getattr(order, "id", None) if order else None
            status = getattr(order, "status", None) if order else "failed"
            send_slack(f":white_check_mark: Placed BUY order {ticker} qty={qty} price={price} status={status} id={order_id}")
            # log (reuse your log function)
            log_trade_row({
                "timestamp": datetime.utcnow().isoformat(),
                "ticker": ticker, "action": "BUY", "signal_price": price,
                "qty": qty, "order_id": order_id, "status": status, "notes": "via webhook", "equity": equity, "cash": acct["cash"]
            })
            return {"status":"ok","order_id":order_id,"qty":qty}
        else:
            # Sell/short flow — safe-mode: if long position exists, exit it
            # Attempt to find open position and sell it
            pos_list = api.list_positions()
            pos = next((p for p in pos_list if p.symbol == ticker), None)
            if not pos:
                send_slack(f":information_source: Sell signal for {ticker} but no position found. Skipping.")
                return {"status":"skipped","reason":"no_position"}
            sell_qty = float(pos.qty)
            order = api.submit_order(symbol=ticker, qty=sell_qty, side='sell', type='market', time_in_force='day')
            order_id = getattr(order, "id", None)
            status = getattr(order, "status", None)
            send_slack(f":white_check_mark: Exit SELL {ticker} qty={sell_qty} status={status} id={order_id}")
            log_trade_row({...})  # similar logging
            return {"status":"ok","order_id":order_id,"qty":sell_qty}
    except Exception as e:
        logging.exception("handle_signal error: %s", e)
        send_slack(f":x: Error handling signal for payload {payload}: {e}")
        return {"status":"error","reason":str(e)}
