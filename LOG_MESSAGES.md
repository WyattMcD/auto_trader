# Log Message Reference: Scan Cycle and Exposure Caps

This document explains the recurring informational log entries that appear in
`auto_trader.py` while the main loop is running. Use it as a quick reference
when reviewing Docker logs in production or during troubleshooting sessions.

## Scan Cycle Pacing

```
INFO Sleeping for 5 minutes before next scan.
```

This message is emitted at the end of each pass through the main loop. After
finishing a scan, the bot pauses for the configured `SCAN_INTERVAL_MINUTES`
value (five minutes by default) before starting the next cycle. The log entry is
produced immediately before the `time.sleep` call that enforces this delay.【F:auto_trader.py†L2829-L2836】

## Beginning of a Scan

```
INFO Beginning scan cycle for 42 symbols — slots used 11/20 (filled=11, pending=0).
```

At the start of each scan, the bot gathers account, position, and pending-order
information to determine how many "slots" (concurrent positions) are currently
in use. The message records:

* The number of symbols in the active watchlist (42 in this example).
* `slots used`: the total slots counted against the concurrency cap.
* `filled`: the number of positions that are already open.
* `pending`: how many buy orders are waiting to fill.

These values come from the concurrency snapshot assembled just before iterating
through the watchlist.【F:auto_trader.py†L2167-L2211】

## Per-Symbol Exposure Cap Skips

```
INFO Skipping XOM/rsi_sma buy — per-symbol exposure cap reached (intended=43).
```

When a strategy generates a buy signal, the system calculates the intended share
quantity using its risk model. Before submitting the order it checks the
per-symbol exposure cap enforced by `capped_qty_to_buy`. If the existing
position value plus any pending buys would exceed the configured maximum
percentage of account equity, the quantity is reduced. If no shares can be added
without breaching the cap, the buy is skipped entirely, producing this log
message. The `intended` field records the original share count proposed by the
risk calculation.【F:auto_trader.py†L2392-L2424】【F:risk.py†L70-L115】

## Scan Summary

```
INFO Scan complete in 40.3s — considered 42 symbols. Signals seen: buy=8, sell=8. Orders placed: buy=0, sell=0. Pending-buy skips=0, concurrency blocks=0, trailing updates=0. Slots remaining: 9/20.
```

At the end of each scan the bot summarizes the work it performed:

* `Scan complete in …`: total runtime of the cycle.
* `considered`: number of tickers processed.
* `Signals seen`: how many buy/sell signals were triggered.
* `Orders placed`: the number of orders actually submitted.
* `Pending-buy skips`: signals ignored because an order is already open for the ticker.
* `Concurrency blocks`: signals suppressed because the global position limit is reached.
* `Trailing updates`: how many trailing stop adjustments succeeded.
* `Slots remaining`: how many position slots are free vs. the maximum.

This report helps confirm that the system is evaluating symbols and enforcing
risk controls as expected.【F:auto_trader.py†L2768-L2826】
