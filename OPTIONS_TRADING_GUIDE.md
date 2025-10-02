# Options vs. Shares Execution Guide

This guide outlines when the auto trader should execute strategies with equity shares versus option contracts. It reflects the recommended operating plan for integrating a focused options layer while keeping core share-based strategies.

## Baseline: Shares First

- **Default mode:** Run all baseline strategies with the underlying shares. This keeps execution simple, robust, and inexpensive.
- **Use shares when:**
  - Signal edge is modest or slow-moving (e.g., trend following, moving average crossovers).
  - The underlying equity is highly liquid but its options chain is illiquid or exhibits wide bid/ask spreads.
  - You need straightforward P&L mapping for backtests, risk analytics, or partial scaling.

## Options as a Tactical Layer

- **Purpose:** Deploy options selectively as a "scalpel" for well-defined, capital-efficient plays.
- **Use options when:**
  - You want defined risk and smaller notional exposure per trade, which suits accounts with limited capital.
  - The signal is tactical or short-horizon (e.g., breakouts, momentum around catalysts) where convex payoff profiles are beneficial.
  - Implied volatility conditions align with the strategy (e.g., buying premium when expecting expansion, selling premium when expecting mean reversion).
  - Liquidity filters are satisfied (tight bid/ask spreads, strong open interest and volume) so the bot can get fills without excessive slippage.

## Implementation Notes

1. **Keep both execution paths:** Shares remain the baseline. Layer in options for the subset of signals that justify the complexity.
2. **Risk controls:**
   - Use defined-risk option structures (debit spreads, credit spreads, long options) to cap downside.
   - Size positions using option Greeks and max-loss calculations instead of share count alone.
3. **Operational safeguards:**
   - Build liquidity checks (spread, open interest, volume) into the options execution pipeline.
   - Monitor Greeks/IV shifts to avoid unexpected exposure.
4. **Backtesting alignment:** Ensure the backtest engine supports both shares and options so historical performance reflects live execution choices.

By following this plan, the trader maintains the simplicity and robustness of share-based strategies while gaining the precision and capital efficiency of options for targeted, high-conviction setups.
