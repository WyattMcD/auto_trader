# strategies/spreads.py
from datetime import date
from typing import Optional, Dict, Any, List, Tuple
from config import (
    SPREADS_MIN_DTE, SPREADS_MAX_DTE, SPREADS_MIN_OI, SPREADS_MAX_REL_SPREAD,
    SPREADS_TARGET_DELTA, SPREADS_MAX_RISK_PER_TRADE
)

# ---------- Helpers ----------
def _why_no(s, reason, extra=None):
    sym = getattr(s, "symbol", "?")
    print(f"[spread-skip] {sym} -> {reason} {extra or ''}")

def _parse_occ(sym: str):
    """Parse OCC symbol -> (right 'C'/'P', expiry: date, strike: float)."""
    try:
        right = sym[-9]
        yy, mm, dd = sym[-15:-13], sym[-13:-11], sym[-11:-9]
        strike = int(sym[-8:]) / 1000.0
        exp = date(int("20" + yy), int(mm), int(dd))
        return right, exp, strike
    except Exception:
        return None, None, None

def _has_quote(snap) -> bool:
    q = getattr(snap, "latest_quote", None)
    return bool(q and q.bid_price and q.ask_price and q.ask_price > 0)

def _rel_spread(snap) -> float:
    q = snap.latest_quote
    return (q.ask_price - q.bid_price) / q.ask_price if q and q.ask_price else 1.0

def _dte(exp: date) -> int:
    return (exp - date.today()).days

def _ok_liquidity(snap) -> bool:
    if not _has_quote(snap):
        return False
    if _rel_spread(snap) > SPREADS_MAX_REL_SPREAD:
        return False
    oi = getattr(snap, "open_interest", 0) or 0
    return oi >= SPREADS_MIN_OI

def _delta_ok(snap) -> bool:
    lo, hi = SPREADS_TARGET_DELTA
    greeks = getattr(snap, "greeks", None)
    d = abs(getattr(greeks, "delta", 0) or 0)
    return lo <= d <= hi

def _mid_price(snap) -> Optional[float]:
    if not _has_quote(snap):
        return None
    q = snap.latest_quote
    return (q.bid_price + q.ask_price) / 2

# ---------- Strategies ----------

def pick_bull_put_spread(od, underlying: str) -> Optional[Dict[str, Any]]:
    """
    Short put (near 0.20-0.35Δ), long lower-strike put same expiry.
    Returns a spread intent with two legs and net credit.
    """
    snaps = od.chain_snapshots(underlying)
    if not snaps:
        return None

    # Index by (expiry -> puts sorted by strike)
    puts_by_exp: Dict[date, List[Tuple[float, Any]]] = {}
    short_candidates: Dict[date, List[Tuple[float, Any]]] = {}

    for s in snaps:
        sym = getattr(s, "symbol", "")
        r, exp, strike = _parse_occ(sym)
        if r != "P" or not exp:
            continue
        dte = _dte(exp)
        if dte < SPREADS_MIN_DTE or dte > SPREADS_MAX_DTE:
            continue
        if not _ok_liquidity(s):
            continue
        # Candidate short put must have delta in range
        puts_by_exp.setdefault(exp, []).append((strike, s))

        if _delta_ok(s):
            short_candidates.setdefault(exp, []).append((strike, s))

    if not short_candidates:
        return None

    # Try nearest expiry first
    for exp in sorted(short_candidates.keys(), key=lambda e: _dte(e)):
        puts = sorted(puts_by_exp.get(exp, []), key=lambda t: t[0])  # by strike
        if not puts:
            continue
        shorts = sorted(short_candidates.get(exp, []), key=lambda t: t[0])
        if not shorts:
            continue
        dte = _dte(exp)
        # For each short put candidate, try to find a lower strike long put
        for short_strike, short_snap in shorts:
            # pick a protection leg 1-3 strikes lower with decent liquidity
            # We'll walk the list for the next lower strike
            lower_candidates = [p for p in puts if p[0] < short_strike]
            if not lower_candidates:
                continue
            # choose the closest lower strike
            long_strike, long_snap = max(lower_candidates, key=lambda t: t[0])

            short_mid = _mid_price(short_snap)
            long_mid  = _mid_price(long_snap)
            if short_mid is None or long_mid is None:
                continue

            net_credit = round(max(short_mid * 0.98 - long_mid * 1.02, 0.01), 2)  # credit, nudge for fill
            width = (short_strike - long_strike) * 100.0  # dollars
            max_loss = round(width - net_credit * 100.0, 2)  # in dollars

            if max_loss <= 0 or max_loss > SPREADS_MAX_RISK_PER_TRADE:
                continue

            entry_credit = round(net_credit * 100, 2)
            tp_credit = round(entry_credit * 0.50, 2)
            sl_credit = round(entry_credit * 2.0, 2)
            meta = {
                "strategy": "bull_put_spread",
                "why": (
                    f"Collect ≈${entry_credit:.2f} credit by shorting {short_strike}P/long {long_strike}P "
                    f"({dte} DTE)."
                ),
                "thesis": (
                    "Defined-risk bullish credit spread: short higher strike put with protection lower strike "
                    "to harvest premium while capping downside."
                ),
                "underlying": underlying,
                "expiry": exp.isoformat(),
                "dte": dte,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "net_credit": entry_credit,
                "max_loss": max_loss,
                "plan": {
                    "entry": (
                        f"Sell {getattr(short_snap, 'symbol')} / buy {getattr(long_snap, 'symbol')} for ≈${entry_credit:.2f} credit"
                    ),
                    "take_profit": (
                        f"Buy-to-close both legs if remaining credit falls to ${tp_credit:.2f} (≈50% of entry credit)"
                    ),
                    "stop_loss": (
                        f"Buy-to-close if spread trades ≥${sl_credit:.2f} credit (≈200% of entry credit)"
                    ),
                },
            }

            return {
                "asset_class": "option_spread",
                "strategy": "bull_put_spread",
                "underlying": underlying,
                "expiry": exp.isoformat(),
                "net_limit": net_credit,
                "tif": "day",
                "legs": [
                    {   # SHORT PUT
                        "symbol": getattr(short_snap, "symbol"),
                        "side": "sell",
                        "type": "limit",
                        "qty": 1,
                        "limit_price": round(short_mid * 0.98, 2),
                    },
                    {   # LONG PUT (protection)
                        "symbol": getattr(long_snap, "symbol"),
                        "side": "buy",
                        "type": "limit",
                        "qty": 1,
                        "limit_price": round(long_mid * 1.02, 2),
                    }
                ],
                "risk": {
                    "width": width,                  # strike diff * 100
                    "net_credit": net_credit * 100,  # in dollars per spread
                    "max_loss": max_loss,            # dollars
                },
                "quality": {
                    "short_rel_spread": _rel_spread(short_snap),
                    "long_rel_spread": _rel_spread(long_snap),
                    "short_oi": getattr(short_snap, "open_interest", 0) or 0,
                    "long_oi": getattr(long_snap, "open_interest", 0) or 0,
                },
                "meta": meta,
            }
    return None


def pick_call_debit_spread(od, underlying: str) -> Optional[Dict[str, Any]]:
    """
    Buy near-ATM call, sell higher-strike call same expiry.
    Returns a spread intent (net debit) with two legs.
    """
    snaps = od.chain_snapshots(underlying)
    if not snaps:
        return None

    # collect calls per expiry
    calls_by_exp: Dict[date, List[Tuple[float, Any]]] = {}
    atms: Dict[date, Any] = {}

    for s in snaps:
        sym = getattr(s, "symbol", "")
        r, exp, strike = _parse_occ(sym)
        if r != "C" or not exp:
            continue
        dte = _dte(exp)
        if dte < SPREADS_MIN_DTE or dte > SPREADS_MAX_DTE:
            continue
        if not _ok_liquidity(s):
            continue
        calls_by_exp.setdefault(exp, []).append((strike, s))

    if not calls_by_exp:
        return None

    for exp in sorted(calls_by_exp.keys(), key=lambda e: _dte(e)):
        calls = sorted(calls_by_exp[exp], key=lambda t: t[0])
        # pick an ATM-ish long call: closest to underlying via greeks or by quote moneyness
        # Without underlying price here, we approximate ATM as median strike
        if not calls:
            continue
        mid_idx = len(calls) // 2
        long_strike, long_snap = calls[mid_idx]

        # pick a short call few strikes above
        higher = [c for c in calls if c[0] > long_strike]
        if not higher:
            continue
        short_strike, short_snap = min(higher, key=lambda t: t[0])

        long_mid = _mid_price(long_snap)
        short_mid = _mid_price(short_snap)
        if long_mid is None or short_mid is None:
            continue

        net_debit = round(max(long_mid * 1.02 - short_mid * 0.98, 0.01), 2)
        width = (short_strike - long_strike) * 100.0
        max_gain = round(width - net_debit * 100.0, 2)

        if net_debit * 100.0 > SPREADS_MAX_RISK_PER_TRADE:
            continue
        if max_gain <= 0:
            continue

        entry_debit = round(net_debit * 100, 2)
        tp_gain = round(max_gain * 0.50, 2)
        sl_debit = round(entry_debit * 0.50, 2)
        meta = {
            "strategy": "call_debit_spread",
            "why": (
                f"Pay ≈${entry_debit:.2f} to buy {long_strike}C / sell {short_strike}C ({dte} DTE) targeting upside breakout."
            ),
            "thesis": (
                "Defined-risk bullish debit spread: long near-ATM call financed with short higher strike call "
                "to capture upside with limited cost."
            ),
            "underlying": underlying,
            "expiry": exp.isoformat(),
            "dte": dte,
            "long_strike": long_strike,
            "short_strike": short_strike,
            "net_debit": entry_debit,
            "max_gain": max_gain,
            "plan": {
                "entry": (
                    f"Buy {getattr(long_snap, 'symbol')} / sell {getattr(short_snap, 'symbol')} for ≈${entry_debit:.2f} debit"
                ),
                "take_profit": (
                    f"Close spread once ~50% of max gain (≈${tp_gain:.2f}) is captured"
                ),
                "stop_loss": (
                    f"Exit if spread value contracts to ${sl_debit:.2f} (≈50% of entry debit)"
                ),
            },
        }

        return {
            "asset_class": "option_spread",
            "strategy": "call_debit_spread",
            "underlying": underlying,
            "expiry": exp.isoformat(),
            "net_limit": -net_debit,  # negative = debit
            "tif": "day",
            "legs": [
                {   # LONG CALL
                    "symbol": getattr(long_snap, "symbol"),
                    "side": "buy",
                    "type": "limit",
                    "qty": 1,
                    "limit_price": round(long_mid * 1.02, 2),
                },
                {   # SHORT CALL
                    "symbol": getattr(short_snap, "symbol"),
                    "side": "sell",
                    "type": "limit",
                    "qty": 1,
                    "limit_price": round(short_mid * 0.98, 2),
                }
            ],
            "risk": {
                "width": width,
                "net_debit": net_debit * 100,  # dollars
                "max_gain": max_gain,          # dollars
            },
            "quality": {
                "long_rel_spread": _rel_spread(long_snap),
                "short_rel_spread": _rel_spread(short_snap),
                "long_oi": getattr(long_snap, "open_interest", 0) or 0,
                "short_oi": getattr(short_snap, "open_interest", 0) or 0,
            },
            "meta": meta,
        }

    return None
