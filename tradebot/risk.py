"""Steps 3-4 of the framework: risk-based sizing, caps and exit rules."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SizeDecision:
    shares: int
    stop: float
    value: float
    risk_dollars: float
    binding_limit: str  # which rule set the final size


def size_position(
    *,
    equity: float,
    price: float,
    atr_value: float,
    adv_usd: float,
    sector_room: float,
    cash_available: float,
    cfg: dict,
) -> SizeDecision:
    """Position = (equity x risk%) / stop distance, then capped.

    Caps applied (smallest wins): per-name % of equity, % of daily traded
    value (liquidity), remaining sector budget and available cash.
    """
    p, u, x = cfg["portfolio"], cfg["universe"], cfg["exits"]
    stop_distance = x["stop_atr_multiple"] * atr_value
    if price <= 0 or not math.isfinite(stop_distance) or stop_distance <= 0:
        return SizeDecision(0, 0.0, 0.0, 0.0, "invalid data")

    candidates = {
        "risk budget": equity * p["risk_per_trade"] / stop_distance * price,
        "max per name": equity * p["max_position_pct"],
        "liquidity": adv_usd * u["max_pct_of_adv"],
        "sector cap": max(sector_room, 0.0),
        "cash": max(cash_available, 0.0),
    }
    binding = min(candidates, key=candidates.get)
    shares = int(candidates[binding] // price)
    stop = round(price - stop_distance, 2)
    return SizeDecision(
        shares=shares,
        stop=stop,
        value=shares * price,
        risk_dollars=shares * stop_distance,
        binding_limit=binding,
    )


def trail_stop(old_stop: float, close: float, atr_value: float, multiple: float) -> float:
    """Ratchet the stop upward only — a stop never moves against you."""
    candidate = close - multiple * atr_value
    if not math.isfinite(candidate):
        return old_stop
    return round(max(old_stop, candidate), 2)


def exit_reason(
    *,
    close: float,
    stop: float,
    days_held: int,
    entry_price: float,
    rank: int | None,
    keep_rank: int,
    is_rebalance_day: bool,
    time_stop_days: int,
) -> str | None:
    """Return why a position should be closed, or None to keep holding.

    Order matters: the hard price stop is checked first.
    """
    if close <= stop:
        return f"price stop hit (close {close:.2f} <= stop {stop:.2f})"
    if days_held >= time_stop_days and close < entry_price:
        return f"time stop: still losing after {days_held} trading days"
    if is_rebalance_day and (rank is None or rank > keep_rank):
        where = "dropped out of universe" if rank is None else f"rank fell to {rank}"
        return f"rebalance: {where} (keep if rank <= {keep_rank})"
    return None


def kill_criteria(symbol: str, stop: float, keep_rank: int, time_stop_days: int) -> str:
    """Pre-committed conditions that prove the trade wrong, written at entry."""
    return (
        f"Exit {symbol} if: close <= ${stop:.2f} (ATR stop, trailing); "
        f"composite rank falls below #{keep_rank} at a monthly rebalance; "
        f"still below entry after {time_stop_days} trading days."
    )
