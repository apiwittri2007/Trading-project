"""The daily run: the whole framework, one step at a time.

1. Universe & liquidity filter
2. Factor scores -> composite rank, trend filter
3. Manage open positions: trail stops, apply exit rules
4. On the monthly rebalance: buy the best-ranked names, sized by risk
5. Record equity vs. benchmark
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import risk, signals
from .state import State

log = logging.getLogger(__name__)


def liquidity_filter(close: pd.DataFrame, volume: pd.DataFrame, cfg: dict) -> pd.Index:
    u = cfg["universe"]
    last = close.iloc[-1]
    adv = signals.avg_dollar_volume(close, volume)
    ok = (last >= u["min_price"]) & (adv >= u["min_adv_usd"])
    return ok[ok].index


def _trading_days_since(index: pd.DatetimeIndex, date: str) -> int:
    return int((index > pd.Timestamp(date)).sum())


def _buy_reason(sym: str, row: pd.Series, n: int, sizing: risk.SizeDecision) -> str:
    parts = [f"{f} z={row[f + '_z']:+.2f}" for f in ("momentum", "value", "quality") if f + "_z" in row and pd.notna(row[f + "_z"])]
    return (
        f"Rank #{int(row['rank'])} of {n}, score {row['score']:+.2f} ({', '.join(parts)}); "
        f"above 200-day MA; size set by {sizing.binding_limit}; "
        f"risk ${sizing.risk_dollars:,.0f} to stop"
    )


def run_day(
    *,
    cfg: dict,
    prices: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    benchmark_close: float | None,
    broker,
    state: State,
) -> dict:
    close, high, low, volume = (prices[k] for k in ("close", "high", "low", "volume"))
    today = close.index[-1]
    today_s = today.strftime("%Y-%m-%d")
    x, p = cfg["exits"], cfg["portfolio"]
    sector_of = universe.set_index("symbol")["sector"].to_dict()

    last_close = close.iloc[-1]
    atr_now = signals.atr(high, low, close, x["atr_period"])
    adv = signals.avg_dollar_volume(close, volume)
    broker.mark(last_close.dropna().to_dict())

    # ---- Steps 1-2: filter and rank -------------------------------------
    eligible = liquidity_filter(close, volume, cfg)
    ranking = signals.rank_universe(close[eligible], fundamentals, cfg)
    for f in ("momentum", "value", "quality"):
        if f in ranking:
            ranking[f + "_z"] = signals.zscore(ranking[f])
    ranking["sector"] = ranking.index.map(sector_of)
    n_ranked = len(ranking)
    keep_rank = int(p["max_positions"] * x["sell_rank_buffer"])

    is_rebalance = state.meta.get("last_rebalance", "")[:7] != today_s[:7]
    already_ran_today = state.meta.get("last_run") == today_s
    log.info("%s: %d eligible, rebalance=%s, broker=%s", today_s, n_ranked, is_rebalance, broker.name)

    # ---- Reconcile with the broker ---------------------------------------
    held = broker.positions()
    pending_sells = set(state.meta.get("pending_sells", []))  # Alpaca: sold after close, fills next open
    for sym in list(state.positions):
        if sym not in held and state.positions[sym].get("entry_date") != today_s:
            log.warning("%s no longer at broker; dropping from state", sym)
            state.positions.pop(sym)
    for sym, bp in held.items():
        if sym in pending_sells and sym not in state.positions:
            continue  # exit order still waiting to fill

        pos = state.positions.setdefault(
            sym,
            {
                "entry_date": today_s,
                "entry_price": bp["avg_price"],
                "stop": round(bp["avg_price"] - x["stop_atr_multiple"] * float(atr_now.get(sym, 0) or 0), 2),
                "sector": sector_of.get(sym, "Unknown"),
                "reason": "adopted from broker",
                "kill_criteria": "",
            },
        )
        pos["qty"], pos["entry_price"] = bp["qty"], bp["avg_price"]

    # ---- Step 3: manage open positions -----------------------------------
    sells = []
    for sym, pos in list(state.positions.items()):
        px = float(last_close.get(sym, np.nan))
        if not np.isfinite(px):
            continue
        a = float(atr_now.get(sym, np.nan))
        if x.get("trail_stop") and not already_ran_today:
            pos["stop"] = risk.trail_stop(pos["stop"], px, a, x["stop_atr_multiple"])
        pos["days_held"] = _trading_days_since(close.index, pos["entry_date"])
        pos["last_price"] = px
        pos["rank"] = int(ranking.at[sym, "rank"]) if sym in ranking.index else None
        why = risk.exit_reason(
            close=px,
            stop=pos["stop"],
            days_held=pos["days_held"],
            entry_price=pos["entry_price"],
            rank=pos["rank"],
            keep_rank=keep_rank,
            is_rebalance_day=is_rebalance,
            time_stop_days=x["time_stop_days"],
        )
        if why and pos.get("qty", 0) > 0:
            sells.append((sym, why))

    for sym, why in sells:
        pos = state.positions.pop(sym)
        fill = broker.submit(sym, pos["qty"], "sell", float(last_close[sym]))
        pnl = (fill.price - pos["entry_price"]) / pos["entry_price"]
        state.record_trade(
            date=today_s, symbol=sym, side="sell", qty=fill.qty, price=fill.price,
            value=round(fill.qty * fill.price, 2), reason=f"{why}; P&L {pnl:+.1%}",
            kill_criteria=pos.get("kill_criteria", ""), broker=broker.name, order_id=fill.order_id,
        )
    state.meta["pending_sells"] = [s for s, _ in sells] if broker.name != "simulator" else []

    # ---- Step 4: monthly rebalance buys ----------------------------------
    acct = broker.account()
    equity = acct["equity"]
    if is_rebalance and not already_ran_today:
        cash = acct["cash"] - p["min_cash_pct"] * equity
        sector_value: dict[str, float] = {}
        for sym, pos in state.positions.items():
            sector_value[pos["sector"]] = sector_value.get(pos["sector"], 0.0) + pos["qty"] * float(last_close.get(sym, pos["entry_price"]))
        slots = p["max_positions"] - len(state.positions)
        candidates = ranking[ranking["trend_ok"]].head(p["max_positions"])
        for sym, row in candidates.iterrows():
            if slots <= 0:
                break
            if sym in state.positions:
                continue
            sector = row["sector"] if isinstance(row["sector"], str) else "Unknown"
            sizing = risk.size_position(
                equity=equity,
                price=float(last_close[sym]),
                atr_value=float(atr_now[sym]),
                adv_usd=float(adv[sym]),
                sector_room=p["max_sector_pct"] * equity - sector_value.get(sector, 0.0),
                cash_available=cash,
                cfg=cfg,
            )
            if sizing.shares < 1:
                log.info("skip %s: size 0 (%s)", sym, sizing.binding_limit)
                continue
            fill = broker.submit(sym, sizing.shares, "buy", float(last_close[sym]))
            kill = risk.kill_criteria(sym, sizing.stop, keep_rank, x["time_stop_days"])
            reason = _buy_reason(sym, row, n_ranked, sizing)
            state.positions[sym] = {
                "qty": fill.qty, "entry_price": fill.price, "entry_date": today_s, "stop": sizing.stop,
                "sector": sector, "score": round(float(row["score"]), 3), "rank": int(row["rank"]),
                "reason": reason, "kill_criteria": kill, "days_held": 0, "last_price": float(last_close[sym]),
            }
            state.record_trade(
                date=today_s, symbol=sym, side="buy", qty=fill.qty, price=fill.price,
                value=round(fill.qty * fill.price, 2), reason=reason, kill_criteria=kill,
                broker=broker.name, order_id=fill.order_id,
            )
            cash -= sizing.value
            sector_value[sector] = sector_value.get(sector, 0.0) + sizing.value
            slots -= 1
        state.meta["last_rebalance"] = today_s

    # ---- Step 5: record ---------------------------------------------------
    acct = broker.account()
    state.meta.update({"last_run": today_s, "broker": broker.name, "eligible": n_ranked})
    broker.save()
    state.save(
        equity_row={
            "date": today_s, "equity": round(acct["equity"], 2), "cash": round(acct["cash"], 2),
            "positions": len(state.positions), "benchmark": benchmark_close,
        },
        ranking=ranking.drop(columns=[c for c in ranking.columns if c.endswith("_z")]),
    )
    return {"date": today_s, "sells": len(sells), "positions": len(state.positions), "equity": acct["equity"]}
