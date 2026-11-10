"""Historical test of the same rules the live bot uses.

Why price-only: free sources only give *today's* fundamentals. Using today's
P/E or ROE to pick stocks in 2015 would leak future information (look-ahead
bias), so the backtest ranks on momentum alone and keeps every other rule
identical: liquidity filter, 200-day trend filter, ATR risk sizing, caps,
trailing stops, time stop and monthly rebalancing, with slippage.

Known bias: the universe is today's index membership, so stocks that were
delisted or dropped are missing (survivorship bias). Treat results as an
upper bound, not a forecast.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import risk, signals

log = logging.getLogger(__name__)
LOOKBACK = 260  # trading days of history needed for 12-1 momentum and the 200-day MA


def performance_stats(equity: pd.Series) -> dict:
    equity = equity.dropna()
    rets = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    dd = equity / equity.cummax() - 1
    vol = rets.std() * np.sqrt(252)
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1),
        "volatility": float(vol),
        "sharpe": float(rets.mean() * 252 / vol) if vol > 0 else 0.0,
        "max_drawdown": float(dd.min()),
    }


def run_backtest(prices: dict[str, pd.DataFrame], benchmark: pd.Series, universe: pd.DataFrame, cfg: dict) -> dict:
    close, high, low, volume = (prices[k] for k in ("close", "high", "low", "volume"))
    x, p, b = cfg["exits"], cfg["portfolio"], cfg["backtest"]
    slip = cfg["execution"]["slippage_bps"] / 10_000
    sector_of = universe.set_index("symbol")["sector"].to_dict()
    bt_cfg = {**cfg, "factors": {**cfg["factors"], "momentum": 1.0, "value": 0, "quality": 0}}
    keep_rank = int(p["max_positions"] * x["sell_rank_buffer"])

    first = max(int(np.searchsorted(close.index, pd.Timestamp(b["start"]))), LOOKBACK)
    dates = close.index[first:]
    if len(dates) == 0:
        raise ValueError("not enough history for the backtest start date")

    cash = float(cfg["starting_equity"])
    book: dict[str, dict] = {}
    trades, curve = [], []
    last_month = None

    for d in dates:
        i = close.index.get_loc(d)
        win = slice(max(0, i - LOOKBACK + 1), i + 1)
        c, h, l, v = close.iloc[win], high.iloc[win], low.iloc[win], volume.iloc[win]
        px = c.iloc[-1]
        a = signals.atr(h, l, c, x["atr_period"])
        rebalance = d.month != last_month

        ranking = None
        if rebalance:
            adv = signals.avg_dollar_volume(c, v)
            liquid = adv[(px >= cfg["universe"]["min_price"]) & (adv >= cfg["universe"]["min_adv_usd"])]
            top = liquid.nlargest(b["universe_size"]).index
            ranking = signals.rank_universe(c[top], None, bt_cfg)

        # exits
        for sym in list(book):
            pos = book[sym]
            price = px.get(sym, np.nan)
            if not np.isfinite(price):
                continue
            if x.get("trail_stop"):
                pos["stop"] = risk.trail_stop(pos["stop"], price, a.get(sym, np.nan), x["stop_atr_multiple"])
            pos["days"] += 1
            rank = int(ranking.at[sym, "rank"]) if ranking is not None and sym in ranking.index else None
            why = risk.exit_reason(
                close=price, stop=pos["stop"], days_held=pos["days"], entry_price=pos["entry"],
                rank=rank, keep_rank=keep_rank, is_rebalance_day=rebalance, time_stop_days=x["time_stop_days"],
            )
            if why:
                fill = price * (1 - slip)
                cash += pos["qty"] * fill
                trades.append({"date": d, "symbol": sym, "side": "sell", "ret": fill / pos["entry"] - 1, "reason": why.split(":")[0].split(" (")[0]})
                del book[sym]

        equity = cash + sum(pos["qty"] * px.get(s, pos["entry"]) for s, pos in book.items())

        # entries
        if rebalance:
            last_month = d.month
            sector_val: dict[str, float] = {}
            for s, pos in book.items():
                sector_val[pos["sector"]] = sector_val.get(pos["sector"], 0) + pos["qty"] * px[s]
            avail = cash - p["min_cash_pct"] * equity
            slots = p["max_positions"] - len(book)
            for sym, row in ranking[ranking["trend_ok"]].head(p["max_positions"]).iterrows():
                if slots <= 0:
                    break
                if sym in book:
                    continue
                sector = sector_of.get(sym, "Unknown")
                size = risk.size_position(
                    equity=equity, price=float(px[sym]), atr_value=float(a[sym]),
                    adv_usd=float(adv[sym]), sector_room=p["max_sector_pct"] * equity - sector_val.get(sector, 0),
                    cash_available=avail, cfg=cfg,
                )
                if size.shares < 1:
                    continue
                fill = float(px[sym]) * (1 + slip)
                cash -= size.shares * fill
                avail -= size.shares * fill
                sector_val[sector] = sector_val.get(sector, 0) + size.shares * fill
                book[sym] = {"qty": size.shares, "entry": fill, "stop": size.stop, "days": 0, "sector": sector}
                trades.append({"date": d, "symbol": sym, "side": "buy", "ret": np.nan, "reason": "entry"})
                slots -= 1

        equity = cash + sum(pos["qty"] * px.get(s, pos["entry"]) for s, pos in book.items())
        curve.append((d, equity, len(book)))

    eq = pd.DataFrame(curve, columns=["date", "equity", "positions"]).set_index("date")["equity"]
    bench = benchmark.reindex(eq.index).ffill()
    bench = bench / bench.iloc[0] * eq.iloc[0]
    tdf = pd.DataFrame(trades)
    closed = tdf[tdf["side"] == "sell"] if len(tdf) else tdf
    stats = performance_stats(eq)
    stats.update(
        {
            "benchmark": performance_stats(bench),
            "trades": int(len(closed)),
            "win_rate": float((closed["ret"] > 0).mean()) if len(closed) else 0.0,
            "avg_win": float(closed.loc[closed["ret"] > 0, "ret"].mean()) if len(closed) else 0.0,
            "avg_loss": float(closed.loc[closed["ret"] <= 0, "ret"].mean()) if len(closed) else 0.0,
            "exit_reasons": closed["reason"].value_counts().to_dict() if len(closed) else {},
            "start": str(eq.index[0].date()),
            "end": str(eq.index[-1].date()),
        }
    )
    return {"equity": eq, "benchmark": bench, "stats": stats, "trades": tdf}
