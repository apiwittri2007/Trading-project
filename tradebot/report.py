"""Build docs/data.json, which the GitHub Pages dashboard (docs/index.html) reads."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .backtest import performance_stats
from .config import DOCS_DIR, STATE_DIR
from .state import State


def _clean(obj):
    """Make numpy / NaN values JSON-safe."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), 6)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def build_report(cfg: dict) -> dict:
    state = State()
    eq = state.equity()
    out: dict = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "broker": state.meta.get("broker", "not run yet"),
        "last_run": state.meta.get("last_run"),
        "last_rebalance": state.meta.get("last_rebalance"),
        "benchmark_symbol": cfg["benchmark"],
        "config": {
            "max_positions": cfg["portfolio"]["max_positions"],
            "risk_per_trade": cfg["portfolio"]["risk_per_trade"],
            "max_position_pct": cfg["portfolio"]["max_position_pct"],
            "max_sector_pct": cfg["portfolio"]["max_sector_pct"],
            "stop_atr_multiple": cfg["exits"]["stop_atr_multiple"],
            "time_stop_days": cfg["exits"]["time_stop_days"],
            "weights": {k: cfg["factors"][k] for k in ("momentum", "value", "quality")},
        },
    }

    # live equity curve vs benchmark (benchmark rescaled to the starting equity)
    if len(eq):
        eq = eq.sort_values("date")
        bench = pd.to_numeric(eq["benchmark"], errors="coerce")
        first_b = bench.dropna().iloc[0] if bench.notna().any() else np.nan
        eq["bench_norm"] = bench / first_b * eq["equity"].iloc[0]
        out["equity"] = [
            {"date": r.date, "equity": r.equity, "benchmark": r.bench_norm, "cash": r.cash}
            for r in eq.itertuples()
        ]
        series = pd.Series(eq["equity"].values, index=pd.to_datetime(eq["date"]))
        out["live_stats"] = performance_stats(series) if len(series) > 1 else {}
        out["live_stats"]["equity"] = float(eq["equity"].iloc[-1])
        out["live_stats"]["cash"] = float(eq["cash"].iloc[-1])
        if eq["bench_norm"].notna().sum() > 1:
            b = pd.Series(eq["bench_norm"].values, index=pd.to_datetime(eq["date"])).dropna()
            out["live_stats"]["benchmark_return"] = float(b.iloc[-1] / b.iloc[0] - 1)

    # open positions
    positions = []
    for sym, p in sorted(state.positions.items()):
        last = p.get("last_price", p["entry_price"])
        positions.append(
            {
                "symbol": sym, "qty": p.get("qty"), "entry_date": p["entry_date"], "entry_price": p["entry_price"],
                "last_price": last, "pnl_pct": last / p["entry_price"] - 1 if p["entry_price"] else None,
                "value": (p.get("qty") or 0) * last, "stop": p["stop"], "sector": p.get("sector"),
                "rank": p.get("rank"), "days_held": p.get("days_held", 0),
                "reason": p.get("reason", ""), "kill_criteria": p.get("kill_criteria", ""),
            }
        )
    out["positions"] = positions

    trades = state.trades()
    out["trades"] = trades.iloc[::-1].head(300).to_dict(orient="records") if len(trades) else []

    rank_path = STATE_DIR / "ranking.csv"
    if rank_path.exists():
        r = pd.read_csv(rank_path).head(30)
        out["ranking"] = r.to_dict(orient="records")

    bt_path = STATE_DIR / "backtest.json"
    if bt_path.exists():
        out["backtest"] = json.loads(bt_path.read_text())

    out = _clean(out)
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "data.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def save_backtest(result: dict) -> None:
    """Store a weekly-sampled backtest curve plus stats for the dashboard."""
    eq = result["equity"].resample("W").last()
    bench = result["benchmark"].resample("W").last()
    payload = {
        "stats": result["stats"],
        "curve": [
            {"date": d.strftime("%Y-%m-%d"), "equity": float(e), "benchmark": float(b) if np.isfinite(b) else None}
            for d, e, b in zip(eq.index, eq.values, bench.reindex(eq.index).values)
        ],
    }
    STATE_DIR.mkdir(exist_ok=True)
    (STATE_DIR / "backtest.json").write_text(json.dumps(_clean(payload), indent=1))
