"""Files that persist between daily runs (committed back to the repo by GitHub Actions).

state/positions.json   open positions with stop, thesis and kill criteria
state/trades.csv       every order with its reason
state/equity.csv       daily equity vs. benchmark
state/ranking.csv      latest full scorecard
state/meta.json        last run / last rebalance dates
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import STATE_DIR

TRADE_COLS = ["date", "symbol", "side", "qty", "price", "value", "reason", "kill_criteria", "broker", "order_id"]
EQUITY_COLS = ["date", "equity", "cash", "positions", "benchmark"]


class State:
    def __init__(self, root: Path = STATE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.positions: dict[str, dict] = self._json("positions.json", {})
        self.meta: dict = self._json("meta.json", {})
        self.new_trades: list[dict] = []

    def _json(self, name: str, default):
        p = self.root / name
        return json.loads(p.read_text()) if p.exists() else default

    def record_trade(self, **row) -> None:
        self.new_trades.append({c: row.get(c, "") for c in TRADE_COLS})

    def trades(self) -> pd.DataFrame:
        p = self.root / "trades.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=TRADE_COLS)

    def equity(self) -> pd.DataFrame:
        p = self.root / "equity.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=EQUITY_COLS)

    def save(self, equity_row: dict | None = None, ranking: pd.DataFrame | None = None) -> None:
        (self.root / "positions.json").write_text(json.dumps(self.positions, indent=2, sort_keys=True))
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=2, sort_keys=True))
        if self.new_trades:
            trades = pd.concat([self.trades(), pd.DataFrame(self.new_trades)], ignore_index=True)
            trades.to_csv(self.root / "trades.csv", index=False)
            self.new_trades = []
        if equity_row:
            eq = self.equity()
            eq = eq[eq["date"].astype(str) != str(equity_row["date"])]  # one row per day, re-runs overwrite
            eq = pd.concat([eq, pd.DataFrame([equity_row])], ignore_index=True)
            eq.to_csv(self.root / "equity.csv", index=False)
        if ranking is not None:
            ranking.to_csv(self.root / "ranking.csv", index_label="symbol", float_format="%.4f")
