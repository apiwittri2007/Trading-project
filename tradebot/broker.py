"""Brokers: Alpaca paper trading, or a local simulator that needs no account.

Both expose the same small interface so the strategy code does not care
which one it is talking to.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from .config import STATE_DIR

log = logging.getLogger(__name__)


@dataclass
class Fill:
    symbol: str
    side: str          # "buy" or "sell"
    qty: int
    price: float       # fill price (simulator) or reference price (Alpaca, filled next open)
    order_id: str = ""


def to_alpaca_symbol(symbol: str) -> str:
    return symbol.replace("-", ".")  # BRK-B (Yahoo) -> BRK.B (Alpaca)


def from_alpaca_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


class SimBroker:
    """Fills at the latest close plus slippage. Account lives in state/sim_account.json."""

    name = "simulator"

    def __init__(self, starting_equity: float, slippage_bps: float, path=None):
        self.path = path or STATE_DIR / "sim_account.json"
        self.slippage = slippage_bps / 10_000
        if self.path.exists():
            self.acct = json.loads(self.path.read_text())
        else:
            self.acct = {"cash": float(starting_equity), "positions": {}}
        self.marks: dict[str, float] = {}

    def mark(self, prices: dict[str, float]) -> None:
        self.marks.update({k: float(v) for k, v in prices.items()})

    def account(self) -> dict:
        value = sum(p["qty"] * self.marks.get(s, p["avg_price"]) for s, p in self.acct["positions"].items())
        return {"equity": self.acct["cash"] + value, "cash": self.acct["cash"]}

    def positions(self) -> dict[str, dict]:
        return {s: dict(p) for s, p in self.acct["positions"].items()}

    def submit(self, symbol: str, qty: int, side: str, ref_price: float) -> Fill:
        px = ref_price * (1 + self.slippage if side == "buy" else 1 - self.slippage)
        pos = self.acct["positions"].get(symbol, {"qty": 0, "avg_price": 0.0})
        if side == "buy":
            new_qty = pos["qty"] + qty
            pos["avg_price"] = (pos["qty"] * pos["avg_price"] + qty * px) / new_qty
            pos["qty"] = new_qty
            self.acct["cash"] -= qty * px
            self.acct["positions"][symbol] = pos
        else:
            qty = min(qty, pos["qty"])
            pos["qty"] -= qty
            self.acct["cash"] += qty * px
            if pos["qty"] == 0:
                self.acct["positions"].pop(symbol, None)
            else:
                self.acct["positions"][symbol] = pos
        return Fill(symbol, side, qty, round(px, 4), order_id="sim")

    def save(self) -> None:
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.acct, indent=2))


class AlpacaBroker:
    """Alpaca paper account. Keys come from env vars ALPACA_API_KEY / ALPACA_SECRET_KEY."""

    name = "alpaca-paper"

    def __init__(self):
        from alpaca.trading.client import TradingClient

        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set for paper mode")
        # paper=True hard-codes the paper endpoint: this bot can never touch real money.
        self.client = TradingClient(key, secret, paper=True)

    def mark(self, prices: dict[str, float]) -> None:  # Alpaca marks its own positions
        pass

    def account(self) -> dict:
        a = self.client.get_account()
        return {"equity": float(a.equity), "cash": float(a.cash)}

    def positions(self) -> dict[str, dict]:
        out = {}
        for p in self.client.get_all_positions():
            out[from_alpaca_symbol(p.symbol)] = {"qty": int(float(p.qty)), "avg_price": float(p.avg_entry_price)}
        return out

    def submit(self, symbol: str, qty: int, side: str, ref_price: float) -> Fill:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order = self.client.submit_order(
            MarketOrderRequest(
                symbol=to_alpaca_symbol(symbol),
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,  # after-hours submissions queue for the next open
            )
        )
        return Fill(symbol, side, qty, ref_price, order_id=str(order.id))

    def save(self) -> None:
        pass


def make_broker(cfg: dict):
    if cfg.get("mode") == "paper":
        if os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"):
            return AlpacaBroker()
        log.warning("mode=paper but Alpaca keys are not set; falling back to the simulator")
    return SimBroker(cfg["starting_equity"], cfg["execution"]["slippage_bps"])
