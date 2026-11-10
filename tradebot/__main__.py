"""Command line entry point.

    python -m tradebot run        # daily: screen, manage exits, rebalance, record
    python -m tradebot backtest   # historical test of the same rules
    python -m tradebot report     # rebuild docs/data.json for the dashboard
"""
from __future__ import annotations

import argparse
import logging

from .config import load_config


def cmd_run(cfg: dict) -> None:
    from .broker import make_broker
    from .data import download_prices, load_fundamentals, load_universe
    from .state import State
    from .strategy import run_day

    universe = load_universe(cfg["universe"]["source"])
    symbols = sorted(set(universe["symbol"]) | {cfg["benchmark"]})
    prices = download_prices(symbols, years=2)
    bench = prices["close"][cfg["benchmark"]].dropna()
    stock_prices = {k: v.drop(columns=[cfg["benchmark"]], errors="ignore") for k, v in prices.items()}
    fundamentals = load_fundamentals([s for s in universe["symbol"] if s in stock_prices["close"].columns])
    summary = run_day(
        cfg=cfg,
        prices=stock_prices,
        universe=universe,
        fundamentals=fundamentals,
        benchmark_close=float(bench.iloc[-1]),
        broker=make_broker(cfg),
        state=State(),
    )
    logging.info("run complete: %s", summary)


def cmd_backtest(cfg: dict) -> None:
    from .backtest import run_backtest
    from .data import download_prices, load_universe
    from .report import save_backtest

    universe = load_universe(cfg["universe"]["source"])
    symbols = sorted(set(universe["symbol"]) | {cfg["benchmark"]})
    start = str(int(cfg["backtest"]["start"][:4]) - 2) + cfg["backtest"]["start"][4:]
    prices = download_prices(symbols, start=start)
    bench = prices["close"][cfg["benchmark"]]
    stock_prices = {k: v.drop(columns=[cfg["benchmark"]], errors="ignore") for k, v in prices.items()}
    result = run_backtest(stock_prices, bench, universe, cfg)
    save_backtest(result)
    s = result["stats"]
    logging.info(
        "backtest %s..%s CAGR %.1f%% (SPY %.1f%%), max DD %.1f%% (SPY %.1f%%), Sharpe %.2f, %d trades",
        s["start"], s["end"], s["cagr"] * 100, s["benchmark"]["cagr"] * 100,
        s["max_drawdown"] * 100, s["benchmark"]["max_drawdown"] * 100, s["sharpe"], s["trades"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="tradebot")
    ap.add_argument("command", choices=["run", "backtest", "report"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)

    if args.command == "run":
        cmd_run(cfg)
    elif args.command == "backtest":
        cmd_backtest(cfg)

    from .report import build_report

    build_report(cfg)


if __name__ == "__main__":
    main()
