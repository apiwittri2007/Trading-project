import numpy as np
import pandas as pd
import pytest

from tradebot import risk, signals
from tradebot.backtest import run_backtest
from tradebot.broker import SimBroker
from tradebot.state import State
from tradebot.strategy import run_day


# ---- signals --------------------------------------------------------------
def test_momentum_skips_last_month():
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.DataFrame({"A": np.linspace(100, 200, 300)}, index=idx)
    expected = close["A"].iloc[-22] / close["A"].iloc[-253] - 1
    assert signals.momentum_12_1(close)["A"] == pytest.approx(expected)


def test_composite_treats_missing_as_neutral():
    f = pd.DataFrame({"momentum": [1.0, 2.0, 3.0], "value": [np.nan, 0.1, 0.0]}, index=list("ABC"))
    s = signals.composite_score(f, {"momentum": 0.5, "value": 0.5})
    assert s.notna().all()
    assert s["C"] > s["A"]


def test_rank_universe_orders_by_drift(synthetic, cfg):
    prices, _ = synthetic
    r = signals.rank_universe(prices["close"], None, cfg)
    # higher-numbered stocks have higher drift, so top-ranked names should skew high
    drift_rank = lambda names: np.mean([int(s[1:]) for s in names])
    assert drift_rank(r.head(10).index) > drift_rank(r.tail(10).index) + 10
    assert list(r["rank"]) == list(range(1, len(r) + 1))


# ---- risk -----------------------------------------------------------------
def test_size_uses_risk_budget(cfg):
    d = risk.size_position(equity=100_000, price=50, atr_value=1.6, adv_usd=1e9,
                           sector_room=1e9, cash_available=1e9, cfg=cfg)
    # risk $500 / stop distance (2.5 x 1.6 = 4.0) = 125 shares
    assert d.shares == 125
    assert d.stop == pytest.approx(46.0)
    assert d.binding_limit == "risk budget"


def test_size_capped_per_name(cfg):
    d = risk.size_position(equity=100_000, price=50, atr_value=0.2, adv_usd=1e9,
                           sector_room=1e9, cash_available=1e9, cfg=cfg)
    assert d.value <= 10_000 and d.binding_limit == "max per name"


def test_size_capped_by_liquidity_and_sector(cfg):
    d = risk.size_position(equity=100_000, price=50, atr_value=1, adv_usd=200_000,
                           sector_room=1e9, cash_available=1e9, cfg=cfg)
    assert d.binding_limit == "liquidity" and d.value <= 2_000
    d = risk.size_position(equity=100_000, price=50, atr_value=1, adv_usd=1e9,
                           sector_room=1_000, cash_available=1e9, cfg=cfg)
    assert d.binding_limit == "sector cap" and d.shares == 20


def test_stop_only_moves_up():
    assert risk.trail_stop(90, close=100, atr_value=2, multiple=2.5) == 95
    assert risk.trail_stop(95, close=96, atr_value=2, multiple=2.5) == 95


def test_exit_rules_priority():
    kw = dict(days_held=5, entry_price=100, rank=3, keep_rank=30, is_rebalance_day=False, time_stop_days=40)
    assert risk.exit_reason(close=89, stop=90, **kw).startswith("price stop")
    assert risk.exit_reason(close=95, stop=90, **{**kw, "days_held": 40}).startswith("time stop")
    assert risk.exit_reason(close=105, stop=90, **{**kw, "days_held": 40}) is None
    assert risk.exit_reason(close=105, stop=90, **{**kw, "rank": 31, "is_rebalance_day": True}).startswith("rebalance")
    assert risk.exit_reason(close=105, stop=90, **{**kw, "rank": 31}) is None


# ---- end to end -------------------------------------------------------------
def test_daily_run_respects_limits(synthetic, cfg, tmp_path):
    prices, universe = synthetic
    cfg = {**cfg, "universe": {**cfg["universe"], "min_adv_usd": 1_000_000}}
    broker = SimBroker(100_000, 5, path=tmp_path / "acct.json")
    state = State(tmp_path)
    run_day(cfg=cfg, prices=prices, universe=universe, fundamentals=None,
            benchmark_close=400.0, broker=broker, state=state)

    assert 0 < len(state.positions) <= cfg["portfolio"]["max_positions"]
    acct = broker.account()
    assert acct["cash"] >= 0
    sector_val = {}
    last = prices["close"].iloc[-1]
    for s, p in state.positions.items():
        val = p["qty"] * last[s]
        assert val <= 0.10 * acct["equity"] * 1.01
        assert p["stop"] < p["entry_price"]
        assert p["kill_criteria"] and p["reason"]
        sector_val[p["sector"]] = sector_val.get(p["sector"], 0) + val
    assert max(sector_val.values()) <= 0.25 * acct["equity"] * 1.01
    assert (tmp_path / "trades.csv").exists() and (tmp_path / "equity.csv").exists()

    # running again on the same day must not buy twice
    n_trades = len(state.trades())
    run_day(cfg=cfg, prices=prices, universe=universe, fundamentals=None,
            benchmark_close=400.0, broker=broker, state=State(tmp_path))
    assert len(State(tmp_path).trades()) == n_trades


def test_backtest_runs(synthetic, cfg):
    prices, universe = synthetic
    cfg = {**cfg, "universe": {**cfg["universe"], "min_adv_usd": 1_000_000},
           "backtest": {"start": "2022-01-01", "universe_size": 40}}
    bench = prices["close"].mean(axis=1)
    res = run_backtest(prices, bench, universe, cfg)
    s = res["stats"]
    assert s["trades"] > 0 and np.isfinite(s["cagr"])
    assert -1 < s["max_drawdown"] <= 0
    assert res["equity"].iloc[0] == pytest.approx(cfg["starting_equity"], rel=0.01)
