import numpy as np
import pandas as pd
import pytest

from tradebot.config import load_config


@pytest.fixture
def cfg():
    return load_config()


def make_prices(n_days=600, n_syms=40, seed=0, start="2022-01-03"):
    """Random-walk OHLCV where each stock has its own drift, so momentum is real."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    syms = [f"S{i:02d}" for i in range(n_syms)]
    drift = np.linspace(-0.0006, 0.0012, n_syms)
    rets = rng.normal(drift, 0.015, size=(n_days, n_syms))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=syms)
    spread = close * 0.01
    high, low = close + spread, close - spread
    volume = pd.DataFrame(rng.integers(1_000_000, 3_000_000, size=(n_days, n_syms)), index=idx, columns=syms)
    open_ = close.shift(1).fillna(close)
    universe = pd.DataFrame({"symbol": syms, "sector": [f"Sector{i % 6}" for i in range(n_syms)]})
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, universe


@pytest.fixture
def synthetic():
    return make_prices()
