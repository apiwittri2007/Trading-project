"""Step 2 of the framework: turn prices and fundamentals into a ranked list.

All functions are pure (DataFrame in, DataFrame/Series out) so they are easy
to test and are shared by the live bot and the backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Price-based building blocks
# --------------------------------------------------------------------------
def momentum_12_1(close: pd.DataFrame) -> pd.Series:
    """12-month return skipping the most recent month (classic 12-1 momentum).

    Skipping the last month avoids the short-term reversal effect.
    `close` is a date x symbol frame; returns the latest value per symbol.
    """
    if len(close) < 253:
        return pd.Series(np.nan, index=close.columns)
    return close.iloc[-22] / close.iloc[-253] - 1.0


def above_trend(close: pd.DataFrame, window: int = 200) -> pd.Series:
    """True where the latest close is above its `window`-day moving average."""
    ma = close.rolling(window).mean().iloc[-1]
    return close.iloc[-1] > ma


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int = 20) -> pd.Series:
    """Average True Range (latest value per symbol)."""
    prev_close = close.shift(1)
    hl = high - low
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    tr = np.fmax(np.fmax(hl, hc), lc)  # true range, ignoring the missing first prev close
    return tr.rolling(period).mean().iloc[-1]


def avg_dollar_volume(close: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.Series:
    """20-day average daily traded value in dollars."""
    return (close * volume).rolling(window).mean().iloc[-1]


# --------------------------------------------------------------------------
# Cross-sectional scoring
# --------------------------------------------------------------------------
def zscore(s: pd.Series, clip: float = 3.0) -> pd.Series:
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / std).clip(-clip, clip)


def composite_score(factors: pd.DataFrame, weights: dict, clip: float = 3.0) -> pd.Series:
    """Weighted sum of z-scored factors.

    `factors` has one column per factor name. A missing factor value counts as
    neutral (z = 0) rather than excluding the stock, and weights are
    renormalised over the factors that are present in the frame.
    """
    used = {k: w for k, w in weights.items() if k in factors.columns and w}
    if not used:
        raise ValueError("no factor columns match the configured weights")
    total = sum(used.values())
    score = pd.Series(0.0, index=factors.index)
    for name, w in used.items():
        score += (w / total) * zscore(factors[name], clip).fillna(0.0)
    return score


def quality_score(roe: pd.Series, gp_assets: pd.Series) -> pd.Series:
    """Average of the z-scores of ROE and gross profit / assets."""
    parts = pd.concat([zscore(roe), zscore(gp_assets)], axis=1)
    return parts.mean(axis=1, skipna=True)


def rank_universe(
    close: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    cfg: dict,
) -> pd.DataFrame:
    """Build the full scorecard for the latest date.

    Returns a frame indexed by symbol with each factor, the composite score,
    the rank (1 = best) and whether the stock passes the trend filter.
    """
    fcfg = cfg["factors"]
    table = pd.DataFrame(index=close.columns)
    table["momentum"] = momentum_12_1(close)

    if fundamentals is not None and not fundamentals.empty:
        f = fundamentals.reindex(table.index)
        table["value"] = f.get("earnings_yield")
        table["quality"] = quality_score(
            f.get("roe", pd.Series(np.nan, index=f.index)),
            f.get("gp_assets", pd.Series(np.nan, index=f.index)),
        )

    weights = {k: fcfg.get(k, 0) for k in ("momentum", "value", "quality")}
    table = table[table["momentum"].notna()]
    table["score"] = composite_score(table, weights, fcfg.get("winsorize_z", 3.0))
    table["trend_ok"] = above_trend(close[table.index]) if fcfg.get("trend_filter", True) else True
    table = table.sort_values("score", ascending=False)
    table["rank"] = np.arange(1, len(table) + 1)
    return table
