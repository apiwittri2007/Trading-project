"""Market data and fundamentals (free sources, no API key needed).

Prices and fundamentals come from Yahoo Finance via `yfinance`.
Alpaca is used only for placing paper orders.
"""
from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import DATA_DIR, STATE_DIR

log = logging.getLogger(__name__)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FIELDS = ("open", "high", "low", "close", "volume")


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
def load_universe(source: str = "sp500") -> pd.DataFrame:
    """Return a frame with columns symbol, sector (Yahoo-style tickers, e.g. BRK-B)."""
    if source == "sp500":
        try:
            import requests

            html = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
            table = pd.read_html(io.StringIO(html))[0]
            uni = pd.DataFrame(
                {
                    "symbol": table["Symbol"].str.replace(".", "-", regex=False),
                    "sector": table["GICS Sector"],
                }
            )
            log.info("Loaded %d S&P 500 constituents from Wikipedia", len(uni))
            return uni
        except Exception as exc:  # network / layout change -> use the bundled list
            log.warning("S&P 500 download failed (%s); using data/universe.csv", exc)
    return pd.read_csv(DATA_DIR / "universe.csv")


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
def download_prices(symbols: list[str], start: str | None = None, years: int = 2) -> dict[str, pd.DataFrame]:
    """Daily split/dividend-adjusted OHLCV as a dict of date x symbol frames."""
    import yfinance as yf

    if start is None:
        start = (datetime.now(timezone.utc) - timedelta(days=365 * years + 30)).strftime("%Y-%m-%d")
    raw = yf.download(
        symbols,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("no price data returned")
    out = {}
    for field in FIELDS:
        frame = raw[field.capitalize()]
        if isinstance(frame, pd.Series):  # single symbol
            frame = frame.to_frame(symbols[0])
        out[field] = frame.sort_index()
    # drop symbols with no data at all
    good = out["close"].columns[out["close"].notna().any()]
    return {k: v[good] for k, v in out.items()}


# --------------------------------------------------------------------------
# Fundamentals (cached for a week — they change quarterly)
# --------------------------------------------------------------------------
FUND_CACHE = STATE_DIR / "fundamentals.csv"


def _fetch_one(symbol: str) -> dict:
    import yfinance as yf

    row = {"symbol": symbol, "earnings_yield": np.nan, "roe": np.nan, "gp_assets": np.nan}
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        pe = info.get("trailingPE")
        eps, price = info.get("trailingEps"), info.get("currentPrice") or info.get("regularMarketPrice")
        if pe and pe > 0:
            row["earnings_yield"] = 1.0 / pe
        elif eps is not None and price:
            row["earnings_yield"] = eps / price  # negative earnings -> negative yield
        row["roe"] = info.get("returnOnEquity", np.nan)
        gross_profit = info.get("grossProfits")
        bs = t.balance_sheet
        if gross_profit and bs is not None and not bs.empty and "Total Assets" in bs.index:
            assets = bs.loc["Total Assets"].dropna()
            if len(assets):
                row["gp_assets"] = gross_profit / float(assets.iloc[0])
    except Exception as exc:
        log.debug("fundamentals failed for %s: %s", symbol, exc)
    return row


def load_fundamentals(symbols: list[str], max_age_days: int = 7, workers: int = 8) -> pd.DataFrame:
    """Earnings yield, ROE and gross-profit/assets per symbol."""
    if FUND_CACHE.exists():
        cached = pd.read_csv(FUND_CACHE, parse_dates=["asof"])
        fresh = cached["asof"].max() > pd.Timestamp.now() - pd.Timedelta(days=max_age_days)
        if fresh and set(symbols) <= set(cached["symbol"]):
            return cached.set_index("symbol")
    log.info("Refreshing fundamentals for %d symbols", len(symbols))
    with ThreadPoolExecutor(workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(_fetch_one, symbols)):
            rows.append(row)
            if i and i % 100 == 0:
                time.sleep(1)  # be gentle with the free endpoint
    df = pd.DataFrame(rows)
    df["asof"] = pd.Timestamp.now().normalize()
    STATE_DIR.mkdir(exist_ok=True)
    df.to_csv(FUND_CACHE, index=False)
    return df.set_index("symbol")
