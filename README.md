# Trading Project — Multi-Factor Screener & Paper-Trading Bot

A small, fully automated version of the stock-selection process professional
equity desks use: **filter a liquid universe → rank it with evidence-based
factors → size every position by risk → exit on pre-committed rules → review
everything in public.**

It runs once per trading day on GitHub Actions, trades an **Alpaca paper
account** (simulated money, real prices), and publishes a dashboard on
GitHub Pages with the equity curve vs. SPY, drawdown, open positions and a
trade log that records *why* each trade was made and *what would prove it wrong*.

> Educational project. Paper trading only — the code hard-codes Alpaca's paper
> endpoint. Nothing here is investment advice.

---

## The framework, as code

| Step | Rule | Where |
|---|---|---|
| **1. Universe & liquidity** | S&P 500 members, price ≥ $10, 20-day average traded value ≥ $50M. A position can never exceed 1% of a stock's daily traded value. | `strategy.liquidity_filter`, `risk.size_position` |
| **2. Factor scores** | Cross-sectional z-scores, clipped at ±3, combined into one rank: **momentum** 12-1 (40%), **value** = earnings yield (30%), **quality** = avg of ROE and gross profit / assets (30%). Missing data counts as neutral. | `signals.py` |
| **3. Entry timing** | Buy only stocks above their 200-day moving average — don't fight the tape. | `signals.above_trend` |
| **4. Risk-based sizing** | `shares = equity × 0.5% ÷ (2.5 × ATR20)` — every trade risks ~0.5% of equity if its stop is hit. Then capped by: 10% per name, 25% per sector, liquidity, available cash. The dashboard shows which limit was binding. | `risk.size_position` |
| **5. Exits** | Trailing 2.5×ATR price stop (only moves up) · time stop if still losing after 40 trading days · at the monthly rebalance, sell anything ranked below #30 (2× the 15 slots, a buffer that cuts turnover). | `risk.exit_reason` |
| **6. Record & review** | Every order is logged with its reason and kill criteria; daily equity vs. SPY; full scorecard saved. | `state.py`, `report.py` |

All parameters live in [`config.yaml`](config.yaml).

### Why these factors

- **Momentum** — Jegadeesh & Titman (1993): past 3–12-month winners keep outperforming. Skipping the last month avoids the short-term reversal effect.
- **Value** — Fama & French (1992): cheap stocks (high earnings yield) have historically outperformed expensive ones.
- **Quality** — Novy-Marx (2013): gross profitability predicts returns about as well as value.
- **Combining them** — Asness, Moskowitz & Pedersen (2013): value and momentum are negatively correlated, so the blend is steadier than either alone.
- **Caveat** — McLean & Pontiff (2016): published anomalies earn roughly half as much after publication. Expect weaker results than the papers.

---

## Repository layout

```
config.yaml              every rule and parameter
tradebot/
  data.py                universe (S&P 500), prices & fundamentals via yfinance
  signals.py             momentum, trend, ATR, z-scores, composite ranking
  risk.py                position sizing, caps, trailing stops, exit rules, kill criteria
  strategy.py            the daily run — the whole framework in order
  broker.py              Alpaca paper broker + offline simulator (same interface)
  backtest.py            historical test of the same rules
  report.py              builds docs/data.json for the dashboard
state/                   positions, trades, equity history (committed by the bot)
docs/index.html          GitHub Pages dashboard
.github/workflows/       daily run, backtest, tests
tests/                   unit tests + end-to-end run on synthetic data
```

---

## Setup (about 10 minutes)

**1. Get Alpaca paper keys** — sign up free at [alpaca.markets](https://alpaca.markets) (email only, available outside the US), switch to the *Paper* account, and generate an API key and secret.

**2. Add them as repository secrets** — *Settings → Secrets and variables → Actions → New repository secret*:
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`

Without keys the bot still works: it falls back to a built-in simulator (fills at the close + 5 bp slippage), so you can try it with no account at all. You can also set `mode: sim` in `config.yaml`.

**3. Turn on GitHub Pages** — *Settings → Pages → Build and deployment → Deploy from a branch → `main` / `/docs`*. The dashboard appears at `https://<your-username>.github.io/<repo-name>/`.

**4. Run it** — *Actions → Daily trading run → Run workflow* for the first run, and *Backtest → Run workflow* to fill in the backtest panel. After that the daily job runs automatically at 21:37 UTC Monday–Friday, after the US close; orders queue for the next open.

### Run locally

```bash
pip install -r requirements.txt
export ALPACA_API_KEY=...  ALPACA_SECRET_KEY=...   # optional; omit to use the simulator
python -m tradebot run        # daily run
python -m tradebot backtest   # historical test
python -m tradebot report     # rebuild docs/data.json
python -m pytest -q           # tests
```

---

## Honest limitations

- **Backtest is momentum-only.** Free sources only give *today's* fundamentals; using them to pick stocks in 2015 would leak future information. So the backtest ranks on momentum and keeps every other rule (trend filter, sizing, caps, stops, rebalancing, slippage) identical. The live bot uses all three factors.
- **Survivorship bias.** The backtest universe is today's S&P 500, which excludes companies that were dropped or went bankrupt. Treat results as an upper bound.
- **Paper ≠ real.** Paper fills ignore market impact and queue position, and Alpaca's simulator does not check order size against available liquidity. The liquidity cap in this bot partly compensates.
- **Daily stops.** Stops are checked on the daily close, not intraday, so a gap down can exit below the stop.
- **Free data.** yfinance is unofficial and occasionally fails; the run then exits without trading and tries again the next day.

## Ideas to extend it

- Sector-neutral ranking (z-scores within each sector)
- Earnings-revision or post-earnings-drift factor
- Volatility-targeting the whole portfolio
- A Korean-market version using KIS 모의투자 and `pykrx` foreign/institutional flow data
- Point-in-time fundamentals for a full three-factor backtest

## License

MIT
