# TSLA NYSE / TESLA_USDT MEXC — Quant Arbitrage

A statistical arbitrage backtester for the spread between TSLA equity (NYSE) and TESLA_USDT perpetual futures (MEXC). Implements a log-spread mean-reversion strategy on 1-minute bars, with a full interactive Streamlit app and a CLI runner.

---

## Quick Start

### Option A — Docker (no Python setup required)

```bash
cd quant_arbitrage/

# Build and launch the Streamlit app
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501).
Data and reports are saved to your local `data/` and `reports/` folders automatically.

```bash
# CLI backtest (live data fetch)
docker compose run --rm cli python main.py

# CLI with a custom date range
docker compose run --rm cli python main.py --start 2026-02-12 --end 2026-03-10

# CLI offline (uses saved Excel files, no internet needed)
docker compose run --rm cli python main_static.py
```

### Option B — Local Python

```bash
cd quant_arbitrage/

# Install dependencies
pip install -r requirements.txt

# Run backtest with live data fetch
python main.py

# Run from saved Excel cache (no internet required)
python main_static.py

# Custom date range
python main.py --start 2026-02-12 --end 2026-03-10

# Print metrics only — no report files generated
python main.py --no-report

# Launch the interactive Streamlit app
streamlit run app/Home.py
```

---

## What It Does

Trades the price spread between TSLA (NYSE) and TESLA_USDT (MEXC futures) using
z-score mean-reversion during NYSE market hours.

| Parameter | Value |
|---|---|
| Spread | `log(mexc_close / tsla_close)` |
| Z-score | `spread.shift(1)` → `rolling(90).mean/std` — strictly lookahead-free |
| Entry | `\|z\| ≥ 2.0σ` → fill at **next bar's OPEN** |
| Exit (normal) | z crosses 0, or 20-bar max hold → fill at **next bar's OPEN** |
| Exit (forced) | Session end or end-of-data → fill at **current bar's CLOSE** |
| Fees | MEXC: 3 bps/side · NYSE: $0.01/share/side |

### Why It Works

- **Stationary spread**: ADF p < 0.001 — the spread has a fixed long-run mean
- **Fast reversion**: OU half-life ~1.9 min — spreads close within minutes
- **High correlation**: 0.92 between TSLA and TESLA_USDT

> **Important**: Profitability depends on low MEXC fees (≤ 3 bps/side). At 4+ bps the strategy turns unprofitable on the reference sample.

---

## Verified Results

Reference period: **2026-02-12 → 2026-03-10** with default parameters.

| Metric | Value |
|---|---|
| Trades | 269 |
| Net PnL | $53.09 |
| Win Rate | 74.0% |
| Sharpe Ratio | 13.932 |
| Max Drawdown | −$1.42 |
| Avg Hold Time | 4.2 min |

---

## Streamlit App

Run `streamlit run app/Home.py` (or use Docker above) to open the interactive UI.

| Page | Description |
|---|---|
| **Home** | Strategy overview, recent reports |
| **EDA** | Correlation, spread stationarity, ADF/Hurst tests, OU half-life, intraday patterns |
| **Backtest** | Full configurable simulator: KPI metrics, 5 chart tabs, trade log, downloadable PDF/PNG/CSV reports |

Both pages support **live data fetch** (API) or **load from saved Excel** (offline).

---

## CLI Options

```
python main.py [OPTIONS]

  --start YYYY-MM-DD       Backtest start (default: 10 days ago)
  --end   YYYY-MM-DD       Backtest end (default: today)
  --refresh                Force re-fetch all data (ignore Excel cache)
  --entry-threshold FLOAT  Z-score entry threshold (default: 2.0)
  --window INT             Rolling z-score window in bars (default: 90)
  --report-dir PATH        Output directory (default: reports/)
  --no-report              Print metrics only, no files generated
```

`main_static.py` accepts the same options except `--refresh`.

---

## Project Structure

```
quant_arbitrage/
├── Dockerfile
├── docker-compose.yml
├── config.py               # All strategy and fee parameters
├── requirements.txt
├── main.py                 # CLI — live data fetch + backtest
├── main_static.py          # CLI — Excel cache only, no API calls
│
├── src/
│   ├── data/
│   │   ├── fetcher.py          # Append-only fetch/load, Excel persistence
│   │   └── preprocessor.py     # align_data(), log-spread, shift(1) z-score
│   ├── strategy/
│   │   └── quant_strategy.py   # Signal generation (vectorized, lookahead-free)
│   ├── backtest/
│   │   ├── engine.py           # Event-loop: next-open entry, if/else exit
│   │   └── metrics.py          # Sharpe, drawdown, win rate, profit factor
│   └── reporting/
│       ├── report.py           # generate_report() → .md + .png + .csv + .pdf
│       └── analysis.py         # Static analysis PNGs
│
├── app/
│   ├── Home.py                 # Streamlit landing page
│   ├── pages/
│   │   ├── 1_EDA.py            # EDA page
│   │   └── 2_Backtest.py       # Backtest page
│   ├── helpers/                # Shared pipeline, plot, and report helpers
│   └── reports/                # App-generated reports (per-run subfolders)
│
├── data/raw/               # Excel data files (auto-managed)
│   ├── tsla_1min.xlsx
│   └── mexc_1min.xlsx
└── reports/                # CLI-generated reports (per-run subfolders)
```

---

*For research purposes only. Past performance does not guarantee future results.*
# tsla-mexc-quant-arbitrage
