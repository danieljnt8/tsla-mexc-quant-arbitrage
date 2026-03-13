# HOWTO — Running the Quant Arbitrage Backtester

---

## Option A — Docker (Recommended)

No Python installation required. Docker handles everything.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### 1. Start the Streamlit App

```bash
cd tsla-mexc-quant_arbitrage/
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

- Data (Excel cache) is saved to `quant_arbitrage/data/raw/` on your machine
- Reports are saved to `quant_arbitrage/reports/` and `quant_arbitrage/app/reports/`
- Changes persist between container restarts

### 2. Run the CLI via Docker

```bash
# Live data fetch + backtest (default date range)
docker compose run --rm cli python main.py

# Custom date range
docker compose run --rm cli python main.py --start 2026-02-12 --end 2026-03-10

# Offline mode (uses saved Excel files, no internet)
docker compose run --rm cli python main_static.py

# Print metrics only, no report files
docker compose run --rm cli python main.py --no-report
```

### 3. Stop the App

```bash
docker compose down
```

---

## Option B — Local Python

### System Requirements

| Package | Version |
|---|---|
| Python | 3.10+ |
| pandas | ≥ 2.1.0 |
| numpy | ≥ 1.26.0 |
| yfinance | ≥ 0.2.50 |
| requests | ≥ 2.31.0 |
| matplotlib | ≥ 3.8.0 |
| scipy | ≥ 1.11.0 |
| statsmodels | ≥ 0.14.0 |
| openpyxl | ≥ 3.1.0 |
| streamlit | ≥ 1.32 (app only) |
| plotly | ≥ 5.20 (app only) |
| hurst | ≥ 0.0.5 (EDA page) |

### 1. Installation (One-Time)

```bash
cd quant_arbitrage/
pip install -r requirements.txt
```

### 2. Run the CLI

**Live fetch mode** (`main.py`) — downloads fresh data from yfinance and MEXC REST API,
appends new bars to the Excel cache, then runs the full backtest pipeline.

```bash
# Default run (last ~10 days of data)
python main.py

# Custom date range
python main.py --start 2026-02-12 --end 2026-03-10

# Print metrics only — no report files written
python main.py --no-report

# Adjust strategy parameters
python main.py --entry-threshold 2.5 --window 100

# Custom report output folder
python main.py --report-dir my_results/
```

**Requires internet.** If MEXC is blocked in your region, use a VPN (Singapore or US).

---

**Offline mode** (`main_static.py`) — reads from saved Excel files, no API calls.
All other steps are identical to `main.py`.

```bash
# Run on all cached data
python main_static.py

# Filter to a specific date range
python main_static.py --start 2026-02-12 --end 2026-03-10
```

Fails with a clear error if Excel files don't exist yet — run `python main.py` at least once first.

### 3. Run the Streamlit App

From the `quant_arbitrage/` directory (not inside `app/`):

```bash
streamlit run app/Home.py
```

Opens at [http://localhost:8501](http://localhost:8501).

---

## CLI Options Reference

```
python main.py [OPTIONS]
python main_static.py [OPTIONS]

  --start YYYY-MM-DD       Backtest window start (default: 10 days ago)
  --end   YYYY-MM-DD       Backtest window end (default: today)
  --refresh                Force re-fetch all data, ignore Excel cache  [main.py only]
  --entry-threshold FLOAT  Z-score entry threshold (default: 2.0)
  --window INT             Rolling z-score window in bars (default: 90)
  --report-dir PATH        Output directory (default: reports/)
  --no-report              Print metrics only, no files generated
```

---

## What the Pipeline Does

```
Step 1  Test connectivity     MEXC API ping + yfinance check
Step 2  Fetch data            TSLA 1-min (yfinance) + MEXC TESLA_USDT 1-min (REST API)
                              Append-only: only downloads new bars since last run
Step 3  Align to NYSE hours   Inner join on NYSE sessions, Mon–Fri, DST-aware
Step 4  Compute features      log(MEXC/TSLA) spread + shift(1) rolling z-score
Step 5  Generate signals      z ≥ +2.0σ → short MEXC | z ≤ −2.0σ → long MEXC
Step 6  Run backtest          Event loop: fill at next bar's OPEN, exit on z→0 or 20-bar hold
Step 7  Metrics + report      Sharpe, drawdown, win rate → .md + .png + .csv + .pdf
```

`main_static.py` skips steps 1–2 (loads from Excel instead).

---

## Expected Output

```
[main] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[main]   TSLA NYSE / TESLA_USDT MEXC — Quant Approach Backtest
[main] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[main] Period   : 2026-02-12 → 2026-03-10
[main] Strategy : entry=±2.0σ, window=90 bars, max_hold=20 bars
[main] Fees     : MEXC 3bps/side, NYSE $0.01/share/side

[preprocessor] Aligned 6630 bars (2026-02-12 → 2026-03-10)
[strategy] QuantStrategy: 448 entry bars (228 short_mexc, 220 long_mexc)
[engine] Backtest complete: 269 trades, net PnL = $53.09

==========================================================
  BACKTEST SUMMARY — Quant Approach (Strategy A)
==========================================================
  Trades          : 269
  Net PnL         : $     53.09
  Win Rate        :      74.0%
  Sharpe Ratio    :      13.932
  Max Drawdown    : $     -1.42
  Avg Hold Time   :      4.2 min
  Profit Factor   :      11.234
==========================================================
```

---

## Streamlit App Pages

### Home
- Strategy overview
- Navigation to EDA and Backtest pages
- List of previously generated reports

### EDA (`pages/1_EDA.py`)

Exploratory data analysis explaining *why* the strategy works.

**Controls:** date range, z-score window, entry threshold, data source (live API or Excel)

| Tab | Content |
|---|---|
| Market Structure | TSLA + MEXC price overlay, return scatter, rolling correlation |
| Spread Analysis | Log-spread time series, histogram, z-score with entry bands |
| Statistical Tests | ADF stationarity, Hurst exponent, OU half-life, ACF |
| Intraday Patterns | Signal count by minute, z-score magnitude by hour, per-session table |

### Backtest (`pages/2_Backtest.py`)

Full configurable backtest simulator.

**Controls:** date range, entry threshold, z-score window, max holding bars, MEXC fee (bps), NYSE commission ($/share), data source

**Results:**
- KPI row: Net PnL, Trades, Win Rate, Sharpe, Max Drawdown, Gross PnL, Avg PnL/Trade, Profit Factor, Avg Hold

| Chart Tab | Content |
|---|---|
| Equity Curve | Cumulative net PnL + direction split (short/long MEXC) |
| Price + Trades | TSLA/MEXC price series with entry/exit markers |
| Z-Score | Z-score with ±σ bands and signal markers |
| Trade Analysis | PnL histogram, entry z-score vs PnL scatter, holding bars vs PnL |
| Exit Breakdown | Pie chart of exit reasons + counts |

**Generate Report** button saves a PDF + PNG + Markdown + CSV to `app/reports/`, with direct download buttons.

---

## Output Files

Each backtest run creates a timestamped subfolder with all artifacts:

```
reports/                              ← CLI reports (main.py / main_static.py)
└── YYYYMMDD_HHMMSS/
    ├── backtest_YYYYMMDD_HHMMSS.md   ← Full text report: params, metrics, trade log
    ├── equity_curve_YYYYMMDD_HHMMSS.png  ← 3-panel chart: PnL / prices / z-score
    ├── trades_YYYYMMDD_HHMMSS.csv    ← Trade log: entry/exit, prices, PnL, fees, hold bars
    └── backtest_YYYYMMDD_HHMMSS.pdf  ← PDF summary: KPIs + equity curve + full trade log

app/reports/                          ← Reports generated from the Streamlit Backtest page
└── YYYYMMDD_HHMMSS/
    ├── backtest_YYYYMMDD_HHMMSS.md
    ├── equity_curve_YYYYMMDD_HHMMSS.png
    ├── trades_YYYYMMDD_HHMMSS.csv
    └── backtest_YYYYMMDD_HHMMSS.pdf
```

---

## Data Files

```
data/raw/
├── tsla_1min.xlsx     ← TSLA 1-min OHLCV — grows with each fetch run
└── mexc_1min.xlsx     ← MEXC TESLA_USDT 1-min OHLCV — grows with each fetch run
```

Both files use an append-only strategy: new bars are added, deduplicated by timestamp, and sorted. Old rows are never deleted.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ConnectionError: MEXC API unreachable` | No internet / regional block | Use a VPN (Singapore or US) |
| `ConnectionError: Yahoo Finance unreachable` | No internet / yfinance down | Check network; `pip install --upgrade yfinance` |
| `FileNotFoundError: No saved TSLA/MEXC data` | Excel files don't exist yet | Run `python main.py` first |
| `No trades generated` | Date range too short (< 180 bars after warm-up) | Use a wider `--start` / `--end` range |
| `ModuleNotFoundError` | Missing dependency | Run `pip install -r requirements.txt` |
| Excel file locked (Windows) | File open in Excel while script runs | Close the Excel file |
| App page blank / spinner forever | Connectivity issue | Switch to "Load from saved Excel" in sidebar |

---

## Daily Usage

```bash
cd quant_arbitrage/
python main.py
```

The system fetches only new bars since the last cached timestamp, appends them to the Excel files, and generates a fresh report in `reports/`.

For offline analysis without re-fetching, use `python main_static.py` or open the Streamlit app with "Load from saved Excel" selected.

---

## Project Structure

```
tsla-mexc-quant_arbitrage/
├── HOWTO.md                     ← You are here
├── README.md                    ← Strategy overview and quick start
├── Dockerfile                   ← Container image (app + CLI)
├── docker-compose.yml           ← App service + CLI runner
├── config.py                    ← All parameters (edit to change defaults)
├── requirements.txt
├── main.py                      ← CLI: live fetch + run pipeline
├── main_static.py               ← CLI: load Excel + run pipeline (no API calls)
│
├── src/
│   ├── data/
│   │   ├── fetcher.py           ← Fetch, append-only Excel save, connectivity test
│   │   └── preprocessor.py      ← align_data(), log-spread, shift(1) z-score
│   ├── strategy/
│   │   └── quant_strategy.py    ← QuantStrategy.generate_signals() — vectorized
│   ├── backtest/
│   │   ├── engine.py            ← BacktestEngine — if/else event loop, next-open fill
│   │   └── metrics.py           ← compute_metrics(), format_metrics_table()
│   └── reporting/
│       ├── report.py            ← generate_report() → .md + .png + .csv + .pdf
│       └── analysis.py          ← Static analysis PNGs
│
├── app/
│   ├── Home.py                  ← Streamlit landing page
│   ├── pages/
│   │   ├── 1_EDA.py             ← EDA: correlation, spread, ADF/Hurst, intraday
│   │   └── 2_Backtest.py        ← Backtest: simulator + charts + report downloads
│   ├── helpers/
│   │   ├── pipeline.py          ← build_config(), run_pipeline(), load_aligned_data()
│   │   ├── eda_plots.py         ← Plotly chart helpers for EDA page
│   │   ├── backtest_plots.py    ← Plotly chart helpers for Backtest page
│   │   └── report_writer.py     ← generate_app_report(), list_app_reports()
│   └── reports/                 ← Reports saved from the Streamlit app
│
├── data/raw/                    ← Excel data files (auto-managed, append-only)
│   ├── tsla_1min.xlsx
│   └── mexc_1min.xlsx
└── reports/                     ← CLI-generated reports (timestamped subfolders)
```
