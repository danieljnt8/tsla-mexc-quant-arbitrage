# TSLA/MEXC Quant Arbitrage — Streamlit App

Interactive web application for exploring and backtesting the TSLA NYSE / TESLA_USDT MEXC spread arbitrage strategy.

---

## Quick Start

```bash
# 1. Navigate to the project root
cd quant_arbitrage/

# 2. Install all dependencies (first time only)
pip install -r requirements.txt -r app/requirements.txt

# 3. Launch the app
streamlit run app/Home.py
```

The app opens automatically at **http://localhost:8501**.

---

## Two Pages

### 🔬 EDA — Exploratory Data Analysis
Explains **why** the strategy works through interactive charts:
- Price overlay + return correlation (ρ ≈ 0.92)
- Log-spread time series + z-score with entry bands
- ADF stationarity test + Hurst exponent + OU half-life
- Intraday signal distribution and spread volatility patterns

**Default date range:** 2026-02-12 → 2026-03-05 (the reference sample)

### ⚡ Backtest — Simulator
Run a full backtest with configurable parameters:
- Entry threshold (σ), z-score window, fee assumptions, date range
- KPI summary: Net PnL, Win Rate, Sharpe, Max Drawdown
- Interactive charts: equity curve, price + trade markers, z-score, trade analysis
- Download report: Markdown + PNG chart + CSV trade log → saved to `app/reports/`

**Default parameters replicate the reference notebook exactly:**
244 trades · $35.52 net PnL · 71.3% win rate

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥ 1.32 | Web app framework |
| plotly | ≥ 5.20 | Interactive charts |
| statsmodels | ≥ 0.14 | ADF test, ACF |
| scipy | ≥ 1.12 | Statistical utilities |

Plus the main project deps from `requirements.txt` (yfinance, pandas, numpy, etc.).

---

## Reports

All app-generated reports are saved to `app/reports/`:
- `backtest_YYYYMMDD_HHMMSS.md` — full Markdown report
- `equity_curve_YYYYMMDD_HHMMSS.png` — 3-panel chart
- `trades_YYYYMMDD_HHMMSS.csv` — complete trade log

Previous reports are listed and downloadable from the Backtest page.

---

## Notes

- Data is fetched **live** from **yfinance** (TSLA) and the **MEXC REST API** (TESLA_USDT) every time — no API keys required, no caching.
- The app must be launched from the `quant_arbitrage/` directory (not from inside `app/`).
