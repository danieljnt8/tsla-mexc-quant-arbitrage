"""
main_static.py
==============
CLI entry point for the TSLA NYSE / TESLA_USDT MEXC Quant Approach backtester.
Loads data from the saved Excel cache (data/raw/) — no API calls made.

Usage
-----
  # Default: last 10 days from cache
  python main_static.py

  # Custom date range (filters cached data)
  python main_static.py --start 2026-02-12 --end 2026-03-10

  # Adjust strategy parameters
  python main_static.py --entry-threshold 2.5 --window 100

  # Skip report generation (just print metrics)
  python main_static.py --no-report

  # Custom output directory
  python main_static.py --report-dir my_reports/

Full pipeline
-------------
  Step 1: Load TSLA + MEXC from Excel cache (data/raw/tsla_1min.xlsx, mexc_1min.xlsx)
  Step 2: Align to NYSE hours only (14:30–21:00 UTC, Mon–Fri)
  Step 3: Compute log-spread + shift(1) z-score
  Step 4: Generate entry/exit signals (QuantStrategy)
  Step 5: Run event-loop backtest (BacktestEngine)
  Step 6: Compute metrics + generate report (Markdown + PNG + CSV)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import Config
from src.data.fetcher import load_excel_tsla, load_excel_mexc
from src.data.preprocessor import align_data, compute_spread_features
from src.strategy.quant_strategy import QuantStrategy
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute_metrics
from src.reporting.report import generate_report


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TSLA NYSE / TESLA_USDT MEXC — Quant Approach (Strategy A) Backtester [Static/Excel mode]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_static.py                                 # full cached sample
  python main_static.py --start 2026-02-12 --end 2026-03-10
  python main_static.py --entry-threshold 2.5 --window 100
  python main_static.py --no-report                     # metrics only, no files
        """,
    )
    parser.add_argument(
        "--start", default=None,
        help="Backtest start date (YYYY-MM-DD). Default: 10 days ago.",
    )
    parser.add_argument(
        "--end", default=None,
        help="Backtest end date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--report-dir", default=None,
        help="Output directory for the report files. Default: reports/",
    )
    parser.add_argument(
        "--entry-threshold", type=float, default=None,
        help="Z-score entry threshold. Default: 2.0 (per Quant approach spec).",
    )
    parser.add_argument(
        "--window", type=int, default=None,
        help="Rolling z-score window in bars (minutes). Default: 90.",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip report/chart/CSV generation — just print metrics to stdout.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> Config:
    """Merge CLI overrides into the default Config."""
    cfg = Config()

    if args.report_dir:
        cfg.report_dir = args.report_dir
    if args.entry_threshold is not None:
        cfg.strategy.entry_threshold = args.entry_threshold
    if args.window is not None:
        cfg.strategy.zscore_window = args.window

    return cfg


def _resolve_dates(args: argparse.Namespace) -> tuple[str, str]:
    """
    Determine the start and end dates for filtering the cached data.

    Logic:
    - If --start and --end are both provided: use them directly.
    - If neither is provided: last 10 days.
    - If only --start is provided: use --start to today.
    - If only --end is provided: use 10 days before --end as start.
    """
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    ten_days_ago = (datetime.now(tz=timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")

    if args.start and args.end:
        return args.start, args.end
    elif args.start:
        return args.start, today
    elif args.end:
        return ten_days_ago, args.end
    else:
        return ten_days_ago, today


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(metrics: dict, report_path: Path | None) -> None:
    sep = "=" * 58
    print(f"\n{sep}")
    print("  BACKTEST SUMMARY — Quant Approach (Strategy A)")
    print(sep)
    print(f"  Trades          : {metrics.get('num_trades', 0)}")
    print(f"  Net PnL         : ${metrics.get('total_pnl_net_usd',   0):>10,.2f}")
    print(f"  Gross PnL       : ${metrics.get('total_pnl_gross_usd', 0):>10,.2f}")
    print(f"  Fees (total)    : ${metrics.get('total_fees_usd',      0):>10,.2f}")
    print(f"    MEXC fees     : ${metrics.get('total_fees_mexc_usd', 0):>10,.2f}")
    print(f"    NYSE fees     : ${metrics.get('total_fees_nyse_usd', 0):>10,.2f}")
    print(f"  Win Rate        : {metrics.get('win_rate_pct',         0):>9.1f}%")
    print(f"  Sharpe Ratio    : {metrics.get('sharpe_ratio',         0):>11.3f}")
    print(f"  Max Drawdown    : ${metrics.get('max_drawdown_usd',    0):>10,.2f}")
    print(f"  Avg Hold Time   : {metrics.get('avg_holding_minutes',  0):>7.1f} min")
    print(f"  Profit Factor   : {metrics.get('profit_factor',        0):>11.3f}")
    print(sep)

    reasons = metrics.get("exit_reasons", {})
    if reasons:
        print("  Exit Reasons:")
        for reason, count in sorted(reasons.items()):
            print(f"    {reason:<18}: {count}")
    print(sep)

    if report_path:
        print(f"  Report → {report_path}")
        print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg  = build_config(args)

    start, end = _resolve_dates(args)

    print("\n[main] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("[main]   TSLA NYSE / TESLA_USDT MEXC — Quant Approach Backtest (Static/Excel)")
    print("[main] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"[main] Period   : {start} → {end}")
    print(f"[main] Strategy : entry=±{cfg.strategy.entry_threshold}σ, "
          f"window={cfg.strategy.zscore_window} bars, "
          f"max_hold={cfg.strategy.max_holding_bars} bars")
    print(f"[main] Fees     : MEXC {cfg.fees.mexc_per_side*10000:.0f}bps/side, "
          f"NYSE ${cfg.fees.nyse_commission_per_share:.2f}/share/side\n")

    # ── Step 1: Load from Excel cache ───────────────────────────────────────
    print("─── Step 1/6: Loading data from Excel cache ───")
    try:
        tsla_df = load_excel_tsla(cfg.data, start=start, end=end)
        mexc_df  = load_excel_mexc(cfg.data, start=start, end=end)
    except FileNotFoundError as e:
        print(f"[main] ERROR: {e}")
        return 1

    if tsla_df.empty or mexc_df.empty:
        print("[main] ERROR: One or both data sources returned empty DataFrames for the given date range.")
        return 1

    # ── Step 2: Align to NYSE hours ────────────────────────────────────────
    print("\n─── Step 2/6: Aligning to NYSE trading hours ───")
    aligned = align_data(tsla_df, mexc_df)

    min_required = cfg.strategy.zscore_window + cfg.strategy.zscore_window  # 2× window
    if len(aligned) < min_required:
        print(
            f"[main] ERROR: Only {len(aligned)} aligned bars — not enough for a meaningful "
            f"backtest (need >{min_required}).  Try extending the date range."
        )
        return 1

    # ── Step 3: Compute spread features ───────────────────────────────────
    print("\n─── Step 3/6: Computing log-spread + shift(1) z-score ───")
    featured = compute_spread_features(aligned, window=cfg.strategy.zscore_window, open_cooldown=cfg.strategy.open_cooldown)

    # ── Step 4: Generate signals ──────────────────────────────────────────
    print("\n─── Step 4/6: Generating signals (QuantStrategy) ───")
    strategy    = QuantStrategy(cfg.strategy)
    signal_data = strategy.generate_signals(featured)

    # ── Step 5: Run backtest ──────────────────────────────────────────────
    print("\n─── Step 5/6: Running event-loop backtest ───")
    engine = BacktestEngine(cfg)
    result = engine.run(signal_data)

    if not result.trades:
        print(
            "[main] WARNING: No trades generated.  "
            "Consider lowering --entry-threshold or extending the date range."
        )

    # ── Step 6: Metrics + report ──────────────────────────────────────────
    print("\n─── Step 6/6: Computing metrics and generating report ───")
    metrics = compute_metrics(result)

    report_path = None
    if not args.no_report and result.trades:
        report_path = generate_report(
            result=result,
            cfg=cfg,
            start=start,
            end=end,
            report_dir=cfg.report_dir,
        )
    elif args.no_report:
        print("[main] --no-report: skipping file generation.")

    print_summary(metrics, report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
