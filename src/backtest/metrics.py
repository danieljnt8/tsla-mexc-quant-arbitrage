from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from src.backtest.engine import EngineResult, Trade


def compute_metrics(result: EngineResult) -> Dict[str, Any]:
    """
    Compute all performance metrics from a backtest result.

    Parameters
    ----------
    result : EngineResult from BacktestEngine.run()

    Returns
    -------
    dict of metrics.  Returns {"num_trades": 0} if there are no trades.
    """
    trades = result.trades
    equity = result.equity_curve

    if not trades:
        print("[metrics] No trades — nothing to compute.")
        return {"num_trades": 0}

    # -----------------------------------------------------------------------
    # PnL vectors
    # -----------------------------------------------------------------------
    pnls_net   = [t.pnl_net   for t in trades]
    pnls_gross = [t.pnl_gross for t in trades]
    fees_mexc  = [t.pnl_fees_mexc for t in trades]
    fees_nyse  = [t.pnl_fees_nyse for t in trades]
    fees_total = [t.pnl_fees  for t in trades]

    winners = [p for p in pnls_net if p > 0]
    losers  = [p for p in pnls_net if p <= 0]

    # -----------------------------------------------------------------------
    # Sharpe ratio (annualized, daily absolute PnL)
    # Group by exit date — matches notebook's backtest_pairs() exactly.
    # Only days with at least one closed trade are included; idle days,
    # weekends, and holidays are excluded (same as notebook).
    # -----------------------------------------------------------------------
    daily_pnl = (
        pd.Series(
            [t.pnl_net for t in trades],
            index=[t.exit_time.date() for t in trades],
        )
        .groupby(level=0)
        .sum()
    )

    if len(daily_pnl) > 1 and daily_pnl.std(ddof=1) > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std(ddof=1)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # -----------------------------------------------------------------------
    # Max drawdown (USD and %)
    # -----------------------------------------------------------------------
    peak            = equity.cummax()
    drawdown_series = equity - peak

    max_dd_usd  = drawdown_series.min()
    worst_idx   = drawdown_series.idxmin()
    peak_at_worst = peak.loc[worst_idx]

    max_dd_pct = (max_dd_usd / peak_at_worst * 100) if peak_at_worst > 0 else 0.0

    # -----------------------------------------------------------------------
    # Exit reasons
    # -----------------------------------------------------------------------
    exit_reasons = Counter(t.exit_reason for t in trades)

    # -----------------------------------------------------------------------
    # Holding time
    # -----------------------------------------------------------------------
    holding_bars = [t.holding_bars for t in trades]

    # -----------------------------------------------------------------------
    # Assemble
    # -----------------------------------------------------------------------
    metrics = {
        "total_pnl_net_usd":   round(sum(pnls_net),   2),
        "total_pnl_gross_usd": round(sum(pnls_gross), 2),
        "total_fees_usd":      round(sum(fees_total), 2),
        "total_fees_mexc_usd": round(sum(fees_mexc),  2),
        "total_fees_nyse_usd": round(sum(fees_nyse),  2),
        "num_trades":          len(trades),
        "win_rate_pct":        round(len(winners) / len(trades) * 100, 1),
        "avg_pnl_per_trade":   round(float(np.mean(pnls_net)), 2),
        "best_trade_usd":      round(max(pnls_net),  2),
        "worst_trade_usd":     round(min(pnls_net),  2),
        "sharpe_ratio":        round(float(sharpe),  3),
        "max_drawdown_usd":    round(float(max_dd_usd), 2),
        "max_drawdown_pct":    round(float(max_dd_pct), 2),
        "avg_holding_minutes": round(float(np.mean(holding_bars)), 1),
        "max_holding_minutes": int(max(holding_bars)),
        "min_holding_minutes": int(min(holding_bars)),
        "profit_factor":       round(
            abs(sum(winners) / sum(losers)), 3
        ) if losers else float("inf"),
        "exit_reasons":        dict(exit_reasons),
    }

    return metrics


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dict as a Markdown table string.

    Fee breakdown shows MEXC and NYSE fees separately to help
    identify which venue dominates the cost structure.
    """
    rows = [
        ("Total Net PnL",           f"${metrics.get('total_pnl_net_usd',   0):>10,.2f}"),
        ("Total Gross PnL",         f"${metrics.get('total_pnl_gross_usd', 0):>10,.2f}"),
        ("Total Fees (MEXC + NYSE)",f"${metrics.get('total_fees_usd',      0):>10,.2f}"),
        ("  ↳ MEXC fees",           f"${metrics.get('total_fees_mexc_usd', 0):>10,.2f}"),
        ("  ↳ NYSE fees",           f"${metrics.get('total_fees_nyse_usd', 0):>10,.2f}"),
        ("Number of Trades",        f"{metrics.get('num_trades',           0):>11}"),
        ("Win Rate",                f"{metrics.get('win_rate_pct',         0):>10.1f}%"),
        ("Avg PnL per Trade",       f"${metrics.get('avg_pnl_per_trade',   0):>10,.2f}"),
        ("Best Trade",              f"${metrics.get('best_trade_usd',      0):>10,.2f}"),
        ("Worst Trade",             f"${metrics.get('worst_trade_usd',     0):>10,.2f}"),
        ("Sharpe Ratio",            f"{metrics.get('sharpe_ratio',         0):>11.3f}"),
        ("Max Drawdown (USD)",      f"${metrics.get('max_drawdown_usd',    0):>10,.2f}"),
        ("Max Drawdown (%)",        f"{metrics.get('max_drawdown_pct',     0):>10.2f}%"),
        ("Avg Holding Time",        f"{metrics.get('avg_holding_minutes',  0):>7.1f} min"),
        ("Profit Factor",           f"{metrics.get('profit_factor',        0):>11.3f}"),
    ]

    lines = ["| Metric | Value |", "|:---|---:|"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")

    lines.append("| **Exit Reasons** | |")
    for reason, count in sorted(metrics.get("exit_reasons", {}).items()):
        lines.append(f"| &nbsp;&nbsp;{reason} | {count} |")

    return "\n".join(lines)
