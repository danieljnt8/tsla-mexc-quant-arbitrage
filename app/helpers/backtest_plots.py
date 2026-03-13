"""
app/helpers/backtest_plots.py
=============================
Plotly chart builders for the Backtest page.

All functions are pure: they take EngineResult / Trade lists / metrics dicts
and return plotly.graph_objects.Figure.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.backtest.engine import EngineResult, Trade  # noqa: E402


# ---------------------------------------------------------------------------
# Color palette (matching eda_plots)
# ---------------------------------------------------------------------------

TSLA_COLOR   = "#00C853"
MEXC_COLOR   = "#FF6D00"
SPREAD_COLOR = "#7C4DFF"
SHORT_COLOR  = "#1565C0"
LONG_COLOR   = "#AD1457"
EQUITY_COLOR = "#0288D1"
WIN_COLOR    = "#2E7D32"
LOSS_COLOR   = "#C62828"


# ---------------------------------------------------------------------------
# Equity Curve
# ---------------------------------------------------------------------------

def plot_equity_curve(result: EngineResult) -> go.Figure:
    """
    Cumulative net PnL (equity curve) with:
    - Running peak overlay
    - Drawdown fill (shaded area between equity and peak)
    """
    equity = result.equity_curve
    peak   = equity.cummax()
    dd     = equity - peak

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=("Cumulative Net PnL (USD)", "Drawdown (USD)"),
    )

    # Panel 1: Equity curve
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Net PnL",
        line=dict(color=EQUITY_COLOR, width=2),
        hovertemplate="%{x}<br>PnL: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=peak.index, y=peak.values,
        name="Running Peak",
        line=dict(color="gray", width=1.0, dash="dot"),
        hovertemplate="Peak: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Fill positive / negative regions
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        fill="tozeroy",
        fillcolor="rgba(2,136,209,0.10)",
        line=dict(width=0),
        showlegend=False,
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color="#AAA", row=1, col=1)

    # Panel 2: Drawdown
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name="Drawdown",
        fill="tozeroy",
        fillcolor="rgba(198,40,40,0.25)",
        line=dict(color=LOSS_COLOR, width=1.0),
        hovertemplate="%{x}<br>Drawdown: $%{y:.2f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=60, r=40, t=60, b=40),
    )
    fig.update_yaxes(title_text="PnL (USD)", tickprefix="$", row=1, col=1, showgrid=True, gridcolor="#EEE")
    fig.update_yaxes(title_text="Drawdown ($)", tickprefix="$", row=2, col=1, showgrid=True, gridcolor="#EEE")
    return fig


# ---------------------------------------------------------------------------
# Price with Trade Markers
# ---------------------------------------------------------------------------

def plot_price_with_trades(result: EngineResult) -> go.Figure:
    """
    TSLA close price + MEXC close price (dual axis) with entry/exit markers.
    """
    data   = result.data
    trades = result.trades

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Price series
    fig.add_trace(go.Scatter(
        x=data.index, y=data["tsla_close"],
        name="TSLA", line=dict(color=TSLA_COLOR, width=1.0),
        hovertemplate="TSLA: $%{y:.2f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data.index, y=data["mexc_close"],
        name="MEXC", line=dict(color=MEXC_COLOR, width=1.0),
        hovertemplate="MEXC: $%{y:.2f}<extra></extra>",
    ), secondary_y=True)

    # Trade markers
    entry_short_x, entry_short_y = [], []
    entry_long_x,  entry_long_y  = [], []
    exit_x,        exit_y        = [], []

    for t in trades:
        # Use TSLA price for marker position
        tsla_at_entry = data["tsla_close"].asof(t.entry_time) if t.entry_time in data.index or True else t.entry_tsla
        tsla_at_exit  = data["tsla_close"].asof(t.exit_time)  if t.exit_time  in data.index or True else t.exit_tsla

        if t.direction == "short_mexc":
            entry_short_x.append(t.entry_time)
            entry_short_y.append(t.entry_tsla)
        else:
            entry_long_x.append(t.entry_time)
            entry_long_y.append(t.entry_tsla)
        exit_x.append(t.exit_time)
        exit_y.append(t.exit_tsla)

    if entry_short_x:
        fig.add_trace(go.Scatter(
            x=entry_short_x, y=entry_short_y,
            mode="markers", name="Entry: Short MEXC",
            marker=dict(color=SHORT_COLOR, size=8, symbol="triangle-up",
                        line=dict(color="white", width=0.5)),
            hovertemplate="Short MEXC entry<br>TSLA: $%{y:.2f}<br>%{x}<extra></extra>",
        ), secondary_y=False)

    if entry_long_x:
        fig.add_trace(go.Scatter(
            x=entry_long_x, y=entry_long_y,
            mode="markers", name="Entry: Long MEXC",
            marker=dict(color=LONG_COLOR, size=8, symbol="triangle-down",
                        line=dict(color="white", width=0.5)),
            hovertemplate="Long MEXC entry<br>TSLA: $%{y:.2f}<br>%{x}<extra></extra>",
        ), secondary_y=False)

    if exit_x:
        fig.add_trace(go.Scatter(
            x=exit_x, y=exit_y,
            mode="markers", name="Exit",
            marker=dict(color="#888", size=5, symbol="x",
                        line=dict(color="white", width=0.5)),
            hovertemplate="Exit<br>TSLA: $%{y:.2f}<br>%{x}<extra></extra>",
        ), secondary_y=False)

    fig.update_layout(
        title="Price Series with Trade Entries & Exits",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=60, r=60, t=60, b=40),
    )
    fig.update_yaxes(title_text="TSLA (USD)", tickprefix="$", secondary_y=False,
                     showgrid=True, gridcolor="#EEE")
    fig.update_yaxes(title_text="MEXC (USDT)", tickprefix="$", secondary_y=True,
                     showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# Z-Score with Signal Markers
# ---------------------------------------------------------------------------

def plot_zscore_with_signals(result: EngineResult, entry_threshold: float = 2.0) -> go.Figure:
    """
    Z-score time series with ±entry_threshold bands and actual entry/exit dots.
    """
    data   = result.data
    trades = result.trades
    z      = data["z_score"].dropna()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=z.index, y=z.values,
        name="Z-score",
        line=dict(color=SPREAD_COLOR, width=1.0),
        hovertemplate="%{x}<br>z=%{y:.3f}<extra></extra>",
    ))

    # Threshold bands
    fig.add_hline(y=+entry_threshold, line_dash="dash", line_color=SHORT_COLOR,
                  annotation_text=f"+{entry_threshold}σ", annotation_position="right")
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color=LONG_COLOR,
                  annotation_text=f"-{entry_threshold}σ", annotation_position="right")
    fig.add_hline(y=0, line_dash="dot", line_color="#999")

    # Entry signal markers
    short_trades = [t for t in trades if t.direction == "short_mexc"]
    long_trades  = [t for t in trades if t.direction == "long_mexc"]

    if short_trades:
        fig.add_trace(go.Scatter(
            x=[t.entry_time for t in short_trades],
            y=[t.entry_z    for t in short_trades],
            mode="markers", name="Short MEXC entries",
            marker=dict(color=SHORT_COLOR, size=7, symbol="triangle-up"),
            hovertemplate="Short entry<br>z=%{y:.2f}<br>%{x}<extra></extra>",
        ))
    if long_trades:
        fig.add_trace(go.Scatter(
            x=[t.entry_time for t in long_trades],
            y=[t.entry_z    for t in long_trades],
            mode="markers", name="Long MEXC entries",
            marker=dict(color=LONG_COLOR, size=7, symbol="triangle-down"),
            hovertemplate="Long entry<br>z=%{y:.2f}<br>%{x}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Z-Score (Lookahead-Free) with ±{entry_threshold}σ Entry Bands",
        xaxis_title="Time (UTC)",
        yaxis_title="Z-Score",
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=60, r=80, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Trade Analysis
# ---------------------------------------------------------------------------

def plot_pnl_distribution(trades: List[Trade]) -> go.Figure:
    """
    PnL distribution: histogram + box plot side-by-side.
    """
    pnls = [t.pnl_net for t in trades]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Net PnL Distribution", "Net PnL Box Plot"),
        column_widths=[0.65, 0.35],
    )

    # Histogram
    colors = [WIN_COLOR if p > 0 else LOSS_COLOR for p in pnls]
    fig.add_trace(go.Histogram(
        x=pnls,
        name="Net PnL",
        nbinsx=40,
        marker_color=[WIN_COLOR if p > 0 else LOSS_COLOR for p in pnls],
        opacity=0.8,
        hovertemplate="PnL: $%{x:.3f}<br>Count: %{y}<extra></extra>",
    ), row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#333", row=1, col=1)

    # Box plot
    short_pnls = [t.pnl_net for t in trades if t.direction == "short_mexc"]
    long_pnls  = [t.pnl_net for t in trades if t.direction == "long_mexc"]
    fig.add_trace(go.Box(
        y=short_pnls, name="Short MEXC",
        marker_color=SHORT_COLOR, boxpoints="all",
        jitter=0.3, pointpos=-1.5,
    ), row=1, col=2)
    fig.add_trace(go.Box(
        y=long_pnls, name="Long MEXC",
        marker_color=LONG_COLOR, boxpoints="all",
        jitter=0.3, pointpos=-1.5,
    ), row=1, col=2)

    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=50, r=50, t=60, b=50),
    )
    fig.update_xaxes(title_text="Net PnL (USD)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Net PnL (USD)", tickprefix="$", row=1, col=2)
    return fig


def plot_entry_z_vs_pnl(trades: List[Trade]) -> go.Figure:
    """
    Scatter: entry z-score vs net PnL, colored by direction.
    Shows whether high |z| entries are more profitable.
    """
    short_trades = [t for t in trades if t.direction == "short_mexc"]
    long_trades  = [t for t in trades if t.direction == "long_mexc"]

    fig = go.Figure()

    if short_trades:
        fig.add_trace(go.Scatter(
            x=[t.entry_z for t in short_trades],
            y=[t.pnl_net  for t in short_trades],
            mode="markers",
            name="Short MEXC",
            marker=dict(
                color=[WIN_COLOR if t.pnl_net > 0 else LOSS_COLOR for t in short_trades],
                size=7, opacity=0.8, symbol="circle",
                line=dict(color=SHORT_COLOR, width=1),
            ),
            hovertemplate="Entry z: %{x:.2f}<br>Net PnL: $%{y:.3f}<extra></extra>",
        ))
    if long_trades:
        fig.add_trace(go.Scatter(
            x=[t.entry_z for t in long_trades],
            y=[t.pnl_net  for t in long_trades],
            mode="markers",
            name="Long MEXC",
            marker=dict(
                color=[WIN_COLOR if t.pnl_net > 0 else LOSS_COLOR for t in long_trades],
                size=7, opacity=0.8, symbol="diamond",
                line=dict(color=LONG_COLOR, width=1),
            ),
            hovertemplate="Entry z: %{x:.2f}<br>Net PnL: $%{y:.3f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#999")

    fig.update_layout(
        title="Entry Z-Score vs Net PnL (green=win, red=loss)",
        xaxis_title="Entry Z-Score (|z| at signal bar)",
        yaxis_title="Net PnL (USD)",
        height=400,
        hovermode="closest",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=60, r=50, t=60, b=60),
    )
    fig.update_yaxes(tickprefix="$")
    return fig


def plot_holding_bars_vs_pnl(trades: List[Trade]) -> go.Figure:
    """
    Scatter: holding time (bars) vs net PnL. Reveals if quick exits are better.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[t.holding_bars for t in trades],
        y=[t.pnl_net       for t in trades],
        mode="markers",
        marker=dict(
            color=[WIN_COLOR if t.pnl_net > 0 else LOSS_COLOR for t in trades],
            size=7, opacity=0.7,
        ),
        text=[f"Dir: {t.direction}<br>Exit: {t.exit_reason}" for t in trades],
        hovertemplate="Hold: %{x} bars<br>PnL: $%{y:.3f}<br>%{text}<extra></extra>",
        name="Trades",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#999")

    fig.update_layout(
        title="Holding Time vs Net PnL (green=win, red=loss)",
        xaxis_title="Holding Duration (bars = minutes)",
        yaxis_title="Net PnL (USD)",
        height=380,
        hovermode="closest",
        margin=dict(l=60, r=50, t=60, b=60),
    )
    fig.update_yaxes(tickprefix="$")
    return fig


# ---------------------------------------------------------------------------
# Exit Breakdown
# ---------------------------------------------------------------------------

def plot_exit_breakdown(metrics: Dict[str, Any]) -> go.Figure:
    """
    Two-panel: pie chart of exit reason counts + bar chart of avg PnL per reason.
    """
    exit_reasons = metrics.get("exit_reasons", {})
    if not exit_reasons:
        return go.Figure()

    labels = list(exit_reasons.keys())
    counts = list(exit_reasons.values())

    colors = {
        "signal":      "#2196F3",
        "session_end": "#FF9800",
        "max_holding": "#9C27B0",
        "eod":         "#607D8B",
    }
    pie_colors = [colors.get(l, "#999") for l in labels]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Exit Reason Distribution", "Count by Exit Reason"),
        specs=[[{"type": "pie"}, {"type": "bar"}]],
    )

    fig.add_trace(go.Pie(
        labels=labels, values=counts,
        name="Exit Reasons",
        marker=dict(colors=pie_colors),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} trades (%{percent})<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=labels, y=counts,
        name="Count",
        marker_color=pie_colors, opacity=0.85,
        hovertemplate="%{x}: %{y} trades<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        height=380,
        showlegend=False,
        margin=dict(l=50, r=50, t=60, b=60),
    )
    fig.update_yaxes(title_text="Number of Trades", row=1, col=2)
    return fig


def plot_cumulative_pnl_by_direction(trades: List[Trade]) -> go.Figure:
    """
    Cumulative PnL broken down by short_mexc vs long_mexc trades over time.
    """
    short_trades = sorted([t for t in trades if t.direction == "short_mexc"],
                           key=lambda t: t.exit_time)
    long_trades  = sorted([t for t in trades if t.direction == "long_mexc"],
                           key=lambda t: t.exit_time)

    def cum_pnl_series(tlist):
        if not tlist:
            return pd.Series(dtype=float)
        times = [t.exit_time for t in tlist]
        pnls  = [t.pnl_net   for t in tlist]
        s = pd.Series(pnls, index=times).cumsum()
        return s

    fig = go.Figure()

    s_short = cum_pnl_series(short_trades)
    s_long  = cum_pnl_series(long_trades)

    if not s_short.empty:
        fig.add_trace(go.Scatter(
            x=s_short.index, y=s_short.values,
            name="Short MEXC (cumulative)",
            line=dict(color=SHORT_COLOR, width=1.8),
            hovertemplate="%{x}<br>Cum PnL: $%{y:.2f}<extra></extra>",
        ))
    if not s_long.empty:
        fig.add_trace(go.Scatter(
            x=s_long.index, y=s_long.values,
            name="Long MEXC (cumulative)",
            line=dict(color=LONG_COLOR, width=1.8),
            hovertemplate="%{x}<br>Cum PnL: $%{y:.2f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="#999")

    fig.update_layout(
        title="Cumulative PnL by Direction",
        xaxis_title="Time (UTC)",
        yaxis_title="Cumulative Net PnL (USD)",
        height=360,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=60, r=40, t=60, b=40),
    )
    fig.update_yaxes(tickprefix="$")
    return fig
