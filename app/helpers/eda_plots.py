"""
app/helpers/eda_plots.py
========================
Plotly chart builders for the EDA page.

All functions are pure: they take DataFrames and return plotly.graph_objects.Figure.
No Streamlit imports here — charts are consumed by pages/1_EDA.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

TSLA_COLOR  = "#00C853"   # green
MEXC_COLOR  = "#FF6D00"   # orange
SPREAD_COLOR = "#7C4DFF"  # purple
SHORT_COLOR = "#1565C0"   # blue  (short_mexc = MEXC expensive)
LONG_COLOR  = "#AD1457"   # pink  (long_mexc  = MEXC cheap)
NEUTRAL_COLOR = "#546E7A" # slate


# ---------------------------------------------------------------------------
# Market Structure Charts
# ---------------------------------------------------------------------------

def plot_price_overlay(df: pd.DataFrame) -> go.Figure:
    """
    Dual-axis overlay of TSLA and MEXC close prices.
    Shows how tightly the two instruments track each other.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["tsla_close"],
            name="TSLA (NYSE)",
            line=dict(color=TSLA_COLOR, width=1.5),
            hovertemplate="TSLA: $%{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["mexc_close"],
            name="TESLA_USDT (MEXC)",
            line=dict(color=MEXC_COLOR, width=1.5),
            hovertemplate="MEXC: $%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="TSLA (NYSE) vs TESLA_USDT (MEXC) — Price Overlay",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=50, r=50, t=60, b=40),
    )
    fig.update_yaxes(title_text="TSLA Price (USD)", secondary_y=False,
                     tickprefix="$", showgrid=True, gridcolor="#E0E0E0")
    fig.update_yaxes(title_text="MEXC Price (USDT)", secondary_y=True,
                     tickprefix="$", showgrid=False)
    return fig


def plot_individual_prices(df: pd.DataFrame) -> go.Figure:
    """
    Two-panel chart: TSLA price+volume and MEXC price+volume.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("TSLA (NYSE) — 1-min OHLCV", "TESLA_USDT (MEXC) — 1-min OHLCV"),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
    )

    # TSLA price
    fig.add_trace(
        go.Scatter(x=df.index, y=df["tsla_close"],
                   name="TSLA Close", line=dict(color=TSLA_COLOR, width=1.2)),
        row=1, col=1, secondary_y=False,
    )
    # TSLA volume
    fig.add_trace(
        go.Bar(x=df.index, y=df["tsla_volume"],
               name="TSLA Volume", marker_color=TSLA_COLOR, opacity=0.25),
        row=1, col=1, secondary_y=True,
    )

    # MEXC price
    fig.add_trace(
        go.Scatter(x=df.index, y=df["mexc_close"],
                   name="MEXC Close", line=dict(color=MEXC_COLOR, width=1.2)),
        row=2, col=1, secondary_y=False,
    )
    # MEXC volume
    fig.add_trace(
        go.Bar(x=df.index, y=df["mexc_volume"],
               name="MEXC Volume", marker_color=MEXC_COLOR, opacity=0.25),
        row=2, col=1, secondary_y=True,
    )

    fig.update_layout(
        height=550,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=60, t=60, b=40),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1, secondary_y=False,
                     tickprefix="$")
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True,
                     showgrid=False)
    fig.update_yaxes(title_text="Price (USDT)", row=2, col=1, secondary_y=False,
                     tickprefix="$")
    fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=True,
                     showgrid=False)
    return fig


def plot_returns_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of MEXC 1-min returns vs TSLA 1-min returns.
    Includes OLS regression line + ρ annotation.
    """
    tsla_ret = df["tsla_close"].pct_change().dropna()
    mexc_ret = df["mexc_close"].pct_change().dropna()
    both = pd.concat([tsla_ret.rename("tsla"), mexc_ret.rename("mexc")], axis=1).dropna()

    corr = both["tsla"].corr(both["mexc"])

    # OLS line
    m, b = np.polyfit(both["tsla"], both["mexc"], 1)
    x_line = np.array([both["tsla"].min(), both["tsla"].max()])
    y_line = m * x_line + b

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=both["tsla"] * 100, y=both["mexc"] * 100,
        mode="markers",
        marker=dict(color=NEUTRAL_COLOR, size=3, opacity=0.4),
        name="1-min returns",
        hovertemplate="TSLA: %{x:.3f}%<br>MEXC: %{y:.3f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_line * 100, y=y_line * 100,
        mode="lines",
        line=dict(color="red", width=2, dash="dash"),
        name=f"OLS fit (β={m:.2f})",
    ))
    fig.add_annotation(
        x=0.05, y=0.95, xref="paper", yref="paper",
        text=f"<b>ρ = {corr:.4f}</b><br>β = {m:.3f}  |  n = {len(both):,}",
        showarrow=False, bgcolor="white", bordercolor="#999", borderwidth=1,
        font=dict(size=13),
    )

    fig.update_layout(
        title="1-Minute Return Correlation: TSLA vs TESLA_USDT",
        xaxis_title="TSLA Return (%)",
        yaxis_title="MEXC Return (%)",
        height=430,
        hovermode="closest",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=50, r=50, t=60, b=50),
    )
    return fig


def plot_rolling_correlation(df: pd.DataFrame, window: int = 60) -> go.Figure:
    """
    Rolling correlation between TSLA and MEXC 1-min returns.
    """
    tsla_ret = df["tsla_close"].pct_change()
    mexc_ret = df["mexc_close"].pct_change()
    rolling_corr = tsla_ret.rolling(window).corr(mexc_ret)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_corr.index, y=rolling_corr.values,
        name=f"Rolling {window}-bar Correlation",
        line=dict(color=TSLA_COLOR, width=1.5),
        fill="tozeroy", fillcolor="rgba(0,200,83,0.12)",
    ))
    fig.add_hline(y=0.8, line_dash="dash", line_color="orange",
                  annotation_text="0.80 threshold", annotation_position="bottom right")
    fig.add_hline(y=1.0, line_dash="dot", line_color="#CCC")

    fig.update_layout(
        title=f"Rolling {window}-bar Return Correlation (TSLA vs MEXC)",
        xaxis_title="Time (UTC)",
        yaxis_title="Pearson Correlation",
        yaxis=dict(range=[0.4, 1.1]),
        height=350,
        hovermode="x unified",
        margin=dict(l=50, r=50, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Spread Analysis Charts
# ---------------------------------------------------------------------------

def plot_log_spread(df: pd.DataFrame) -> go.Figure:
    """
    Log-spread time series = log(MEXC / TSLA).
    Shows mean-reversion behavior.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["log_spread"],
        name="log(MEXC / TSLA)",
        line=dict(color=SPREAD_COLOR, width=1.0),
        hovertemplate="%{x}<br>log_spread: %{y:.5f}<extra></extra>",
    ))
    if "rolling_mean" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rolling_mean"],
            name="Rolling Mean",
            line=dict(color="red", width=1.2, dash="dash"),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="#999",
                  annotation_text="Equilibrium (0)", annotation_position="bottom right")

    fig.update_layout(
        title="Log-Spread: log(MEXC / TSLA) — Mean-Reverting Process",
        xaxis_title="Time (UTC)",
        yaxis_title="Log-Spread",
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=50, r=50, t=60, b=40),
    )
    return fig


def plot_spread_histogram(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of log-spread values with fitted normal distribution overlay.
    """
    spread = df["log_spread"].dropna()
    mu, sigma = spread.mean(), spread.std()

    x_norm = np.linspace(spread.min(), spread.max(), 300)
    y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)

    # Scale normal PDF to histogram density
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=spread, nbinsx=80, name="Log-Spread Distribution",
        histnorm="probability density",
        marker_color=SPREAD_COLOR, opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=x_norm, y=y_norm,
        name=f"Normal fit (μ={mu:.4f}, σ={sigma:.4f})",
        line=dict(color="red", width=2.5),
    ))
    fig.add_vline(x=mu, line_dash="dash", line_color="#333",
                  annotation_text=f"Mean={mu:.4f}", annotation_position="top right")

    fig.update_layout(
        title="Log-Spread Distribution (approx. Normal — supports z-score approach)",
        xaxis_title="log(MEXC / TSLA)",
        yaxis_title="Probability Density",
        height=380,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=50, r=50, t=60, b=50),
    )
    return fig


def plot_zscore_with_bands(df: pd.DataFrame, entry_threshold: float = 2.0) -> go.Figure:
    """
    Z-score time series with ±entry_threshold bands.
    Highlights short_mexc and long_mexc signal regions.
    """
    z = df["z_score"].dropna()

    fig = go.Figure()

    # Fill regions above/below threshold
    above = z.copy()
    above[above <= entry_threshold] = np.nan
    below = z.copy()
    below[below >= -entry_threshold] = np.nan

    fig.add_trace(go.Scatter(
        x=z.index, y=z.values,
        name="Z-score",
        line=dict(color=SPREAD_COLOR, width=1.0),
        hovertemplate="%{x}<br>z=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=above.index, y=above.values,
        name=f"Short MEXC zone (z≥+{entry_threshold}σ)",
        line=dict(color=SHORT_COLOR, width=0),
        fill="tonexty", fillcolor="rgba(21,101,192,0.15)",
        showlegend=True,
    ))

    # Threshold lines
    fig.add_hline(y=+entry_threshold, line_dash="dash", line_color=SHORT_COLOR,
                  annotation_text=f"+{entry_threshold}σ (short MEXC)", annotation_position="right")
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color=LONG_COLOR,
                  annotation_text=f"-{entry_threshold}σ (long MEXC)", annotation_position="right")
    fig.add_hline(y=0, line_dash="dot", line_color="#999",
                  annotation_text="Mean (0)", annotation_position="bottom right")

    # Mark actual entry signals if 'signal' column present
    if "signal" in df.columns:
        short_bars = df[df["signal"] == "short_mexc"]
        long_bars  = df[df["signal"] == "long_mexc"]
        if not short_bars.empty:
            fig.add_trace(go.Scatter(
                x=short_bars.index, y=short_bars["z_score"],
                mode="markers",
                name="Short MEXC entry",
                marker=dict(color=SHORT_COLOR, size=5, symbol="triangle-up"),
            ))
        if not long_bars.empty:
            fig.add_trace(go.Scatter(
                x=long_bars.index, y=long_bars["z_score"],
                mode="markers",
                name="Long MEXC entry",
                marker=dict(color=LONG_COLOR, size=5, symbol="triangle-down"),
            ))

    fig.update_layout(
        title=f"Z-Score (shift-1 lookahead-free) with ±{entry_threshold}σ Entry Bands",
        xaxis_title="Time (UTC)",
        yaxis_title="Z-Score",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=50, r=80, t=60, b=40),
    )
    return fig


def plot_rolling_spread_stats(df: pd.DataFrame) -> go.Figure:
    """
    Rolling mean and std of the log-spread — shows stationarity.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Rolling Mean of Log-Spread", "Rolling Std of Log-Spread"),
                        vertical_spacing=0.08)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["rolling_mean"],
        name="Rolling Mean", line=dict(color=SPREAD_COLOR, width=1.2),
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#999", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["rolling_std"],
        name="Rolling Std", line=dict(color=MEXC_COLOR, width=1.2),
    ), row=2, col=1)

    fig.update_layout(
        height=400,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=50, r=50, t=60, b=40),
    )
    fig.update_yaxes(title_text="Mean", row=1, col=1)
    fig.update_yaxes(title_text="Std Dev", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Statistical Tests Charts
# ---------------------------------------------------------------------------

def compute_adf_stats(spread_series: pd.Series) -> dict:
    """
    Run Augmented Dickey-Fuller test on the spread.

    Returns dict with: test_stat, p_value, lags_used, critical_values, is_stationary
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {"error": "statsmodels not installed"}

    clean = spread_series.dropna()
    result = adfuller(clean, autolag="AIC")
    return {
        "test_stat": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


def compute_hurst_ou(spread_series: pd.Series) -> dict:
    """
    Estimate Hurst exponent and OU half-life from the log-spread.

    Hurst < 0.5 → mean-reverting (anti-persistent)
    OU half-life = -ln(2) / ln(φ) where φ is the AR(1) coefficient
    """
    clean = spread_series.dropna()

    # Hurst exponent via R/S statistic
    lags = [2, 4, 8, 16, 32, 64, 128]
    lags = [l for l in lags if l < len(clean) // 2]
    rs_vals = []
    for lag in lags:
        chunks = [clean.iloc[i:i+lag] for i in range(0, len(clean)-lag, lag)]
        rs_per_chunk = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_ = chunk.mean()
            dev   = (chunk - mean_).cumsum()
            r     = dev.max() - dev.min()
            s     = chunk.std()
            if s > 0:
                rs_per_chunk.append(r / s)
        if rs_per_chunk:
            rs_vals.append(np.mean(rs_per_chunk))

    if len(lags) >= 2 and len(rs_vals) >= 2:
        hurst_coef = np.polyfit(np.log(lags[:len(rs_vals)]), np.log(rs_vals), 1)[0]
    else:
        hurst_coef = float("nan")

    # OU half-life via AR(1) regression
    y  = clean.values[1:]
    x  = clean.values[:-1]
    try:
        phi = np.polyfit(x, y, 1)[0]
        if phi > 0 and phi < 1:
            half_life = -np.log(2) / np.log(phi)
        else:
            half_life = float("nan")
    except Exception:
        half_life = float("nan")
        phi = float("nan")

    return {
        "hurst": hurst_coef,
        "is_mean_reverting": hurst_coef < 0.5 if not np.isnan(hurst_coef) else None,
        "ar1_phi": phi,
        "half_life_bars": half_life,
    }




# ---------------------------------------------------------------------------
# Intraday Patterns
# ---------------------------------------------------------------------------

def plot_intraday_zscore_magnitude(df: pd.DataFrame) -> go.Figure:
    """
    Average |z-score| by minute-of-day (ET local time).
    Shows when the spread is most extreme.
    """
    df2 = df.copy()
    # Convert UTC index to ET (UTC-5 winter / UTC-4 summer) — approximate with -5
    et_index = df2.index.tz_convert("America/New_York") if df2.index.tz else df2.index - pd.Timedelta(hours=5)
    df2["et_minute"] = et_index.hour * 60 + et_index.minute
    df2["abs_z"] = df2["z_score"].abs()

    by_minute = df2.groupby("et_minute")["abs_z"].mean().dropna()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_minute.index, y=by_minute.values,
        name="|Z| Average",
        marker_color=SPREAD_COLOR, opacity=0.8,
        hovertemplate="Minute %{x} ET<br>Avg |z|: %{y:.3f}<extra></extra>",
    ))
    # Highlight first 5 minutes (open volatility)
    fig.add_vrect(x0=570, x1=575, fillcolor="rgba(255,165,0,0.2)",
                  annotation_text="Open (5 min)", annotation_position="top right",
                  line_width=0)

    fig.update_layout(
        title="Average |Z-Score| by Minute of Trading Day (ET)",
        xaxis=dict(
            title="Minute of Day (ET)",
            tickmode="array",
            tickvals=[570, 600, 660, 720, 780, 840, 900, 960],
            ticktext=["09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"],
        ),
        yaxis_title="Avg |Z-Score|",
        height=350,
        margin=dict(l=50, r=50, t=60, b=60),
    )
    return fig


def plot_intraday_signal_count(df: pd.DataFrame) -> go.Figure:
    """
    Number of entry signals (short + long) per minute-of-day.
    """
    if "signal" not in df.columns:
        return go.Figure()

    df2 = df.copy()
    et_index = df2.index.tz_convert("America/New_York") if df2.index.tz else df2.index - pd.Timedelta(hours=5)
    df2["et_minute"] = et_index.hour * 60 + et_index.minute
    df2["has_signal"] = df2["signal"].isin(["short_mexc", "long_mexc"]).astype(int)
    df2["is_short"] = (df2["signal"] == "short_mexc").astype(int)
    df2["is_long"]  = (df2["signal"] == "long_mexc").astype(int)

    by_min_short = df2.groupby("et_minute")["is_short"].sum()
    by_min_long  = df2.groupby("et_minute")["is_long"].sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_min_short.index, y=by_min_short.values,
        name="Short MEXC signals",
        marker_color=SHORT_COLOR, opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=by_min_long.index, y=by_min_long.values,
        name="Long MEXC signals",
        marker_color=LONG_COLOR, opacity=0.8,
    ))

    fig.update_layout(
        title="Entry Signals by Minute of Day (ET)",
        barmode="stack",
        xaxis=dict(
            title="Minute of Day (ET)",
            tickmode="array",
            tickvals=[570, 600, 660, 720, 780, 840, 900, 960],
            ticktext=["09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"],
        ),
        yaxis_title="Signal Count",
        height=350,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=50, r=50, t=60, b=60),
    )
    return fig


def plot_spread_volatility_by_hour(df: pd.DataFrame) -> go.Figure:
    """
    Log-spread volatility (std) by hour of day. Shows intraday risk pattern.
    """
    df2 = df.copy()
    et_index = df2.index.tz_convert("America/New_York") if df2.index.tz else df2.index - pd.Timedelta(hours=5)
    df2["et_hour"] = et_index.hour
    df2["abs_spread"] = df2["log_spread"].abs()

    by_hour = df2.groupby("et_hour")["log_spread"].std().dropna()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{h:02d}:00 ET" for h in by_hour.index], y=by_hour.values,
        name="Spread Volatility", marker_color=MEXC_COLOR, opacity=0.8,
    ))
    fig.update_layout(
        title="Log-Spread Std Dev by Hour (ET) — Intraday Risk Pattern",
        xaxis_title="Hour (ET)",
        yaxis_title="Log-Spread Std Dev",
        height=330,
        margin=dict(l=50, r=50, t=60, b=60),
    )
    return fig
