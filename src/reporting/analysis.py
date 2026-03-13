"""
src/reporting/analysis.py
=========================
Generates four research-quality PNG charts for the TSLA/MEXC arbitrage analysis.

Charts produced
---------------
1. price_intraday.png
   Side-by-side intraday price panels for two sample (non-consecutive) trading days.
   TSLA on the left y-axis (blue), MEXC on the right y-axis (orange).
   Prices plotted during NYSE hours only (9:30 AM – 4:00 PM ET).

2. log_return_correlation.png
   Scatter plot of MEXC 1-min log returns vs TSLA 1-min log returns across the
   entire aligned sample. Includes Pearson r, regression line, and 2D density contours.

3. log_spread_intraday.png
   Intraday log-spread panels for two sample (non-consecutive) trading days.
   Shows the spread relative to its rolling mean ± 2σ entry bands.

4. spread_distribution.png
   Full-sample histogram + KDE of the log-spread (log(MEXC/TSLA)).
   Mean, ±1σ, and ±2σ lines annotated.

Usage
-----
    cd quant_arbitrage/
    python -m src.reporting.analysis

    # or directly:
    python src/reporting/analysis.py

Output
------
    reports/analysis/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── sys.path bootstrap ───────────────────────────────────────────────────────
_SRC_DIR      = Path(__file__).resolve().parent.parent           # src/
_PROJECT_ROOT = _SRC_DIR.parent                                   # quant_arbitrage/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── stdlib / third-party imports ─────────────────────────────────────────────
import zoneinfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

# ── project imports ───────────────────────────────────────────────────────────
from config import Config, DataConfig
from src.data.fetcher import load_excel_tsla, load_excel_mexc
from src.data.preprocessor import align_data, compute_spread_features

# ── constants ─────────────────────────────────────────────────────────────────
_EASTERN   = zoneinfo.ZoneInfo("America/New_York")
_OUT_DIR   = _PROJECT_ROOT / "reports" / "analysis"

# Matplotlib style
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#fafafa",
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
})

TSLA_COLOR = "#1565C0"   # deep blue
MEXC_COLOR = "#E65100"   # deep orange
SPREAD_COLOR = "#4A148C" # deep purple


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_aligned() -> pd.DataFrame:
    """Load full Excel data, align to NYSE hours, compute spread features."""
    cfg = Config()
    tsla_df = load_excel_tsla(cfg.data)
    mexc_df  = load_excel_mexc(cfg.data)
    aligned  = align_data(tsla_df, mexc_df)
    featured = compute_spread_features(aligned, window=90, open_cooldown=0)
    return featured


def _get_sample_days(df: pd.DataFrame, n: int = 2, gap: int = 3) -> list[str]:
    """
    Pick n sample days from the aligned data, separated by at least `gap` days.
    Returns dates as YYYY-MM-DD strings (ET local date).
    """
    et_dates = df.index.tz_convert(_EASTERN).normalize().unique()
    # Sort and pick well-spaced days: first day and a day far from the end
    et_dates = sorted(et_dates)
    chosen = [et_dates[2]]  # skip the very first couple (z-score warm-up)
    for d in et_dates:
        if (d - chosen[-1]).days >= gap:
            chosen.append(d)
        if len(chosen) == n:
            break
    return [str(d.date()) for d in chosen[:n]]


def _filter_day(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Return rows for a single NYSE trading day (ET local date)."""
    et_idx = df.index.tz_convert(_EASTERN)
    mask   = et_idx.normalize() == pd.Timestamp(date_str, tz=_EASTERN).normalize()
    return df[mask]


# ---------------------------------------------------------------------------
# Chart 1 — Intraday price movement (2 days)
# ---------------------------------------------------------------------------

def plot_price_intraday(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Two-panel figure showing TSLA and MEXC intraday prices on two sample days.
    """
    days = _get_sample_days(df, n=2, gap=4)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    fig.suptitle("TSLA (NYSE) vs MEXC TESLA_USDT — Intraday Price Movement",
                 fontsize=13, fontweight="bold")

    for ax, date_str in zip(axes, days):
        day_df  = _filter_day(df, date_str)
        times   = day_df.index.tz_convert(_EASTERN)

        ax2 = ax.twinx()

        # TSLA on left axis
        l1, = ax.plot(times, day_df["tsla_close"], color=TSLA_COLOR,
                      linewidth=1.4, label="TSLA (NYSE, USD)")
        ax.set_ylabel("TSLA Price (USD)", color=TSLA_COLOR, fontsize=9)
        ax.tick_params(axis="y", labelcolor=TSLA_COLOR)

        # MEXC on right axis
        l2, = ax2.plot(times, day_df["mexc_close"], color=MEXC_COLOR,
                       linewidth=1.4, linestyle="--", label="MEXC (USDT)")
        ax2.set_ylabel("MEXC Price (USDT)", color=MEXC_COLOR, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=MEXC_COLOR)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(MEXC_COLOR)
        ax2.spines["right"].set_alpha(0.5)

        # Format x-axis as ET time
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=_EASTERN))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=_EASTERN))
        ax.set_xlabel("Time (ET)", fontsize=9)

        # Compute price range for annotation
        tsla_open  = day_df["tsla_close"].iloc[0]
        tsla_close = day_df["tsla_close"].iloc[-1]
        tsla_ret   = (tsla_close / tsla_open - 1) * 100

        ax.set_title(
            f"{date_str}  |  TSLA: ${tsla_open:.2f} → ${tsla_close:.2f} "
            f"({tsla_ret:+.2f}%)  |  {len(day_df)} bars",
            fontsize=10,
        )

        lines  = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left", fontsize=8, framealpha=0.8)

        # Grid only on primary axis
        ax.grid(True, alpha=0.3, linestyle="--")
        ax2.grid(False)

    out_path = out_dir / "price_intraday.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Chart 2 — Log return correlation
# ---------------------------------------------------------------------------

def plot_log_return_correlation(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Scatter plot of MEXC vs TSLA 1-min log returns with regression line and
    2D kernel density estimate contours.
    """
    tsla_ret = np.log(df["tsla_close"] / df["tsla_close"].shift(1)).dropna()
    mexc_ret = np.log(df["mexc_close"] / df["mexc_close"].shift(1)).dropna()

    # Align on common index
    common   = tsla_ret.index.intersection(mexc_ret.index)
    x        = tsla_ret.loc[common].values
    y        = mexc_ret.loc[common].values

    # Remove outliers beyond 5 IQR for a clean plot (keep for stats)
    iqr_x = np.percentile(np.abs(x), 99)
    iqr_y = np.percentile(np.abs(y), 99)
    mask  = (np.abs(x) <= iqr_x) & (np.abs(y) <= iqr_y)

    xp, yp = x[mask], y[mask]

    # Pearson r (on full data)
    r, pval = stats.pearsonr(x, y)

    # OLS regression (on clipped data for line display)
    slope, intercept, *_ = stats.linregress(xp, yp)
    xline = np.linspace(xp.min(), xp.max(), 200)
    yline = slope * xline + intercept

    # KDE density colours
    xy_stack = np.vstack([xp, yp])
    kde_vals  = gaussian_kde(xy_stack)(xy_stack)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    sc = ax.scatter(xp, yp, c=kde_vals, cmap="Blues", s=4, alpha=0.5,
                    rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Density", fontsize=9)

    ax.plot(xline, yline, color="crimson", linewidth=1.8,
            label=f"OLS fit: slope={slope:.3f}")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.4)

    ax.set_xlabel("TSLA 1-min Log Return", fontsize=10)
    ax.set_ylabel("MEXC TESLA_USDT 1-min Log Return", fontsize=10)
    ax.set_title(
        f"Log Return Correlation — TSLA (NYSE) vs MEXC\n"
        f"Pearson r = {r:.4f}   (n = {len(x):,} bars, p < 0.001)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)

    # Annotate stats box
    stats_text = (
        f"r = {r:.4f}\n"
        f"slope = {slope:.4f}\n"
        f"n = {len(x):,}"
    )
    ax.text(0.97, 0.05, stats_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    out_path = out_dir / "log_return_correlation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Chart 3 — Intraday log-spread (2 days)
# ---------------------------------------------------------------------------

def plot_log_spread_intraday(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Two-panel figure showing intraday log-spread and z-score bands for two
    sample trading days.
    """
    days = _get_sample_days(df, n=2, gap=4)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    fig.suptitle("MEXC / TSLA Log-Spread — Intraday Sample (NYSE Hours)",
                 fontsize=13, fontweight="bold")

    for ax, date_str in zip(axes, days):
        day_df = _filter_day(df, date_str)
        times  = day_df.index.tz_convert(_EASTERN)
        spread = day_df["log_spread"]

        # Use rolling mean / std from the full aligned data (already computed)
        roll_mean = day_df["rolling_mean"]
        roll_std  = day_df["rolling_std"]
        upper2    = roll_mean + 2 * roll_std
        lower2    = roll_mean - 2 * roll_std

        ax.plot(times, spread * 100, color=SPREAD_COLOR, linewidth=1.4,
                label="Log-spread (%)")
        ax.plot(times, roll_mean * 100, color="gray", linewidth=1.0,
                linestyle="--", alpha=0.7, label="Rolling mean (90-bar)")
        ax.fill_between(times, lower2 * 100, upper2 * 100,
                        color="gold", alpha=0.2, label="±2σ band")
        ax.plot(times, upper2 * 100, color="goldenrod", linewidth=0.8,
                linestyle=":", alpha=0.8)
        ax.plot(times, lower2 * 100, color="goldenrod", linewidth=0.8,
                linestyle=":", alpha=0.8)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=_EASTERN))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=_EASTERN))
        ax.set_xlabel("Time (ET)", fontsize=9)
        ax.set_ylabel("Log-spread (%)", fontsize=9)

        spread_bps = spread.mean() * 10_000
        ax.set_title(
            f"{date_str}  |  Mean spread: {spread_bps:+.1f} bps  |  {len(day_df)} bars",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f%%"))

    out_path = out_dir / "log_spread_intraday.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Chart 4 — Spread distribution (full sample)
# ---------------------------------------------------------------------------

def plot_spread_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Histogram + KDE of the full-sample log-spread with mean and ±1σ / ±2σ markers.
    """
    spread = df["log_spread"].dropna().values * 100  # convert to %

    mu  = spread.mean()
    sig = spread.std()

    kde_x = np.linspace(spread.min(), spread.max(), 500)
    kde_y = gaussian_kde(spread)(kde_x)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Histogram
    ax.hist(spread, bins=80, density=True, color=SPREAD_COLOR,
            alpha=0.35, edgecolor="white", linewidth=0.4, label="Empirical distribution")

    # KDE
    ax.plot(kde_x, kde_y, color=SPREAD_COLOR, linewidth=2.0, label="KDE")

    # Reference lines
    vline_styles = [
        (mu,        "black",    2.0, "-",  f"Mean ({mu:+.4f}%)"),
        (mu + sig,  "#1976D2",  1.4, "--", f"+1σ ({mu+sig:+.4f}%)"),
        (mu - sig,  "#1976D2",  1.4, "--", f"−1σ ({mu-sig:+.4f}%)"),
        (mu + 2*sig,"#D32F2F",  1.4, ":",  f"+2σ ({mu+2*sig:+.4f}%)"),
        (mu - 2*sig,"#D32F2F",  1.4, ":",  f"−2σ ({mu-2*sig:+.4f}%)"),
    ]
    for val, col, lw, ls, lbl in vline_styles:
        ax.axvline(val, color=col, linewidth=lw, linestyle=ls, alpha=0.85,
                   label=lbl)

    # Fill ±2σ region
    ax.fill_between(kde_x, kde_y,
                    where=(kde_x >= mu - 2*sig) & (kde_x <= mu + 2*sig),
                    color="#E3F2FD", alpha=0.5, label="±2σ region (95.4%)")

    ax.set_xlabel("Log-spread: log(MEXC / TSLA)  ×100  (%)", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.set_title(
        f"Distribution of MEXC/TSLA Log-Spread — Full Sample\n"
        f"n = {len(spread):,} bars  |  "
        f"μ = {mu:+.4f}%  |  σ = {sig:.4f}%  |  "
        f"skew = {float(stats.skew(spread)):.3f}  |  "
        f"kurt = {float(stats.kurtosis(spread)):.3f}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    out_path = out_dir / "spread_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Chart 5 — System design (building-blocks flow)
# ---------------------------------------------------------------------------

def plot_system_design(out_dir: Path) -> Path:
    """
    Building-blocks pipeline diagram.

    Left track  : Core data → compute → save pipeline (7 numbered stages).
    Right track : Streamlit app layer (3 blocks) tapping into the pipeline.
    Both tracks share config.py at the bottom.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    FIG_W, FIG_H = 16, 20
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    # ── colour palette ────────────────────────────────────────────────────────
    CLR = {
        "api":      ("#1565C0", "#E3F2FD"),   # edge, face
        "fetch":    ("#2E7D32", "#E8F5E9"),
        "preproc":  ("#6A1B9A", "#F3E5F5"),
        "signal":   ("#E65100", "#FFF3E0"),
        "engine":   ("#AD1457", "#FCE4EC"),
        "metrics":  ("#00695C", "#E0F2F1"),
        "save":     ("#37474F", "#ECEFF1"),
        "app":      ("#F57F17", "#FFFDE7"),
        "config":   ("#455A64", "#ECEFF1"),
    }
    ARROW_CLR = "#546E7A"
    DARK      = "#212121"
    GRAY      = "#78909C"

    # ── helpers ───────────────────────────────────────────────────────────────
    def block(cx, cy, w, h, num, title, detail_lines, key,
              radius=0.35, fontsize_title=11):
        edge, face = CLR[key]
        patch = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle=f"round,pad={radius}",
            facecolor=face, edgecolor=edge,
            linewidth=2.5, zorder=3,
        )
        ax.add_patch(patch)

        # Step number badge
        badge_r = 0.38
        badge = plt.Circle((cx - w / 2 + 0.55, cy + h / 2 - 0.55),
                            badge_r, color=edge, zorder=5)
        ax.add_patch(badge)
        ax.text(cx - w / 2 + 0.55, cy + h / 2 - 0.55, str(num),
                ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="white", zorder=6)

        # Title
        n_lines = len(detail_lines)
        title_y = cy + (n_lines * 0.32) / 2 if n_lines else cy
        ax.text(cx, title_y, title,
                ha="center", va="center",
                fontsize=fontsize_title, fontweight="bold",
                color=edge, zorder=4)

        # Detail lines
        for i, line in enumerate(detail_lines):
            ly = title_y - 0.50 - i * 0.42
            ax.text(cx, ly, line,
                    ha="center", va="center",
                    fontsize=8.5, color=GRAY, zorder=4,
                    style="italic")

    def vblock(cx, cy, w, h, title, detail_lines, key, radius=0.3):
        """Block without step number (for app track)."""
        edge, face = CLR[key]
        patch = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle=f"round,pad={radius}",
            facecolor=face, edgecolor=edge,
            linewidth=2.0, zorder=3,
        )
        ax.add_patch(patch)
        n_lines = len(detail_lines)
        title_y = cy + (n_lines * 0.32) / 2 if n_lines else cy
        ax.text(cx, title_y, title,
                ha="center", va="center",
                fontsize=10.5, fontweight="bold",
                color=edge, zorder=4)
        for i, line in enumerate(detail_lines):
            ly = title_y - 0.50 - i * 0.42
            ax.text(cx, ly, line,
                    ha="center", va="center",
                    fontsize=8, color=GRAY, zorder=4, style="italic")

    def down_arrow(cx, y_top, y_bot, label=""):
        ax.annotate(
            "", xy=(cx, y_bot), xytext=(cx, y_top),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_CLR,
                            lw=2.0, mutation_scale=18),
            zorder=2,
        )
        if label:
            ax.text(cx + 0.25, (y_top + y_bot) / 2, label,
                    ha="left", va="center", fontsize=7.5,
                    color=GRAY, style="italic")

    def h_arrow(x1, y, x2, label="", rad=0.0):
        ax.annotate(
            "", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle="-|>", color="#F57F17",
                            lw=2.0, mutation_scale=18,
                            connectionstyle=f"arc3,rad={rad}"),
            zorder=2,
        )
        if label:
            ax.text((x1 + x2) / 2, y + 0.18, label,
                    ha="center", va="bottom", fontsize=7.5,
                    color="#F57F17", style="italic")

    def divider(y, label=""):
        ax.axhline(y, color="#CFD8DC", linewidth=1.0, xmin=0.04, xmax=0.96,
                   zorder=1)
        if label:
            ax.text(FIG_W * 0.5, y + 0.12, label,
                    ha="center", va="bottom", fontsize=8,
                    color="#90A4AE", style="italic")

    # =========================================================================
    # TITLE
    # =========================================================================
    ax.text(FIG_W / 2, FIG_H - 0.55,
            "TSLA / MEXC Quant Arbitrage — Building-Blocks Flow",
            ha="center", va="center",
            fontsize=16, fontweight="bold", color=DARK)
    divider(FIG_H - 1.0)

    # ── track x-centres and block geometry ───────────────────────────────────
    LX   = 5.5    # core pipeline x-centre
    RX   = 12.5   # app track x-centre
    BW   = 8.0    # core block width
    ABW  = 5.5    # app block width
    BH   = 2.0    # block height (core)
    ABH  = 2.0    # block height (app)
    GAP  = 0.75   # vertical gap between blocks

    # y-positions (top to bottom)
    ys = [FIG_H - 2.2 - i * (BH + GAP) for i in range(8)]

    # =========================================================================
    # CORE PIPELINE — left track (numbered steps)
    # =========================================================================
    # Column header
    ax.text(LX, ys[0] + BH / 2 + 0.45, "  CORE PIPELINE",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="#37474F",
            bbox=dict(facecolor="#ECEFF1", edgecolor="#90A4AE",
                      boxstyle="round,pad=0.3", linewidth=1.2))

    block(LX, ys[0], BW, BH, 1,
          "DATA SOURCES",
          ["yfinance API  ·  TSLA 1-min OHLCV (NYSE hours, up to 30-day lookback)",
           "MEXC REST API  ·  TESLA_USDT 1-min futures  (contract.mexc.com)"],
          "api")

    down_arrow(LX, ys[0] - BH / 2, ys[1] + BH / 2, "HTTP / REST")

    block(LX, ys[1], BW, BH, 2,
          "FETCH & CACHE",
          ["fetch_tsla()  /  fetch_mexc()  →  src/data/fetcher.py",
           "Append-only Excel cache:  data/raw/tsla_1min.xlsx  +  mexc_1min.xlsx"],
          "fetch")

    down_arrow(LX, ys[1] - BH / 2, ys[2] + BH / 2, "DataFrame (UTC)")

    block(LX, ys[2], BW, BH, 3,
          "ALIGN  ·  PREPROCESS",
          ["align_data()  →  inner join on NYSE hours  (ET-aware, DST-safe)",
           "compute_spread_features()  →  log(MEXC/TSLA)  ·  shift(1) z-score  ·  session markers"],
          "preproc")

    down_arrow(LX, ys[2] - BH / 2, ys[3] + BH / 2)

    block(LX, ys[3], BW, BH, 4,
          "SIGNAL GENERATION",
          ["QuantStrategy.generate_signals()  →  src/strategy/quant_strategy.py",
           "|z| ≥ 2.0σ  →  short_mexc  /  long_mexc  /  flat   (vectorised, lookahead-free)"],
          "signal")

    down_arrow(LX, ys[3] - BH / 2, ys[4] + BH / 2)

    block(LX, ys[4], BW, BH, 5,
          "BACKTEST ENGINE",
          ["BacktestEngine.run()  →  src/backtest/engine.py",
           "Fill at next-bar OPEN  ·  EOD forced exit at CLOSE  ·  MEXC + NYSE fees"],
          "engine")

    down_arrow(LX, ys[4] - BH / 2, ys[5] + BH / 2)

    block(LX, ys[5], BW, BH, 6,
          "METRICS",
          ["compute_metrics()  →  src/backtest/metrics.py",
           "Sharpe  ·  Max drawdown  ·  Win rate  ·  Profit factor  ·  Exit reasons"],
          "metrics")

    down_arrow(LX, ys[5] - BH / 2, ys[6] + BH / 2)

    block(LX, ys[6], BW, BH, 7,
          "SAVE  &  REPORT",
          ["generate_report()  →  src/reporting/report.py",
           "equity_curve.png  ·  backtest_YYYYMMDD.md  ·  trades_YYYYMMDD.csv"],
          "save")

    # =========================================================================
    # APP TRACK — right column
    # =========================================================================
    ax.text(RX, ys[0] + BH / 2 + 0.45, "  STREAMLIT APP  (app/)",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="#F57F17",
            bbox=dict(facecolor="#FFFDE7", edgecolor="#FFB300",
                      boxstyle="round,pad=0.3", linewidth=1.2))

    # App block 1 — UI pages
    ay1 = (ys[0] + ys[1]) / 2
    vblock(RX, ay1, ABW, ABH * 1.5, "UI PAGES",
           ["app/Home.py  ·  1_EDA.py  ·  2_Backtest.py",
            "User sets: date range · entry σ · window",
            "max hold · fees · data source"],
           "app")

    # App block 2 — pipeline.py
    ay2 = (ys[2] + ys[4]) / 2
    vblock(RX, ay2, ABW, ABH * 2.5, "ORCHESTRATOR",
           ["app/helpers/pipeline.py",
            "build_config()  →  Config object",
            "run_pipeline()  →  calls src/ modules",
            "Connects UI params to core pipeline"],
           "app")

    # App block 3 — Display
    ay3 = (ys[5] + ys[6]) / 2
    vblock(RX, ay3, ABW, ABH * 1.5, "DISPLAY",
           ["Equity curve chart",
            "Metrics table  ·  Trade log",
            "Download CSV  /  PDF report"],
           "app")

    # ── App internal arrows ───────────────────────────────────────────────────
    down_arrow(RX, ay1 - ABH * 0.75, ay2 + ABH * 1.25)
    down_arrow(RX, ay2 - ABH * 1.25, ay3 + ABH * 0.75)

    # ── App ↔ Core arrows ─────────────────────────────────────────────────────
    # Orchestrator → core (calls src/ at preprocess level)
    h_arrow(RX - ABW / 2, ay2, LX + BW / 2 + 0.1,
            label="calls src/ modules", rad=0.0)
    # Core metrics → Display
    h_arrow(LX + BW / 2, ys[5], RX - ABW / 2,
            label="PipelineResult", rad=0.0)

    # =========================================================================
    # CONFIG.PY — shared foundation
    # =========================================================================
    cfg_y = ys[6] - BH / 2 - GAP - 0.8
    edge, face = CLR["config"]
    cfg_w, cfg_h = 13.0, 1.4
    ax.add_patch(FancyBboxPatch(
        (FIG_W / 2 - cfg_w / 2, cfg_y - cfg_h / 2), cfg_w, cfg_h,
        boxstyle="round,pad=0.3",
        facecolor=face, edgecolor=edge,
        linewidth=2.0, linestyle="dashed", zorder=3,
    ))
    ax.text(FIG_W / 2, cfg_y + 0.18, "config.py  —  Shared Configuration",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=edge, zorder=4)
    ax.text(FIG_W / 2, cfg_y - 0.22,
            "Config  |  DataConfig  |  StrategyConfig  |  FeeConfig   "
            "(project root — imported by all src/ modules and app/helpers/pipeline.py)",
            ha="center", va="center", fontsize=8.5, color=GRAY,
            style="italic", zorder=4)

    # Dashed line from core pipeline bottom to config
    ax.plot([LX, LX], [ys[6] - BH / 2, cfg_y + cfg_h / 2],
            color="#90A4AE", linewidth=1.2, linestyle=":", zorder=1)

    # =========================================================================
    # DATA FLOW SUMMARY STRIP (bottom)
    # =========================================================================
    strip_y = cfg_y - cfg_h / 2 - 0.55
    ax.text(FIG_W / 2, strip_y,
            "Flow:  API  →  Fetch & Cache  →  Align  →  Z-Score  →  "
            "Signals  →  Backtest  →  Metrics  →  Save / Display",
            ha="center", va="center", fontsize=9,
            color="#546E7A", style="italic")

    out_path = out_dir / "system_design.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")
    return out_path

    # ── colour palette ────────────────────────────────────────────────────────
    C = {
        "api":      "#BBDEFB",
        "fetch":    "#C8E6C9",
        "cache":    "#FFF9C4",
        "core":     "#E1BEE7",
        "strategy": "#FFCCBC",
        "engine":   "#F8BBD0",
        "metrics":  "#B2EBF2",
        "output":   "#DCEDC8",
        "app":      "#FFF3E0",
        "ui":       "#FFE0B2",
    }
    EDGE  = "#546E7A"
    ARROW = "#37474F"
    DARK  = "#212121"

    def box(cx, cy, w, h, label, color, sublabel=None, fontsize=8.5):
        patch = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.012",
            facecolor=color, edgecolor=EDGE,
            linewidth=1.3, zorder=3,
        )
        ax.add_patch(patch)
        dy = h * 0.12 if sublabel else 0
        ax.text(cx, cy + dy, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=DARK, zorder=4)
        if sublabel:
            ax.text(cx, cy - h * 0.22, sublabel,
                    ha="center", va="center", fontsize=fontsize - 1.5,
                    color="#546E7A", style="italic", zorder=4)

    def arrow(x1, y1, x2, y2, label="", rad=0.0):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.4,
                            connectionstyle=f"arc3,rad={rad}"),
            zorder=2,
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.012, my, label, ha="left", va="center",
                    fontsize=7, color="#546E7A", style="italic", zorder=4)

    def section_label(cx, y, text):
        ax.text(cx, y, text, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#37474F",
                bbox=dict(facecolor="#ECEFF1", edgecolor="#90A4AE",
                          boxstyle="round,pad=0.3", linewidth=1.0),
                zorder=4)

    BW, BH = 0.24, 0.058
    NW      = 0.18
    LX, CX, RX = 0.18, 0.50, 0.82

    # ── title ─────────────────────────────────────────────────────────────────
    ax.text(0.50, 0.975,
            "TSLA / MEXC Quant Arbitrage — System Architecture",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=DARK, zorder=4)
    ax.axhline(0.955, color="#B0BEC5", linewidth=0.8)

    # =========================================================================
    # LEFT — Data ingestion
    # =========================================================================
    section_label(LX, 0.925, "① Data Ingestion")

    box(LX - 0.08, 0.865, NW, BH, "yfinance API",   C["api"],
        "TSLA 1-min OHLCV", fontsize=8)
    box(LX + 0.08, 0.865, NW, BH, "MEXC REST API",  C["api"],
        "TESLA_USDT 1-min", fontsize=8)

    box(LX - 0.08, 0.780, NW, BH, "fetch_tsla()",   C["fetch"],
        "src/data/fetcher.py", fontsize=8)
    box(LX + 0.08, 0.780, NW, BH, "fetch_mexc()",   C["fetch"],
        "src/data/fetcher.py", fontsize=8)

    box(LX - 0.08, 0.695, NW, BH, "tsla_1min.xlsx", C["cache"],
        "data/raw/  (append-only)", fontsize=8)
    box(LX + 0.08, 0.695, NW, BH, "mexc_1min.xlsx", C["cache"],
        "data/raw/  (append-only)", fontsize=8)

    arrow(LX - 0.08, 0.836, LX - 0.08, 0.809)
    arrow(LX + 0.08, 0.836, LX + 0.08, 0.809)
    arrow(LX - 0.08, 0.751, LX - 0.08, 0.724)
    arrow(LX + 0.08, 0.751, LX + 0.08, 0.724)

    box(LX, 0.610, BW, BH, "load_excel_tsla/mexc()", C["fetch"],
        "Excel → DataFrame (UTC index)", fontsize=8)
    arrow(LX - 0.08, 0.666, LX - 0.04, 0.639)
    arrow(LX + 0.08, 0.666, LX + 0.04, 0.639)

    # =========================================================================
    # CENTRE — Core pipeline
    # =========================================================================
    section_label(CX, 0.925, "② Core Processing Pipeline  (src/)")

    box(CX, 0.845, BW + 0.06, BH, "align_data()",  C["core"],
        "src/data/preprocessor.py  |  inner join · NYSE hours (ET-aware, DST-safe)",
        fontsize=8)
    box(CX, 0.760, BW + 0.06, BH, "compute_spread_features()",  C["core"],
        "log(MEXC/TSLA)  |  shift(1) z-score  |  session markers",
        fontsize=8)
    box(CX, 0.675, BW + 0.06, BH, "QuantStrategy.generate_signals()",  C["strategy"],
        "src/strategy/quant_strategy.py  |  |z| ≥ 2.0σ  →  short/long/flat",
        fontsize=8)
    box(CX, 0.590, BW + 0.06, BH, "BacktestEngine.run()",  C["engine"],
        "src/backtest/engine.py  |  next-bar-open fill  |  EOD forced exit",
        fontsize=8)
    box(CX, 0.505, BW + 0.06, BH, "compute_metrics()",  C["metrics"],
        "src/backtest/metrics.py  |  Sharpe · drawdown · win rate · profit factor",
        fontsize=8)
    box(CX, 0.415, BW + 0.06, BH + 0.01, "generate_report()",  C["output"],
        "src/reporting/report.py  →  .md report · equity_curve.png · trades.csv",
        fontsize=8)

    arrow(CX, 0.816, CX, 0.789)
    arrow(CX, 0.731, CX, 0.704)
    arrow(CX, 0.646, CX, 0.619)
    arrow(CX, 0.561, CX, 0.534)
    arrow(CX, 0.476, CX, 0.450)

    # load_excel → align_data
    arrow(LX + 0.12, 0.610, CX - 0.16, 0.845, rad=-0.25,
          label="tsla_df, mexc_df")

    # config.py shared reference
    box(CX, 0.320, BW + 0.08, BH, "config.py",  "#ECEFF1",
        "Config  |  DataConfig  |  StrategyConfig  |  FeeConfig  (project root)",
        fontsize=8)
    ax.annotate(
        "", xy=(CX - 0.06, 0.845 - BH / 2),
        xytext=(CX - 0.06, 0.320 + BH / 2),
        arrowprops=dict(arrowstyle="-", color="#90A4AE", lw=1.2,
                        linestyle="dashed",
                        connectionstyle="arc3,rad=0"),
        zorder=1,
    )
    ax.text(CX - 0.10, 0.582, "cfg", ha="center", va="center",
            fontsize=7.5, color="#90A4AE", style="italic")

    # =========================================================================
    # RIGHT — Streamlit app
    # =========================================================================
    section_label(RX, 0.925, "③ Streamlit App  (app/)")

    box(RX, 0.860, BW, BH, "Browser / User",         C["app"],
        "localhost:8501", fontsize=8)
    box(RX, 0.775, BW, BH, "app/Home.py",             C["ui"],
        "Landing page + connectivity check", fontsize=8)
    box(RX, 0.690, BW, BH, "app/pages/1_EDA.py",      C["ui"],
        "load_aligned_data() + compute_features_for_eda()", fontsize=8)
    box(RX, 0.605, BW, BH, "app/pages/2_Backtest.py", C["ui"],
        "Sidebar params  →  run_pipeline()", fontsize=8)
    box(RX, 0.508, BW, BH + 0.01, "app/helpers/pipeline.py", C["app"],
        "build_config()  |  run_pipeline()\norchestrates src/ imports",
        fontsize=8)
    box(RX, 0.395, BW, BH + 0.02, "Streamlit Display", C["output"],
        "Equity curve  |  Metrics table\nTrade log  |  Download CSV",
        fontsize=8)

    arrow(RX, 0.831, RX, 0.804)
    arrow(RX, 0.746, RX, 0.719)
    arrow(RX, 0.661, RX, 0.634)
    arrow(RX, 0.576, RX, 0.538)
    arrow(RX, 0.479, RX, 0.428)

    # pipeline.py calls src/ modules
    arrow(RX - 0.12, 0.508, CX + 0.18, 0.760, rad=0.20,
          label="calls src/ modules")
    # PipelineResult returned
    arrow(CX + 0.18, 0.505, RX - 0.12, 0.495, rad=-0.15,
          label="PipelineResult")

    # =========================================================================
    # Legend
    # =========================================================================
    legend_items = [
        (C["api"],      "External API"),
        (C["fetch"],    "Data fetcher  (src/data/)"),
        (C["cache"],    "Excel cache  (data/raw/)"),
        (C["core"],     "Preprocessing  (src/data/)"),
        (C["strategy"], "Strategy  (src/strategy/)"),
        (C["engine"],   "Backtest engine  (src/backtest/)"),
        (C["metrics"],  "Metrics  (src/backtest/)"),
        (C["output"],   "Output / reporting"),
        (C["ui"],       "Streamlit UI pages"),
        (C["app"],      "App helpers  (app/)"),
    ]
    lx0, ly0 = 0.012, 0.285
    ax.add_patch(FancyBboxPatch(
        (lx0 - 0.008, ly0 - len(legend_items) * 0.026 + 0.003),
        0.178, len(legend_items) * 0.026 + 0.036,
        boxstyle="round,pad=0.005",
        facecolor="white", edgecolor="#B0BEC5",
        linewidth=1.0, zorder=2,
    ))
    ax.text(lx0, ly0 + 0.022, "Legend",
            ha="left", va="center", fontsize=8.5, fontweight="bold",
            color=DARK)
    for i, (color, label) in enumerate(legend_items):
        ly = ly0 - i * 0.026
        ax.add_patch(FancyBboxPatch(
            (lx0, ly - 0.009), 0.022, 0.018,
            boxstyle="round,pad=0.003",
            facecolor=color, edgecolor=EDGE,
            linewidth=0.8, zorder=3,
        ))
        ax.text(lx0 + 0.028, ly, label,
                ha="left", va="center", fontsize=7.5, color=DARK, zorder=4)

    ax.axhline(0.300, color="#B0BEC5", linewidth=0.8, xmin=0.20, xmax=0.80)
    ax.text(
        0.50, 0.270,
        "Data flow:  yfinance / MEXC API  →  Excel cache  →  align  →  "
        "z-score features  →  signals  →  backtest  →  metrics  →  report / UI",
        ha="center", va="center", fontsize=8,
        color="#546E7A", style="italic",
    )

    out_path = out_dir / "system_design.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Chart 6 — Analysis Part 1: Data → Preprocess → Backtest overview
# ---------------------------------------------------------------------------

def plot_analysis_part1(out_dir: Path) -> Path:
    """
    Three-stage visual overview:
      Stage 1 — Data Ingestion: two API sources → running Excel files
      Stage 2 — Preprocessing: align + spread features
      Stage 3 — Backtesting & Reporting: engine + output artefacts
    """
    from matplotlib.patches import FancyBboxPatch

    FW, FH = 14, 20
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    DARK  = "#212121"
    GRAY  = "#546E7A"
    LGRAY = "#90A4AE"

    # Stage colour palette: (header_bg, header_text, body_bg, border)
    STAGES = {
        1: ("#1565C0", "white",   "#E3F2FD", "#1565C0"),   # blue
        2: ("#6A1B9A", "white",   "#F3E5F5", "#6A1B9A"),   # purple
        3: ("#2E7D32", "white",   "#E8F5E9", "#2E7D32"),   # green
    }

    def outer_block(cx, cy, w, h, stage_num, stage_title, border):
        """Draw a large stage block with a coloured header band."""
        hdr_h   = 1.0
        edge    = STAGES[stage_num][3]
        body_bg = STAGES[stage_num][2]
        hdr_bg  = STAGES[stage_num][0]
        hdr_txt = STAGES[stage_num][1]

        # Outer frame
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.18",
            facecolor=body_bg, edgecolor=edge,
            linewidth=2.8, zorder=2,
        ))
        # Header band
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy + h / 2 - hdr_h), w, hdr_h,
            boxstyle="round,pad=0.18",
            facecolor=hdr_bg, edgecolor=edge,
            linewidth=0, zorder=3,
        ))
        # Badge circle
        badge_r = 0.42
        badge = plt.Circle(
            (cx - w / 2 + 0.72, cy + h / 2 - hdr_h / 2),
            badge_r, color="white", zorder=5,
        )
        ax.add_patch(badge)
        ax.text(cx - w / 2 + 0.72, cy + h / 2 - hdr_h / 2,
                str(stage_num),
                ha="center", va="center", fontsize=13,
                fontweight="bold", color=hdr_bg, zorder=6)
        # Stage title
        ax.text(cx + 0.3, cy + h / 2 - hdr_h / 2,
                stage_title,
                ha="center", va="center", fontsize=13,
                fontweight="bold", color=hdr_txt, zorder=5)

    def inner_box(cx, cy, w, h, title, lines, edge, face, fontsize_t=9.5):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=face, edgecolor=edge,
            linewidth=1.8, zorder=4,
        ))
        n = len(lines)
        ty = cy + (n * 0.36) / 2 if n else cy
        ax.text(cx, ty, title,
                ha="center", va="center", fontsize=fontsize_t,
                fontweight="bold", color=edge, zorder=5)
        for i, ln in enumerate(lines):
            ax.text(cx, ty - 0.52 - i * 0.40, ln,
                    ha="center", va="center", fontsize=8.0,
                    color=GRAY, style="italic", zorder=5)

    def v_arrow(cx, y1, y2, label=""):
        ax.annotate("", xy=(cx, y2), xytext=(cx, y1),
                    arrowprops=dict(arrowstyle="-|>", color=LGRAY,
                                   lw=2.2, mutation_scale=20),
                    zorder=2)
        if label:
            ax.text(cx + 0.22, (y1 + y2) / 2, label,
                    ha="left", va="center", fontsize=8,
                    color=LGRAY, style="italic")

    def h_connector(x1, y, x2, label=""):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=LGRAY,
                                   lw=1.8, mutation_scale=16),
                    zorder=2)
        if label:
            ax.text((x1 + x2) / 2, y + 0.2, label,
                    ha="center", va="bottom", fontsize=7.5,
                    color=LGRAY, style="italic")

    # ── layout constants ──────────────────────────────────────────────────────
    CX   = FW / 2       # centre x of all stage blocks
    SW   = 12.5         # stage block width
    IW   = 3.8          # inner box width (narrow)
    IWW  = 5.4          # inner box width (wide)
    STAGE_GAP = 1.2     # vertical gap between stage blocks

    # Stage heights and y-centres (top to bottom)
    SH  = [5.6, 4.2, 5.0]          # heights: stage 1, 2, 3
    sy0 = FH - 1.6                  # top of stage 1
    y1  = sy0 - SH[0] / 2
    y2  = y1 - SH[0] / 2 - STAGE_GAP - SH[1] / 2
    y3  = y2 - SH[1] / 2 - STAGE_GAP - SH[2] / 2

    # =========================================================================
    # TITLE
    # =========================================================================
    ax.text(CX, FH - 0.62,
            "TSLA / MEXC Quant Arbitrage — From Data to Backtest",
            ha="center", va="center",
            fontsize=15, fontweight="bold", color=DARK)
    ax.axhline(FH - 1.05, color="#CFD8DC", linewidth=1.0)

    # =========================================================================
    # STAGE 1 — DATA INGESTION
    # =========================================================================
    outer_block(CX, y1, SW, SH[0], 1,
                "DATA INGESTION", STAGES[1][3])

    # Two API source boxes side-by-side
    src_y = y1 + SH[0] / 2 - 2.2
    inner_box(CX - 3.2, src_y, IW, 1.65,
              "yfinance API",
              ["TSLA 1-min OHLCV",
               "NYSE hours  ·  up to 30-day lookback",
               "Auto-paginated in 7-day chunks"],
              "#1565C0", "#BBDEFB")

    inner_box(CX + 3.2, src_y, IW, 1.65,
              "MEXC REST API",
              ["TESLA_USDT 1-min futures",
               "contract.mexc.com/api/v1/contract/kline",
               "No auth required for market data"],
              "#1565C0", "#BBDEFB")

    # Down arrows from APIs to cache
    cache_y = y1 - SH[0] / 2 + 1.55
    h_connector(CX - 3.2, src_y - 0.82,
                CX - 1.4, cache_y + 0.55)
    h_connector(CX + 3.2, src_y - 0.82,
                CX + 1.4, cache_y + 0.55)

    # Running Excel cache box (wide, centred)
    inner_box(CX, cache_y, IWW + 2.4, 1.9,
              "Running Excel Cache  (data/raw/)",
              ["tsla_1min.xlsx  ·  mexc_1min.xlsx",
               "Append-only: new rows added each run — never overwrites history",
               "Enables analysis beyond the 30-day API limit"],
              "#0D47A1", "#BBDEFB", fontsize_t=10)

    # =========================================================================
    # Arrow Stage 1 → Stage 2
    # =========================================================================
    v_arrow(CX, y1 - SH[0] / 2, y2 + SH[1] / 2, "load_excel_tsla/mexc()")

    # =========================================================================
    # STAGE 2 — PREPROCESSING & ALIGNMENT
    # =========================================================================
    outer_block(CX, y2, SW, SH[1], 2,
                "PREPROCESSING  &  ALIGNMENT", STAGES[2][3])

    align_y = y2 + SH[1] / 2 - 1.95
    inner_box(CX, align_y, IWW + 2.8, 1.65,
              "align_data()",
              ["Inner join TSLA + MEXC on UTC timestamp",
               "Filter to NYSE hours  (9:30–16:00 ET, DST-aware via zoneinfo)",
               "Output: common 1-min bars only  →  same index, same length"],
              "#6A1B9A", "#EDE7F6")

    feat_y = y2 - SH[1] / 2 + 1.55
    inner_box(CX, feat_y, IWW + 2.8, 1.65,
              "compute_spread_features()",
              ["log_spread = log(mexc_close / tsla_close)",
               "shift(1) z-score  →  strictly lookahead-free  (window = 90 bars)",
               "session_end  /  session_cooldown markers added"],
              "#6A1B9A", "#EDE7F6")

    v_arrow(CX, align_y - 0.82, feat_y + 0.82)

    # =========================================================================
    # Arrow Stage 2 → Stage 3
    # =========================================================================
    v_arrow(CX, y2 - SH[1] / 2, y3 + SH[2] / 2, "signal_data DataFrame")

    # =========================================================================
    # STAGE 3 — BACKTESTING & REPORTING
    # =========================================================================
    outer_block(CX, y3, SW, SH[2], 3,
                "BACKTESTING  &  REPORTING", STAGES[3][3])

    # Parameters box (left)
    param_y = y3 + SH[2] / 2 - 2.2
    inner_box(CX - 3.2, param_y, IW + 0.3, 2.2,
              "Parameters",
              ["entry threshold  (σ)",
               "z-score window  (bars)",
               "max holding bars",
               "MEXC bps / NYSE commission"],
              "#2E7D32", "#C8E6C9")

    # Engine box (centre)
    inner_box(CX + 0.4, param_y, IW + 0.5, 2.2,
              "BacktestEngine.run()",
              ["Fill at next-bar OPEN",
               "EOD forced exit at CLOSE",
               "MEXC + NYSE fees deducted",
               "EngineResult + equity curve"],
              "#2E7D32", "#C8E6C9")

    h_connector(CX - 3.2 + (IW + 0.3) / 2,
                param_y,
                CX + 0.4 - (IW + 0.5) / 2,
                "cfg")

    # Output artefacts box (bottom, wide)
    out_y = y3 - SH[2] / 2 + 1.55
    inner_box(CX, out_y, IWW + 2.8, 1.85,
              "reports/   Output Artefacts",
              ["equity_curve_YYYYMMDD.png  —  3-panel PnL + price + z-score chart",
               "backtest_YYYYMMDD.md  —  strategy summary + metrics table + trade log",
               "trades_YYYYMMDD.csv  —  full trade log  (entry/exit prices, PnL, fees, hold time)"],
              "#1B5E20", "#C8E6C9", fontsize_t=10)

    v_arrow(CX, param_y - 1.1, out_y + 0.92)

    # =========================================================================
    # FOOTER
    # =========================================================================
    footer_y = y3 - SH[2] / 2 - 0.55
    ax.text(CX, footer_y,
            "Each stage is independently reusable: load cached Excel → preprocess → "
            "backtest without re-fetching from the API",
            ha="center", va="center", fontsize=8.5,
            color=LGRAY, style="italic")

    out_path = out_dir / "analysis_part1.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all(out_dir: Path | None = None) -> list[Path]:
    """
    Generate all four analysis charts.

    Parameters
    ----------
    out_dir : output directory. Defaults to reports/analysis/ inside project root.

    Returns
    -------
    List of Path objects for the saved PNG files.
    """
    if out_dir is None:
        out_dir = _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[analysis] Loading and aligning data from Excel…")
    df = _load_aligned()
    print(f"[analysis] Dataset: {len(df):,} aligned NYSE-hours bars "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print()

    results = []
    results.append(plot_price_intraday(df, out_dir))
    results.append(plot_log_return_correlation(df, out_dir))
    results.append(plot_log_spread_intraday(df, out_dir))
    results.append(plot_spread_distribution(df, out_dir))
    results.append(plot_system_design(out_dir))
    results.append(plot_analysis_part1(out_dir))

    print(f"\n[analysis] Done. All charts saved to: {out_dir}")
    return results


if __name__ == "__main__":
    generate_all()
