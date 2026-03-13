"""
pages/1_EDA.py
==============
Exploratory Data Analysis page.

Explains WHY the TSLA/MEXC spread arbitrage strategy works through
interactive Plotly charts and statistical test results.

Default date range: 2026-02-12 → 2026-03-05 (the known interesting region
with ρ ≈ 0.92, ADF p < 0.001, OU half-life ≈ 1.9 min).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_PAGES_DIR    = Path(__file__).resolve().parent
_APP_DIR      = _PAGES_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

for p in [str(_PROJECT_ROOT), str(_APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd

from helpers.pipeline import (
    load_aligned_data,
    compute_features_for_eda,
    Config,
)
from helpers.eda_plots import (
    plot_price_overlay,
    plot_individual_prices,
    plot_returns_scatter,
    plot_rolling_correlation,
    plot_log_spread,
    plot_spread_histogram,
    plot_zscore_with_bands,
    plot_rolling_spread_stats,
    compute_adf_stats,
    compute_hurst_ou,
    plot_intraday_zscore_magnitude,
    plot_intraday_signal_count,
    plot_spread_volatility_by_hour,
)
from helpers.pipeline import compute_features_for_eda

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA — TSLA/MEXC Arbitrage",
    page_icon="🔬",
    layout="wide",
)

# ── Sidebar: Controls ─────────────────────────────────────────────────────────
st.sidebar.header("📅 Data Settings")

DEFAULT_START = date(2026, 2, 12)
DEFAULT_END   = date(2026, 3, 5)

start_date = st.sidebar.date_input(
    "Start Date",
    value=DEFAULT_START,
    help="Default: 2026-02-12 — known interesting region (ρ≈0.92, ADF p<0.001)",
)
end_date = st.sidebar.date_input(
    "End Date",
    value=DEFAULT_END,
    help="Default: 2026-03-05 — 15 trading days of data",
)
st.sidebar.divider()
zscore_window = st.sidebar.slider(
    "Z-Score Window (bars)",
    min_value=30, max_value=200, value=90, step=5,
    help="Rolling window for spread z-score. Default 90 ≈ 4× OU half-life.",
)
entry_threshold = st.sidebar.number_input(
    "Entry Threshold (σ)",
    min_value=1.0, max_value=4.0, value=2.0, step=0.1,
    help="Z-score threshold to highlight entry signals on charts.",
)

st.sidebar.divider()
data_source = st.sidebar.radio(
    "Data Source",
    ["Fetch live (API)", "Load from saved Excel"],
    help=(
        "**Fetch live**: calls yfinance + MEXC API (requires VPN in some regions).\n\n"
        "**Load from saved Excel**: uses data/raw/tsla_1min.xlsx & mexc_1min.xlsx "
        "saved from a previous fetch — no internet required."
    ),
)
use_excel = data_source == "Load from saved Excel"

load_btn = st.sidebar.button("🔄 Load / Refresh Data", use_container_width=True, type="primary")

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🔬 Exploratory Data Analysis")
st.caption(
    "This page explains **why** the TSLA/MEXC spread arbitrage works. "
    "Explore price correlation, spread stationarity, and statistical tests."
)

# ── Session state ─────────────────────────────────────────────────────────────
if "eda_df" not in st.session_state:
    st.session_state.eda_df = None
if "eda_featured" not in st.session_state:
    st.session_state.eda_featured = None
if "eda_start" not in st.session_state:
    st.session_state.eda_start = None
if "eda_end" not in st.session_state:
    st.session_state.eda_end = None

# ── Data loading logic ────────────────────────────────────────────────────────
start_str = start_date.strftime("%Y-%m-%d")
end_str   = end_date.strftime("%Y-%m-%d")

# Auto-load on first visit or when button is pressed
need_load = (
    load_btn
    or st.session_state.eda_df is None
    or st.session_state.eda_start != start_str
    or st.session_state.eda_end   != end_str
)

if need_load:
    with st.spinner(f"Fetching and aligning data ({start_str} → {end_str})…"):
        try:
            aligned_df = load_aligned_data(
                start=start_str,
                end=end_str,
                use_excel=use_excel,
            )
            if aligned_df.empty:
                st.error("No data returned for this date range. Try expanding the dates.")
                st.stop()

            featured_df = compute_features_for_eda(
                aligned_df,
                window=zscore_window,
                open_cooldown=0,
            )
            st.session_state.eda_df      = aligned_df
            st.session_state.eda_featured = featured_df
            st.session_state.eda_start   = start_str
            st.session_state.eda_end     = end_str

        except FileNotFoundError as e:
            st.error("**No saved Excel data found.**")
            st.warning(
                "Switch to **Fetch live (API)** to download data first. "
                "After a successful fetch, the Excel file will be saved automatically "
                "and you can use **Load from saved Excel** for all future runs.\n\n"
                f"Detail: `{e}`"
            )
            st.stop()
        except (ConnectionError, TimeoutError) as e:
            st.error("**Cannot reach MEXC API** — this is usually regional blocking.")
            st.warning(
                "**Fix:** Connect to a VPN set to **Singapore** or **US**, then click Load / Refresh Data again.\n\n"
                f"Detail: `{e}`"
            )
            st.stop()
        except Exception as e:
            st.error(f"Data loading failed: {e}")
            st.stop()

df       = st.session_state.eda_df
feat_df  = st.session_state.eda_featured

if df is None or df.empty:
    st.info("Click **Load / Refresh Data** in the sidebar to begin.")
    st.stop()

# ── Data summary bar ─────────────────────────────────────────────────────────
n_bars   = len(df)
n_days   = df.index.normalize().nunique()
first_ts = df.index[0].strftime("%Y-%m-%d %H:%M")
last_ts  = df.index[-1].strftime("%Y-%m-%d %H:%M")
valid_z  = feat_df["z_score"].notna().sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Aligned Bars",   f"{n_bars:,}")
col2.metric("Trading Days",   f"{n_days}")
col3.metric("First Bar",      first_ts)
col4.metric("Last Bar",       last_ts)
col5.metric("Valid Z-Scores", f"{valid_z:,} ({valid_z/n_bars*100:.0f}%)")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Structure",
    "📉 Spread Analysis",
    "🧪 Statistical Tests",
    "⏰ Intraday Patterns",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: MARKET STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Price Correlation: TSLA and TESLA_USDT")
    st.markdown("""
    The two instruments represent the **same underlying asset (Tesla)**.
    During NYSE hours, they should price identically. Deviations are the
    source of our trading edge.
    """)

    st.plotly_chart(plot_price_overlay(df), use_container_width=True)

    st.divider()

    st.subheader("Individual Price Series")
    st.plotly_chart(plot_individual_prices(df), use_container_width=True)

    st.divider()

    st.subheader("Return Correlation Analysis")
    c1, c2 = st.columns([2, 1])

    with c1:
        st.plotly_chart(plot_returns_scatter(df), use_container_width=True)

    with c2:
        # Compute correlation stats
        tsla_ret = df["tsla_close"].pct_change().dropna()
        mexc_ret = df["mexc_close"].pct_change().dropna()
        corr_df  = pd.concat([tsla_ret.rename("tsla"), mexc_ret.rename("mexc")], axis=1).dropna()
        rho      = corr_df["tsla"].corr(corr_df["mexc"])

        st.markdown("**Correlation Statistics**")
        st.metric("Pearson ρ (returns)", f"{rho:.4f}", help="1-min return correlation over the full sample")
        st.metric("Sample Size", f"{len(corr_df):,} bars")
        st.metric("TSLA Mean Return", f"{corr_df['tsla'].mean()*100:.4f}%")
        st.metric("MEXC Mean Return", f"{corr_df['mexc'].mean()*100:.4f}%")

        if rho > 0.90:
            st.success(f"✅ Very high correlation ({rho:.2f}).")
        elif rho > 0.80:
            st.info(f"ℹ️ High correlation ({rho:.2f}) — strategy may work but monitor for regime changes.")
        else:
            st.warning(f"⚠️ Low correlation ({rho:.2f}) — spread may not be reliable in this period.")

    st.divider()
    st.subheader("Rolling Correlation Stability")
    st.markdown("A stable rolling correlation above 0.80 indicates the pair relationship is robust.")
    st.plotly_chart(plot_rolling_correlation(df, window=60), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SPREAD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Log-Spread: log(MEXC / TSLA)")
    st.markdown("""
    The **log-ratio spread** is symmetric: a +5% MEXC premium and a −5% discount
    produce equal-magnitude z-scores.  A stationary spread (one that reverts to its mean)
    is the foundational requirement for mean-reversion trading.
    """)

    st.plotly_chart(plot_log_spread(feat_df), use_container_width=True)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Spread Distribution")
        st.markdown("Approximately normal distribution supports the z-score approach.")
        st.plotly_chart(plot_spread_histogram(feat_df), use_container_width=True)

    with c2:
        st.subheader("Rolling Mean & Std")
        st.markdown("Stable rolling statistics confirm stationarity over time.")
        st.plotly_chart(plot_rolling_spread_stats(feat_df), use_container_width=True)

    st.divider()

    st.subheader(f"Z-Score with ±{entry_threshold}σ Entry Bands")
    st.markdown(f"""
    When `|z| ≥ {entry_threshold}σ`, the spread is statistically anomalous:
    - **z ≥ +{entry_threshold}** (blue ▲): MEXC overpriced → **SHORT MEXC, LONG TSLA**
    - **z ≤ −{entry_threshold}** (red ▼): MEXC underpriced → **LONG MEXC, SHORT TSLA**
    """)

    # Generate signals for visualization
    from helpers.pipeline import QuantStrategy, Config as _Config
    _cfg = _Config()
    _cfg.strategy.entry_threshold = entry_threshold
    _cfg.strategy.zscore_window   = zscore_window
    _strategy = QuantStrategy(_cfg.strategy)
    _signal_data = _strategy.generate_signals(feat_df)

    st.plotly_chart(plot_zscore_with_bands(_signal_data, entry_threshold), use_container_width=True)

    # Signal count summary
    n_short = (_signal_data["signal"] == "short_mexc").sum()
    n_long  = (_signal_data["signal"] == "long_mexc").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Short MEXC Signals", f"{n_short}")
    c2.metric("Long MEXC Signals",  f"{n_long}")
    c3.metric("Total Entry Signals", f"{n_short + n_long}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    spread_series = feat_df["log_spread"].dropna()

    st.subheader("Augmented Dickey-Fuller Test (Stationarity)")
    st.markdown("""
    The ADF test checks whether the spread has a **unit root** (non-stationary)
    or is stationary (mean-reverting).
    - **p < 0.05** → reject unit root → spread is **stationary** ✅
    - **p > 0.05** → fail to reject → spread may not be mean-reverting ⚠️
    """)

    with st.spinner("Running ADF test…"):
        adf = compute_adf_stats(spread_series)

    if "error" in adf:
        st.error(f"ADF test failed: {adf['error']}")
    else:
        cols = st.columns(4)
        cols[0].metric("Test Statistic",  f"{adf['test_stat']:.4f}")
        cols[1].metric("p-value",         f"{adf['p_value']:.2e}",
                        delta="Stationary" if adf["is_stationary"] else "NOT Stationary",
                        delta_color="normal" if adf["is_stationary"] else "inverse")
        cols[2].metric("Lags Used",       str(adf["lags_used"]))
        cols[3].metric("Observations",    f"{adf['n_obs']:,}")

        cv = adf["critical_values"]
        st.markdown("**Critical Values:**")
        cv_cols = st.columns(3)
        for i, (level, val) in enumerate(cv.items()):
            cv_cols[i % 3].metric(f"Critical @ {level}", f"{val:.4f}",
                                   delta="Test stat below → stationary" if adf["test_stat"] < val else None)

        if adf["is_stationary"]:
            st.success(
                f"✅ **Stationary spread confirmed** (p = {adf['p_value']:.2e} << 0.05). "
                "The spread reverts to its mean — the core requirement for this strategy."
            )
        else:
            st.warning(
                f"⚠️ **Spread may not be stationary** in this period (p = {adf['p_value']:.4f}). "
                "Results may not be reliable."
            )

    st.divider()

    


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: INTRADAY PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Intraday Signal Distribution")
    st.markdown("""
    Understanding **when** signals occur during the trading day helps
    identify optimal trading windows and potential structural risks
    (e.g. open volatility, lunch lull, close volatility).
    """)

    # Need signal data
    from helpers.pipeline import QuantStrategy, Config as _Cfg2
    _cfg2 = _Cfg2()
    _cfg2.strategy.entry_threshold = entry_threshold
    _cfg2.strategy.zscore_window   = zscore_window
    _s2   = QuantStrategy(_cfg2.strategy)
    _sd2  = _s2.generate_signals(feat_df)

    st.plotly_chart(plot_intraday_signal_count(_sd2), use_container_width=True)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("|Z-Score| Magnitude by Minute")
        st.markdown("""
        Shows when the spread is most extreme during the trading day.
        Higher values = larger spread = potentially larger trades.
        """)
        st.plotly_chart(plot_intraday_zscore_magnitude(feat_df), use_container_width=True)

    with c2:
        st.subheader("Spread Volatility by Hour")
        st.markdown("""
        Log-spread standard deviation by hour.
        Higher volatility = more entry opportunities but also more risk.
        """)
        st.plotly_chart(plot_spread_volatility_by_hour(feat_df), use_container_width=True)

    st.divider()

    st.subheader("Session Summary Statistics")
    # Per-session (per-day) summary
    feat_df2 = feat_df.copy()
    et_index = feat_df2.index.tz_convert("America/New_York") if feat_df2.index.tz else feat_df2.index - pd.Timedelta(hours=5)
    feat_df2["et_date"] = et_index.date

    daily_stats = feat_df2.groupby("et_date").agg(
        bars=("z_score", "count"),
        abs_z_mean=("z_score", lambda x: x.abs().mean()),
        abs_z_max=("z_score", lambda x: x.abs().max()),
        spread_std=("log_spread", "std"),
    ).reset_index()
    daily_stats.columns = ["Date", "Bars", "Avg |Z|", "Max |Z|", "Spread Std"]
    daily_stats = daily_stats.round(4)

    st.dataframe(daily_stats, use_container_width=True, hide_index=True)

st.divider()
st.caption("EDA page · TSLA/MEXC Quant Arbitrage System · For research purposes only")
