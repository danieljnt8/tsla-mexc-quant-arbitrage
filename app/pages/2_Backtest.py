"""
pages/2_Backtest.py
===================
Backtest Simulator page.

Configure parameters, run the full pipeline, view all metrics and charts,
and download reports (PNG, Markdown, CSV) saved to app/reports/.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Optional

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_PAGES_DIR    = Path(__file__).resolve().parent
_APP_DIR      = _PAGES_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

for p in [str(_PROJECT_ROOT), str(_APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd

from helpers.pipeline import run_pipeline, PipelineResult
from helpers.backtest_plots import (
    plot_equity_curve,
    plot_price_with_trades,
    plot_zscore_with_signals,
    plot_pnl_distribution,
    plot_entry_z_vs_pnl,
    plot_holding_bars_vs_pnl,
    plot_exit_breakdown,
    plot_cumulative_pnl_by_direction,
)
from helpers.report_writer import generate_app_report, list_app_reports

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Backtest — TSLA/MEXC Arbitrage",
    page_icon="⚡",
    layout="wide",
)

# ── Sidebar: Parameters ───────────────────────────────────────────────────────
st.sidebar.header("⚙️ Backtest Parameters")

st.sidebar.subheader("Date Range")
DEFAULT_START = date(2026, 2, 12)
DEFAULT_END   = date(2026, 3, 5)

start_date = st.sidebar.date_input("Start Date", value=DEFAULT_START)
end_date   = st.sidebar.date_input("End Date",   value=DEFAULT_END)

st.sidebar.subheader("Strategy Parameters")
entry_threshold = st.sidebar.slider(
    "Entry Threshold (σ)",
    min_value=1.0, max_value=3.5, value=2.0, step=0.1,
    help="Enter when |z-score| ≥ this value. Higher = fewer, better-quality trades.",
)
zscore_window = st.sidebar.slider(
    "Z-Score Window (bars)",
    min_value=50, max_value=200, value=90, step=5,
    help="Rolling window for computing spread z-score. Default 90 ≈ 4× OU half-life.",
)
max_holding_bars = st.sidebar.slider(
    "Max Holding (bars)",
    min_value=1, max_value=60, value=20, step=1,
    help="Force-close position after this many bars. Default 20 (matches reference notebook).",
)
open_cooldown = st.sidebar.slider(
    "Open Cooldown (bars)",
    min_value=0, max_value=30, value=0, step=1,
    help="Suppress signals in first N bars of each session (0 = disabled, matches reference).",
)

st.sidebar.subheader("Fee Assumptions")
mexc_bps = st.sidebar.slider(
    "MEXC Fee (bps/side)",
    min_value=1.0, max_value=10.0, value=3.0, step=0.5,
    help="Total MEXC cost per side: maker fee + slippage. Default 3 bps (2 maker + 1 slip).",
)
nyse_commission = st.sidebar.slider(
    "NYSE Commission ($/share/side)",
    min_value=0.0, max_value=0.10, value=0.01, step=0.005,
    format="$%.3f",
    help="Per-share per-side NYSE commission. Default $0.01 (IBKR/Alpaca tiered). Set to $0.00 for zero-commission brokers.",
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

run_btn = st.sidebar.button(
    "▶  Run Backtest",
    use_container_width=True,
    type="primary",
    help="Runs the full pipeline with the parameters above.",
)

# ── Page header ───────────────────────────────────────────────────────────────
st.title("⚡ Backtest Simulator")
st.caption(
    "Configure parameters, run the pipeline, view results and charts, "
    "then download a full report (Markdown + PNG chart + CSV trade log)."
)

# ── Session state ─────────────────────────────────────────────────────────────
if "bt_result" not in st.session_state:
    st.session_state.bt_result = None
if "bt_params" not in st.session_state:
    st.session_state.bt_params = {}

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    progress_bar = st.progress(0, text="Initializing…")

    try:
        progress_bar.progress(10, text="Fetching data…")
        result = run_pipeline(
            start                     = start_str,
            end                       = end_str,
            entry_threshold           = entry_threshold,
            zscore_window             = zscore_window,
            max_holding_bars          = max_holding_bars,
            mexc_bps_per_side         = mexc_bps,
            nyse_commission_per_share = nyse_commission,
            open_cooldown             = open_cooldown,
            report_dir                = str(_APP_DIR / "reports"),
            use_excel                 = use_excel,
        )
        progress_bar.progress(100, text="Done!")
        st.session_state.bt_result = result
        st.session_state.bt_params = {
            "start": start_str, "end": end_str,
            "entry_threshold": entry_threshold,
            "zscore_window": zscore_window,
            "mexc_bps": mexc_bps,
            "open_cooldown": open_cooldown,
        }
        progress_bar.empty()
        st.success(f"✅ Backtest complete: {len(result.result.trades)} trades in {start_str} → {end_str}")

    except FileNotFoundError as e:
        progress_bar.empty()
        st.error("**No saved Excel data found.**")
        st.warning(
            "Switch to **Fetch live (API)** to download data first. "
            "After a successful fetch, the Excel file is saved automatically "
            "and you can use **Load from saved Excel** for all future runs.\n\n"
            f"Detail: `{e}`"
        )
        st.stop()
    except (ConnectionError, TimeoutError) as e:
        progress_bar.empty()
        st.error("**Cannot reach MEXC API** — this is usually regional blocking.")
        st.warning(
            "**Fix:** Connect to a VPN set to **Singapore** or **US**, then click Run Backtest again.\n\n"
            f"Detail: `{e}`"
        )
        st.stop()
    except ValueError as e:
        progress_bar.empty()
        st.error(f"Backtest failed: {e}")
        st.stop()
    except Exception as e:
        progress_bar.empty()
        st.error(f"Unexpected error: {e}")
        st.stop()

# ── Results area ──────────────────────────────────────────────────────────────
pr: Optional[PipelineResult] = st.session_state.bt_result

if pr is None:
    st.info(
        "👈 Set parameters in the sidebar and click **▶ Run Backtest** to begin.\n\n"
        "**Default settings** replicate the reference notebook exactly:\n"
        "- Period: 2026-02-12 → 2026-03-05\n"
        "- Entry: ±2.0σ · Window: 90 bars · Fees: 3 bps/side\n"
        "- Expected: **244 trades**, **$35.52 net PnL**, **71.3% win rate**"
    )
    st.stop()

result  = pr.result
metrics = pr.metrics
cfg     = pr.cfg
trades  = result.trades

# ── Section 1: Data + Signal Summary ─────────────────────────────────────────
st.subheader("Data & Signal Summary")

data = result.data
n_bars   = len(data)
n_valid  = data["z_score"].notna().sum()
n_short  = (data["signal"] == "short_mexc").sum()
n_long   = (data["signal"] == "long_mexc").sum()
n_flat   = (data["signal"] == "flat").sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Aligned Bars",    f"{n_bars:,}")
c2.metric("Valid Z-scores",  f"{n_valid:,}")
c3.metric("Short MEXC Sigs", f"{n_short:,}")
c4.metric("Long MEXC Sigs",  f"{n_long:,}")
c5.metric("Flat Bars",       f"{n_flat:,}")

st.divider()

# ── Section 2: KPI Metrics ────────────────────────────────────────────────────
if not trades:
    st.warning(
        "No trades were generated with these parameters. "
        "Try lowering the entry threshold or extending the date range."
    )
    st.stop()

st.subheader("Performance Summary")

m = metrics
kpi_cols = st.columns(5)
kpi_cols[0].metric(
    "Net PnL",
    f"${m.get('total_pnl_net_usd', 0):,.2f}",
    delta=f"${m.get('total_fees_usd', 0):,.2f} fees" if m.get("total_fees_usd") else None,
    delta_color="inverse",
)
kpi_cols[1].metric("Trades",    str(m.get("num_trades", 0)))
kpi_cols[2].metric("Win Rate",  f"{m.get('win_rate_pct', 0):.1f}%")
kpi_cols[3].metric("Sharpe",    f"{m.get('sharpe_ratio', 0):.3f}")
kpi_cols[4].metric("Max DD",    f"${m.get('max_drawdown_usd', 0):,.2f}")

kpi2_cols = st.columns(5)
kpi2_cols[0].metric("Gross PnL",       f"${m.get('total_pnl_gross_usd', 0):,.2f}")
kpi2_cols[1].metric("Avg PnL/Trade",   f"${m.get('avg_pnl_per_trade', 0):,.3f}")
kpi2_cols[2].metric("Profit Factor",   f"{m.get('profit_factor', 0):.3f}")
kpi2_cols[3].metric("Avg Hold Time",   f"{m.get('avg_holding_minutes', 0):.1f} min")
kpi2_cols[4].metric("Best / Worst",    f"${m.get('best_trade_usd',0):,.3f} / ${m.get('worst_trade_usd',0):,.3f}")

st.divider()

# ── Section 3: Charts ─────────────────────────────────────────────────────────
st.subheader("Charts")

chart_tab1, chart_tab2, chart_tab3, chart_tab4, chart_tab5 = st.tabs([
    "📈 Equity Curve",
    "💹 Price + Trades",
    "📊 Z-Score",
    "🎯 Trade Analysis",
    "🥧 Exit Breakdown",
])

with chart_tab1:
    st.plotly_chart(plot_equity_curve(result), use_container_width=True)
    st.plotly_chart(plot_cumulative_pnl_by_direction(trades), use_container_width=True)

with chart_tab2:
    st.plotly_chart(plot_price_with_trades(result), use_container_width=True)
    st.caption(
        "▲ = Short MEXC entry (MEXC overpriced) · "
        "▼ = Long MEXC entry (MEXC underpriced) · "
        "× = Exit"
    )

with chart_tab3:
    st.plotly_chart(
        plot_zscore_with_signals(result, entry_threshold=cfg.strategy.entry_threshold),
        use_container_width=True,
    )

with chart_tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_pnl_distribution(trades), use_container_width=True)
    with c2:
        st.plotly_chart(plot_entry_z_vs_pnl(trades), use_container_width=True)
    st.plotly_chart(plot_holding_bars_vs_pnl(trades), use_container_width=True)

with chart_tab5:
    st.plotly_chart(plot_exit_breakdown(metrics), use_container_width=True)
    exit_reasons = metrics.get("exit_reasons", {})
    if exit_reasons:
        st.markdown("**Exit Reason Breakdown:**")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / len(trades) * 100
            st.markdown(f"- **{reason}**: {count} trades ({pct:.1f}%)")

st.divider()

# ── Section 4: Full Metrics Table ────────────────────────────────────────────
with st.expander("📋 Full Metrics Table", expanded=False):
    from src.backtest.metrics import format_metrics_table  # noqa: F401
    st.markdown(format_metrics_table(metrics))

# ── Section 5: Trade Log ──────────────────────────────────────────────────────
with st.expander(f"📜 Trade Log ({len(trades)} trades)", expanded=False):
    trade_rows = []
    for i, t in enumerate(trades, start=1):
        trade_rows.append({
            "#":           i,
            "Entry Time":  t.entry_time.strftime("%Y-%m-%d %H:%M"),
            "Exit Time":   t.exit_time.strftime("%Y-%m-%d %H:%M"),
            "Direction":   t.direction,
            "Entry Z":     round(t.entry_z, 3),
            "Exit Z":      round(t.exit_z, 3),
            "Hold (bars)": t.holding_bars,
            "Exit Reason": t.exit_reason,
            "Gross ($)":   round(t.pnl_gross, 4),
            "Fees ($)":    round(t.pnl_fees, 4),
            "Net ($)":     round(t.pnl_net, 4),
        })

    trade_df = pd.DataFrame(trade_rows)

    def color_pnl(val):
        if isinstance(val, (int, float)):
            color = "#2E7D32" if val > 0 else ("#C62828" if val < 0 else "gray")
            return f"color: {color}; font-weight: bold"
        return ""

    styled = trade_df.style.map(color_pnl, subset=["Net ($)", "Gross ($)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ── Section 6: Download Reports ──────────────────────────────────────────────
st.subheader("📥 Download Reports")
st.markdown(
    "Generate and download the full backtest report. "
    "Files are saved to `app/reports/` and available for future reference."
)

if st.button("Generate Report (PDF + PNG + Markdown + CSV)", type="secondary"):
    with st.spinner("Generating report…"):
        try:
            paths = generate_app_report(
                result = result,
                cfg    = cfg,
                start  = pr.start,
                end    = pr.end,
            )
            st.success("✅ Report saved to `app/reports/`")

            # Primary: PDF (full-width)
            if paths.get("pdf") and paths["pdf"].exists():
                with open(paths["pdf"], "rb") as f:
                    st.download_button(
                        label     = "⬇ Download PDF Report (Summary + Chart + Trade Log)",
                        data      = f.read(),
                        file_name = paths["pdf"].name,
                        mime      = "application/pdf",
                        use_container_width=True,
                        type      = "primary",
                    )

            st.caption("Individual files:")
            col1, col2, col3 = st.columns(3)

            if paths.get("md") and paths["md"].exists():
                with open(paths["md"], "r", encoding="utf-8") as f:
                    md_content = f.read()
                col1.download_button(
                    label     = "⬇ Markdown Report",
                    data      = md_content,
                    file_name = paths["md"].name,
                    mime      = "text/markdown",
                    use_container_width=True,
                )

            if paths.get("png") and paths["png"].exists():
                with open(paths["png"], "rb") as f:
                    png_content = f.read()
                col2.download_button(
                    label     = "⬇ Equity Curve PNG",
                    data      = png_content,
                    file_name = paths["png"].name,
                    mime      = "image/png",
                    use_container_width=True,
                )

            if paths.get("csv") and paths["csv"].exists():
                with open(paths["csv"], "r", encoding="utf-8") as f:
                    csv_content = f.read()
                col3.download_button(
                    label     = "⬇ Trade Log CSV",
                    data      = csv_content,
                    file_name = paths["csv"].name,
                    mime      = "text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Report generation failed: {e}")

# ── Previous Reports ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Previous Reports")

prev_reports = list_app_reports()
if not prev_reports:
    st.info("No reports saved yet. Generate a report above.")
else:
    for rpt in prev_reports[:10]:
        ts = rpt["timestamp"]
        label = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]} UTC"
        with st.expander(f"📄 Report from {label}", expanded=False):
            if rpt.get("pdf") and rpt["pdf"].exists():
                with open(rpt["pdf"], "rb") as f:
                    st.download_button(
                        "⬇ PDF Report",
                        data=f.read(),
                        file_name=rpt["pdf"].name,
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"dl_pdf_{ts}",
                    )
            rcols = st.columns(3)
            if rpt.get("md") and rpt["md"].exists():
                with open(rpt["md"], "r", encoding="utf-8") as f:
                    rcols[0].download_button(
                        "⬇ Markdown",
                        data=f.read(),
                        file_name=rpt["md"].name,
                        mime="text/markdown",
                        key=f"dl_md_{ts}",
                    )
            if rpt.get("png") and rpt["png"].exists():
                with open(rpt["png"], "rb") as f:
                    rcols[1].download_button(
                        "⬇ PNG Chart",
                        data=f.read(),
                        file_name=rpt["png"].name,
                        mime="image/png",
                        key=f"dl_png_{ts}",
                    )
            if rpt.get("csv") and rpt["csv"].exists():
                with open(rpt["csv"], "r", encoding="utf-8") as f:
                    rcols[2].download_button(
                        "⬇ Trade CSV",
                        data=f.read(),
                        file_name=rpt["csv"].name,
                        mime="text/csv",
                        key=f"dl_csv_{ts}",
                    )

st.divider()
st.caption("Backtest page · TSLA/MEXC Quant Arbitrage System · For research purposes only")
