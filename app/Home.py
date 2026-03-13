"""
app/Home.py
===========
Entry point for the TSLA/MEXC Quant Arbitrage Streamlit app.

Run from the quant_arbitrage/ directory:
    streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_APP_DIR      = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA/MEXC Quant Arbitrage",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .big-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0D47A1;
        line-height: 1.2;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #546E7A;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }
    .nav-card {
        background: #F8F9FA;
        border: 1.5px solid #E0E0E0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: border-color 0.2s;
    }
    .nav-card:hover { border-color: #1565C0; }
    .nav-card h3 { color: #1565C0; margin-bottom: 0.4rem; }
    .metric-pill {
        display: inline-block;
        background: #E3F2FD;
        color: #0D47A1;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .metric-pill.green { background: #E8F5E9; color: #2E7D32; }
    .metric-pill.orange { background: #FFF3E0; color: #E65100; }
    .insight-box {
        background: #E8F5E9;
        border-left: 4px solid #43A047;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="big-title">📈 TSLA/MEXC Quant Arbitrage</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Statistical spread mean-reversion · TSLA (NYSE) vs TESLA_USDT (MEXC) · 1-minute bars</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Navigation cards ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="nav-card">
        <h3>🔬 Exploratory Data Analysis</h3>
        <p>Understand <em>why</em> this strategy works through interactive charts:
        price correlation, log-spread stationarity, ADF/Hurst tests,
        OU half-life, and intraday signal patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_EDA.py", label="→ Open EDA", use_container_width=True)

with col2:
    st.markdown("""
    <div class="nav-card">
        <h3>⚡ Backtest Simulator</h3>
        <p>Run a full backtest with configurable parameters: entry threshold,
        z-score window, date range, and fee assumptions.
        Download Markdown report, equity chart, and trade CSV.</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Backtest.py", label="→ Open Backtest", use_container_width=True)

st.divider()

# ── Strategy summary ──────────────────────────────────────────────────────────
st.subheader("Strategy Overview")

c1, c2 = st.columns([1.5, 1])

with c1:
    st.markdown("""
    **The Quant Approach** exploits temporary price divergences between
    TSLA equity (NYSE) and TESLA_USDT perpetual futures (MEXC).

    **Key statistical properties of this sample (Feb–Mar 2026):**
    """)
    st.markdown("""
    <span class="metric-pill green">ρ ≈ 0.92 return correlation</span>
    <span class="metric-pill">ADF p < 0.001 (stationary spread)</span>
    <span class="metric-pill orange">OU half-life ≈ 1.9 min</span>
    <span class="metric-pill green">244 trades · 71.3% win rate</span>
    <span class="metric-pill">$35.52 net PnL (1-share sizing)</span>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("**Execution Model**")
    st.markdown("""
    | Step | Rule |
    |:-----|:-----|
    | Spread | `log(MEXC / TSLA)` |
    | Z-score | `shift(1)` before rolling — no lookahead |
    | Entry | `\|z\| ≥ 2.0σ` → fill at **next bar's OPEN** |
    | Exit | z crosses 0 or 20-bar max hold → **next bar OPEN** |
    | Forced exit | Session end → **current bar CLOSE** |
    | Fees | MEXC: 3 bps/side · NYSE: $0.01/share/side |
    """)

st.divider()


st.info(
    "💡 Data is fetched live from **yfinance** (TSLA) and the **MEXC REST API** (TESLA_USDT) "
    "each time you run. Always up to date."
)

# ── Saved reports ─────────────────────────────────────────────────────────────
reports_dir = _APP_DIR / "reports"
if reports_dir.exists():
    # Each run lives in its own subfolder: reports/YYYYMMDD_HHMMSS/
    run_dirs = sorted(
        [d for d in reports_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
    if run_dirs:
        st.subheader("Recent Reports")
        for rd in run_dirs[:5]:
            ts = rd.name   # YYYYMMDD_HHMMSS
            st.caption(f"📁 {ts}  (generated {ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]} UTC)")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "TSLA/MEXC Quant Arbitrage System · "
)
