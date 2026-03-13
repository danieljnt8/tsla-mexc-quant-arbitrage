"""
app/helpers/pipeline.py
=======================
Thin wrappers around the quant_arbitrage src/ pipeline.

No logic changes — only orchestration so the Streamlit pages don't need to
manage sys.path or import order themselves.

All functions add the quant_arbitrage project root to sys.path so that
`from config import Config` and `from src.data.fetcher import ...` work
exactly as they do in main.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# ── sys.path bootstrap ────────────────────────────────────────────────────────
# app/ is one level below quant_arbitrage/, so parent.parent is the project root
_APP_DIR     = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── project imports (safe after sys.path patch) ───────────────────────────────
from config import Config, DataConfig, StrategyConfig, FeeConfig   # noqa: E402
from src.data.fetcher import (                                          # noqa: E402
    fetch_tsla, fetch_mexc, test_connectivity,
    load_excel_tsla, load_excel_mexc,
)
from src.data.preprocessor import align_data, compute_spread_features  # noqa: E402
from src.strategy.quant_strategy import QuantStrategy                   # noqa: E402
from src.backtest.engine import BacktestEngine, EngineResult            # noqa: E402
from src.backtest.metrics import compute_metrics                        # noqa: E402


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(
    entry_threshold: float = 2.0,
    zscore_window: int = 90,
    max_holding_bars: int = 20,
    mexc_bps_per_side: float = 3.0,
    nyse_commission_per_share: float = 0.01,
    open_cooldown: int = 0,
    report_dir: Optional[str] = None,
) -> Config:
    """
    Build a Config object from UI parameters.

    `mexc_bps_per_side` is split equally into maker (2/3) and slippage (1/3)
    to preserve the FeeConfig structure, but the total per-side rate is what
    the user controls.

    `nyse_commission_per_share` is the per-share per-side NYSE commission in USD.
    Default $0.01/share/side matches the reference notebook.
    """
    cfg = Config()
    cfg.strategy.entry_threshold  = entry_threshold
    cfg.strategy.zscore_window    = zscore_window
    cfg.strategy.max_holding_bars = max_holding_bars
    cfg.strategy.open_cooldown    = open_cooldown

    # Split user-supplied bps evenly into maker + slippage
    per_side_frac = mexc_bps_per_side / 10_000
    # Keep the 2:1 maker/slip ratio from the reference model
    cfg.fees.mexc_maker_bps          = per_side_frac * (2 / 3)
    cfg.fees.mexc_slippage_bps       = per_side_frac * (1 / 3)
    cfg.fees.nyse_commission_per_share = nyse_commission_per_share

    if report_dir:
        cfg.report_dir = report_dir

    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data(
    start: str,
    end: str,
    cfg: Optional[Config] = None,
    use_excel: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load TSLA and MEXC raw OHLCV DataFrames.

    Parameters
    ----------
    use_excel : if True, load from saved Excel files (no API call).
                If False (default), fetch live from yfinance / MEXC API.

    Returns
    -------
    (tsla_df, mexc_df)  — raw, unaligned DataFrames
    """
    if cfg is None:
        cfg = Config()
    if use_excel:
        tsla_df = load_excel_tsla(cfg.data, start=start, end=end)
        mexc_df  = load_excel_mexc(cfg.data, start=start, end=end)
    else:
        tsla_df = fetch_tsla(cfg.data, start=start, end=end)
        mexc_df  = fetch_mexc(cfg.data, start=start, end=end)
    return tsla_df, mexc_df


def load_aligned_data(
    start: str,
    end: str,
    cfg: Optional[Config] = None,
    use_excel: bool = False,
) -> pd.DataFrame:
    """
    Load + align TSLA and MEXC to NYSE trading hours.

    Parameters
    ----------
    use_excel : if True, load from saved Excel files instead of fetching live.

    Returns
    -------
    aligned DataFrame with columns:
        tsla_open/high/low/close/volume, mexc_open/high/low/close/volume
    """
    tsla_df, mexc_df = load_raw_data(start, end, cfg, use_excel=use_excel)
    if tsla_df.empty or mexc_df.empty:
        return pd.DataFrame()
    return align_data(tsla_df, mexc_df)


def compute_features_for_eda(
    aligned_df: pd.DataFrame,
    window: int = 90,
    open_cooldown: int = 0,
) -> pd.DataFrame:
    """
    Compute log-spread + shift(1) z-score + session markers.

    Wrapper around compute_spread_features for the EDA page.
    """
    return compute_spread_features(aligned_df, window=window, open_cooldown=open_cooldown)


# ---------------------------------------------------------------------------
# Full backtest pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Return type for run_pipeline()."""
    result:  EngineResult
    metrics: dict
    cfg:     Config
    start:   str
    end:     str


def run_pipeline(
    start: str,
    end: str,
    entry_threshold: float = 2.0,
    zscore_window: int = 90,
    max_holding_bars: int = 20,
    mexc_bps_per_side: float = 3.0,
    nyse_commission_per_share: float = 0.01,
    open_cooldown: int = 0,
    report_dir: Optional[str] = None,
    use_excel: bool = False,
) -> PipelineResult:
    """
    Run the complete backtest pipeline:
      fetch → align → features → signals → backtest → metrics

    Parameters
    ----------
    start / end          : date strings YYYY-MM-DD
    entry_threshold      : z-score entry threshold (default 2.0)
    zscore_window        : rolling window in bars (default 90)
    mexc_bps_per_side    : MEXC total cost per side in bps (default 3)
    open_cooldown        : suppress signals in first N bars of session (default 0)
    report_dir           : override report output directory

    Returns
    -------
    PipelineResult with .result, .metrics, .cfg, .start, .end
    """
    cfg = build_config(
        entry_threshold           = entry_threshold,
        zscore_window             = zscore_window,
        max_holding_bars          = max_holding_bars,
        mexc_bps_per_side         = mexc_bps_per_side,
        nyse_commission_per_share = nyse_commission_per_share,
        open_cooldown             = open_cooldown,
        report_dir                = report_dir,
    )

    # Step 1: Fetch
    tsla_df, mexc_df = load_raw_data(start, end, cfg, use_excel=use_excel)

    if tsla_df.empty or mexc_df.empty:
        raise ValueError("One or both data sources returned empty DataFrames.")

    # Step 2: Align
    aligned = align_data(tsla_df, mexc_df)

    min_required = cfg.strategy.zscore_window * 2
    if len(aligned) < min_required:
        raise ValueError(
            f"Only {len(aligned)} aligned bars — need >{min_required}. "
            "Extend the date range."
        )

    # Step 3: Features
    featured = compute_spread_features(
        aligned,
        window       = cfg.strategy.zscore_window,
        open_cooldown = cfg.strategy.open_cooldown,
    )

    # Step 4: Signals
    strategy    = QuantStrategy(cfg.strategy)
    signal_data = strategy.generate_signals(featured)

    # Step 5: Backtest
    engine = BacktestEngine(cfg)
    result = engine.run(signal_data)

    # Step 6: Metrics
    metrics = compute_metrics(result)

    return PipelineResult(result=result, metrics=metrics, cfg=cfg, start=start, end=end)
