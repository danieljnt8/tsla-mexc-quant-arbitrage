"""
config.py
=========
Single source of truth for all Quant Approach parameters.

Design notes
------------
- backtest_start / backtest_end are NOT stored here — they come from CLI
  args or notebook variables.  The fetcher discovers the stored date range
  from existing parquet filenames and only fetches the gap.
- Fee constants are fixed per the Quant strategy spec (not configurable at
  runtime to avoid accidentally weakening the fee model).
- No notional_per_leg: the engine sizes to "1 TSLA share equivalent" per
  the strategy specification.  Scale positions in your own risk layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Absolute path to the project root (where config.py lives).
# This ensures data/raw and reports/ resolve correctly regardless of
# the current working directory (e.g. running from notebooks/).
_PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Data source and caching configuration."""

    mexc_symbol: str = "TESLA_USDT"
    yf_ticker:   str = "TSLA"
    interval:    str = "Min1"          # MEXC kline interval
    cache_dir:   str = str(_PROJECT_ROOT / "data" / "raw")

    # NOTE: no backtest_start / backtest_end here.
    # Those come from CLI --start / --end or from notebook cells.
    # The fetcher discovers and extends the stored range automatically.


# ---------------------------------------------------------------------------
# StrategyConfig — Quant Approach parameters
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """
    Quant Approach parameters.

    Window / threshold choices are grounded in the research notebook:
    - 90-bar window: ~4× the Ornstein-Uhlenbeck half-life of 1.9 min
    - 2.0σ entry: grid-search optimal, balances trade frequency vs edge
    - 20-bar max hold: OU process reverts in <5 min; 20 min is generous buffer
    """

    zscore_window:    int   = 90    # rolling window in bars (= minutes)
    entry_threshold:  float = 2.0   # enter when |z| >= this value
    max_holding_bars: int   = 20    # force-exit after this many bars
    open_cooldown:    int   = 0     # suppress signals in first N bars of session (0 = disabled, matches reference notebook)

    # Exit threshold is ALWAYS 0.0 for the Quant approach: we exit when z
    # crosses zero (full reversion to rolling mean), not at some partial exit.
    # This is a design constant, not a tunable parameter.
    exit_threshold: float = 0.0


# ---------------------------------------------------------------------------
# FeeConfig — Quant Approach fee model
# ---------------------------------------------------------------------------

@dataclass
class FeeConfig:
    """
    Fee structure for the Quant Approach.

    MEXC: taker fee (2 bps) + slippage (1 bp) = 3 bps/side
    NYSE: stock execution cost ($0.01/share/side)

    Round-trip total per share:
        MEXC: 2 × 3 bps × entry_mexc_price  (entry + exit)
        NYSE: 2 × $0.01                      (entry + exit)
        Total: 6 bps on MEXC notional + $0.02/share NYSE

    At a typical TSLA price of ~$350 and 1-share sizing:
        MEXC fees: 0.06% × $350 ≈ $0.21
        NYSE fees: $0.02
        Total fees per round-trip: ~$0.23 (≈ 8 bps of $350 notional)
    """

    mexc_maker_bps:          float = 0.0002   # 2 bps maker fee per side
    mexc_slippage_bps:       float = 0.0001   # 1 bp slippage per side
    nyse_commission_per_share: float = 0.01   # $0.01/share per side (entry + exit)

    @property
    def mexc_per_side(self) -> float:
        """Total MEXC cost per side (maker + slippage)."""
        return self.mexc_maker_bps + self.mexc_slippage_bps  # 3 bps

    @property
    def mexc_round_trip(self) -> float:
        """Total MEXC cost for a full round-trip (entry + exit)."""
        return 2 * self.mexc_per_side  # 6 bps

    @property
    def nyse_round_trip(self) -> float:
        """Total NYSE cost for a full round-trip (2 × $0.01/share)."""
        return 2 * self.nyse_commission_per_share  # $0.02/share


# ---------------------------------------------------------------------------
# Root Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Root configuration object.  Pass this (or its sub-configs) throughout
    the codebase.  Never scatter magic numbers in strategy/engine code.
    """

    data:       DataConfig     = field(default_factory=DataConfig)
    strategy:   StrategyConfig = field(default_factory=StrategyConfig)
    fees:       FeeConfig      = field(default_factory=FeeConfig)
    report_dir: str            = str(_PROJECT_ROOT / "reports")
