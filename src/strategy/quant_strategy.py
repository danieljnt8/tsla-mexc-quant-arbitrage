"""
src/strategy/quant_strategy.py
============================
Quant Approach (Strategy A) signal generation.

Signal semantics
----------------
  signal[t] in {"short_mexc", "long_mexc", "flat"}

  "short_mexc" at bar t means:
    - z_score[t] >= +entry_threshold
    - MEXC is overpriced relative to TSLA (log-spread above its rolling mean)
    - Trade action: SHORT MEXC, LONG TSLA

  "long_mexc" at bar t means:
    - z_score[t] <= -entry_threshold
    - MEXC is underpriced relative to TSLA
    - Trade action: LONG MEXC, SHORT TSLA

  "flat" means: no entry signal (NaN z, cooldown period, or |z| < threshold)

Execution model
---------------
  The engine reads signal[t] at bar t's close and opens the position at
  bar t+1's OPEN price.  This is the key realism property of the Quant approach:
  you read the signal, send the order, and it fills at the next bar's open.
  The strategy only generates signals — timing of fills is the engine's job.

No abstract base class
----------------------
  Since this project implements exactly one strategy, there is no need for an
  abstract BaseStrategy.  Adding one would be a premature abstraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import StrategyConfig


class QuantStrategy:
    """
    Quant Approach: z-score spread mean reversion with next-bar-open fill.

    Parameters
    ----------
    cfg : StrategyConfig
        Contains entry_threshold, max_holding_bars, open_cooldown.
    """

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized signal generation over the full aligned dataset.

        Rules (applied in priority order):
          1. NaN z-score (warm-up / insufficient history) → "flat"
          2. Session open cooldown bars → "flat"
          3. z >= +entry_threshold  → "short_mexc"
          4. z <= -entry_threshold  → "long_mexc"
          5. Otherwise             → "flat"

        Parameters
        ----------
        data : pd.DataFrame
            Output of compute_spread_features() — must contain z_score,
            and optionally session_cooldown.

        Returns
        -------
        pd.DataFrame
            Same as input with an added 'signal' column.
        """
        data = data.copy()
        z = data["z_score"]

        # Priority ordering via np.select (first match wins)
        conditions = [
            z >= self.cfg.entry_threshold,    # MEXC overpriced → short spread
            z <= -self.cfg.entry_threshold,   # MEXC underpriced → long spread
        ]
        choices = ["short_mexc", "long_mexc"]

        data["signal"] = np.select(conditions, choices, default="flat")

        # NaN z-scores (warm-up bars) → force flat
        data.loc[z.isna(), "signal"] = "flat"

        # Session open cooldown → force flat (open prints are noisy)
        if "session_cooldown" in data.columns:
            data.loc[data["session_cooldown"], "signal"] = "flat"

        # --- Diagnostic summary ---
        n_short = (data["signal"] == "short_mexc").sum()
        n_long  = (data["signal"] == "long_mexc").sum()
        n_total = n_short + n_long
        print(
            f"[strategy] QuantStrategy: {n_total} entry bars "
            f"({n_short} short_mexc, {n_long} long_mexc) "
            f"out of {len(data)} total bars "
            f"(threshold ±{self.cfg.entry_threshold}σ, window={self.cfg.zscore_window})"
        )
        return data

    def __repr__(self) -> str:
        return (
            f"QuantStrategy("
            f"entry=±{self.cfg.entry_threshold}σ, "
            f"window={self.cfg.zscore_window}, "
            f"max_hold={self.cfg.max_holding_bars})"
        )
