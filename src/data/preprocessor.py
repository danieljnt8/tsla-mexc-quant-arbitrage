from __future__ import annotations

import zoneinfo

import numpy as np
import pandas as pd

# NYSE session: 9:30 AM – 4:00 PM ET.
# Using US/Eastern (not hardcoded UTC offsets) so DST transitions are handled
# correctly: EST = UTC-5 (14:30–21:00 UTC), EDT = UTC-4 (13:30–20:00 UTC).
_EASTERN       = zoneinfo.ZoneInfo("America/New_York")
_NYSE_OPEN_ET  = 9 * 60 + 30   # 570 min from midnight ET  (9:30 AM)
_NYSE_CLOSE_ET = 16 * 60        # 960 min from midnight ET  (4:00 PM)


# ---------------------------------------------------------------------------
# Session markers
# ---------------------------------------------------------------------------

def _add_session_markers(df: pd.DataFrame, open_cooldown: int = 0) -> pd.DataFrame:
    """
    Add boolean columns marking session open cooldown and session end bars.

    session_end      — True for the last bar of each NYSE session day.
                       The engine uses this to force-close positions before
                       the market closes, avoiding overnight TSLA equity exposure.

    session_cooldown — True for the first `open_cooldown` bars of each day.
                       The spread is erratic right after the open bell; we
                       suppress signals during this period.
    """
    time_diff = df.index.to_series().diff()

    # Detect session boundaries using time gaps larger than 3× the typical bar gap
    typical_gap = time_diff.dropna().median()
    boundary    = typical_gap * 3

    is_start    = time_diff > boundary
    is_start.iloc[0] = True

    is_end = time_diff.shift(-1) > boundary
    is_end.iloc[-1] = True

    session_id     = is_start.cumsum()
    bar_in_session = df.groupby(session_id).cumcount()

    df["session_end"]      = is_end.values
    df["session_cooldown"] = (bar_in_session < open_cooldown).values
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align_data(tsla_df: pd.DataFrame, mexc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align TSLA and MEXC DataFrames to a common UTC timestamp index,
    filtered to NYSE trading hours and weekdays only.

    Parameters
    ----------
    tsla_df : yfinance output (UTC DatetimeIndex, columns include 'close')
    mexc_df : MEXC output (UTC DatetimeIndex, columns include 'close')

    Returns
    -------
    pd.DataFrame
        Index: open_time (UTC, NYSE hours only)
        Columns: tsla_open, tsla_high, tsla_low, tsla_close, tsla_volume,
                 mexc_open, mexc_high, mexc_low, mexc_close, mexc_volume
    """
    tsla_df.index = pd.to_datetime(tsla_df.index, utc=True)
    mexc_df.index = pd.to_datetime(mexc_df.index, utc=True)

    def _is_nyse_hours(idx: pd.DatetimeIndex) -> pd.Series:
        idx_et  = idx.tz_convert(_EASTERN)
        mins_et = idx_et.hour * 60 + idx_et.minute
        return (
            (idx_et.dayofweek < 5)            # Mon–Fri only
            & (mins_et >= _NYSE_OPEN_ET)      # ≥ 9:30 AM ET (EST: 14:30 UTC, EDT: 13:30 UTC)
            & (mins_et <  _NYSE_CLOSE_ET)     # <  4:00 PM ET (EST: 21:00 UTC, EDT: 20:00 UTC)
        )

    tsla_nyse = tsla_df[_is_nyse_hours(tsla_df.index)]
    mexc_nyse = mexc_df[_is_nyse_hours(mexc_df.index)]

    tsla_renamed = tsla_nyse[["open", "high", "low", "close", "volume"]].rename(
        columns=lambda c: f"tsla_{c}"
    )
    mexc_renamed = mexc_nyse[["open", "high", "low", "close", "volume"]].rename(
        columns=lambda c: f"mexc_{c}"
    )

    merged = tsla_renamed.join(mexc_renamed, how="inner")
    merged = merged.dropna(subset=["tsla_close", "mexc_close"]).sort_index()

    print(
        f"[preprocessor] Aligned {len(merged)} bars "
        f"({merged.index[0]} → {merged.index[-1]})"
    )
    return merged


def compute_spread_features(df: pd.DataFrame, window: int = 90, open_cooldown: int = 0) -> pd.DataFrame:
    """
    Compute log-spread and the Quant approach's strictly lookahead-free z-score.

    The key innovation: we shift the spread back by 1 bar before computing
    the rolling mean and std.  This means at time t, the z-score uses only
    bars [t-window, t-1] — never including bar t itself.

    Comparison with the standard (biased) approach:
        # Standard (mild lookahead — includes current bar in window):
        mean = spread.rolling(W).mean()   # at t: uses [t-W+1, t]

        # Quant approach (strict lookahead-free):
        shifted = spread.shift(1)          # shifted[t] = spread[t-1]
        mean    = shifted.rolling(W).mean()  # at t: uses [t-W, t-1] ✓

    Parameters
    ----------
    df     : aligned DataFrame from align_data()
    window : rolling window in bars (minutes). Default: 90 (≈ 4× OU half-life)

    Adds columns
    ------------
    log_spread    : log(mexc_close / tsla_close)
    shifted_spread: log_spread.shift(1) — used internally for rolling stats
    rolling_mean  : rolling mean of shifted_spread (window bars)
    rolling_std   : rolling std of shifted_spread (window bars)
    z_score       : (log_spread - rolling_mean) / rolling_std

    First (window + 1) bars will have NaN z_score (1 bar from shift + window
    bars from rolling warm-up).
    """
    df = df.copy()

    # --- Step 1: Log-ratio spread ---
    # log(MEXC / TSLA): positive when MEXC is expensive, negative when cheap
    df["log_spread"] = np.log(df["mexc_close"] / df["tsla_close"])

    # --- Step 2: Shift by 1 bar (lookahead prevention) ---
    # At bar t, shifted_spread[t] = log_spread[t-1]
    # So rolling(window) at bar t uses log_spread[t-window .. t-1]
    shifted = df["log_spread"].shift(1)

    # --- Step 3: Rolling mean and std over the shifted series ---
    df["rolling_mean"] = shifted.rolling(window=window, min_periods=window).mean()
    df["rolling_std"]  = shifted.rolling(window=window, min_periods=window).std()

    # --- Step 4: Z-score (today's spread vs. yesterday's distribution) ---
    df["z_score"] = np.where(
        df["rolling_std"] > 1e-10,
        (df["log_spread"] - df["rolling_mean"]) / df["rolling_std"],
        np.nan,
    )

    n_valid = int(df["z_score"].notna().sum())
    warmup  = window + 1
    print(
        f"[preprocessor] Log-spread + shift(1) z-score computed.\n"
        f"  Window: {window} bars | Warm-up: {warmup} bars\n"
        f"  Valid z-scores: {n_valid}/{len(df)} bars "
        f"({n_valid/len(df)*100:.1f}% of data usable)"
    )

    # --- Step 5: Session markers ---
    df = _add_session_markers(df, open_cooldown=open_cooldown)
    return df
