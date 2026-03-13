from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from config import DataConfig

# Fixed filenames inside cfg.cache_dir
_TSLA_EXCEL = "tsla_1min.xlsx"
_MEXC_EXCEL = "mexc_1min.xlsx"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEXC_BASE_URL = "https://contract.mexc.com"
MEXC_PING_URL = f"{MEXC_BASE_URL}/api/v1/contract/ping"


# ---------------------------------------------------------------------------
# Connectivity test
# ---------------------------------------------------------------------------

def test_connectivity() -> None:
    """
    Verify that both MEXC API and Yahoo Finance are reachable before fetching.

    Raises
    ------
    ConnectionError
        With a descriptive message identifying which service failed.
    """
    try:
        resp = requests.get(MEXC_PING_URL, timeout=10)
        resp.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise ConnectionError(
            f"[fetcher] MEXC API unreachable — this is usually caused by regional blocking.\n"
            f"  URL: {MEXC_PING_URL}\n"
            f"  Error: {e}\n"
            f"  → Connect to a VPN (Singapore or US server) and retry."
        ) from e
    except requests.exceptions.HTTPError as e:
        raise ConnectionError(f"[fetcher] MEXC API returned HTTP error: {e}") from e

    try:
        t = yf.Ticker("TSLA")
        _ = t.fast_info["lastPrice"]
    except Exception as e:
        raise ConnectionError(
            f"[fetcher] Yahoo Finance unreachable or TSLA data unavailable.\n  Error: {e}"
        ) from e

    print("[fetcher] Connectivity OK — MEXC API and Yahoo Finance reachable.")


# ---------------------------------------------------------------------------
# Preview helper
# ---------------------------------------------------------------------------

def _print_preview(df: pd.DataFrame, label: str) -> None:
    print(f"\n[fetcher] {label} — shape: {df.shape}")
    print(df.head(5).to_string())
    print()


# ---------------------------------------------------------------------------
# Excel persistence helpers
# ---------------------------------------------------------------------------

def _save_excel(df: pd.DataFrame, path: Path) -> None:
    """
    Append new rows to an Excel file, deduplicating by index (open_time).

    If the file already exists, existing rows are merged with the new data
    so the file grows incrementally over time without duplicates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_excel(path, index_col=0, parse_dates=True)
        existing.index = pd.to_datetime(existing.index, utc=True)
        existing.index.name = "open_time"
        combined = pd.concat([existing, df])
    else:
        combined = df.copy()

    combined = combined[~combined.index.duplicated()].sort_index()

    # openpyxl cannot store timezone-aware datetimes — strip tz before saving
    to_save = combined.copy()
    to_save.index = to_save.index.tz_localize(None)
    to_save.to_excel(path)
    print(f"[fetcher] Saved {len(to_save):,} rows → {path}")


# ---------------------------------------------------------------------------
# Excel load functions (public API)
# ---------------------------------------------------------------------------

def load_excel_tsla(
    cfg: DataConfig,
    start: Optional[str] = None,
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Load TSLA 1-min OHLCV from the saved Excel file.

    Parameters
    ----------
    cfg   : DataConfig (used for cache_dir path)
    start : optional filter — return rows >= start (YYYY-MM-DD)
    end   : optional filter — return rows <= end (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame — Index: open_time (UTC), Columns: open, high, low, close, volume

    Raises
    ------
    FileNotFoundError if no Excel file exists yet (run fetch_tsla first).
    """
    path = Path(cfg.cache_dir) / _TSLA_EXCEL
    if not path.exists():
        raise FileNotFoundError(
            f"[fetcher] No saved TSLA data found at {path}.\n"
            f"  → Run fetch_tsla() at least once to create the file."
        )

    df = pd.read_excel(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "open_time"

    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]

    print(f"[fetcher] Loaded TSLA from Excel: {len(df):,} rows"
          + (f" | {df.index[0]} → {df.index[-1]}" if not df.empty else " (empty after filter)"))
    return df


def load_excel_mexc(
    cfg: DataConfig,
    start: Optional[str] = None,
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Load MEXC TESLA_USDT 1-min OHLCV from the saved Excel file.

    Parameters
    ----------
    cfg   : DataConfig (used for cache_dir path)
    start : optional filter — return rows >= start (YYYY-MM-DD)
    end   : optional filter — return rows <= end (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame — Index: open_time (UTC), Columns: open, high, low, close, volume, amount

    Raises
    ------
    FileNotFoundError if no Excel file exists yet (run fetch_mexc first).
    """
    path = Path(cfg.cache_dir) / _MEXC_EXCEL
    if not path.exists():
        raise FileNotFoundError(
            f"[fetcher] No saved MEXC data found at {path}.\n"
            f"  → Run fetch_mexc() at least once to create the file."
        )

    df = pd.read_excel(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "open_time"

    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]

    print(f"[fetcher] Loaded MEXC from Excel: {len(df):,} rows"
          + (f" | {df.index[0]} → {df.index[-1]}" if not df.empty else " (empty after filter)"))
    return df


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

def _yf_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance DataFrame to standard schema."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "open_time"
    return df.sort_index()


def _yf_fetch_range(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch 1-min yfinance bars over a date range, paginating in 7-day chunks.
    Max lookback from today: ~30 calendar days.
    """
    end_dt   = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)

    t = yf.Ticker(ticker)
    frames, cursor = [], start_dt

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=7), end_dt)
        s = cursor.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        print(f"[yfinance] Fetching {ticker} {s} → {e} ...")
        df = t.history(start=s, end=e, interval="1m", auto_adjust=True)
        if not df.empty:
            frames.append(_yf_clean(df))
        cursor = chunk_end

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    return combined[~combined.index.duplicated()].sort_index()


# ---------------------------------------------------------------------------
# MEXC REST helpers
# ---------------------------------------------------------------------------

def _mexc_fetch_chunk(symbol: str, interval: str, start: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch up to 2000 klines from MEXC REST API.
    Retries up to 3 times with increasing timeouts.

    If this consistently times out, MEXC is likely blocked in your region.
    Use a VPN (e.g. set to Singapore or US) and retry.
    """
    params: dict = {"interval": interval}
    if start is not None:
        params["start"] = start

    url = f"{MEXC_BASE_URL}/api/v1/contract/kline/{symbol}"
    timeouts = [15, 30, 60]   # seconds — progressively longer per retry

    last_err: Exception = RuntimeError("No attempts made")
    for attempt, timeout in enumerate(timeouts, start=1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            break  # success — exit retry loop
        except requests.exceptions.Timeout:
            last_err = TimeoutError(
                f"MEXC API timed out after {timeout}s (attempt {attempt}/{len(timeouts)}).\n"
                f"  URL: {url}\n"
                f"  → MEXC may be blocked in your region. Try using a VPN (Singapore / US)."
            )
            print(f"[MEXC] Timeout on attempt {attempt} ({timeout}s). "
                  + ("Retrying..." if attempt < len(timeouts) else "Giving up."))
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"[MEXC] Cannot connect to {url}\n"
                f"  Error: {e}\n"
                f"  → MEXC may be blocked in your region. Try using a VPN (Singapore / US)."
            ) from e
    else:
        raise last_err

    data = resp.json()
    if not data.get("success"):
        raise ValueError(f"[MEXC] API error: {data}")

    d = data["data"]
    df = pd.DataFrame({
        "open_time": d["time"],
        "open":      d["open"],
        "high":      d["high"],
        "low":       d["low"],
        "close":     d["close"],
        "volume":    d["vol"],
        "amount":    d["amount"],
    })
    df["open_time"] = pd.to_datetime(df["open_time"], unit="s", utc=True)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = df[col].astype(float)

    return df.set_index("open_time").sort_index()


def _mexc_fetch_range(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """Paginate MEXC klines over an arbitrary date range."""
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    start_s  = int(start_dt.timestamp())
    end_s    = int(end_dt.timestamp())

    frames, cursor = [], start_s

    while cursor < end_s:
        cursor_dt = datetime.fromtimestamp(cursor, tz=timezone.utc)
        print(f"[MEXC] Fetching {symbol} from {cursor_dt.strftime('%Y-%m-%d %H:%M')} UTC ...")
        df = _mexc_fetch_chunk(symbol, interval, start=cursor)
        if df.empty:
            break
        frames.append(df)
        last_ts = int(df.index[-1].timestamp())
        if last_ts <= cursor:
            break
        cursor = last_ts + 60

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated()].sort_index()
    combined = combined[combined.index <= end_dt]
    return combined


# ---------------------------------------------------------------------------
# Public fetch API  (always fetches live, then saves to Excel)
# ---------------------------------------------------------------------------

def fetch_tsla(
    cfg: DataConfig,
    start: str,
    end: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch 1-min TSLA OHLCV from yfinance and persist to Excel.

    Every successful fetch appends new rows to data/raw/tsla_1min.xlsx so
    the file grows over time.  Use load_excel_tsla() to load without an API call.

    Parameters
    ----------
    cfg           : DataConfig
    start         : window start (YYYY-MM-DD)
    end           : window end (YYYY-MM-DD)
    force_refresh : ignored (kept for API compatibility)

    Returns
    -------
    pd.DataFrame — Index: open_time (UTC), Columns: open, high, low, close, volume
    """
    today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    fetch_end = end if end <= today_str else today_str

    print(f"[fetcher] Fetching TSLA {start} → {fetch_end} ...")
    df = _yf_fetch_range(cfg.yf_ticker, start, fetch_end)

    if df.empty:
        print("[fetcher] WARNING: yfinance returned empty DataFrame for TSLA.")
        return df

    # Filter to requested window
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt   = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    df = df[(df.index >= start_dt) & (df.index < end_dt)]

    print(f"[fetcher] TSLA: {len(df)} rows | {df.index[0]} → {df.index[-1]}")
    _print_preview(df, "TSLA (first 5 rows)")

    # Persist to Excel
    _save_excel(df, Path(cfg.cache_dir) / _TSLA_EXCEL)

    return df


def fetch_mexc(
    cfg: DataConfig,
    start: str,
    end: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch 1-min MEXC TESLA_USDT OHLCV from the REST API and persist to Excel.

    Every successful fetch appends new rows to data/raw/mexc_1min.xlsx so
    the file grows over time.  Use load_excel_mexc() to load without an API call.

    Parameters
    ----------
    cfg           : DataConfig
    start         : window start (YYYY-MM-DD)
    end           : window end (YYYY-MM-DD)
    force_refresh : ignored (kept for API compatibility)

    Returns
    -------
    pd.DataFrame — Index: open_time (UTC), Columns: open, high, low, close, volume, amount
    """
    today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    fetch_end = end if end <= today_str else today_str

    print(f"[fetcher] Fetching MEXC {cfg.mexc_symbol} {start} → {fetch_end} ...")
    df = _mexc_fetch_range(cfg.mexc_symbol, cfg.interval, start, fetch_end)

    if df.empty:
        print("[fetcher] WARNING: MEXC returned empty DataFrame.")
        return df

    # Filter to requested window
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt   = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    df = df[(df.index >= start_dt) & (df.index < end_dt)]

    print(f"[fetcher] MEXC: {len(df)} rows | {df.index[0]} → {df.index[-1]}")
    _print_preview(df, "MEXC TESLA_USDT (first 5 rows)")

    # Persist to Excel
    _save_excel(df, Path(cfg.cache_dir) / _MEXC_EXCEL)

    return df
