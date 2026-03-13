from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from config import Config


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Immutable record of one complete round-trip pair trade."""

    entry_time:    pd.Timestamp
    exit_time:     pd.Timestamp

    direction:     str            # "short_mexc" | "long_mexc"

    # Prices (USD)
    entry_tsla:    float          # TSLA fill at bar t+1 open
    entry_mexc:    float          # MEXC fill at bar t+1 open
    exit_tsla:     float          # TSLA fill: next bar open (normal) or current close (session_end/eod)
    exit_mexc:     float          # MEXC fill: next bar open (normal) or current close (session_end/eod)

    # Z-scores for analytics
    entry_z:       float          # z-score at SIGNAL bar (bar t)
    exit_z:        float          # z-score at exit signal bar

    holding_bars:  int            # bars held: 1 at first bar after entry fill, 20 at max-hold

    exit_reason:   str            # "session_end" | "max_holding" | "signal" | "eod"

    # PnL breakdown (USD)
    pnl_gross:     float          # gross PnL (no fees)
    pnl_fees_mexc: float          # MEXC fees (entry + exit notional × 3 bps)
    pnl_fees_nyse: float          # NYSE fees ($0.02/share round-trip)
    pnl_fees:      float          # total fees = pnl_fees_mexc + pnl_fees_nyse
    pnl_net:       float          # net PnL = pnl_gross - pnl_fees


@dataclass
class EngineResult:
    """Full output of one backtest run."""

    trades:       List[Trade]
    equity_curve: pd.Series    # cumulative net PnL indexed by timestamp
    data:         pd.DataFrame # signal-enriched data (useful for plotting)


# ---------------------------------------------------------------------------
# Internal position state
# ---------------------------------------------------------------------------

@dataclass
class _OpenPosition:
    signal_bar:  int            # index i where signal fired (bar t)
    entry_bar:   int            # index i+1 where fill occurred (bar t+1)
    entry_time:  pd.Timestamp   # timestamp of fill bar (t+1)
    direction:   str
    entry_tsla:  float          # open price at bar t+1
    entry_mexc:  float          # open price at bar t+1
    entry_z:     float          # z-score at signal bar t


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Simulates the Quant approach bar-by-bar on pre-computed signal data.

    Usage
    -----
    >>> engine = BacktestEngine(config)
    >>> result = engine.run(signal_data)
    >>> print(f"Trades: {len(result.trades)}, Net PnL: ${sum(t.pnl_net for t in result.trades):.2f}")
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self, data: pd.DataFrame) -> EngineResult:
        """
        Main event loop.

        Parameters
        ----------
        data : pd.DataFrame
            Output of QuantStrategy.generate_signals() — must contain:
            tsla_open, tsla_close, mexc_open, mexc_close, z_score, signal

        Returns
        -------
        EngineResult
        """
        trades: List[Trade] = []
        equity_points: List[tuple] = []  # (timestamp, cumulative_pnl)

        cumulative_pnl = 0.0
        position: Optional[_OpenPosition] = None

        rows = list(data.iterrows())
        n    = len(rows)

        for i, (ts, row) in enumerate(rows):
            is_last_bar    = (i == n - 1)
            is_session_end = bool(row.get("session_end", False))

            equity_points.append((ts, cumulative_pnl))

            # ── IN POSITION: check exit conditions ────────────────────────
            # bars_held starts at 1 at the entry fill bar, matching the
            # reference notebook's hold_bars = i - entry_idx convention.
            if position is not None:
                bars_held = i - position.entry_bar + 1
                z_now     = row["z_score"]

                exit_reason = self._check_exit(
                    pos            = position,
                    z_now          = z_now,
                    bars_held      = bars_held,
                    is_last_bar    = is_last_bar,
                    is_session_end = is_session_end,
                )

                if exit_reason is not None:
                    if exit_reason in ("session_end", "eod"):
                        # Forced close: fill at current bar's CLOSE
                        # (no next bar available, or must avoid overnight)
                        exit_tsla = row["tsla_close"]
                        exit_mexc = row["mexc_close"]
                        exit_time = ts
                    else:
                        # Normal exit (z-reversion or max-hold):
                        # fill at NEXT bar's OPEN — matches reference notebook.
                        # Exception: if the next bar is itself a session-end bar,
                        # use its CLOSE (session_end priority, same as notebook).
                        next_ts, next_row = rows[i + 1]
                        next_is_session_end = bool(next_row.get("session_end", False))
                        if next_is_session_end or (i + 1 == n - 1):
                            exit_tsla = next_row["tsla_close"]
                            exit_mexc = next_row["mexc_close"]
                        else:
                            exit_tsla = next_row["tsla_open"]
                            exit_mexc = next_row["mexc_open"]
                        exit_time = next_ts

                    trade = self._close_position(
                        position     = position,
                        exit_time    = exit_time,
                        exit_tsla    = exit_tsla,
                        exit_mexc    = exit_mexc,
                        exit_z       = float(z_now) if pd.notna(z_now) else 0.0,
                        holding_bars = bars_held,
                        exit_reason  = exit_reason,
                    )
                    trades.append(trade)
                    cumulative_pnl      += trade.pnl_net
                    equity_points[-1]    = (ts, cumulative_pnl)
                    position             = None

            # ── FLAT: look for entry signal ────────────────────────────────
            # Uses else (not a second if) to match the reference notebook's
            # if/else structure: a bar processes either an exit OR an entry,
            # never both.  Same-bar re-entry after an exit is not allowed.
            # Guard: no entry on the last bar (no fill bar available).
            else:
                if not is_last_bar:
                    signal = row["signal"]
                    if signal in ("short_mexc", "long_mexc"):
                        next_ts, next_row = rows[i + 1]
                        position = _OpenPosition(
                            signal_bar  = i,
                            entry_bar   = i + 1,
                            entry_time  = next_ts,
                            direction   = signal,
                            entry_tsla  = next_row["tsla_open"],   # fill: t+1 open
                            entry_mexc  = next_row["mexc_open"],   # fill: t+1 open
                            entry_z     = row["z_score"],          # z at signal bar t
                        )

        equity_curve = pd.Series(
            [v for _, v in equity_points],
            index=pd.DatetimeIndex([t for t, _ in equity_points]),
            name="cumulative_pnl",
        )

        print(
            f"[engine] Backtest complete: {len(trades)} trades, "
            f"net PnL = ${cumulative_pnl:,.2f}"
        )
        return EngineResult(trades=trades, equity_curve=equity_curve, data=data)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _check_exit(
        self,
        pos:            _OpenPosition,
        z_now:          float,
        bars_held:      int,
        is_last_bar:    bool,
        is_session_end: bool,
    ) -> Optional[str]:
        """
        Evaluate exit conditions in priority order.

        Priority (matching reference notebook):
          1. session_end  — forced close, CLOSE price (prevent overnight)
          2. max_holding  — 20-bar time limit, next bar OPEN
          3. signal       — z-score crossed zero, next bar OPEN
          4. eod          — end of data, CLOSE price
        """
        cfg = self.cfg.strategy

        # 1. Session end — always force close at last NYSE bar
        if is_session_end:
            return "session_end"

        # 2. Max holding time — fires when bars_held reaches max_holding_bars
        #    (bars_held=1 at entry fill bar, so fires at entry_bar + max_hold - 1,
        #    filling at entry_bar + max_hold open — identical to notebook)
        if bars_held >= cfg.max_holding_bars:
            return "max_holding"

        # 3. Z-score reversion — exit when z crosses through zero
        #    short_mexc: entered because z was too high → exit when z ≤ 0
        #    long_mexc:  entered because z was too low  → exit when z ≥ 0
        if pd.notna(z_now):
            if pos.direction == "short_mexc" and z_now <= cfg.exit_threshold:
                return "signal"
            if pos.direction == "long_mexc"  and z_now >= -cfg.exit_threshold:
                return "signal"

        # 4. End of data — force close to avoid open positions at end of run
        if is_last_bar:
            return "eod"

        return None  # keep holding

    def _close_position(
        self,
        position:     _OpenPosition,
        exit_time:    pd.Timestamp,
        exit_tsla:    float,
        exit_mexc:    float,
        exit_z:       float,
        holding_bars: int,
        exit_reason:  str,
    ) -> Trade:
        """
        Calculate PnL (gross, fees split, net) and create a Trade record.

        Sizing: 1 TSLA share equivalent.
          - TSLA leg: 1 share
          - MEXC leg: 1 TSLA-share-equivalent (MEXC is quoted in USDT ≈ USD)

        For "short_mexc" (MEXC expensive → sell MEXC, buy TSLA):
          gross = (entry_mexc - exit_mexc)    short MEXC: sell high, buy low
                + (exit_tsla  - entry_tsla)   long TSLA:  buy low,  sell high

        For "long_mexc" (MEXC cheap → buy MEXC, sell TSLA):
          gross = (exit_mexc  - entry_mexc)   long MEXC:  buy low,  sell high
                + (entry_tsla - exit_tsla)    short TSLA: sell high, buy low
        """
        fees_cfg = self.cfg.fees

        if position.direction == "short_mexc":
            pnl_gross = (
                (position.entry_mexc - exit_mexc)    # short MEXC
                + (exit_tsla - position.entry_tsla)  # long TSLA
            )
        else:  # long_mexc
            pnl_gross = (
                (exit_mexc - position.entry_mexc)    # long MEXC
                + (position.entry_tsla - exit_tsla)  # short TSLA
            )

        # MEXC fees: 3 bps per side × 2 sides (entry + exit)
        mexc_fees = (position.entry_mexc + exit_mexc) * fees_cfg.mexc_per_side

        # NYSE fees: $0.01/share per side × 2 sides = $0.02/share round-trip
        nyse_fees = fees_cfg.nyse_round_trip  # $0.02

        total_fees = mexc_fees + nyse_fees
        pnl_net    = pnl_gross - total_fees

        return Trade(
            entry_time    = position.entry_time,
            exit_time     = exit_time,
            direction     = position.direction,
            entry_tsla    = position.entry_tsla,
            entry_mexc    = position.entry_mexc,
            exit_tsla     = exit_tsla,
            exit_mexc     = exit_mexc,
            entry_z       = position.entry_z,
            exit_z        = exit_z,
            holding_bars  = holding_bars,
            exit_reason   = exit_reason,
            pnl_gross     = pnl_gross,
            pnl_fees_mexc = mexc_fees,
            pnl_fees_nyse = nyse_fees,
            pnl_fees      = total_fees,
            pnl_net       = pnl_net,
        )
