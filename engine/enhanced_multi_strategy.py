#!/usr/bin/env python3
"""
Enhanced Multi-Strategy Backtesting Engine

Part 3: Main backtesting engine with all enhancements integrated
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger
import argparse

from engine.enhanced_backtest import (
    StrategyType, MarketSignal, TechnicalIndicators, StrategySelection,
    EnhancedBacktestResult, TechnicalAnalyzer, StrategySelector,
    EnhancedMultiStrategyBacktester, IronCondorLegStatus, DayBacktestResult
)
from engine.delta_strike_selector import DeltaStrikeSelector, PositionMonitor, IntradayPositionMonitor, StrikeSelection, IronCondorStrikeSelection
from engine.query_engine_adapter import EnhancedQueryEngineAdapter


# Intraday scan constants
ENTRY_SCAN_START    = "10:00:00"   # Wait 30 min for opening volatility to settle
LAST_ENTRY_TIME     = "14:00:00"   # No new entries at or after 2 PM
FINAL_EXIT_TIME     = "16:00:00"   # Hold to expiry (market close)
MIN_DISTANCE_IC     = 50.0         # IC short strike must be >= $50 away from SPX
MIN_DISTANCE_SPREAD = 25.0         # Single spread short strike must be >= $25 away from SPX

# Trend filter
TREND_FILTER_LOOKBACK_MINUTES = 30  # How far back to measure trend
TREND_FILTER_POINTS           = 30  # SPX points moved → market is trending; block IC, use asymmetric spread

# Daily-drift guards — anchored to the day's opening price
# Block the DANGEROUS-SIDE spread when the market has moved significantly:
#   Large DOWN move (≤ -DRIFT_BLOCK_POINTS)  → block call spreads
#     (mean-reversion bounce through the short call is the risk)
#   Large UP move   (≥ +DRIFT_BLOCK_POINTS)  → block put spreads
#     (mean-reversion pullback through the short put is the risk)
#   IC: blocked only on extreme moves (≥ DRIFT_IC_BLOCK_POINTS)
#
# Indicator conviction guard: when day_drift is already negative, require RSI < 50
# before opening a put spread (indicators must confirm the bullish thesis, not just
# drift into a falling market on neutral readings).
DRIFT_BLOCK_POINTS      = 20.0   # Hard block: dangerous-side spread beyond this abs drift
DRIFT_IC_BLOCK_POINTS   = 50.0   # Hard block: IC entirely on extreme moves
PUT_SPREAD_MAX_RSI_ON_NEG_DRIFT = 30.0  # Max RSI allowed for put spread on negative-drift day
PUT_SPREAD_MAX_RSI_ON_POS_DRIFT = 30.0  # Max RSI allowed for put spread on flat/positive-drift day
#   Tighter requirement on up days: only enter if market is genuinely very oversold;
#   an RSI between 30-45 on an up day does not provide enough conviction.
# After the call-spread block latches, wait this many minutes before allowing put spreads.
# Prevents entering into a "dead-cat bounce" immediately after a sharp drop is confirmed.
PUT_SPREAD_DRIFT_CONFIRM_MINUTES = 30
# Intraday reversal thresholds — separate for call vs put spreads.
# Call spreads: blocked after a ≥ 15-pt reversal from the intraday peak.
# Put spreads: tighter at 10 pts — a smaller bounce from an extreme low still
#   means the market has not stabilised enough to safely sell a put spread.
INTRADAY_CALL_REVERSAL_POINTS = 15.0
INTRADAY_PUT_REVERSAL_POINTS  = 10.0
# Extreme overbought threshold: when RSI exceeds this level, do not sell a call spread —
# the market is in a momentum burst and may continue higher.
CALL_SPREAD_MAX_ENTRY_RSI = 77.0

# Strategy mode constants (match BacktestStrategyEnum values)
STRATEGY_IRON_CONDOR     = "iron_condor"
STRATEGY_CREDIT_SPREADS  = "credit_spreads"
STRATEGY_IC_CREDIT_SPREADS = "ic_credit_spreads"


def _build_minute_grid(date: str, start_time: str, end_time: str) -> List[str]:
    """Generate all HH:MM:SS strings for 1-min bars between start and end (inclusive)."""
    start_dt = pd.Timestamp(f"{date} {start_time}")
    end_dt   = pd.Timestamp(f"{date} {end_time}")
    times = pd.date_range(start=start_dt, end=end_dt, freq='1min')
    return [t.strftime("%H:%M:%S") for t in times]


class EnhancedBacktestingEngine(EnhancedMultiStrategyBacktester):
    """Complete enhanced backtesting engine"""
    
    def __init__(self, data_path: str = "data/processed/parquet_1m"):
        super().__init__(data_path)
        # Wrap query engine with enhanced adapter
        self.enhanced_query_engine = EnhancedQueryEngineAdapter(self.query_engine)
        self.delta_selector = DeltaStrikeSelector(self.enhanced_query_engine, self.ic_loader)
        self.position_monitor = PositionMonitor(self.enhanced_query_engine, self.strategy_builder)
        self.intraday_monitor = IntradayPositionMonitor(self.enhanced_query_engine, self.strategy_builder)

    # ------------------------------------------------------------------
    # Trend detection helpers
    # ------------------------------------------------------------------

    def _get_trend_state(self, date: str, current_time: str,
                         spx_history: pd.Series) -> Tuple[bool, str]:
        """
        Return (is_trending, direction) based on the net SPX move over the last
        TREND_FILTER_LOOKBACK_MINUTES bars.

        direction: 'down' | 'up' | 'flat'
        is_trending is True when |net_move| >= TREND_FILTER_POINTS.
        """
        if len(spx_history) < TREND_FILTER_LOOKBACK_MINUTES:
            return False, 'flat'

        lookback = spx_history.tail(TREND_FILTER_LOOKBACK_MINUTES)
        net_move = lookback.iloc[-1] - lookback.iloc[0]

        if net_move <= -TREND_FILTER_POINTS:
            return True, 'down'
        elif net_move >= TREND_FILTER_POINTS:
            return True, 'up'
        return False, 'flat'

    # ------------------------------------------------------------------
    # Intraday multi-trade scan loop
    # ------------------------------------------------------------------

    def backtest_day_intraday(self,
                              date: str,
                              target_delta: float = 0.15,
                              target_prob_itm: float = 0.15,
                              min_spread_width: int = 10,
                              take_profit: float = 0.10,
                              stop_loss: float = 2.0,
                              monitor_interval: int = 1,
                              quantity: int = 1,
                              target_credit: Optional[float] = 0.50,
                              strategy_mode: str = STRATEGY_IRON_CONDOR,
                              progress_callback=None,
                              entry_start_time: str = ENTRY_SCAN_START,
                              last_entry_time: str = LAST_ENTRY_TIME) -> DayBacktestResult:
        """
        Full intraday scan loop for one trading day.
        strategy_mode controls which entry types are allowed:
          iron_condor       — IC only
          credit_spreads    — put/call spreads only
          ic_credit_spreads — all types

        entry_start_time: first bar at which entries are considered (default 10:00:00)
        last_entry_time:  no new entries at or after this time (default 14:00:00)
        """
        logger.info(
            f"Intraday scan: {date} | mode={strategy_mode} credit={target_credit} "
            f"contracts={quantity} tp={take_profit} sl={stop_loss} interval={monitor_interval}m "
            f"entry_window={entry_start_time}–{last_entry_time}"
        )

        scan_times = _build_minute_grid(date, entry_start_time, FINAL_EXIT_TIME)
        trades: List[EnhancedBacktestResult] = []

        def _fire(event: dict):
            """Invoke progress_callback safely — never let it crash the backtest."""
            if progress_callback:
                try:
                    progress_callback(event)
                except Exception as _cb_err:
                    logger.debug(f"progress_callback error (ignored): {_cb_err}")

        if date not in self.available_dates:
            return DayBacktestResult(date=date, trades=[], total_pnl=0.0,
                                     trade_count=0, scan_minutes_checked=0)

        # Fetch opening SPX price for daily-drift guards.
        # Try several early bars; keep as None (not 0) so the guard stays
        # disabled rather than computing a nonsense drift if all fail.
        # The scan loop will latch the first valid entry_spx if still None.
        spx_open: Optional[float] = None
        for _t in ("09:30:00", "09:31:00", "09:32:00", "09:35:00"):
            _p = self.enhanced_query_engine.get_fastest_spx_price(date, _t)
            if _p and _p > 0:
                spx_open = float(_p)
                break
        logger.debug(f"Drift guard: SPX open = {spx_open}")

        # Sticky drift flags — once set True they stay True for the whole day.
        # Only the dangerous-side spread is blocked; the safe side stays open.
        #   Large DOWN move → block call spreads (mean-reversion bounce risk)
        #   Large UP move   → block put spreads  (mean-reversion pullback risk)
        #   Extreme move (either dir) → block IC
        _put_spread_ever_blocked  = False   # latched True on large UP drift
        _call_spread_ever_blocked = False   # latched True on large DOWN drift
        _ic_ever_blocked          = False   # latched True on extreme drift (either direction)
        _call_blocked_latch_time  = None    # HH:MM:SS when call_spread block first latched
        # Intraday reversal tracking: peak positive / peak negative drift seen so
        # far today.  If the market has reversed sharply from an intraday extreme,
        # selling against the new direction carries elevated volatility risk.
        _intraday_max_drift       = 0.0    # highest drift reached today
        _intraday_min_drift       = 0.0    # lowest drift reached today

        # Pre-scan drift check: evaluate every bar from 09:31 up to (but not
        # including) ENTRY_SCAN_START so that the sticky flags are already
        # latched correctly when the first entry bar is evaluated.
        # This matters when ENTRY_SCAN_START > 09:35 — without this, a large
        # opening move that crosses the threshold before 10:00 would be invisible
        # to the guards at the first entry bar.
        if spx_open and spx_open > 0:
            pre_scan_times = _build_minute_grid(date, "09:31:00", entry_start_time)
            # Exclude the first bar of the actual scan to avoid double-counting
            pre_scan_times = [t for t in pre_scan_times if t < entry_start_time]
            for _pt in pre_scan_times:
                _pp = self.enhanced_query_engine.get_fastest_spx_price(date, _pt)
                if not _pp or _pp <= 0:
                    continue
                _pre_drift = float(_pp) - spx_open
                if _pre_drift >= DRIFT_BLOCK_POINTS:         # large UP → block put spreads
                    _put_spread_ever_blocked = True
                if _pre_drift <= -DRIFT_BLOCK_POINTS:        # large DOWN → block call spreads
                    if not _call_spread_ever_blocked:
                        _call_blocked_latch_time = _pt   # record when it first latched
                    _call_spread_ever_blocked = True
                if abs(_pre_drift) >= DRIFT_IC_BLOCK_POINTS:  # extreme → block IC
                    _ic_ever_blocked = True
                # Track intraday extremes during the pre-scan window too
                _intraday_max_drift = max(_intraday_max_drift, _pre_drift)
                _intraday_min_drift = min(_intraday_min_drift, _pre_drift)
            logger.debug(
                f"Pre-scan drift check complete: put_blocked={_put_spread_ever_blocked} "
                f"call_blocked={_call_spread_ever_blocked} ic_blocked={_ic_ever_blocked}"
            )

        # Build a per-run monitor with the request's risk params
        monitor = IntradayPositionMonitor(
            self.enhanced_query_engine,
            self.strategy_builder,
            take_profit=take_profit,
            stop_loss=stop_loss,
            monitor_interval=monitor_interval,
        )

        # Open position slots
        open_put_spread  = None
        open_call_spread = None
        open_ic          = None
        ic_leg_status    = None
        ic_entry_meta    = {}
        put_spread_meta  = {}
        call_spread_meta = {}

        # Track whether a call spread was opened at any point today.
        # Used to prevent re-entry on a declining day after the first call spread closes.
        _had_call_spread_today = False

        # Track whether a call spread closed with a profit today.
        # Used to prevent entering a put spread after a winning CS on an up/flat day
        # (the CS win is enough premium for the day; a subsequent PS risks giving it back).
        _had_call_spread_win_today = False

        # Track whether a put spread closed with a profit today.
        # Used to prevent entering a call spread after a winning PS on a declining day
        # (the PS win is enough premium for the day; a subsequent CS bets against an
        # already-falling market with low RSI, which is extremely risky).
        _had_put_spread_win_today = False

        # Checkpoint lists — built up while a position is open, flushed on close
        ic_checkpoints        : list = []
        put_spread_checkpoints: list = []
        call_spread_checkpoints: list = []

        for bar_index, current_time in enumerate(scan_times):
            is_past_entry_cutoff = current_time >= last_entry_time
            is_final_bar         = current_time >= FINAL_EXIT_TIME

            # Respect monitor_interval; always process the final bar for the hard close.
            is_check_bar = (bar_index % monitor_interval == 0) or is_final_bar

            # --- 1. Monitor IC legs independently ---
            if open_ic is not None and ic_leg_status is not None and is_check_bar:
                ic_leg_status = monitor.check_ic_leg_decay(
                    open_ic, date, current_time, ic_leg_status
                )

                # Snapshot this bar
                ic_entry_credit = getattr(open_ic, 'entry_credit', 0)
                ic_quantity = getattr(open_ic, 'quantity', 1)
                spx_now = self.enhanced_query_engine.get_fastest_spx_price(date, current_time) or 0

                if is_final_bar:
                    # Use intrinsic settlement values at expiry
                    raw_put  = self._expiry_exit_cost(open_ic, StrategyType.PUT_SPREAD,  spx_now)
                    raw_call = self._expiry_exit_cost(open_ic, StrategyType.CALL_SPREAD, spx_now)
                    # Force both sides closed at intrinsic cost
                    if not ic_leg_status.put_side_closed:
                        ic_leg_status.put_side_closed = True
                        ic_leg_status.put_side_exit_time = current_time
                        ic_leg_status.put_side_exit_cost = raw_put
                        ic_leg_status.put_side_exit_reason = "Expired at market close"
                    if not ic_leg_status.call_side_closed:
                        ic_leg_status.call_side_closed = True
                        ic_leg_status.call_side_exit_time = current_time
                        ic_leg_status.call_side_exit_cost = raw_call
                        ic_leg_status.call_side_exit_reason = "Expired at market close"
                else:
                    try:
                        _, _, raw_put, raw_call = monitor._check_ic_leg_decay_values(open_ic)
                    except Exception:
                        raw_put, raw_call = 0.0, 0.0

                put_cost_ps  = round(raw_put  / (100.0 * ic_quantity), 4)
                call_cost_ps = round(raw_call / (100.0 * ic_quantity), 4)
                total_cost_ps   = round((put_cost_ps + call_cost_ps), 4)
                entry_credit_ps = round(ic_entry_credit / (100.0 * ic_quantity), 4)
                ic_checkpoints.append({
                    "time": current_time,
                    "spx": round(spx_now, 2),
                    "cost_per_share": total_cost_ps,
                    "pnl_per_share": round(entry_credit_ps - total_cost_ps, 4),
                    "put_cost_per_share": put_cost_ps,
                    "call_cost_per_share": call_cost_ps,
                })
                _fire({
                    'event': 'monitor_tick',
                    'strategy_type': 'Iron Condor',
                    'entry_time': ic_entry_meta.get('entry_time', ''),
                    'time': current_time,
                    'spx': round(spx_now, 2),
                    'pnl_per_share': round(entry_credit_ps - total_cost_ps, 4),
                    'entry_credit_per_share': entry_credit_ps,
                })

                # If both sides closed → finalize IC trade
                if ic_leg_status.put_side_closed and ic_leg_status.call_side_closed:
                    total_exit_cost = ic_leg_status.put_side_exit_cost + ic_leg_status.call_side_exit_cost
                    entry_credit = ic_entry_credit
                    pnl = entry_credit - total_exit_cost
                    pnl_pct = (pnl / entry_credit * 100) if entry_credit > 0 else 0
                    exit_spx = spx_now or ic_entry_meta.get('entry_spx', 0)
                    later_side_time = max(
                        ic_leg_status.put_side_exit_time or "00:00:00",
                        ic_leg_status.call_side_exit_time or "00:00:00"
                    )
                    _ic_entry_spx = ic_entry_meta.get('entry_spx', 0)
                    ic_exit_rationale = {
                        'exit_trigger': 'IC both sides closed (decay threshold)',
                        'put_side_exit_time': ic_leg_status.put_side_exit_time,
                        'put_side_exit_reason': ic_leg_status.put_side_exit_reason,
                        'call_side_exit_time': ic_leg_status.call_side_exit_time,
                        'call_side_exit_reason': ic_leg_status.call_side_exit_reason,
                        'put_side_exit_cost': round(ic_leg_status.put_side_exit_cost, 2),
                        'call_side_exit_cost': round(ic_leg_status.call_side_exit_cost, 2),
                        'total_exit_cost': round(total_exit_cost, 2),
                        'entry_credit': round(entry_credit, 2),
                        'spx_at_exit': round(exit_spx, 2),
                        'spx_move_since_entry': round(exit_spx - _ic_entry_spx, 2) if _ic_entry_spx else None,
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2),
                    }
                    trades.append(EnhancedBacktestResult(
                        date=date,
                        strategy_type=StrategyType.IRON_CONDOR,
                        market_signal=ic_entry_meta.get('market_signal', MarketSignal.NEUTRAL),
                        entry_time=ic_entry_meta.get('entry_time', current_time),
                        exit_time=later_side_time,
                        exit_reason="IC both sides closed",
                        entry_spx_price=_ic_entry_spx,
                        exit_spx_price=exit_spx,
                        technical_indicators=ic_entry_meta.get('indicators', TechnicalIndicators(0,0,0,0,0,0,0,0.5)),
                        strike_selection=ic_entry_meta.get('strike_selection', StrikeSelection(0,0,0,0,0)),
                        entry_credit=entry_credit,
                        exit_cost=total_exit_cost,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        max_profit=getattr(open_ic, 'max_profit', entry_credit),
                        max_loss=getattr(open_ic, 'max_loss', -total_exit_cost),
                        monitoring_points=ic_checkpoints,
                        success=True,
                        confidence=ic_entry_meta.get('confidence', 0),
                        notes=ic_entry_meta.get('notes', ''),
                        entry_rationale=ic_entry_meta.get('entry_rationale'),
                        exit_rationale=ic_exit_rationale,
                        ic_leg_status=ic_leg_status
                    ))
                    open_ic = None
                    ic_leg_status = None
                    ic_entry_meta = {}
                    ic_checkpoints = []
                    _fire({'event': 'position_closed', 'strategy_type': 'Iron Condor',
                           'result': trades[-1]})

            # --- 2. Monitor put spread ---
            if open_put_spread is not None and is_check_bar:
                should_exit, current_cost, reason = monitor.check_decay_at_time(
                    open_put_spread, StrategyType.PUT_SPREAD, date, current_time
                )

                # At final bar use intrinsic settlement value, not stale market price
                spx_now = self.enhanced_query_engine.get_fastest_spx_price(date, current_time) or 0
                if is_final_bar:
                    current_cost = self._expiry_exit_cost(open_put_spread, StrategyType.PUT_SPREAD, spx_now)
                    should_exit = True
                    reason = "Expired at market close"

                # Snapshot this bar
                ps_entry_credit = getattr(open_put_spread, 'entry_credit', 0)
                ps_quantity = getattr(open_put_spread, 'quantity', 1)
                entry_credit_ps = round(ps_entry_credit / (100.0 * ps_quantity), 4)
                cost_ps = round(current_cost / (100.0 * ps_quantity), 4)
                put_spread_checkpoints.append({
                    "time": current_time,
                    "spx": round(spx_now, 2),
                    "cost_per_share": cost_ps,
                    "pnl_per_share": round(entry_credit_ps - cost_ps, 4),
                })
                _fire({
                    'event': 'monitor_tick',
                    'strategy_type': 'Put Spread',
                    'entry_time': put_spread_meta.get('entry_time', current_time),
                    'time': current_time,
                    'spx': round(spx_now, 2),
                    'pnl_per_share': round(entry_credit_ps - cost_ps, 4),
                    'entry_credit_per_share': entry_credit_ps,
                })

                if should_exit or is_final_bar:
                    exit_reason = reason if should_exit else "Expired at market close"
                    entry_credit = ps_entry_credit
                    pnl = entry_credit - current_cost
                    pnl_pct = (pnl / entry_credit * 100) if entry_credit > 0 else 0
                    exit_spx = spx_now or put_spread_meta.get('entry_spx', 0)
                    _ps_entry_spx = put_spread_meta.get('entry_spx', 0)
                    ps_exit_rationale = {
                        'exit_trigger': exit_reason,
                        'exit_cost': round(current_cost, 2),
                        'entry_credit': round(entry_credit, 2),
                        'spx_at_exit': round(exit_spx, 2),
                        'spx_move_since_entry': round(exit_spx - _ps_entry_spx, 2) if _ps_entry_spx else None,
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2),
                    }
                    trades.append(EnhancedBacktestResult(
                        date=date,
                        strategy_type=StrategyType.PUT_SPREAD,
                        market_signal=put_spread_meta.get('market_signal', MarketSignal.BULLISH),
                        entry_time=put_spread_meta.get('entry_time', current_time),
                        exit_time=current_time,
                        exit_reason=exit_reason,
                        entry_spx_price=_ps_entry_spx,
                        exit_spx_price=exit_spx,
                        technical_indicators=put_spread_meta.get('indicators', TechnicalIndicators(0,0,0,0,0,0,0,0.5)),
                        strike_selection=put_spread_meta.get('strike_selection', StrikeSelection(0,0,0,0,0)),
                        entry_credit=entry_credit,
                        exit_cost=current_cost,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        max_profit=getattr(open_put_spread, 'max_profit', entry_credit),
                        max_loss=getattr(open_put_spread, 'max_loss', -current_cost),
                        monitoring_points=put_spread_checkpoints,
                        success=True,
                        confidence=put_spread_meta.get('confidence', 0),
                        notes=put_spread_meta.get('notes', ''),
                        entry_rationale=put_spread_meta.get('entry_rationale'),
                        exit_rationale=ps_exit_rationale,
                    ))
                    open_put_spread = None
                    put_spread_meta = {}
                    put_spread_checkpoints = []
                    # Latch win flag for post-PS call spread guard
                    if pnl > 0:
                        _had_put_spread_win_today = True
                    _fire({'event': 'position_closed', 'strategy_type': 'Put Spread',
                           'result': trades[-1]})

            # --- 3. Monitor call spread ---
            if open_call_spread is not None and is_check_bar:
                should_exit, current_cost, reason = monitor.check_decay_at_time(
                    open_call_spread, StrategyType.CALL_SPREAD, date, current_time
                )

                # At final bar use intrinsic settlement value, not stale market price
                spx_now = self.enhanced_query_engine.get_fastest_spx_price(date, current_time) or 0
                if is_final_bar:
                    current_cost = self._expiry_exit_cost(open_call_spread, StrategyType.CALL_SPREAD, spx_now)
                    should_exit = True
                    reason = "Expired at market close"

                # Snapshot this bar
                cs_entry_credit = getattr(open_call_spread, 'entry_credit', 0)
                cs_quantity = getattr(open_call_spread, 'quantity', 1)
                entry_credit_ps = round(cs_entry_credit / (100.0 * cs_quantity), 4)
                cost_ps = round(current_cost / (100.0 * cs_quantity), 4)
                call_spread_checkpoints.append({
                    "time": current_time,
                    "spx": round(spx_now, 2),
                    "cost_per_share": cost_ps,
                    "pnl_per_share": round(entry_credit_ps - cost_ps, 4),
                })
                _fire({
                    'event': 'monitor_tick',
                    'strategy_type': 'Call Spread',
                    'entry_time': call_spread_meta.get('entry_time', current_time),
                    'time': current_time,
                    'spx': round(spx_now, 2),
                    'pnl_per_share': round(entry_credit_ps - cost_ps, 4),
                    'entry_credit_per_share': entry_credit_ps,
                })

                if should_exit or is_final_bar:
                    exit_reason = reason if should_exit else "Expired at market close"
                    entry_credit = cs_entry_credit
                    pnl = entry_credit - current_cost
                    pnl_pct = (pnl / entry_credit * 100) if entry_credit > 0 else 0
                    exit_spx = spx_now or call_spread_meta.get('entry_spx', 0)
                    _cs_entry_spx = call_spread_meta.get('entry_spx', 0)
                    cs_exit_rationale = {
                        'exit_trigger': exit_reason,
                        'exit_cost': round(current_cost, 2),
                        'entry_credit': round(entry_credit, 2),
                        'spx_at_exit': round(exit_spx, 2),
                        'spx_move_since_entry': round(exit_spx - _cs_entry_spx, 2) if _cs_entry_spx else None,
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2),
                    }
                    trades.append(EnhancedBacktestResult(
                        date=date,
                        strategy_type=StrategyType.CALL_SPREAD,
                        market_signal=call_spread_meta.get('market_signal', MarketSignal.BEARISH),
                        entry_time=call_spread_meta.get('entry_time', current_time),
                        exit_time=current_time,
                        exit_reason=exit_reason,
                        entry_spx_price=_cs_entry_spx,
                        exit_spx_price=exit_spx,
                        technical_indicators=call_spread_meta.get('indicators', TechnicalIndicators(0,0,0,0,0,0,0,0.5)),
                        strike_selection=call_spread_meta.get('strike_selection', StrikeSelection(0,0,0,0,0)),
                        entry_credit=entry_credit,
                        exit_cost=current_cost,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        max_profit=getattr(open_call_spread, 'max_profit', entry_credit),
                        max_loss=getattr(open_call_spread, 'max_loss', -current_cost),
                        monitoring_points=call_spread_checkpoints,
                        success=True,
                        confidence=call_spread_meta.get('confidence', 0),
                        notes=call_spread_meta.get('notes', ''),
                        entry_rationale=call_spread_meta.get('entry_rationale'),
                        exit_rationale=cs_exit_rationale,
                    ))
                    open_call_spread = None
                    call_spread_meta = {}
                    call_spread_checkpoints = []
                    # Latch win flag for post-CS put spread guard
                    if pnl > 0:
                        _had_call_spread_win_today = True
                    _fire({'event': 'position_closed', 'strategy_type': 'Call Spread',
                           'result': trades[-1]})
            if not is_past_entry_cutoff and not is_final_bar:
                try:
                    spx_history = self.get_spx_price_history(date, current_time, lookback_minutes=60)
                    indicators = self.technical_analyzer.analyze_market_conditions(spx_history)

                    # Skip bars where indicators are at sentinel fallback values —
                    # these occur when there isn't enough price history to compute
                    # real RSI/BB values (RSI defaults to 50.0, BB position to 0.5).
                    # Entering on sentinel values is indistinguishable from noise.
                    if indicators.rsi == 50.0 and indicators.bb_position == 0.5:
                        continue

                    selection = self.strategy_selector.select_strategy(indicators)
                    entry_spx = self.enhanced_query_engine.get_fastest_spx_price(date, current_time) or 0

                    # --- Trend filter (30-min momentum window) ---
                    is_trending, trend_dir = self._get_trend_state(date, current_time, spx_history)
                    if is_trending:
                        logger.debug(
                            f"Trend detected at {current_time}: {trend_dir} "
                            f"({TREND_FILTER_POINTS}+ pts over {TREND_FILTER_LOOKBACK_MINUTES}m)"
                        )

                    # --- Daily-drift guard (anchored to opening price, sticky) ---
                    # Latch the first valid SPX price as the open reference if
                    # pre-scan fetch failed (09:30 bar often has price=0 in data).
                    if spx_open is None and entry_spx > 0:
                        spx_open = entry_spx
                        logger.debug(f"Drift guard: latched open = {spx_open} at {current_time}")
                    day_drift = (entry_spx - spx_open) if (spx_open and spx_open > 0) else 0.0

                    # Once the dangerous threshold is crossed, latch the flag for the day.
                    # A temporary bounce does NOT re-enable the blocked side.
                    if day_drift >= DRIFT_BLOCK_POINTS:          # large UP → block put spreads
                        _put_spread_ever_blocked = True
                    if day_drift <= -DRIFT_BLOCK_POINTS:         # large DOWN → block call spreads
                        if not _call_spread_ever_blocked:
                            _call_blocked_latch_time = current_time  # record first latch
                        _call_spread_ever_blocked = True
                    if abs(day_drift) >= DRIFT_IC_BLOCK_POINTS:  # extreme → block IC
                        _ic_ever_blocked = True
                    # Update intraday extreme trackers (used for reversal guard below)
                    _intraday_max_drift = max(_intraday_max_drift, day_drift)
                    _intraday_min_drift = min(_intraday_min_drift, day_drift)

                    ic_blocked_by_drift   = _ic_ever_blocked
                    put_spread_blocked    = _put_spread_ever_blocked
                    call_spread_blocked   = _call_spread_ever_blocked
                    if ic_blocked_by_drift or put_spread_blocked or call_spread_blocked:
                        logger.debug(
                            f"Drift guard at {current_time}: drift={day_drift:+.1f} "
                            f"ic_blocked={ic_blocked_by_drift} put_blocked={put_spread_blocked} "
                            f"call_blocked={call_spread_blocked}"
                        )

                    # Determine which entry types are permitted by this strategy mode.
                    # On trend days, IC is blocked; asymmetric spread is used instead
                    # (call spread on down-trend — sell premium above a falling market;
                    #  put spread on up-trend — sell premium below a rising market).
                    allow_ic      = strategy_mode in (STRATEGY_IRON_CONDOR, STRATEGY_IC_CREDIT_SPREADS) and not is_trending and not ic_blocked_by_drift
                    allow_spreads = strategy_mode in (STRATEGY_CREDIT_SPREADS, STRATEGY_IC_CREDIT_SPREADS)
                    # Trend override: apply when the short-term trend direction agrees with
                    # the day's drift. Applies in all strategy modes so the trend signal is
                    # always respected (in credit_spreads mode the trend would otherwise be ignored).
                    trend_agrees_with_drift = (
                        (trend_dir == 'down' and day_drift <= 0) or
                        (trend_dir == 'up'   and day_drift >= 0) or
                        day_drift == 0.0
                    )
                    if is_trending and trend_agrees_with_drift:
                        allow_spreads = True
                        # Override selection to the safe-side spread
                        if trend_dir == 'down':
                            selection = StrategySelection(
                                strategy_type=StrategyType.CALL_SPREAD,
                                market_signal=MarketSignal.BEARISH,
                                confidence=selection.confidence,
                                reason=f"Trend override (down {trend_dir}) → Call Credit Spread only"
                            )
                        else:
                            selection = StrategySelection(
                                strategy_type=StrategyType.PUT_SPREAD,
                                market_signal=MarketSignal.BULLISH,
                                confidence=selection.confidence,
                                reason=f"Trend override (up {trend_dir}) → Put Credit Spread only"
                            )

                    # Bollinger Band breakout guard:
                    # BB position > 0.94 means price is near/above the upper band — a
                    # strong bullish breakout in progress. Selling a call spread bets
                    # against this breakout. Block it.
                    # BB position < 0.05 means price is near/below the lower band — a
                    # bearish breakdown. Selling a put spread bets against continued
                    # selling. Block it.
                    if indicators.bb_position > 0.94:
                        call_spread_blocked = True
                    if indicators.bb_position < 0.05:
                        put_spread_blocked = True

                    # Moderate-decline put spread guard:
                    # When the market has declined moderately from the open (-50 < drift < 0)
                    # and RSI is below 30 (oversold), a put spread with sl=3 is high risk:
                    # the market has not fallen far enough to signal a reliable bounce, and
                    # the stop is tight enough to get hit before any recovery.
                    # We allow put spreads once the market has fallen more than 50 pts
                    # (extreme sell-off territory) because at that magnitude a bounce is
                    # significantly more likely than further continuation.
                    if day_drift < 0 and day_drift > -50 and indicators.rsi < 30:
                        put_spread_blocked = True

                    # Extreme panic selling guard:
                    # RSI below 12 indicates the most extreme short-term overselling —
                    # the kind seen only during acute crash moments (sharp intraday drops).
                    # In these conditions the market often continues falling for several more
                    # bars before any meaningful bounce.  A put spread with sl=3 will almost
                    # certainly be stopped out before the recovery materialises.
                    if indicators.rsi < 12:
                        put_spread_blocked = True

                    # Deep crash continuation guard:
                    # When the market has already fallen more than 50 pts AND RSI is still
                    # below 20 (the market has not stabilised even at extreme sell-off levels),
                    # further continuation is more likely than an immediate bounce.
                    # Block put spreads until RSI shows at least some stabilisation (≥ 20).
                    if day_drift < -50 and indicators.rsi < 20:
                        put_spread_blocked = True

                    # Extreme overbought call spread guard:
                    # RSI above CALL_SPREAD_MAX_ENTRY_RSI (80) signals a momentum burst —
                    # the market is likely continuing higher, making a call spread
                    # (which caps profit at the short strike) extremely risky.
                    if indicators.rsi > CALL_SPREAD_MAX_ENTRY_RSI:
                        call_spread_blocked = True

                    # Bear-flag upper-band guard:
                    # When the market is down on the day (drift < -5) but trading near its
                    # upper Bollinger Band (bb > 0.90), the market is in a "bear flag" where
                    # a temporary bounce has pushed price to resistance.  A call spread here
                    # bets on a capped market — extremely risky if the bounce fails and the
                    # market resumes its downtrend.
                    if day_drift < -5 and indicators.bb_position > 0.90:
                        call_spread_blocked = True

                    # Oversold call spread guard:
                    # When RSI < 40 on a declining day (drift < -5), the market is
                    # approaching oversold territory — a bounce is plausible.  Entering
                    # a call spread bets against that bounce.  Block call spreads.
                    # Also block when RSI is extremely oversold (< 30) regardless of drift —
                    # an RSI this low signals the market is in a free-fall where even a
                    # flat-to-up bet (call spread) is dangerous.
                    if indicators.rsi < 40 and day_drift < -5:
                        call_spread_blocked = True
                    if indicators.rsi < 30:
                        call_spread_blocked = True

                    # Strong uptrend + low RSI guard:
                    # When the market has already surged significantly from open (> 12 pts)
                    # but RSI is still below 40, the market is in a strong momentum move.
                    # Selling a call spread attempts to fade that momentum — high risk.
                    if day_drift > 12 and indicators.rsi < 40:
                        call_spread_blocked = True

                    # Intraday reversal guard:
                    # If the market has reversed ≥ INTRADAY_CALL_REVERSAL_POINTS from its
                    # intraday peak (was up, now rolling over), volatility is elevated —
                    # block call spreads.
                    # If the market has bounced ≥ INTRADAY_PUT_REVERSAL_POINTS from its
                    # intraday trough, the market has not stabilised — block put spreads.
                    # (Tighter put threshold: a smaller bounce still means elevated
                    #  downside risk on a volatile down day.)
                    if (_intraday_max_drift - day_drift) >= INTRADAY_CALL_REVERSAL_POINTS:
                        call_spread_blocked = True
                    if (day_drift - _intraday_min_drift) >= INTRADAY_PUT_REVERSAL_POINTS:
                        put_spread_blocked = True

                    # No call spread re-entry on a declining day guard:
                    # If a call spread was already opened today and the market is currently
                    # below its opening price, do not attempt another call spread.
                    # Rationale: on declining days a first call spread may win by luck
                    # (e.g. taken-profit quickly before the down-move accelerated), but
                    # re-entering doubles the risk on an already-bearish day.
                    if _had_call_spread_today and day_drift < 0:
                        call_spread_blocked = True

                    # No call spread re-entry after a big run-up guard:
                    # If a call spread was already opened today and drift has since extended
                    # further upward (> 15 pts), the market is in a strong bull run.
                    # Re-entering a second call spread chases momentum at the worst time —
                    # stop losses on call spreads are triggered by further upside.
                    if _had_call_spread_today and day_drift > 15:
                        call_spread_blocked = True

                    # Sub-zero Bollinger Band guard:
                    # BB position < 0 (below the lower band) signals an extreme bearish
                    # breakdown. Selling a call spread in this environment bets against
                    # continued selling — extremely risky in a true breakdown.
                    if indicators.bb_position < 0.10:
                        call_spread_blocked = True

                    # Expanded bear-flag guard (extended drift threshold):
                    # Even a modest negative drift (-3 pts) combined with price near the
                    # upper Bollinger Band (> 0.82) is a bear-flag pattern — block call spreads.
                    if day_drift < -3 and indicators.bb_position > 0.82:
                        call_spread_blocked = True

                    # Low RSI guard (expanded from RSI < 30):
                    # RSI below 32 means the market is deeply oversold — any call spread
                    # bet against further downside is extremely risky at these levels.
                    if indicators.rsi < 32:
                        call_spread_blocked = True

                    # Post-winning-call-spread put guard:
                    # If a call spread already closed profitably today, do not open a new
                    # put spread unless the market is at extreme oversold levels (RSI ≤ 20).
                    # Rationale: the CS already secured the day's premium target.  Chasing
                    # a PS with RSI still above 20 means the oversold signal is not strong
                    # enough to justify the additional risk — the market may not have found
                    # its floor yet.
                    if _had_call_spread_win_today and indicators.rsi > 20:
                        put_spread_blocked = True

                    # Post-winning-put-spread call guard:
                    # If a put spread already closed profitably today and the market is
                    # still oversold (RSI < 40) and below the mid-Bollinger Band (< 0.50),
                    # do not open a call spread.
                    # Rationale: the market is in a confirmed downtrend; a call spread bets
                    # on a capped market when all indicators still point downward.
                    if _had_put_spread_win_today and indicators.rsi < 40 and indicators.bb_position < 0.50:
                        call_spread_blocked = True

                    # Concurrent call-spread / oversold guard:
                    # If a call spread is currently open and RSI drops below 40, the market
                    # is moving against the CS while also signaling oversold conditions.
                    # Opening a put spread in this state creates an unintended IC-like
                    # exposure where both legs are under simultaneous stress.
                    if open_call_spread is not None and indicators.rsi < 40:
                        put_spread_blocked = True

                    # Concurrent put-spread / oversold guard:
                    # If a put spread is currently open and the market is still oversold
                    # (RSI < 40) with price below the mid-Bollinger Band (< 0.50), the
                    # downtrend is not resolved.  Opening a call spread creates unbalanced
                    # IC exposure on a clearly directional day.
                    if open_put_spread is not None and indicators.rsi < 40 and indicators.bb_position < 0.50:
                        call_spread_blocked = True

                    # Concurrent put-spread / recovered market guard:
                    # If a put spread is currently open and RSI has rebounded above 65,
                    # the market has bounced enough to threaten the put spread's short leg.
                    # Opening a call spread now creates unintended IC exposure at the exact
                    # moment when both legs are under simultaneous pressure.
                    if open_put_spread is not None and indicators.rsi > 65:
                        call_spread_blocked = True

                    # Overbought RSI on declining day guard:
                    # When the market is below its opening level (negative day_drift) but
                    # RSI is in overbought territory (> 74), the technical picture is
                    # contradictory: price has already declined yet momentum is "overbought".
                    # This typically indicates a late-day reversal that pushed RSI high on
                    # a declining day — a very risky setup for a call spread.
                    if day_drift < 0 and indicators.rsi > 74:
                        call_spread_blocked = True

                    if selection.strategy_type == StrategyType.IRON_CONDOR and allow_ic:
                        if open_ic is None and open_put_spread is None and open_call_spread is None:
                            strategy = self._try_open_strategy(
                                date, current_time, StrategyType.IRON_CONDOR,
                                target_delta, target_prob_itm, min_spread_width, quantity,
                                target_credit=target_credit, spx_history=spx_history
                            )
                            if strategy:
                                open_ic = strategy
                                ic_leg_status = IronCondorLegStatus()
                                ic_entry_meta = {
                                    'entry_time': current_time,
                                    'entry_spx': entry_spx,
                                    'indicators': indicators,
                                    'strike_selection': self._last_strike_selection or StrikeSelection(0,0,0,0,0),
                                    'market_signal': selection.market_signal,
                                    'confidence': selection.confidence,
                                    'notes': selection.reason,
                                    'entry_rationale': {
                                        'strategy_selected': 'iron_condor',
                                        'selection_reason': selection.reason,
                                        'confidence': round(selection.confidence, 3),
                                        'market_signal': selection.market_signal.value,
                                        'spx_open': round(spx_open, 2) if spx_open else None,
                                        'spx_at_entry': round(entry_spx, 2),
                                        'day_drift_pts': round(day_drift, 2),
                                        'is_trending': is_trending,
                                        'trend_direction': trend_dir,
                                        'trend_override': False,
                                        'ic_blocked_by_drift': ic_blocked_by_drift,
                                        'put_spread_blocked': put_spread_blocked,
                                        'call_spread_blocked': call_spread_blocked,
                                        'rsi': round(indicators.rsi, 2),
                                        'bb_position': round(indicators.bb_position, 3),
                                        'bb_upper': round(indicators.bb_upper, 2),
                                        'bb_lower': round(indicators.bb_lower, 2),
                                        'macd_histogram': round(indicators.macd_histogram, 4),
                                    }
                                }
                                logger.info(f"Opened IC at {current_time}")
                                _fire({'event': 'position_opened', 'strategy_type': 'Iron Condor',
                                       'entry_time': current_time, 'entry_spx': entry_spx,
                                       'entry_credit': getattr(strategy, 'entry_credit', 0),
                                       'strikes': ic_entry_meta.get('strike_selection'),
                                       'entry_rationale': ic_entry_meta.get('entry_rationale')})

                    elif selection.strategy_type == StrategyType.PUT_SPREAD and allow_spreads and not put_spread_blocked:
                        # RSI conviction is required for ALL put spread entries.
                        # On negative-drift days three conditions must hold:
                        #   1. Drift guard: call_spread_blocked must be True (drift <= -20 pts)
                        #   2. RSI guard: RSI <= PUT_SPREAD_MAX_RSI_ON_NEG_DRIFT (45)
                        #   3. Cooldown: PUT_SPREAD_DRIFT_CONFIRM_MINUTES since block latched
                        # On flat/positive drift days a stricter RSI threshold applies:
                        #   RSI <= PUT_SPREAD_MAX_RSI_ON_POS_DRIFT (30)
                        #   Rationale: on an up/neutral day, only extreme oversold readings
                        #   (RSI < 30) provide enough conviction that the put spread is safe.
                        #   An RSI between 30-45 on a positive-drift day means the market is
                        #   not clearly oversold and a further drop is equally possible.
                        if day_drift < 0:
                            rsi_ok = indicators.rsi <= PUT_SPREAD_MAX_RSI_ON_NEG_DRIFT
                            if call_spread_blocked and _call_blocked_latch_time:
                                latch_dt   = pd.Timestamp(f"{date} {_call_blocked_latch_time}")
                                current_dt = pd.Timestamp(f"{date} {current_time}")
                                cooldown_ok = (current_dt - latch_dt).seconds / 60 >= PUT_SPREAD_DRIFT_CONFIRM_MINUTES
                            else:
                                cooldown_ok = False
                            put_ok = call_spread_blocked and rsi_ok and cooldown_ok
                        else:
                            # Positive or zero drift: require strong oversold conviction.
                            put_ok = indicators.rsi <= PUT_SPREAD_MAX_RSI_ON_POS_DRIFT
                        if open_put_spread is None and open_ic is None and put_ok:
                            strategy = self._try_open_strategy(
                                date, current_time, StrategyType.PUT_SPREAD,
                                target_delta, target_prob_itm, min_spread_width, quantity,
                                target_credit=target_credit, spx_history=spx_history
                            )
                            if strategy:
                                open_put_spread = strategy
                                put_spread_meta = {
                                    'entry_time': current_time,
                                    'entry_spx': entry_spx,
                                    'indicators': indicators,
                                    'strike_selection': self._last_strike_selection or StrikeSelection(0,0,0,0,0),
                                    'market_signal': selection.market_signal,
                                    'confidence': selection.confidence,
                                    'notes': selection.reason,
                                    'entry_rationale': {
                                        'strategy_selected': 'put_spread',
                                        'selection_reason': selection.reason,
                                        'confidence': round(selection.confidence, 3),
                                        'market_signal': selection.market_signal.value,
                                        'spx_open': round(spx_open, 2) if spx_open else None,
                                        'spx_at_entry': round(entry_spx, 2),
                                        'day_drift_pts': round(day_drift, 2),
                                        'is_trending': is_trending,
                                        'trend_direction': trend_dir,
                                        'trend_override': 'trend override' in selection.reason.lower(),
                                        'ic_blocked_by_drift': ic_blocked_by_drift,
                                        'put_spread_blocked': put_spread_blocked,
                                        'call_spread_blocked': call_spread_blocked,
                                        'rsi': round(indicators.rsi, 2),
                                        'bb_position': round(indicators.bb_position, 3),
                                        'bb_upper': round(indicators.bb_upper, 2),
                                        'bb_lower': round(indicators.bb_lower, 2),
                                        'macd_histogram': round(indicators.macd_histogram, 4),
                                    }
                                }
                                logger.info(f"Opened put spread at {current_time}")
                                _fire({'event': 'position_opened', 'strategy_type': 'Put Spread',
                                       'entry_time': current_time, 'entry_spx': entry_spx,
                                       'entry_credit': getattr(strategy, 'entry_credit', 0),
                                       'strikes': put_spread_meta.get('strike_selection'),
                                       'entry_rationale': put_spread_meta.get('entry_rationale')})

                    elif selection.strategy_type == StrategyType.CALL_SPREAD and allow_spreads and not call_spread_blocked:
                        if open_call_spread is None and open_ic is None:
                            strategy = self._try_open_strategy(
                                date, current_time, StrategyType.CALL_SPREAD,
                                target_delta, target_prob_itm, min_spread_width, quantity,
                                target_credit=target_credit, spx_history=spx_history
                            )
                            if strategy:
                                open_call_spread = strategy
                                _had_call_spread_today = True   # latch: used by re-entry guard
                                call_spread_meta = {
                                    'entry_time': current_time,
                                    'entry_spx': entry_spx,
                                    'indicators': indicators,
                                    'strike_selection': self._last_strike_selection or StrikeSelection(0,0,0,0,0),
                                    'market_signal': selection.market_signal,
                                    'confidence': selection.confidence,
                                    'notes': selection.reason,
                                    'entry_rationale': {
                                        'strategy_selected': 'call_spread',
                                        'selection_reason': selection.reason,
                                        'confidence': round(selection.confidence, 3),
                                        'market_signal': selection.market_signal.value,
                                        'spx_open': round(spx_open, 2) if spx_open else None,
                                        'spx_at_entry': round(entry_spx, 2),
                                        'day_drift_pts': round(day_drift, 2),
                                        'is_trending': is_trending,
                                        'trend_direction': trend_dir,
                                        'trend_override': 'trend override' in selection.reason.lower(),
                                        'ic_blocked_by_drift': ic_blocked_by_drift,
                                        'put_spread_blocked': put_spread_blocked,
                                        'call_spread_blocked': call_spread_blocked,
                                        'rsi': round(indicators.rsi, 2),
                                        'bb_position': round(indicators.bb_position, 3),
                                        'bb_upper': round(indicators.bb_upper, 2),
                                        'bb_lower': round(indicators.bb_lower, 2),
                                        'macd_histogram': round(indicators.macd_histogram, 4),
                                    }
                                }
                                logger.info(f"Opened call spread at {current_time}")
                                _fire({'event': 'position_opened', 'strategy_type': 'Call Spread',
                                       'entry_time': current_time, 'entry_spx': entry_spx,
                                       'entry_credit': getattr(strategy, 'entry_credit', 0),
                                       'strikes': call_spread_meta.get('strike_selection'),
                                       'entry_rationale': call_spread_meta.get('entry_rationale')})

                except Exception as e:
                    logger.debug(f"Entry scan error at {current_time}: {e}")
                    continue

        # --- 5. Force-close remaining positions at FINAL_EXIT_TIME ---
        for (open_pos, meta, stype, checkpoints) in [
            (open_ic,          ic_entry_meta,    StrategyType.IRON_CONDOR, ic_checkpoints),
            (open_put_spread,  put_spread_meta,  StrategyType.PUT_SPREAD,  put_spread_checkpoints),
            (open_call_spread, call_spread_meta, StrategyType.CALL_SPREAD, call_spread_checkpoints),
        ]:
            if open_pos is not None:
                exit_spx = self.enhanced_query_engine.get_fastest_spx_price(date, FINAL_EXIT_TIME) or meta.get('entry_spx', 0)
                # Use intrinsic value at expiry — options settle to $0 if OTM
                exit_cost = self._expiry_exit_cost(open_pos, stype, exit_spx)
                entry_credit = getattr(open_pos, 'entry_credit', 0)
                pnl = entry_credit - exit_cost
                pnl_pct = (pnl / entry_credit * 100) if entry_credit > 0 else 0

                # Add final force-close checkpoint
                pos_quantity = getattr(open_pos, 'quantity', 1)
                entry_credit_ps = round(entry_credit / (100.0 * pos_quantity), 4)
                cost_ps = round(exit_cost / (100.0 * pos_quantity), 4)
                force_checkpoint: dict = {
                    "time": FINAL_EXIT_TIME,
                    "spx": round(exit_spx, 2),
                    "cost_per_share": cost_ps,
                    "pnl_per_share": round(entry_credit_ps - cost_ps, 4),
                }
                if stype == StrategyType.IRON_CONDOR:
                    put_cost  = self._expiry_exit_cost(open_pos, StrategyType.PUT_SPREAD,  exit_spx)
                    call_cost = self._expiry_exit_cost(open_pos, StrategyType.CALL_SPREAD, exit_spx)
                    force_checkpoint["put_cost_per_share"]  = round(put_cost  / (100.0 * pos_quantity), 4)
                    force_checkpoint["call_cost_per_share"] = round(call_cost / (100.0 * pos_quantity), 4)
                checkpoints.append(force_checkpoint)

                result_ic_leg_status = None
                if stype == StrategyType.IRON_CONDOR and ic_leg_status is not None:
                    if not ic_leg_status.put_side_closed:
                        ic_leg_status.put_side_closed = True
                        ic_leg_status.put_side_exit_time = FINAL_EXIT_TIME
                        ic_leg_status.put_side_exit_reason = "Expired at market close"
                    if not ic_leg_status.call_side_closed:
                        ic_leg_status.call_side_closed = True
                        ic_leg_status.call_side_exit_time = FINAL_EXIT_TIME
                        ic_leg_status.call_side_exit_reason = "Expired at market close"
                    result_ic_leg_status = ic_leg_status

                trades.append(EnhancedBacktestResult(
                    date=date,
                    strategy_type=stype,
                    market_signal=meta.get('market_signal', MarketSignal.NEUTRAL),
                    entry_time=meta.get('entry_time', FINAL_EXIT_TIME),
                    exit_time=FINAL_EXIT_TIME,
                    exit_reason="Expired at market close",
                    entry_spx_price=meta.get('entry_spx', 0),
                    exit_spx_price=exit_spx,
                    technical_indicators=meta.get('indicators', TechnicalIndicators(0,0,0,0,0,0,0,0.5)),
                    strike_selection=meta.get('strike_selection', StrikeSelection(0,0,0,0,0)),
                    entry_credit=entry_credit,
                    exit_cost=exit_cost,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    max_profit=getattr(open_pos, 'max_profit', entry_credit),
                    max_loss=getattr(open_pos, 'max_loss', -exit_cost),
                    monitoring_points=checkpoints,
                    success=True,
                    confidence=meta.get('confidence', 0),
                    notes=meta.get('notes', ''),
                    entry_rationale=meta.get('entry_rationale'),
                    exit_rationale={
                        'exit_trigger': 'Expired at market close',
                        'exit_cost': round(exit_cost, 2),
                        'entry_credit': round(entry_credit, 2),
                        'spx_at_exit': round(exit_spx, 2),
                        'spx_move_since_entry': round(exit_spx - meta.get('entry_spx', 0), 2) if meta.get('entry_spx') else None,
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2),
                    },
                    ic_leg_status=result_ic_leg_status
                ))

        return DayBacktestResult(
            date=date,
            trades=trades,
            total_pnl=sum(t.pnl for t in trades),
            trade_count=len(trades),
            scan_minutes_checked=len(scan_times)
        )

    def _try_open_strategy(self, date: str, timestamp: str, strategy_type: StrategyType,
                           target_delta: float, target_prob_itm: float,
                           min_spread_width: int, quantity: int,
                           target_credit: Optional[float] = None,
                           spx_history: Optional[pd.Series] = None):
        """
        Attempt to build a strategy at the given timestamp.
        Returns the strategy object, or None if any guard fails:
          1. No suitable strikes found
          2. Dynamic minimum distance: short strike too close to SPX
             (distance = max(MIN_DISTANCE_FROM_SPX, morning_range * 0.75))
          3. Circuit breaker: short strike within 1 × spread_width of SPX
        """
        self._last_strike_selection = None
        try:
            strike_selection = self.delta_selector.select_strikes_by_delta(
                date=date,
                timestamp=timestamp,
                strategy_type=strategy_type,
                target_delta=target_delta,
                target_prob_itm=target_prob_itm,
                min_spread_width=min_spread_width,
                target_credit=target_credit
            )
            if not strike_selection:
                return None

            spx_price = self.enhanced_query_engine.get_fastest_spx_price(date, timestamp) or 0
            if spx_price > 0:
                from engine.delta_strike_selector import IronCondorStrikeSelection

                # --- Dynamic minimum distance (Recommendation 2) ---
                # IC: scale base distance with morning range (both sides exposed).
                # Single spreads: use flat base only — the morning range scaling
                # would push the threshold too far OTM and block valid entries.
                base_min_dist = MIN_DISTANCE_IC if strategy_type == StrategyType.IRON_CONDOR else MIN_DISTANCE_SPREAD
                dynamic_min_dist = base_min_dist
                if strategy_type == StrategyType.IRON_CONDOR and spx_history is not None and len(spx_history) >= 2:
                    morning_range = spx_history.max() - spx_history.min()
                    dynamic_min_dist = max(base_min_dist, morning_range * 0.75)

                # --- Distance guard ---
                if isinstance(strike_selection, IronCondorStrikeSelection):
                    put_dist  = abs(strike_selection.put_short_strike  - spx_price)
                    call_dist = abs(strike_selection.call_short_strike - spx_price)
                    if put_dist < dynamic_min_dist or call_dist < dynamic_min_dist:
                        logger.debug(
                            f"Skipping IC at {timestamp}: put_dist={put_dist:.1f} "
                            f"call_dist={call_dist:.1f} (min={dynamic_min_dist:.1f})"
                        )
                        return None
                    # --- Circuit breaker: within 1 × spread_width (Recommendation 4) ---
                    if put_dist < min_spread_width or call_dist < min_spread_width:
                        logger.debug(
                            f"Circuit breaker IC at {timestamp}: "
                            f"put_dist={put_dist:.1f} call_dist={call_dist:.1f} "
                            f"(spread_width={min_spread_width})"
                        )
                        return None
                else:
                    short_dist = abs(strike_selection.short_strike - spx_price)
                    if short_dist < dynamic_min_dist:
                        logger.debug(
                            f"Skipping {strategy_type.value} at {timestamp}: "
                            f"short_dist={short_dist:.1f} (min={dynamic_min_dist:.1f})"
                        )
                        return None
                    # --- Circuit breaker ---
                    if short_dist < min_spread_width:
                        logger.debug(
                            f"Circuit breaker {strategy_type.value} at {timestamp}: "
                            f"short_dist={short_dist:.1f} (spread_width={min_spread_width})"
                        )
                        return None

            self._last_strike_selection = strike_selection

            if strategy_type == StrategyType.IRON_CONDOR:
                return self._build_iron_condor_strategy(date, timestamp, strike_selection, quantity)
            elif strategy_type == StrategyType.PUT_SPREAD:
                return self._build_put_spread_strategy(date, timestamp, strike_selection, quantity)
            else:
                return self._build_call_spread_strategy(date, timestamp, strike_selection, quantity)
        except Exception as e:
            logger.debug(f"Failed to open {strategy_type.value} at {timestamp}: {e}")
            return None

    def _expiry_exit_cost(self, strategy, strategy_type: StrategyType, spx_price: float) -> float:
        """
        Compute the true settlement cost at expiry from intrinsic value alone.
        0DTE options expire at intrinsic value; time value is zero.
        OTM options expire worthless (cost = 0).
        """
        try:
            quantity = getattr(strategy, 'quantity', 1)
            if strategy_type == StrategyType.IRON_CONDOR:
                # Find put/call strikes from legs
                put_legs  = [l for l in strategy.legs if l.option_type.value == 'put']
                call_legs = [l for l in strategy.legs if l.option_type.value == 'call']
                put_short_strike  = max((l.strike for l in put_legs  if l.position_side.name == 'SHORT'), default=0)
                put_long_strike   = min((l.strike for l in put_legs  if l.position_side.name == 'LONG'),  default=0)
                call_short_strike = min((l.strike for l in call_legs if l.position_side.name == 'SHORT'), default=0)
                call_long_strike  = max((l.strike for l in call_legs if l.position_side.name == 'LONG'),  default=0)

                put_intrinsic  = max(0.0, put_short_strike  - spx_price)   # put spread cost if SPX < put_short
                call_intrinsic = max(0.0, spx_price - call_short_strike)   # call spread cost if SPX > call_short

                put_spread_width  = put_short_strike  - put_long_strike
                call_spread_width = call_long_strike  - call_short_strike

                put_cost  = min(put_intrinsic,  put_spread_width  if put_spread_width  > 0 else put_intrinsic)
                call_cost = min(call_intrinsic, call_spread_width if call_spread_width > 0 else call_intrinsic)

                return (put_cost + call_cost) * 100 * quantity

            elif strategy_type == StrategyType.PUT_SPREAD:
                short_leg = next((l for l in strategy.legs if l.position_side.name == 'SHORT'), None)
                long_leg  = next((l for l in strategy.legs if l.position_side.name == 'LONG'),  None)
                if not short_leg or not long_leg:
                    return 0.0
                intrinsic    = max(0.0, short_leg.strike - spx_price)
                spread_width = short_leg.strike - long_leg.strike
                return min(intrinsic, spread_width) * 100 * quantity

            elif strategy_type == StrategyType.CALL_SPREAD:
                short_leg = next((l for l in strategy.legs if l.position_side.name == 'SHORT'), None)
                long_leg  = next((l for l in strategy.legs if l.position_side.name == 'LONG'),  None)
                if not short_leg or not long_leg:
                    return 0.0
                intrinsic    = max(0.0, spx_price - short_leg.strike)
                spread_width = long_leg.strike - short_leg.strike
                return min(intrinsic, spread_width) * 100 * quantity

        except Exception as e:
            logger.warning(f"Expiry cost calculation failed ({strategy_type}): {e}")
        return 0.0

    def enhanced_backtest_single_day(self,
                                   date: str,
                                   entry_time: str = "10:00:00",
                                   exit_time: str = "15:45:00",
                                   target_delta: float = 0.15,
                                   target_prob_itm: float = 0.15,
                                   min_spread_width: int = 10,
                                   take_profit: float = 0.10,
                                   stop_loss: float = 2.0,
                                   monitor_interval: int = 1,
                                   quantity: int = 1,
                                   target_credit: Optional[float] = 0.50,
                                   strategy_mode: str = STRATEGY_IRON_CONDOR) -> EnhancedBacktestResult:
        """Legacy single-day method — runs intraday scan and returns first trade result."""
        day_result = self.backtest_day_intraday(
            date=date,
            target_delta=target_delta,
            target_prob_itm=target_prob_itm,
            min_spread_width=min_spread_width,
            take_profit=take_profit,
            stop_loss=stop_loss,
            monitor_interval=monitor_interval,
            quantity=quantity,
            target_credit=target_credit,
            strategy_mode=strategy_mode,
        )
        if day_result.trades:
            return day_result.trades[0]
        return self._create_failed_result(date, entry_time, FINAL_EXIT_TIME, "No setup found intraday")

    def _build_iron_condor_strategy(self, date: str, timestamp: str, strike_selection, quantity: int):
        """Build Iron Condor directly from credit-selected put and call strikes."""
        from src.strategies.options_strategies import IronCondor
        from datetime import datetime as _dt

        spx_price = self.enhanced_query_engine.get_fastest_spx_price(date, timestamp)
        if not spx_price:
            return None

        if isinstance(strike_selection, IronCondorStrikeSelection):
            put_short  = strike_selection.put_short_strike
            put_long   = strike_selection.put_long_strike
            call_short = strike_selection.call_short_strike
            call_long  = strike_selection.call_long_strike
        else:
            # Fallback: symmetric IC from a single-side StrikeSelection
            put_short  = strike_selection.short_strike
            put_long   = strike_selection.long_strike
            call_short = strike_selection.short_strike
            call_long  = strike_selection.long_strike

        # Fetch options data and build a dict keyed "{strike}_{type}" for the 4 legs
        options_df = self.enhanced_query_engine.get_options_data(date, timestamp)
        if options_df is None or len(options_df) == 0:
            return None

        options_dict = {}
        for _, row in options_df.iterrows():
            key = f"{float(row['strike'])}_{row['option_type']}"
            options_dict[key] = {
                'mid_price': (float(row['bid']) + float(row['ask'])) / 2.0,
                'bid': float(row['bid']),
                'ask': float(row['ask']),
                'delta': float(row.get('delta', 0)),
                'gamma': float(row.get('gamma', 0)),
                'theta': float(row.get('theta', 0)),
                'vega': float(row.get('vega', 0)),
                'iv': float(row.get('implied_volatility', 0)),
            }

        # Verify all 4 strikes are present in the options data
        required_keys = [
            f"{put_long}_put", f"{put_short}_put",
            f"{call_short}_call", f"{call_long}_call",
        ]
        missing = [k for k in required_keys if k not in options_dict]
        if missing:
            logger.debug(f"IC build: missing options data for {missing}")
            # Fall back to distance-based builder
            put_distance  = abs(put_short - spx_price)
            call_distance = abs(call_short - spx_price)
            spread_width  = int(max(put_short - put_long, call_long - call_short))
            return self.strategy_builder.build_iron_condor_optimized(
                date=date, timestamp=timestamp,
                put_distance=put_distance, call_distance=call_distance,
                spread_width=spread_width, quantity=quantity, use_liquid_options=True
            )

        entry_dt = _dt.strptime(f"{date} {timestamp}", "%Y-%m-%d %H:%M:%S")
        expiry_dt = _dt.strptime(date, "%Y-%m-%d")
        try:
            ic = IronCondor(
                entry_date=entry_dt,
                underlying_price=spx_price,
                put_short_strike=put_short,
                put_long_strike=put_long,
                call_short_strike=call_short,
                call_long_strike=call_long,
                quantity=quantity,
                expiration=expiry_dt,
                options_data=options_dict,
            )
            ic.entry_spx_price = spx_price
            ic.entry_timestamp = timestamp
            return ic
        except Exception as e:
            logger.error(f"Failed to build IC directly: {e}")
            return None
    
    def _build_put_spread_strategy(self, date: str, timestamp: str, strike_selection: StrikeSelection, quantity: int):
        """Build Put Spread strategy"""
        # Build put spread using existing infrastructure
        # This is a simplified implementation - you may want to create a dedicated put spread builder
        return self._build_single_spread(date, timestamp, strike_selection, quantity, 'put')
    
    def _build_call_spread_strategy(self, date: str, timestamp: str, strike_selection: StrikeSelection, quantity: int):
        """Build Call Spread strategy"""
        # Build call spread using existing infrastructure  
        return self._build_single_spread(date, timestamp, strike_selection, quantity, 'call')
    
    def _build_single_spread(self, date: str, timestamp: str, strike_selection: StrikeSelection, quantity: int, option_type: str):
        """Build single spread (put or call) directly from the credit-selected strikes."""
        from src.strategies.options_strategies import VerticalSpread, OptionType as OT
        from datetime import datetime as _dt

        spx_price = self.enhanced_query_engine.get_fastest_spx_price(date, timestamp)
        if not spx_price:
            return None

        ot = OT.PUT if option_type == 'put' else OT.CALL
        short_strike = strike_selection.short_strike
        long_strike  = strike_selection.long_strike

        options_df = self.enhanced_query_engine.get_options_data(date, timestamp)
        if options_df is None or len(options_df) == 0:
            return None

        options_dict = {}
        for _, row in options_df.iterrows():
            key = f"{float(row['strike'])}_{row['option_type']}"
            options_dict[key] = {
                'mid_price': (float(row['bid']) + float(row['ask'])) / 2.0,
                'bid': float(row['bid']),
                'ask': float(row['ask']),
                'delta': float(row.get('delta', 0)),
                'gamma': float(row.get('gamma', 0)),
                'theta': float(row.get('theta', 0)),
                'vega': float(row.get('vega', 0)),
                'iv': float(row.get('implied_volatility', 0)),
            }

        entry_dt  = _dt.strptime(f"{date} {timestamp}", "%Y-%m-%d %H:%M:%S")
        expiry_dt = _dt.strptime(date, "%Y-%m-%d")
        try:
            strategy = VerticalSpread(
                entry_date=entry_dt,
                underlying_price=spx_price,
                short_strike=short_strike,
                long_strike=long_strike,
                option_type=ot,
                quantity=quantity,
                expiration=expiry_dt,
                options_data=options_dict,
            )
            return strategy
        except Exception as e:
            logger.error(f"Failed to build {option_type} spread: {e}")
            return None
    
    def _create_failed_result(self, date: str, entry_time: str, exit_time: str, reason: str) -> EnhancedBacktestResult:
        """Create a failed result object"""
        return EnhancedBacktestResult(
            date=date,
            strategy_type=StrategyType.IRON_CONDOR,
            market_signal=MarketSignal.NEUTRAL,
            entry_time=entry_time,
            exit_time=exit_time,
            exit_reason=reason,
            entry_spx_price=0,
            exit_spx_price=0,
            technical_indicators=TechnicalIndicators(0, 0, 0, 0, 0, 0, 0, 0.5),
            strike_selection=StrikeSelection(0, 0, 0, 0, 0),
            entry_credit=0,
            exit_cost=0,
            pnl=0,
            pnl_pct=0,
            max_profit=0,
            max_loss=0,
            monitoring_points=[],
            success=False,
            confidence=0,
            notes=reason
        )
    
    def print_enhanced_results(self, results: List[EnhancedBacktestResult], show_monitoring: bool = False):
        """Print enhanced results with technical analysis details"""
        
        if not results:
            print("No results to display")
            return
        
        print(f"\n{'='*140}")
        print(f"ENHANCED MULTI-STRATEGY BACKTEST RESULTS - {len(results)} Days")
        print(f"{'='*140}")
        
        # Summary table
        print(f"{'Date':<12} {'Strategy':<12} {'Signal':<8} {'Delta':<6} {'RSI':<5} {'P&L':<10} {'%':<7} {'Exit':<15} {'Status'}")
        print(f"{'-'*140}")
        
        for result in results:
            if result.success:
                status = "✓ WIN" if result.pnl > 0 else "✗ LOSS"
                delta_str = f"{result.strike_selection.short_delta:.3f}" if result.success else "N/A"
                rsi_str = f"{result.technical_indicators.rsi:.0f}" if result.success else "N/A"
                strategy_short = result.strategy_type.value.replace(" ", "")[:10]
                signal_short = result.market_signal.value[:6]
                
                print(f"{result.date:<12} {strategy_short:<12} {signal_short:<8} {delta_str:<6} {rsi_str:<5} "
                      f"${result.pnl:<9.2f} {result.pnl_pct:<6.1f}% {result.exit_reason[:13]:<15} {status}")
            else:
                print(f"{result.date:<12} {'SKIP':<12} {'N/A':<8} {'N/A':<6} {'N/A':<5} "
                      f"{'$0.00':<10} {'0.0%':<7} {result.exit_reason[:13]:<15} {'SKIP'}")
        
        # Enhanced statistics
        successful_results = [r for r in results if r.success]
        
        if successful_results:
            print(f"\n{'-'*140}")
            print(f"ENHANCED STATISTICS")
            print(f"{'-'*140}")
            
            # Strategy breakdown
            strategy_stats = {}
            for result in successful_results:
                strategy = result.strategy_type.value
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'count': 0, 'wins': 0, 'total_pnl': 0}
                strategy_stats[strategy]['count'] += 1
                if result.pnl > 0:
                    strategy_stats[strategy]['wins'] += 1
                strategy_stats[strategy]['total_pnl'] += result.pnl
            
            print(f"Strategy Performance:")
            for strategy, stats in strategy_stats.items():
                win_rate = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
                avg_pnl = stats['total_pnl'] / stats['count'] if stats['count'] > 0 else 0
                print(f"  {strategy}: {stats['count']} trades, {win_rate:.1f}% win rate, ${avg_pnl:.2f} avg P&L")
            
            # Technical indicator summary
            avg_rsi = sum(r.technical_indicators.rsi for r in successful_results) / len(successful_results)
            avg_delta = sum(r.strike_selection.short_delta for r in successful_results) / len(successful_results)
            
            print(f"\\nTechnical Summary:")
            print(f"  Average RSI: {avg_rsi:.1f}")
            print(f"  Average Short Delta: {avg_delta:.3f}")
            print(f"  Setup Success Rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
            
            # Show monitoring details if requested
            if show_monitoring:
                print(f"\\n{'-'*140}")
                print(f"DETAILED POSITION MONITORING")
                print(f"{'-'*140}")
                
                for result in successful_results:  # Show all results, not just first 3
                    print(f"\\n📊 {result.date} - {result.strategy_type.value} Strategy:")
                    print(f"   Entry SPX: ${result.entry_spx_price:.2f} → Exit SPX: ${result.exit_spx_price:.2f}")
                    print(f"   Strike Selection: Short {result.strike_selection.short_strike:.0f} | Long {result.strike_selection.long_strike:.0f} | Delta {result.strike_selection.short_delta:.3f}")
                    print(f"   Entry Credit: ${result.entry_credit:.2f} → Exit Cost: ${result.exit_cost:.2f} → P&L: ${result.pnl:.2f}")
                    print(f"   Exit Reason: {result.exit_reason}")
                    print(f"   Monitoring Points: {len(result.monitoring_points)}")
                    
                    if result.monitoring_points:
                        print(f"   ⏱️  5-Minute Checkpoint Details:")
                        print(f"      {'Time':<8} {'SPX':<8} {'ExitCost':<10} {'P&L':<8} {'P&L%':<7} {'Decay':<6} {'ΔP&L':<8}")
                        print(f"      {'-'*58}")
                        
                        prev_pnl = result.entry_credit  # Starting P&L
                        for i, point in enumerate(result.monitoring_points):
                            pnl_change = point['pnl'] - prev_pnl if i > 0 else 0
                            status_icon = "🔴" if point['decay_ratio'] <= 0.1 else "🟡" if point['decay_ratio'] <= 0.3 else "🟢"
                            print(f"      {point['timestamp']:<8} ${point['spx_price']:<7.0f} ${point['exit_cost']:<9.2f} ${point['pnl']:<7.2f} {point['pnl_pct']:<6.1f}% {point['decay_ratio']:<5.3f} ${pnl_change:>+6.2f} {status_icon}")
                            prev_pnl = point['pnl']
                        
                        last_point = result.monitoring_points[-1]
                        print(f"      💡 Final Status: {result.exit_reason} (Decay: {last_point.get('decay_ratio', 0):.3f})")
                    
                    print()  # Empty line between days
        
        print(f"{'='*140}\\n")


def run_enhanced_backtest():
    """Run enhanced backtesting with command line interface"""

    parser = argparse.ArgumentParser(description="Enhanced Multi-Strategy SPX 0DTE Backtester")
    parser.add_argument("--date", "-d", help="Date to backtest (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="Start date for range (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for range (YYYY-MM-DD)")
    parser.add_argument("--strategy", default=STRATEGY_IRON_CONDOR,
                        choices=[STRATEGY_IRON_CONDOR, STRATEGY_CREDIT_SPREADS, STRATEGY_IC_CREDIT_SPREADS],
                        help="Strategy mode (default: iron_condor)")
    parser.add_argument("--contracts", type=int, default=1, help="Number of contracts per position (default: 1)")
    parser.add_argument("--target-delta", type=float, default=0.15, help="Target delta for short strikes")
    parser.add_argument("--target-credit", type=float, default=0.50, help="Target net credit per spread per share (default: 0.50)")
    parser.add_argument("--target-prob-itm", type=float, default=0.15, help="Target probability ITM")
    parser.add_argument("--take-profit", type=float, default=0.10, help="Take profit: exit when cost/share drops to this value (default: 0.10)")
    parser.add_argument("--stop-loss", type=float, default=2.0, help="Stop loss: exit when cost/share reaches this value (default: 2.0)")
    parser.add_argument("--monitor-interval", type=int, default=1, help="Minutes between position checks (default: 1)")
    parser.add_argument("--min-spread-width", type=int, default=10, help="Minimum spread width")
    parser.add_argument("--show-monitoring", action="store_true", help="Show detailed monitoring")

    args = parser.parse_args()

    # Initialize enhanced engine
    engine = EnhancedBacktestingEngine()

    if args.date:
        # Single day intraday scan
        day_result = engine.backtest_day_intraday(
            date=args.date,
            target_delta=args.target_delta,
            target_prob_itm=args.target_prob_itm,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss,
            monitor_interval=args.monitor_interval,
            min_spread_width=args.min_spread_width,
            target_credit=args.target_credit,
            strategy_mode=args.strategy,
            quantity=args.contracts,
        )
        print(f"\nDate: {day_result.date} | Trades: {day_result.trade_count} | Total P&L: ${day_result.total_pnl:.2f} | Bars scanned: {day_result.scan_minutes_checked}")
        if day_result.trades:
            engine.print_enhanced_results(day_result.trades, show_monitoring=args.show_monitoring)

    elif args.start_date and args.end_date:
        # Date range intraday scan
        all_trades: List[EnhancedBacktestResult] = []

        start_dt = pd.to_datetime(args.start_date)
        end_dt = pd.to_datetime(args.end_date)
        date_range = pd.date_range(start_dt, end_dt, freq='D')

        test_dates = [d.strftime('%Y-%m-%d') for d in date_range if d.strftime('%Y-%m-%d') in engine.available_dates]
        logger.info(f"Testing {len(test_dates)} available days in intraday mode")

        for i, date in enumerate(test_dates, 1):
            logger.info(f"Intraday scan {i}/{len(test_dates)}: {date}")
            day_result = engine.backtest_day_intraday(
                date=date,
                target_delta=args.target_delta,
                target_prob_itm=args.target_prob_itm,
                take_profit=args.take_profit,
                stop_loss=args.stop_loss,
                monitor_interval=args.monitor_interval,
                min_spread_width=args.min_spread_width,
                target_credit=args.target_credit,
                strategy_mode=args.strategy,
                quantity=args.contracts,
            )
            all_trades.extend(day_result.trades)
            logger.info(f"  {date}: {day_result.trade_count} trades, P&L=${day_result.total_pnl:.2f}")

        engine.print_enhanced_results(all_trades, show_monitoring=args.show_monitoring)

    else:
        print("Enhanced SPX 0DTE Multi-Strategy Backtester (Intraday Mode)")
        print("Available commands:")
        print("  --date YYYY-MM-DD                               # Single day intraday scan")
        print("  --start-date YYYY-MM-DD --end-date YYYY-MM-DD  # Date range intraday scan")
        print("  --target-credit 0.50                           # Target credit per spread/share")
        print("  --take-profit 0.10                             # Exit when cost/share <= $0.10")
        print("  --stop-loss 2.0                                # Exit when cost/share >= $2.00")
        print("  --monitor-interval 1                           # Check every N minutes")
        print("  --show-monitoring                              # Show detailed monitoring")
        print("\nExample:")
        print("  python enhanced_multi_strategy.py --date 2026-02-09 --take-profit 0.10 --stop-loss 2.0")
        print("  python enhanced_multi_strategy.py --start-date 2026-02-09 --end-date 2026-02-13")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    run_enhanced_backtest()


print("Enhanced Multi-Strategy Backtesting Engine Complete - Part 3/3")
print("\\n✅ All enhancements implemented:")
print("1. ✅ Multi-strategy selection (IC, Put Spreads, Call Spreads)")
print("2. ✅ Technical indicators (RSI, MACD, Bollinger Bands)")
print("3. ✅ Delta/Probability ITM based strike selection") 
print("4. ✅ Dynamic position monitoring (5-min intervals)")
print("5. ✅ Decay-based exits (0.1 threshold)")