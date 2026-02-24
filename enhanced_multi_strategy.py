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

from enhanced_backtest import (
    StrategyType, MarketSignal, TechnicalIndicators, StrategySelection,
    EnhancedBacktestResult, TechnicalAnalyzer, StrategySelector,
    EnhancedMultiStrategyBacktester, IronCondorLegStatus, DayBacktestResult
)
from delta_strike_selector import DeltaStrikeSelector, PositionMonitor, IntradayPositionMonitor, StrikeSelection, IronCondorStrikeSelection
from query_engine_adapter import EnhancedQueryEngineAdapter


# Intraday scan constants
ENTRY_SCAN_START    = "09:35:00"   # 9:30 + first 5-min bar
LAST_ENTRY_TIME     = "14:00:00"   # No new entries at or after 2 PM
FINAL_EXIT_TIME     = "16:00:00"   # Hold to expiry (market close)
MIN_DISTANCE_FROM_SPX = 50.0       # Short strike must be >= $50 away from SPX

# Trend filter
TREND_FILTER_LOOKBACK_MINUTES = 30  # How far back to measure trend
TREND_FILTER_POINTS           = 30  # SPX points moved → market is trending; block IC, use asymmetric spread
TREND_TIGHTEN_CONSECUTIVE     = 3   # Consecutive adverse bars before SL tightens
TREND_TIGHTEN_FACTOR          = 0.5 # SL tightens to this fraction of configured stop_loss

# IC daily-drift guards
IC_DAILY_DRIFT_BLOCK    = 40.0      # Block IC entirely if SPX already down >= this many pts from 9:31 open
IC_MIN_DRIFT_FOR_DELAY  = 30.0      # If drifted >= this, no IC before IC_MIN_ENTRY_HOUR
IC_MIN_ENTRY_HOUR       = "12:00:00"  # Earliest IC entry allowed on a drifted day

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
                              strategy_mode: str = STRATEGY_IRON_CONDOR) -> DayBacktestResult:
        """
        Full intraday scan loop for one trading day.
        strategy_mode controls which entry types are allowed:
          iron_condor       — IC only
          credit_spreads    — put/call spreads only
          ic_credit_spreads — all types
        """
        logger.info(
            f"Intraday scan: {date} | mode={strategy_mode} credit={target_credit} "
            f"contracts={quantity} tp={take_profit} sl={stop_loss} interval={monitor_interval}m"
        )

        scan_times = _build_minute_grid(date, ENTRY_SCAN_START, FINAL_EXIT_TIME)
        trades: List[EnhancedBacktestResult] = []

        if date not in self.available_dates:
            return DayBacktestResult(date=date, trades=[], total_pnl=0.0,
                                     trade_count=0, scan_minutes_checked=0)

        # Fetch the opening SPX price once for daily-drift IC guards.
        # 09:31 is the first real bar; fall back to 0 (guard is disabled) if unavailable.
        spx_open = self.enhanced_query_engine.get_fastest_spx_price(date, "09:31:00") or 0.0
        logger.debug(f"Daily drift guard: SPX open @ 09:31 = {spx_open}")

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

        # Checkpoint lists — built up while a position is open, flushed on close
        ic_checkpoints        : list = []
        put_spread_checkpoints: list = []
        call_spread_checkpoints: list = []

        # Adverse-bar counters for dynamic SL tightening (Recommendation 5).
        # Incremented each check bar where cost increased vs. previous bar;
        # reset to 0 when cost decreases or a new position is opened.
        _ic_adverse_bars   = 0
        _ps_adverse_bars   = 0
        _cs_adverse_bars   = 0
        _ic_prev_cost      = 0.0
        _ps_prev_cost      = 0.0
        _cs_prev_cost      = 0.0

        for bar_index, current_time in enumerate(scan_times):
            is_past_entry_cutoff = current_time >= LAST_ENTRY_TIME
            is_final_bar         = current_time >= FINAL_EXIT_TIME

            # Respect monitor_interval; always process the final bar for the hard close.
            is_check_bar = (bar_index % monitor_interval == 0) or is_final_bar

            # --- 1. Monitor IC legs independently ---
            if open_ic is not None and ic_leg_status is not None and is_check_bar:
                # Dynamic SL tightening: after N consecutive adverse bars,
                # tighten stop_loss to TREND_TIGHTEN_FACTOR × configured value
                ic_total_cost = (ic_leg_status.put_side_exit_cost if ic_leg_status.put_side_closed else 0) + \
                                (ic_leg_status.call_side_exit_cost if ic_leg_status.call_side_closed else 0)
                if not ic_leg_status.put_side_closed or not ic_leg_status.call_side_closed:
                    try:
                        _, _, _rp, _rc = monitor._check_ic_leg_decay_values(open_ic)
                        ic_total_cost = (_rp if not ic_leg_status.put_side_closed else 0) + \
                                        (_rc if not ic_leg_status.call_side_closed else 0)
                    except Exception:
                        pass
                if ic_total_cost > _ic_prev_cost and _ic_prev_cost > 0:
                    _ic_adverse_bars += 1
                else:
                    _ic_adverse_bars = 0
                _ic_prev_cost = ic_total_cost

                if _ic_adverse_bars >= TREND_TIGHTEN_CONSECUTIVE:
                    monitor.stop_loss = stop_loss * TREND_TIGHTEN_FACTOR
                    logger.debug(f"IC SL tightened to {monitor.stop_loss:.2f} after {_ic_adverse_bars} adverse bars at {current_time}")
                else:
                    monitor.stop_loss = stop_loss

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
                    trades.append(EnhancedBacktestResult(
                        date=date,
                        strategy_type=StrategyType.IRON_CONDOR,
                        market_signal=ic_entry_meta.get('market_signal', MarketSignal.NEUTRAL),
                        entry_time=ic_entry_meta.get('entry_time', current_time),
                        exit_time=later_side_time,
                        exit_reason="IC both sides closed",
                        entry_spx_price=ic_entry_meta.get('entry_spx', 0),
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
                        ic_leg_status=ic_leg_status
                    ))
                    open_ic = None
                    ic_leg_status = None
                    ic_entry_meta = {}
                    ic_checkpoints = []

            # --- 2. Monitor put spread ---
            if open_put_spread is not None and is_check_bar:
                # Dynamic SL tightening for put spread
                if current_time != FINAL_EXIT_TIME:
                    try:
                        _ps_current = monitor._calculate_exit_cost(open_put_spread)
                    except Exception:
                        _ps_current = _ps_prev_cost
                    if _ps_current > _ps_prev_cost and _ps_prev_cost > 0:
                        _ps_adverse_bars += 1
                    else:
                        _ps_adverse_bars = 0
                    _ps_prev_cost = _ps_current
                    monitor.stop_loss = stop_loss * TREND_TIGHTEN_FACTOR if _ps_adverse_bars >= TREND_TIGHTEN_CONSECUTIVE else stop_loss

                should_exit, current_cost, reason = monitor.check_decay_at_time(
                    open_put_spread, StrategyType.PUT_SPREAD, date, current_time
                )
                monitor.stop_loss = stop_loss  # restore after check

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

                if should_exit or is_final_bar:
                    exit_reason = reason if should_exit else "Expired at market close"
                    entry_credit = ps_entry_credit
                    pnl = entry_credit - current_cost
                    pnl_pct = (pnl / entry_credit * 100) if entry_credit > 0 else 0
                    exit_spx = spx_now or put_spread_meta.get('entry_spx', 0)
                    trades.append(EnhancedBacktestResult(
                        date=date,
                        strategy_type=StrategyType.PUT_SPREAD,
                        market_signal=put_spread_meta.get('market_signal', MarketSignal.BULLISH),
                        entry_time=put_spread_meta.get('entry_time', current_time),
                        exit_time=current_time,
                        exit_reason=exit_reason,
                        entry_spx_price=put_spread_meta.get('entry_spx', 0),
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
                        notes=put_spread_meta.get('notes', '')
                    ))
                    open_put_spread = None
                    put_spread_meta = {}
                    put_spread_checkpoints = []

            # --- 3. Monitor call spread ---
            if open_call_spread is not None and is_check_bar:
                # Dynamic SL tightening for call spread
                if current_time != FINAL_EXIT_TIME:
                    try:
                        _cs_current = monitor._calculate_exit_cost(open_call_spread)
                    except Exception:
                        _cs_current = _cs_prev_cost
                    if _cs_current > _cs_prev_cost and _cs_prev_cost > 0:
                        _cs_adverse_bars += 1
                    else:
                        _cs_adverse_bars = 0
                    _cs_prev_cost = _cs_current
                    monitor.stop_loss = stop_loss * TREND_TIGHTEN_FACTOR if _cs_adverse_bars >= TREND_TIGHTEN_CONSECUTIVE else stop_loss

                should_exit, current_cost, reason = monitor.check_decay_at_time(
                    open_call_spread, StrategyType.CALL_SPREAD, date, current_time
                )
                monitor.stop_loss = stop_loss  # restore after check

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

                if should_exit or is_final_bar:
                    exit_reason = reason if should_exit else "Expired at market close"
                    entry_credit = cs_entry_credit
                    pnl = entry_credit - current_cost
                    pnl_pct = (pnl / entry_credit * 100) if entry_credit > 0 else 0
                    exit_spx = spx_now or call_spread_meta.get('entry_spx', 0)
                    trades.append(EnhancedBacktestResult(
                        date=date,
                        strategy_type=StrategyType.CALL_SPREAD,
                        market_signal=call_spread_meta.get('market_signal', MarketSignal.BEARISH),
                        entry_time=call_spread_meta.get('entry_time', current_time),
                        exit_time=current_time,
                        exit_reason=exit_reason,
                        entry_spx_price=call_spread_meta.get('entry_spx', 0),
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
                        notes=call_spread_meta.get('notes', '')
                    ))
                    open_call_spread = None
                    call_spread_meta = {}
                    call_spread_checkpoints = []

            # --- 4. Scan for new entry (only before 2 PM and before final bar) ---
            if not is_past_entry_cutoff and not is_final_bar:
                try:
                    spx_history = self.get_spx_price_history(date, current_time, lookback_minutes=60)
                    indicators = self.technical_analyzer.analyze_market_conditions(spx_history)
                    selection = self.strategy_selector.select_strategy(indicators)
                    entry_spx = self.enhanced_query_engine.get_fastest_spx_price(date, current_time) or 0

                    # --- Trend filter ---
                    is_trending, trend_dir = self._get_trend_state(date, current_time, spx_history)
                    if is_trending:
                        logger.debug(
                            f"Trend detected at {current_time}: {trend_dir} "
                            f"({TREND_FILTER_POINTS}+ pts over {TREND_FILTER_LOOKBACK_MINUTES}m)"
                        )

                    # --- IC daily-drift guards ---
                    # Guard 1: absolute drift block — if SPX has fallen >= IC_DAILY_DRIFT_BLOCK
                    #          from the opening price, no IC for the rest of the day.
                    # Guard 2: time-of-day delay — if SPX has fallen >= IC_MIN_DRIFT_FOR_DELAY
                    #          from open AND it is still before IC_MIN_ENTRY_HOUR, skip IC
                    #          (allow re-entry after noon if market has stabilised).
                    day_drift = (entry_spx - spx_open) if spx_open > 0 else 0.0
                    ic_blocked_by_drift = (
                        day_drift <= -IC_DAILY_DRIFT_BLOCK or
                        (day_drift <= -IC_MIN_DRIFT_FOR_DELAY and current_time < IC_MIN_ENTRY_HOUR)
                    )
                    if ic_blocked_by_drift:
                        logger.debug(
                            f"IC drift guard active at {current_time}: "
                            f"drift={day_drift:+.1f} pts from open ({spx_open:.1f})"
                        )

                    # Determine which entry types are permitted by this strategy mode.
                    # On trend days, IC is blocked; asymmetric spread is used instead
                    # (call spread on down-trend — sell premium above a falling market;
                    #  put spread on up-trend — sell premium below a rising market).
                    allow_ic      = strategy_mode in (STRATEGY_IRON_CONDOR, STRATEGY_IC_CREDIT_SPREADS) and not is_trending and not ic_blocked_by_drift
                    allow_spreads = strategy_mode in (STRATEGY_CREDIT_SPREADS, STRATEGY_IC_CREDIT_SPREADS)

                    # Asymmetric override: in iron_condor or ic_credit_spreads mode,
                    # when trending, allow the safe-side spread only
                    if is_trending and strategy_mode in (STRATEGY_IRON_CONDOR, STRATEGY_IC_CREDIT_SPREADS):
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
                                _ic_adverse_bars = 0
                                _ic_prev_cost    = 0.0
                                ic_entry_meta = {
                                    'entry_time': current_time,
                                    'entry_spx': entry_spx,
                                    'indicators': indicators,
                                    'strike_selection': self._last_strike_selection or StrikeSelection(0,0,0,0,0),
                                    'market_signal': selection.market_signal,
                                    'confidence': selection.confidence,
                                    'notes': selection.reason
                                }
                                logger.info(f"Opened IC at {current_time}")

                    elif selection.strategy_type == StrategyType.PUT_SPREAD and allow_spreads:
                        if open_put_spread is None and open_ic is None:
                            strategy = self._try_open_strategy(
                                date, current_time, StrategyType.PUT_SPREAD,
                                target_delta, target_prob_itm, min_spread_width, quantity,
                                target_credit=target_credit, spx_history=spx_history
                            )
                            if strategy:
                                open_put_spread = strategy
                                _ps_adverse_bars = 0
                                _ps_prev_cost    = 0.0
                                put_spread_meta = {
                                    'entry_time': current_time,
                                    'entry_spx': entry_spx,
                                    'indicators': indicators,
                                    'strike_selection': self._last_strike_selection or StrikeSelection(0,0,0,0,0),
                                    'market_signal': selection.market_signal,
                                    'confidence': selection.confidence,
                                    'notes': selection.reason
                                }
                                logger.info(f"Opened put spread at {current_time}")

                    elif selection.strategy_type == StrategyType.CALL_SPREAD and allow_spreads:
                        if open_call_spread is None and open_ic is None:
                            strategy = self._try_open_strategy(
                                date, current_time, StrategyType.CALL_SPREAD,
                                target_delta, target_prob_itm, min_spread_width, quantity,
                                target_credit=target_credit, spx_history=spx_history
                            )
                            if strategy:
                                open_call_spread = strategy
                                _cs_adverse_bars = 0
                                _cs_prev_cost    = 0.0
                                call_spread_meta = {
                                    'entry_time': current_time,
                                    'entry_spx': entry_spx,
                                    'indicators': indicators,
                                    'strike_selection': self._last_strike_selection or StrikeSelection(0,0,0,0,0),
                                    'market_signal': selection.market_signal,
                                    'confidence': selection.confidence,
                                    'notes': selection.reason
                                }
                                logger.info(f"Opened call spread at {current_time}")

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
                from delta_strike_selector import IronCondorStrikeSelection

                # --- Dynamic minimum distance (Recommendation 2) ---
                # Base distance scales with the morning's actual range so far
                dynamic_min_dist = MIN_DISTANCE_FROM_SPX
                if spx_history is not None and len(spx_history) >= 2:
                    morning_range = spx_history.max() - spx_history.min()
                    dynamic_min_dist = max(MIN_DISTANCE_FROM_SPX, morning_range * 0.75)

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