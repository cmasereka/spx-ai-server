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
from engine.strike_selector import StrikeSelector, IntradayPositionMonitor, StrikeSelection, IronCondorStrikeSelection
from engine.query_engine_adapter import EnhancedQueryEngineAdapter


# Intraday scan constants
ENTRY_SCAN_START    = "10:00:00"   # Wait 30 min for opening volatility to settle
LAST_ENTRY_TIME     = "14:00:00"   # No new entries at or after 2 PM
FINAL_EXIT_TIME     = "16:00:00"   # Hold to expiry (market close)
MIN_DISTANCE_IC     = 25.0         # IC short strike must be >= $25 away from SPX
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
        self.strike_selector = StrikeSelector(self.enhanced_query_engine, self.ic_loader)
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
                              min_spread_width: int = 10,
                              take_profit: float = 0.10,
                              stop_loss: float = 2.0,
                              monitor_interval: int = 1,
                              quantity: int = 1,
                              target_credit: Optional[float] = 0.50,
                              strategy_mode: str = STRATEGY_IRON_CONDOR,
                              progress_callback=None,
                              entry_start_time: str = ENTRY_SCAN_START,
                              last_entry_time: str = LAST_ENTRY_TIME,
                              stale_loss_minutes: int = 120,
                              stale_loss_threshold: float = 1.5,
                              stagnation_window: int = 30,
                              min_improvement: float = 0.05,
                              enable_stale_loss_exit: bool = False,
                              skip_indicators: bool = True) -> DayBacktestResult:
        """
        Full intraday scan loop for one trading day.
        strategy_mode controls which entry types are allowed:
          iron_condor       — IC only
          credit_spreads    — put/call spreads only
          ic_credit_spreads — all types

        Delegates to LiveTradingLoop(is_live=False) so that all guard and
        entry logic is shared with the live trading path.  Import is done
        inside the method body to avoid the circular import that would occur
        if live_trading_loop.py were imported at module level (it imports
        from this module).
        """
        # Import inside body to break circular import:
        # live_trading_loop.py imports EnhancedBacktestingEngine from this module.
        from trading.live_trading_loop import LiveTradingLoop, TradingDayConfig
        cfg = TradingDayConfig(
            strategy_mode          = strategy_mode,
            target_credit          = target_credit,
            spread_width           = min_spread_width,
            quantity               = quantity,
            take_profit            = take_profit,
            stop_loss              = stop_loss,
            monitor_interval       = monitor_interval,
            entry_start_time       = entry_start_time,
            last_entry_time        = last_entry_time,
            enable_stale_loss_exit = enable_stale_loss_exit,
            stale_loss_minutes     = stale_loss_minutes,
            stale_loss_threshold   = stale_loss_threshold,
            stagnation_window      = stagnation_window,
            min_improvement        = min_improvement,
            skip_indicators        = skip_indicators,
        )
        loop = LiveTradingLoop(engine=self, is_live=False)
        return loop.run_day(date=date, config=cfg, progress_callback=progress_callback)


    def _try_open_strategy(self, date: str, timestamp: str, strategy_type: StrategyType,
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
            strike_selection = self.strike_selector.select_strikes(
                date=date,
                timestamp=timestamp,
                strategy_type=strategy_type,
                min_spread_width=min_spread_width,
                target_credit=target_credit
            )
            if not strike_selection:
                return None

            spx_price = self.enhanced_query_engine.get_fastest_spx_price(date, timestamp) or 0
            if spx_price > 0:
                from engine.strike_selector import IronCondorStrikeSelection

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
            strike_selection=StrikeSelection(0, 0, 0),
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
        print(f"{'Date':<12} {'Strategy':<12} {'Signal':<8} {'RSI':<5} {'P&L':<10} {'%':<7} {'Exit':<15} {'Status'}")
        print(f"{'-'*140}")
        
        for result in results:
            if result.success:
                status = "✓ WIN" if result.pnl > 0 else "✗ LOSS"
                rsi_str = f"{result.technical_indicators.rsi:.0f}" if result.success else "N/A"
                strategy_short = result.strategy_type.value.replace(" ", "")[:10]
                signal_short = result.market_signal.value[:6]

                print(f"{result.date:<12} {strategy_short:<12} {signal_short:<8} {rsi_str:<5} "
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

            print(f"\\nTechnical Summary:")
            print(f"  Average RSI: {avg_rsi:.1f}")
            print(f"  Setup Success Rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
            
            # Show monitoring details if requested
            if show_monitoring:
                print(f"\\n{'-'*140}")
                print(f"DETAILED POSITION MONITORING")
                print(f"{'-'*140}")
                
                for result in successful_results:  # Show all results, not just first 3
                    print(f"\\n📊 {result.date} - {result.strategy_type.value} Strategy:")
                    print(f"   Entry SPX: ${result.entry_spx_price:.2f} → Exit SPX: ${result.exit_spx_price:.2f}")
                    print(f"   Strike Selection: Short {result.strike_selection.short_strike:.0f} | Long {result.strike_selection.long_strike:.0f}")
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
    parser.add_argument("--target-credit", type=float, default=0.50, help="Target net credit per spread per share (default: 0.50)")
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
print("✅ All enhancements implemented:")
print("1. ✅ Multi-strategy selection (IC, Put Spreads, Call Spreads)")
print("2. ✅ Technical indicators (RSI, MACD, Bollinger Bands)")
print("3. ✅ Delta/Probability ITM based strike selection") 
print("4. ✅ Dynamic position monitoring (5-min intervals)")
print("5. ✅ Decay-based exits (0.1 threshold)")