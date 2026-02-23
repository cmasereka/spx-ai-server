#!/usr/bin/env python3
"""
Delta-Based Strike Selection and Position Monitoring

Part 2: Strike selection using delta/probability ITM and dynamic monitoring
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger

from dataclasses import dataclass

from enhanced_backtest import (
    StrategyType, MarketSignal, StrikeSelection, EnhancedBacktestResult,
    TechnicalIndicators, StrategySelection, IronCondorLegStatus
)


@dataclass
class IronCondorStrikeSelection:
    """Strike selection for an Iron Condor — both put and call sides."""
    put_short_strike: float
    put_long_strike: float
    call_short_strike: float
    call_long_strike: float
    put_short_delta: float
    call_short_delta: float
    put_prob_itm: float
    call_prob_itm: float
    put_spread_width: float
    call_spread_width: float

    # Expose the same interface that downstream code uses from StrikeSelection
    @property
    def short_strike(self) -> float:
        """Put short strike (representative short strike for logging)."""
        return self.put_short_strike

    @property
    def long_strike(self) -> float:
        """Put long strike (representative long strike for logging)."""
        return self.put_long_strike

    @property
    def short_delta(self) -> float:
        return self.put_short_delta

    @property
    def short_prob_itm(self) -> float:
        return self.put_prob_itm

    @property
    def spread_width(self) -> float:
        return self.put_spread_width


class DeltaStrikeSelector:
    """Select strikes based on delta and probability ITM"""
    
    def __init__(self, query_engine, ic_loader):
        self.query_engine = query_engine
        self.ic_loader = ic_loader
    
    def select_strikes_by_delta(self,
                              date: str,
                              timestamp: str,
                              strategy_type: StrategyType,
                              target_delta: float = 0.15,
                              target_prob_itm: float = 0.15,
                              min_spread_width: int = 10,
                              target_credit: Optional[float] = None) -> Optional[StrikeSelection]:
        """
        Select strikes based on target credit per spread (preferred) or target delta.

        Args:
            target_delta: Target delta for short strike — used only when target_credit is None
            target_prob_itm: Target probability ITM — used only when target_credit is None
            min_spread_width: Spread width in strike points
            target_credit: Desired net credit per spread per share (e.g. 0.50).
                           Primary selection mode when provided.
        """
        
        try:
            # Get current SPX price
            spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
            if not spx_price:
                return None

            # Get options data
            options_data = self.query_engine.get_options_data(date, timestamp)
            if options_data is None or len(options_data) == 0:
                return None

            # Filter for 0DTE options
            options_data = options_data[options_data['expiration'] == date]

            # Choose selection method
            if target_credit is not None:
                put_selector  = lambda od, sp: self._select_put_spread_by_credit(od, sp, min_spread_width, target_credit)
                call_selector = lambda od, sp: self._select_call_spread_by_credit(od, sp, min_spread_width, target_credit)
            else:
                put_selector  = lambda od, sp: self._select_put_spread_strikes(od, sp, target_delta, target_prob_itm, min_spread_width)
                call_selector = lambda od, sp: self._select_call_spread_strikes(od, sp, target_delta, target_prob_itm, min_spread_width)

            if strategy_type == StrategyType.PUT_SPREAD:
                return put_selector(options_data, spx_price)
            elif strategy_type == StrategyType.CALL_SPREAD:
                return call_selector(options_data, spx_price)
            else:  # IRON_CONDOR — select both sides independently
                put_strikes  = put_selector(options_data, spx_price)
                call_strikes = call_selector(options_data, spx_price)

                if put_strikes is None or call_strikes is None:
                    return None

                return IronCondorStrikeSelection(
                    put_short_strike=put_strikes.short_strike,
                    put_long_strike=put_strikes.long_strike,
                    call_short_strike=call_strikes.short_strike,
                    call_long_strike=call_strikes.long_strike,
                    put_short_delta=put_strikes.short_delta,
                    call_short_delta=call_strikes.short_delta,
                    put_prob_itm=put_strikes.short_prob_itm,
                    call_prob_itm=call_strikes.short_prob_itm,
                    put_spread_width=put_strikes.spread_width,
                    call_spread_width=call_strikes.spread_width,
                )
                
        except Exception as e:
            logger.error(f"Strike selection failed: {e}")
            return None
    
    # ------------------------------------------------------------------
    # Credit-based selection (primary)
    # ------------------------------------------------------------------

    def _select_put_spread_by_credit(self, options_data: pd.DataFrame, spx_price: float,
                                     spread_width: int,
                                     target_credit: float) -> Optional[StrikeSelection]:
        """
        Select OTM put spread whose net credit (short_mid - long_mid) is closest
        to target_credit.  Tolerance: ±0.10 first, widened to ±0.20 if no match.
        """
        puts = options_data[
            (options_data['option_type'] == 'put') &
            (options_data['strike'] < spx_price) &
            (options_data['ask'] > 0)
        ].copy()
        if len(puts) == 0:
            return None

        puts['mid'] = (puts['bid'] + puts['ask']) / 2.0

        candidates = []
        for _, short_row in puts.iterrows():
            short_strike = float(short_row['strike'])
            long_strike  = short_strike - spread_width
            long_rows    = puts[puts['strike'] == long_strike]
            if len(long_rows) == 0:
                continue
            long_mid   = float(long_rows.iloc[0]['mid'])
            short_mid  = float(short_row['mid'])
            net_credit = short_mid - long_mid
            candidates.append((abs(net_credit - target_credit), short_strike, long_strike,
                                short_mid, long_mid, net_credit))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])

        # Try ±0.10 first, fall back to ±0.20
        for tolerance in (0.10, 0.20):
            valid = [c for c in candidates if c[0] <= tolerance]
            if valid:
                _, short_strike, long_strike, short_mid, long_mid, net_credit = valid[0]
                logger.debug(f"Put spread: {short_strike:.0f}/{long_strike:.0f} credit=${net_credit:.3f} (target=${target_credit:.2f})")
                return StrikeSelection(
                    short_strike=short_strike,
                    long_strike=long_strike,
                    short_delta=abs(puts[puts['strike'] == short_strike].iloc[0]['delta']),
                    short_prob_itm=abs(puts[puts['strike'] == short_strike].iloc[0]['delta']),
                    spread_width=short_strike - long_strike
                )

        # Outside both tolerances — return best available
        _, short_strike, long_strike, short_mid, long_mid, net_credit = candidates[0]
        logger.debug(f"Put spread (outside tolerance): {short_strike:.0f}/{long_strike:.0f} credit=${net_credit:.3f}")
        return StrikeSelection(
            short_strike=short_strike,
            long_strike=long_strike,
            short_delta=abs(puts[puts['strike'] == short_strike].iloc[0]['delta']),
            short_prob_itm=abs(puts[puts['strike'] == short_strike].iloc[0]['delta']),
            spread_width=short_strike - long_strike
        )

    def _select_call_spread_by_credit(self, options_data: pd.DataFrame, spx_price: float,
                                      spread_width: int,
                                      target_credit: float) -> Optional[StrikeSelection]:
        """
        Select OTM call spread whose net credit (short_mid - long_mid) is closest
        to target_credit.  Tolerance: ±0.10 first, widened to ±0.20 if no match.
        """
        calls = options_data[
            (options_data['option_type'] == 'call') &
            (options_data['strike'] > spx_price) &
            (options_data['ask'] > 0)
        ].copy()
        if len(calls) == 0:
            return None

        calls['mid'] = (calls['bid'] + calls['ask']) / 2.0

        candidates = []
        for _, short_row in calls.iterrows():
            short_strike = float(short_row['strike'])
            long_strike  = short_strike + spread_width
            long_rows    = calls[calls['strike'] == long_strike]
            if len(long_rows) == 0:
                continue
            long_mid   = float(long_rows.iloc[0]['mid'])
            short_mid  = float(short_row['mid'])
            net_credit = short_mid - long_mid
            candidates.append((abs(net_credit - target_credit), short_strike, long_strike,
                                short_mid, long_mid, net_credit))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])

        for tolerance in (0.10, 0.20):
            valid = [c for c in candidates if c[0] <= tolerance]
            if valid:
                _, short_strike, long_strike, short_mid, long_mid, net_credit = valid[0]
                logger.debug(f"Call spread: {short_strike:.0f}/{long_strike:.0f} credit=${net_credit:.3f} (target=${target_credit:.2f})")
                return StrikeSelection(
                    short_strike=short_strike,
                    long_strike=long_strike,
                    short_delta=calls[calls['strike'] == short_strike].iloc[0]['delta'],
                    short_prob_itm=calls[calls['strike'] == short_strike].iloc[0]['delta'],
                    spread_width=long_strike - short_strike
                )

        _, short_strike, long_strike, short_mid, long_mid, net_credit = candidates[0]
        logger.debug(f"Call spread (outside tolerance): {short_strike:.0f}/{long_strike:.0f} credit=${net_credit:.3f}")
        return StrikeSelection(
            short_strike=short_strike,
            long_strike=long_strike,
            short_delta=calls[calls['strike'] == short_strike].iloc[0]['delta'],
            short_prob_itm=calls[calls['strike'] == short_strike].iloc[0]['delta'],
            spread_width=long_strike - short_strike
        )

    # ------------------------------------------------------------------
    # Delta-based selection (legacy fallback)
    # ------------------------------------------------------------------

    def _select_put_spread_strikes(self, options_data: pd.DataFrame, spx_price: float,
                                 target_delta: float, target_prob_itm: float,
                                 min_spread_width: int) -> Optional[StrikeSelection]:
        """Select put spread strikes"""

        # Only consider OTM puts (strike < SPX) — ITM puts have no place as short strikes
        puts = options_data[
            (options_data['option_type'] == 'put') &
            (options_data['strike'] < spx_price)
        ].copy()
        if len(puts) == 0:
            return None

        # Find puts with delta magnitude closest to target (puts have negative delta)
        puts['abs_delta_diff'] = abs(abs(puts['delta']) - target_delta)
        puts = puts.sort_values('abs_delta_diff')

        # Select short strike (sell this)
        short_candidates = puts.head(5)  # Top 5 closest to target delta

        for _, short_option in short_candidates.iterrows():
            short_strike = short_option['strike']
            short_delta = abs(short_option['delta'])

            short_prob_itm = short_delta  # delta ≈ prob ITM for 0DTE

            # Long strike is further OTM (lower for puts)
            long_strike = short_strike - min_spread_width
            long_options = puts[puts['strike'] == long_strike]

            if len(long_options) > 0:
                return StrikeSelection(
                    short_strike=short_strike,
                    long_strike=long_strike,
                    short_delta=short_delta,
                    short_prob_itm=short_prob_itm,
                    spread_width=short_strike - long_strike
                )

        return None

    def _select_call_spread_strikes(self, options_data: pd.DataFrame, spx_price: float,
                                  target_delta: float, target_prob_itm: float,
                                  min_spread_width: int) -> Optional[StrikeSelection]:
        """Select call spread strikes"""

        # Only consider OTM calls (strike > SPX) — ITM calls have no place as short strikes
        calls = options_data[
            (options_data['option_type'] == 'call') &
            (options_data['strike'] > spx_price)
        ].copy()
        if len(calls) == 0:
            return None

        # Find calls with delta closest to target
        calls['abs_delta_diff'] = abs(calls['delta'] - target_delta)
        calls = calls.sort_values('abs_delta_diff')

        # Select short strike (sell this)
        short_candidates = calls.head(5)

        for _, short_option in short_candidates.iterrows():
            short_strike = short_option['strike']
            short_delta = short_option['delta']

            short_prob_itm = short_delta  # delta ≈ prob ITM for 0DTE

            # Long strike is further OTM (higher for calls)
            long_strike = short_strike + min_spread_width
            long_options = calls[calls['strike'] == long_strike]

            if len(long_options) > 0:
                return StrikeSelection(
                    short_strike=short_strike,
                    long_strike=long_strike,
                    short_delta=short_delta,
                    short_prob_itm=short_prob_itm,
                    spread_width=long_strike - short_strike
                )

        return None


class PositionMonitor:
    """Monitor positions with 5-minute intervals and decay-based exits"""
    
    def __init__(self, query_engine, strategy_builder):
        self.query_engine = query_engine
        self.strategy_builder = strategy_builder
    
    def monitor_position(self, 
                        strategy,
                        date: str,
                        entry_time: str,
                        exit_time: str,
                        strategy_type: StrategyType,
                        decay_threshold: float = 0.1) -> Tuple[List[Dict], str, float]:
        """
        Monitor position every 5 minutes with decay-based exits
        
        Returns:
            monitoring_points: List of monitoring data
            exit_reason: Why position was closed
            final_exit_cost: Final cost to close position
        """
        
        monitoring_points = []
        
        # Parse times
        entry_dt = datetime.strptime(f"{date} {entry_time}", "%Y-%m-%d %H:%M:%S")
        exit_dt = datetime.strptime(f"{date} {exit_time}", "%Y-%m-%d %H:%M:%S")
        
        # Monitor every 5 minutes starting 5 minutes after entry
        current_dt = entry_dt + timedelta(minutes=5)
        
        # Track for at least 30 minutes to show progression
        min_monitoring_time = entry_dt + timedelta(minutes=30)
        
        while current_dt <= exit_dt:
            current_time = current_dt.strftime("%H:%M:%S")
            
            try:
                # Update strategy prices
                price_update_success = self.strategy_builder.update_strategy_prices_optimized(strategy, date, current_time)
                logger.info(f"Price update at {current_time}: {'SUCCESS' if price_update_success else 'FAILED'}")
                
                # Log current leg prices for debugging
                for leg in strategy.legs:
                    logger.info(f"  Leg {leg.strike} {leg.option_type.value}: current_price={leg.current_price}, entry_price={leg.entry_price}")
                
                # Calculate current cost to close
                current_cost = self._calculate_exit_cost(strategy)
                
                # Calculate current P&L
                entry_credit = strategy.entry_credit
                current_pnl = entry_credit - current_cost
                current_pnl_pct = (current_pnl / entry_credit * 100) if entry_credit > 0 else 0
                
                # Calculate decay ratio - how much of the entry credit remains as cost
                decay_ratio = current_cost / entry_credit if entry_credit > 0 else 0
                
                monitoring_point = {
                    'timestamp': current_time,
                    'spx_price': self.query_engine.get_fastest_spx_price(date, current_time) or 0,
                    'exit_cost': current_cost,
                    'pnl': current_pnl,
                    'pnl_pct': current_pnl_pct,
                    'decay_ratio': decay_ratio,
                    'minutes_elapsed': (current_dt - entry_dt).total_seconds() / 60
                }
                
                monitoring_points.append(monitoring_point)
                
                # Check exit conditions only after minimum monitoring time
                if current_dt >= min_monitoring_time:
                    # Check decay threshold
                    if decay_ratio <= decay_threshold:
                        return monitoring_points, f"Decay threshold reached ({decay_ratio:.3f} <= {decay_threshold})", current_cost
                    
                    # Early profit taking (optional - 50% of max profit)
                    if strategy_type in [StrategyType.PUT_SPREAD, StrategyType.CALL_SPREAD]:
                        if current_pnl_pct >= 50:  # 50% profit
                            return monitoring_points, f"Early profit taking ({current_pnl_pct:.1f}%)", current_cost
                    
                    # Iron Condor early exit (25% profit)
                    elif strategy_type == StrategyType.IRON_CONDOR:
                        if current_pnl_pct >= 25:  # 25% profit
                            return monitoring_points, f"Early profit taking ({current_pnl_pct:.1f}%)", current_cost
                
            except Exception as e:
                logger.warning(f"Monitoring failed at {current_time}: {e}")
                # Continue monitoring with estimated values
                entry_credit = strategy.entry_credit
                estimated_cost = entry_credit * 0.3  # Assume 30% of entry credit as cost
                estimated_pnl = entry_credit - estimated_cost
                estimated_pnl_pct = (estimated_pnl / entry_credit * 100) if entry_credit > 0 else 0
                
                monitoring_point = {
                    'timestamp': current_time,
                    'spx_price': self.query_engine.get_fastest_spx_price(date, current_time) or 0,
                    'exit_cost': estimated_cost,
                    'pnl': estimated_pnl,
                    'pnl_pct': estimated_pnl_pct,
                    'decay_ratio': estimated_cost / entry_credit if entry_credit > 0 else 0.3,
                    'minutes_elapsed': (current_dt - entry_dt).total_seconds() / 60
                }
                monitoring_points.append(monitoring_point)
            
            # Next 5-minute interval
            current_dt += timedelta(minutes=5)
        
        # Position held to expiration
        final_cost = self._calculate_exit_cost(strategy)
        return monitoring_points, "Held to expiration", final_cost
    
    def _calculate_exit_cost(self, strategy) -> float:
        """Calculate current cost to close position"""
        current_cost = 0.0
        prices_updated = False

        try:
            for leg in strategy.legs:
                leg_price = leg.current_price

                # Check if price was actually updated (greater than 0)
                if leg_price > 0:
                    prices_updated = True
                else:
                    # More realistic fallback: use entry price as estimate
                    # This preserves the original value of the position for monitoring
                    leg_price = leg.entry_price
                    logger.debug(f"Using entry price fallback for {leg.strike} {leg.option_type.value}: {leg_price}")

                if leg.position_side.name == 'SHORT':
                    current_cost += leg_price * 100 * leg.quantity  # Cost to buy back short
                else:
                    current_cost -= leg_price * 100 * leg.quantity  # Credit from selling long

            # If no prices were updated, log warning
            if not prices_updated:
                logger.warning(f"No current prices available for strategy monitoring - using entry prices")

            return max(current_cost, 0.0)  # Ensure non-negative cost

        except Exception as e:
            logger.warning(f"Error calculating exit cost: {e}")
            # Return a reasonable fallback based on entry credit
            return getattr(strategy, 'entry_credit', 0) * 0.5  # Assume 50% of entry credit as cost


class IntradayPositionMonitor:
    """
    Manages multiple concurrent positions intraday.
    - IC: closes each side independently when take_profit or stop_loss is hit
    - Spreads: same per-share absolute thresholds
    - Monitors at monitor_interval-minute intervals
    """

    def __init__(self, query_engine, strategy_builder,
                 take_profit: float = 0.10,
                 stop_loss: float = 2.0,
                 monitor_interval: int = 1):
        self.query_engine = query_engine
        self.strategy_builder = strategy_builder
        self.take_profit = take_profit      # close when cost/share <= this
        self.stop_loss = stop_loss          # close when cost/share >= this
        self.monitor_interval = monitor_interval

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_exit(self, current_cost: float) -> Tuple[bool, str]:
        """
        Given the current cost-to-close for a spread (per contract = ×100),
        return (should_exit, reason) based on per-share absolute thresholds.
        """
        cost_per_share = current_cost / 100.0

        if cost_per_share <= self.take_profit:
            return True, f"Take profit hit (${cost_per_share:.3f}/share <= ${self.take_profit:.2f})"
        if cost_per_share >= self.stop_loss:
            return True, f"Stop loss hit (${cost_per_share:.3f}/share >= ${self.stop_loss:.2f})"
        return False, ""

    def _should_exit_ic_side(self, side_cost: float) -> Tuple[bool, str]:
        """Same check for a single IC side (put or call spread)."""
        return self._should_exit(side_cost)

    def check_decay(self, strategy, strategy_type: StrategyType) -> Tuple[bool, float, str]:
        """
        Check if position should exit.
        Returns: (should_exit, current_cost, reason)
        """
        try:
            self.strategy_builder.update_strategy_prices_optimized(strategy, None, None)
        except Exception:
            pass

        current_cost = self._calculate_exit_cost(strategy)
        should_exit, reason = self._should_exit(current_cost)
        return should_exit, current_cost, reason

    def check_decay_at_time(self, strategy, strategy_type: StrategyType,
                            date: str, current_time: str) -> Tuple[bool, float, str]:
        """
        Check if position should exit, updating prices first.
        Returns: (should_exit, current_cost, reason)
        """
        try:
            self.strategy_builder.update_strategy_prices_optimized(strategy, date, current_time)
        except Exception as e:
            logger.debug(f"Price update failed at {current_time}: {e}")

        current_cost = self._calculate_exit_cost(strategy)
        should_exit, reason = self._should_exit(current_cost)
        return should_exit, current_cost, reason

    def check_ic_leg_decay(self, strategy, date: str, current_time: str,
                           ic_leg_status: IronCondorLegStatus) -> IronCondorLegStatus:
        """
        For IC: update prices and check each side independently.
        Each side exits when its cost/share hits take_profit or stop_loss.
        Updates ic_leg_status in place and returns it.
        """
        try:
            self.strategy_builder.update_strategy_prices_optimized(strategy, date, current_time)
        except Exception as e:
            logger.debug(f"IC price update failed at {current_time}: {e}")

        put_done, call_done, put_cost, call_cost = self._check_ic_leg_decay_values(strategy)

        if put_done and not ic_leg_status.put_side_closed:
            ic_leg_status.put_side_closed = True
            ic_leg_status.put_side_exit_time = current_time
            ic_leg_status.put_side_exit_cost = put_cost
            _, reason = self._should_exit_ic_side(put_cost)
            ic_leg_status.put_side_exit_reason = reason

        if call_done and not ic_leg_status.call_side_closed:
            ic_leg_status.call_side_closed = True
            ic_leg_status.call_side_exit_time = current_time
            ic_leg_status.call_side_exit_cost = call_cost
            _, reason = self._should_exit_ic_side(call_cost)
            ic_leg_status.call_side_exit_reason = reason

        return ic_leg_status

    def _check_ic_leg_decay_values(self, strategy) -> Tuple[bool, bool, float, float]:
        """
        For IC: returns (put_side_done, call_side_done, put_cost, call_cost).
        A side is 'done' when its per-share cost hits take_profit or stop_loss.
        """
        try:
            put_legs = [l for l in strategy.legs if l.option_type.value == 'put']
            call_legs = [l for l in strategy.legs if l.option_type.value == 'call']
        except AttributeError:
            put_legs = [l for l in strategy.legs if str(l.option_type).lower() == 'put']
            call_legs = [l for l in strategy.legs if str(l.option_type).lower() == 'call']

        def _side_cost(legs) -> float:
            """Return current cost to close for a set of legs."""
            current_cost = 0.0
            for leg in legs:
                quantity = getattr(leg, 'quantity', 1)
                entry_price = getattr(leg, 'entry_price', 0)
                curr_price = getattr(leg, 'current_price', 0) or entry_price
                is_short = getattr(leg, 'position_side', None)
                try:
                    short = is_short.name == 'SHORT'
                except AttributeError:
                    short = str(is_short).upper() == 'SHORT'
                sign = 1 if short else -1
                current_cost += curr_price * quantity * 100 * sign
            return max(current_cost, 0.0)

        put_current_cost  = _side_cost(put_legs)
        call_current_cost = _side_cost(call_legs)

        put_done,  _ = self._should_exit_ic_side(put_current_cost)
        call_done, _ = self._should_exit_ic_side(call_current_cost)

        return put_done, call_done, put_current_cost, call_current_cost

    def _calculate_exit_cost(self, strategy) -> float:
        """Calculate current cost to close entire position."""
        current_cost = 0.0
        try:
            for leg in strategy.legs:
                leg_price = getattr(leg, 'current_price', 0) or getattr(leg, 'entry_price', 0)
                quantity = getattr(leg, 'quantity', 1)
                is_short = getattr(leg, 'position_side', None)
                try:
                    short = is_short.name == 'SHORT'
                except AttributeError:
                    short = str(is_short).upper() == 'SHORT'
                if short:
                    current_cost += leg_price * 100 * quantity
                else:
                    current_cost -= leg_price * 100 * quantity
            return max(current_cost, 0.0)
        except Exception as e:
            logger.warning(f"Error calculating exit cost: {e}")
            return getattr(strategy, 'entry_credit', 0) * 0.5