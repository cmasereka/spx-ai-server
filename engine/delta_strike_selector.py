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

from engine.enhanced_backtest import (
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
    """Select strikes based on target credit."""

    def __init__(self, query_engine, ic_loader):
        self.query_engine = query_engine
        self.ic_loader = ic_loader

    def select_strikes_by_delta(self,
                                date: str,
                                timestamp: str,
                                strategy_type: StrategyType,
                                min_spread_width: int = 10,
                                target_credit: Optional[float] = None,
                                **_ignored) -> Optional[StrikeSelection]:
        """
        Select strikes based on target credit per spread per share.

        Args:
            min_spread_width: Spread width in strike points.
            target_credit: Desired net credit per spread per share (e.g. 0.50).
        """
        if target_credit is None:
            logger.warning("select_strikes_by_delta: target_credit is required")
            return None

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

            if strategy_type == StrategyType.PUT_SPREAD:
                return self._select_put_spread_by_credit(options_data, spx_price, min_spread_width, target_credit)
            elif strategy_type == StrategyType.CALL_SPREAD:
                return self._select_call_spread_by_credit(options_data, spx_price, min_spread_width, target_credit)
            else:  # IRON_CONDOR — select both sides independently
                put_strikes  = self._select_put_spread_by_credit(options_data, spx_price, min_spread_width, target_credit)
                call_strikes = self._select_call_spread_by_credit(options_data, spx_price, min_spread_width, target_credit)

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
                    short_delta=0.0,
                    short_prob_itm=0.0,
                    spread_width=short_strike - long_strike
                )

        # Outside both tolerances — return best available
        _, short_strike, long_strike, short_mid, long_mid, net_credit = candidates[0]
        logger.debug(f"Put spread (outside tolerance): {short_strike:.0f}/{long_strike:.0f} credit=${net_credit:.3f}")
        return StrikeSelection(
            short_strike=short_strike,
            long_strike=long_strike,
            short_delta=0.0,
            short_prob_itm=0.0,
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
                    short_delta=0.0,
                    short_prob_itm=0.0,
                    spread_width=long_strike - short_strike
                )

        _, short_strike, long_strike, short_mid, long_mid, net_credit = candidates[0]
        logger.debug(f"Call spread (outside tolerance): {short_strike:.0f}/{long_strike:.0f} credit=${net_credit:.3f}")
        return StrikeSelection(
            short_strike=short_strike,
            long_strike=long_strike,
            short_delta=0.0,
            short_prob_itm=0.0,
            spread_width=long_strike - short_strike
        )

    # ------------------------------------------------------------------
    # Delta-based selection — removed; credit-only selection is used
    # ------------------------------------------------------------------


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
                 monitor_interval: int = 1,
                 stale_loss_minutes: int = 120,
                 stale_loss_threshold: float = 1.5,
                 stagnation_window: int = 30,
                 min_improvement: float = 0.05):
        self.query_engine = query_engine
        self.strategy_builder = strategy_builder
        self.take_profit = take_profit      # close when cost/share <= this
        self.stop_loss = stop_loss          # close when cost/share >= this
        self.monitor_interval = monitor_interval
        # Stale-loss exit parameters
        self.stale_loss_minutes = stale_loss_minutes      # consecutive bars in the red required
        self.stale_loss_threshold = stale_loss_threshold  # cost must exceed entry_credit × this
        self.stagnation_window = stagnation_window        # look-back window for improvement check
        self.min_improvement = min_improvement            # minimum $/share improvement to not be stagnant

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_exit(self, current_cost: float, quantity: int = 1) -> Tuple[bool, str]:
        """
        Given the current cost-to-close for a spread (already scaled by quantity × 100),
        return (should_exit, reason) based on per-share absolute thresholds.
        Divides by (100 × quantity) to get the per-share cost for a single contract.
        """
        cost_per_share = current_cost / (100.0 * max(quantity, 1))

        if cost_per_share <= self.take_profit:
            return True, f"Take profit hit (${cost_per_share:.3f}/share <= ${self.take_profit:.2f})"
        if cost_per_share >= self.stop_loss:
            return True, f"Stop loss hit (${cost_per_share:.3f}/share >= ${self.stop_loss:.2f})"
        return False, ""

    def _should_exit_ic_side(self, side_cost: float, quantity: int = 1) -> Tuple[bool, str]:
        """Same check for a single IC side (put or call spread)."""
        return self._should_exit(side_cost, quantity)

    def check_stale_loss(self, checkpoints: list, entry_credit_per_share: float,
                         cost_key: str = "cost_per_share") -> Tuple[bool, str]:
        """
        Two-condition stale-loss exit check.

        Condition 1: The last `stale_loss_minutes` worth of bars all have
                     cost_per_share > entry_credit_per_share × stale_loss_threshold
                     (position has been meaningfully in the red for a sustained period).

        Condition 2: In the last `stagnation_window` minutes of bars the cost has not
                     improved (decreased) by more than `min_improvement` $/share
                     (position is stuck — no sign of recovery).

        Both conditions must be true to trigger.

        Note: checkpoints are only appended on check bars (every monitor_interval minutes),
        so minute-based thresholds are divided by monitor_interval to get bar counts.

        Returns (should_exit, reason_string).
        """
        # Convert minute-based thresholds to bar counts
        interval = max(1, self.monitor_interval)
        required_bars    = max(1, self.stale_loss_minutes // interval)
        stagnation_bars  = max(1, self.stagnation_window  // interval)

        if len(checkpoints) < required_bars:
            return False, ""

        threshold = entry_credit_per_share * self.stale_loss_threshold

        # Condition 1 — last N bars continuously above threshold
        recent = checkpoints[-required_bars:]
        all_red = all(cp.get(cost_key, 0.0) > threshold for cp in recent)
        if not all_red:
            return False, ""

        # Condition 2 — no meaningful improvement in the stagnation window
        window_size = min(stagnation_bars, len(checkpoints))
        window = checkpoints[-window_size:]
        costs = [cp.get(cost_key, 0.0) for cp in window]
        improvement = max(costs) - min(costs)   # total range; improvement = how much cost dropped

        if improvement < self.min_improvement:
            current_cost = costs[-1]
            elapsed_minutes = required_bars * interval
            window_minutes  = window_size  * interval
            return True, (
                f"Stale loss: cost ${current_cost:.3f}/share has exceeded "
                f"{self.stale_loss_threshold:.1f}× entry credit (${threshold:.3f}/share) "
                f"for {elapsed_minutes}+ minutes with only ${improvement:.3f}/share "
                f"movement in the last {window_minutes} minutes (min ${self.min_improvement:.2f} required)"
            )

        return False, ""

    def _get_ic_side_entry_credits(self, strategy, quantity: int = 1) -> Tuple[float, float]:
        """
        Compute per-share entry credit for each IC leg independently.
        Returns (put_credit_per_share, call_credit_per_share).
        """
        try:
            put_legs = [l for l in strategy.legs if l.option_type.value == 'put']
            call_legs = [l for l in strategy.legs if l.option_type.value == 'call']
        except AttributeError:
            put_legs = [l for l in strategy.legs if str(l.option_type).lower() == 'put']
            call_legs = [l for l in strategy.legs if str(l.option_type).lower() == 'call']

        def _side_credit(legs) -> float:
            credit = 0.0
            for leg in legs:
                leg_qty = getattr(leg, 'quantity', 1)
                entry_price = getattr(leg, 'entry_price', 0)
                is_short = getattr(leg, 'position_side', None)
                try:
                    short = is_short.name == 'SHORT'
                except AttributeError:
                    short = str(is_short).upper() == 'SHORT'
                sign = 1 if short else -1
                credit += entry_price * leg_qty * 100 * sign
            return max(credit, 0.0)

        put_credit = _side_credit(put_legs)
        call_credit = _side_credit(call_legs)
        per_share = 100.0 * max(quantity, 1)
        return put_credit / per_share, call_credit / per_share

    def check_decay(self, strategy, strategy_type: StrategyType) -> Tuple[bool, float, str]:
        """
        Check if position should exit.
        Returns: (should_exit, current_cost, reason)
        """
        try:
            self.strategy_builder.update_strategy_prices_optimized(strategy, None, None)
        except Exception:
            pass

        quantity = getattr(strategy, 'quantity', 1)
        current_cost = self._calculate_exit_cost(strategy)
        should_exit, reason = self._should_exit(current_cost, quantity)
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

        quantity = getattr(strategy, 'quantity', 1)
        current_cost = self._calculate_exit_cost(strategy)
        should_exit, reason = self._should_exit(current_cost, quantity)
        return should_exit, current_cost, reason

    def check_ic_leg_decay(self, strategy, date: str, current_time: str,
                           ic_leg_status: IronCondorLegStatus) -> IronCondorLegStatus:
        """
        For IC: update prices and check each side independently.
        Each side exits when its cost/share hits take_profit or stop_loss.
        stop_loss is per spread — applied identically to each IC leg.
        Updates ic_leg_status in place and returns it.
        """
        try:
            self.strategy_builder.update_strategy_prices_optimized(strategy, date, current_time)
        except Exception as e:
            logger.debug(f"IC price update failed at {current_time}: {e}")

        put_done, call_done, put_cost, call_cost = self._check_ic_leg_decay_values(strategy)
        quantity = getattr(strategy, 'quantity', 1)

        if put_done and not ic_leg_status.put_side_closed:
            ic_leg_status.put_side_closed = True
            ic_leg_status.put_side_exit_time = current_time
            ic_leg_status.put_side_exit_cost = put_cost
            _, reason = self._should_exit_ic_side(put_cost, quantity)
            ic_leg_status.put_side_exit_reason = reason

        if call_done and not ic_leg_status.call_side_closed:
            ic_leg_status.call_side_closed = True
            ic_leg_status.call_side_exit_time = current_time
            ic_leg_status.call_side_exit_cost = call_cost
            _, reason = self._should_exit_ic_side(call_cost, quantity)
            ic_leg_status.call_side_exit_reason = reason

        return ic_leg_status

    def _check_ic_leg_decay_values(self, strategy) -> Tuple[bool, bool, float, float]:
        """
        For IC: returns (put_side_done, call_side_done, put_cost, call_cost).
        A side is 'done' when its per-share cost hits take_profit or stop_loss.
        stop_loss is per spread — applied identically to each IC leg.
        Costs are already scaled by quantity; quantity is extracted from the strategy.
        """
        try:
            put_legs = [l for l in strategy.legs if l.option_type.value == 'put']
            call_legs = [l for l in strategy.legs if l.option_type.value == 'call']
        except AttributeError:
            put_legs = [l for l in strategy.legs if str(l.option_type).lower() == 'put']
            call_legs = [l for l in strategy.legs if str(l.option_type).lower() == 'call']

        quantity = getattr(strategy, 'quantity', 1)

        def _side_cost(legs) -> float:
            """Return current cost to close for a set of legs."""
            current_cost = 0.0
            for leg in legs:
                leg_qty = getattr(leg, 'quantity', 1)
                entry_price = getattr(leg, 'entry_price', 0)
                curr_price = getattr(leg, 'current_price', 0) or entry_price
                is_short = getattr(leg, 'position_side', None)
                try:
                    short = is_short.name == 'SHORT'
                except AttributeError:
                    short = str(is_short).upper() == 'SHORT'
                sign = 1 if short else -1
                current_cost += curr_price * leg_qty * 100 * sign
            return max(current_cost, 0.0)

        put_current_cost  = _side_cost(put_legs)
        call_current_cost = _side_cost(call_legs)

        put_done,  _ = self._should_exit_ic_side(put_current_cost,  quantity)
        call_done, _ = self._should_exit_ic_side(call_current_cost, quantity)

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