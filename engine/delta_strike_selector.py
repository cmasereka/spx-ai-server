#!/usr/bin/env python3
"""
Credit-Based Strike Selection and Position Monitoring
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
    def spread_width(self) -> float:
        return self.put_spread_width


class StrikeSelector:
    """Select strikes based on target credit."""

    def __init__(self, query_engine, ic_loader):
        self.query_engine = query_engine
        self.ic_loader = ic_loader

    def select_strikes(self,
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
            logger.warning("select_strikes: target_credit is required")
            return None

        try:
            spx_price = self.query_engine.get_fastest_spx_price(date, timestamp)
            if not spx_price:
                return None

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
                    put_spread_width=put_strikes.spread_width,
                    call_spread_width=call_strikes.spread_width,
                )

        except Exception as e:
            logger.error(f"Strike selection failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Credit-based selection
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
            candidates.append((abs(net_credit - target_credit), short_strike, long_strike, net_credit))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])

        # Try ±0.10 first, fall back to ±0.20, then best available
        for tolerance in (0.10, 0.20):
            valid = [c for c in candidates if c[0] <= tolerance]
            if valid:
                _, short_strike, long_strike, net_credit = valid[0]
                logger.debug(f"Put spread: {short_strike:.0f}/{long_strike:.0f} credit=${net_credit:.3f} (target=${target_credit:.2f})")
                return StrikeSelection(
                    short_strike=short_strike,
                    long_strike=long_strike,
                    spread_width=short_strike - long_strike,
                )

        _, short_strike, long_strike, net_credit = candidates[0]
        logger.debug(f"Put spread (outside tolerance): {short_strike:.0f}/{long_strike:.0f} credit=${net_credit:.3f}")
        return StrikeSelection(
            short_strike=short_strike,
            long_strike=long_strike,
            spread_width=short_strike - long_strike,
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
            candidates.append((abs(net_credit - target_credit), short_strike, long_strike, net_credit))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])

        for tolerance in (0.10, 0.20):
            valid = [c for c in candidates if c[0] <= tolerance]
            if valid:
                _, short_strike, long_strike, net_credit = valid[0]
                logger.debug(f"Call spread: {short_strike:.0f}/{long_strike:.0f} credit=${net_credit:.3f} (target=${target_credit:.2f})")
                return StrikeSelection(
                    short_strike=short_strike,
                    long_strike=long_strike,
                    spread_width=long_strike - short_strike,
                )

        _, short_strike, long_strike, net_credit = candidates[0]
        logger.debug(f"Call spread (outside tolerance): {short_strike:.0f}/{long_strike:.0f} credit=${net_credit:.3f}")
        return StrikeSelection(
            short_strike=short_strike,
            long_strike=long_strike,
            spread_width=long_strike - short_strike,
        )


# Backward-compatible alias
DeltaStrikeSelector = StrikeSelector


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
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.monitor_interval = monitor_interval
        self.stale_loss_minutes = stale_loss_minutes
        self.stale_loss_threshold = stale_loss_threshold
        self.stagnation_window = stagnation_window
        self.min_improvement = min_improvement

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_exit(self, current_cost: float, quantity: int = 1) -> Tuple[bool, str]:
        cost_per_share = current_cost / (100.0 * max(quantity, 1))
        if cost_per_share <= self.take_profit:
            return True, f"Take profit hit (${cost_per_share:.3f}/share <= ${self.take_profit:.2f})"
        if cost_per_share >= self.stop_loss:
            return True, f"Stop loss hit (${cost_per_share:.3f}/share >= ${self.stop_loss:.2f})"
        return False, ""

    def _should_exit_ic_side(self, side_cost: float, quantity: int = 1) -> Tuple[bool, str]:
        return self._should_exit(side_cost, quantity)

    def check_stale_loss(self, checkpoints: list, entry_credit_per_share: float,
                         cost_key: str = "cost_per_share") -> Tuple[bool, str]:
        """
        Two-condition stale-loss exit check.

        Condition 1: The last `stale_loss_minutes` worth of bars all have
                     cost_per_share > entry_credit_per_share × stale_loss_threshold.

        Condition 2: In the last `stagnation_window` minutes of bars the cost has not
                     improved by more than `min_improvement` $/share.

        Both conditions must be true to trigger.
        """
        interval = max(1, self.monitor_interval)
        required_bars   = max(1, self.stale_loss_minutes // interval)
        stagnation_bars = max(1, self.stagnation_window  // interval)

        if len(checkpoints) < required_bars:
            return False, ""

        threshold = entry_credit_per_share * self.stale_loss_threshold
        recent = checkpoints[-required_bars:]
        all_red = all(cp.get(cost_key, 0.0) > threshold for cp in recent)
        if not all_red:
            return False, ""

        window_size = min(stagnation_bars, len(checkpoints))
        window = checkpoints[-window_size:]
        costs = [cp.get(cost_key, 0.0) for cp in window]
        improvement = max(costs) - min(costs)

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
        try:
            put_legs  = [l for l in strategy.legs if l.option_type.value == 'put']
            call_legs = [l for l in strategy.legs if l.option_type.value == 'call']
        except AttributeError:
            put_legs  = [l for l in strategy.legs if str(l.option_type).lower() == 'put']
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

        put_credit  = _side_credit(put_legs)
        call_credit = _side_credit(call_legs)
        per_share   = 100.0 * max(quantity, 1)
        return put_credit / per_share, call_credit / per_share

    def check_decay_at_time(self, strategy, strategy_type: StrategyType,
                            date: str, current_time: str) -> Tuple[bool, float, str]:
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
        try:
            put_legs  = [l for l in strategy.legs if l.option_type.value == 'put']
            call_legs = [l for l in strategy.legs if l.option_type.value == 'call']
        except AttributeError:
            put_legs  = [l for l in strategy.legs if str(l.option_type).lower() == 'put']
            call_legs = [l for l in strategy.legs if str(l.option_type).lower() == 'call']

        quantity = getattr(strategy, 'quantity', 1)

        def _side_cost(legs) -> float:
            current_cost = 0.0
            for leg in legs:
                leg_qty    = getattr(leg, 'quantity', 1)
                entry_price = getattr(leg, 'entry_price', 0)
                curr_price  = getattr(leg, 'current_price', 0) or entry_price
                is_short    = getattr(leg, 'position_side', None)
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
        current_cost = 0.0
        try:
            for leg in strategy.legs:
                leg_price = getattr(leg, 'current_price', 0) or getattr(leg, 'entry_price', 0)
                quantity  = getattr(leg, 'quantity', 1)
                is_short  = getattr(leg, 'position_side', None)
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
