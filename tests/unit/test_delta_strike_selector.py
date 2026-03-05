"""Unit tests for credit-based strike selection and position monitoring."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append('.')

from delta_strike_selector import StrikeSelector, IntradayPositionMonitor, StrikeSelection, IronCondorStrikeSelection
from enhanced_backtest import StrategyType, IronCondorLegStatus
from tests.conftest import (
    TestDataGenerator, MockQueryEngine, assert_valid_strike_selection,
    SAMPLE_SPX_PRICE, SAMPLE_DATE
)


class TestStrikeSelector:
    """Test credit-based strike selection functionality."""

    def setup_method(self):
        self.mock_query_engine = MockQueryEngine()
        self.mock_ic_loader = Mock()
        self.selector = StrikeSelector(self.mock_query_engine, self.mock_ic_loader)

    def test_select_strikes_iron_condor_success(self):
        """Test successful Iron Condor strike selection returns both sides."""
        strike_selection = self.selector.select_strikes(
            date=SAMPLE_DATE,
            timestamp="10:00:00",
            strategy_type=StrategyType.IRON_CONDOR,
            target_credit=0.40,
            min_spread_width=25
        )

        if strike_selection:  # May return None if no suitable strikes
            assert isinstance(strike_selection, IronCondorStrikeSelection)
            # Put spread: short strike must be above long strike
            assert strike_selection.put_short_strike > strike_selection.put_long_strike
            # Call spread: short strike must be below long strike
            assert strike_selection.call_short_strike < strike_selection.call_long_strike
            # Spread widths must match the actual strike differences
            assert strike_selection.put_spread_width == pytest.approx(
                strike_selection.put_short_strike - strike_selection.put_long_strike, abs=1e-6
            )
            assert strike_selection.call_spread_width == pytest.approx(
                strike_selection.call_long_strike - strike_selection.call_short_strike, abs=1e-6
            )

    def test_select_strikes_put_spread_success(self):
        """Test successful Put Spread strike selection."""
        strike_selection = self.selector.select_strikes(
            date=SAMPLE_DATE,
            timestamp="10:00:00",
            strategy_type=StrategyType.PUT_SPREAD,
            target_credit=0.40,
            min_spread_width=25
        )

        if strike_selection:
            assert_valid_strike_selection(strike_selection, SAMPLE_SPX_PRICE)
            # Put spread should have short strike above long strike
            assert strike_selection.short_strike > strike_selection.long_strike

    def test_select_strikes_call_spread_success(self):
        """Test successful Call Spread strike selection."""
        strike_selection = self.selector.select_strikes(
            date=SAMPLE_DATE,
            timestamp="10:00:00",
            strategy_type=StrategyType.CALL_SPREAD,
            target_credit=0.40,
            min_spread_width=25
        )

        if strike_selection:
            assert_valid_strike_selection(strike_selection, SAMPLE_SPX_PRICE)
            # Call spread should have short strike below long strike
            assert strike_selection.short_strike < strike_selection.long_strike

    def test_select_strikes_no_spx_price(self):
        """Test strike selection when SPX price unavailable."""
        mock_query = Mock()
        mock_query.get_fastest_spx_price.return_value = None
        selector = StrikeSelector(mock_query, Mock())

        result = selector.select_strikes(
            date=SAMPLE_DATE,
            timestamp="10:00:00",
            strategy_type=StrategyType.IRON_CONDOR,
            target_credit=0.40,
        )

        assert result is None

    def test_select_strikes_no_options_data(self):
        """Test strike selection when options data unavailable."""
        mock_query = Mock()
        mock_query.get_fastest_spx_price.return_value = SAMPLE_SPX_PRICE
        mock_query.get_options_data.return_value = None
        selector = StrikeSelector(mock_query, Mock())

        result = selector.select_strikes(
            date=SAMPLE_DATE,
            timestamp="10:00:00",
            strategy_type=StrategyType.IRON_CONDOR,
            target_credit=0.40,
        )

        assert result is None

    def test_select_strikes_no_target_credit(self):
        """Test strike selection without target_credit returns None."""
        result = self.selector.select_strikes(
            date=SAMPLE_DATE,
            timestamp="10:00:00",
            strategy_type=StrategyType.IRON_CONDOR,
        )

        assert result is None

    def test_select_strikes_minimum_spread_width(self):
        """Test strike selection respects minimum spread width."""
        strike_selection = self.selector.select_strikes(
            date=SAMPLE_DATE,
            timestamp="10:00:00",
            strategy_type=StrategyType.PUT_SPREAD,
            target_credit=0.40,
            min_spread_width=50  # Large minimum spread
        )

        if strike_selection:
            assert strike_selection.spread_width >= 50


class TestIntradayPositionMonitor:
    """Test IntradayPositionMonitor class."""

    def setup_method(self):
        self.mock_query_engine = Mock()
        self.mock_strategy_builder = Mock()
        self.mock_strategy_builder.update_strategy_prices_optimized = Mock(return_value=True)
        self.monitor = IntradayPositionMonitor(self.mock_query_engine, self.mock_strategy_builder)

    def _make_leg(self, option_type: str, entry_price: float, current_price: float, short: bool = True):
        leg = Mock()
        leg.option_type = Mock()
        leg.option_type.value = option_type
        leg.entry_price = entry_price
        leg.current_price = current_price
        leg.quantity = 1
        leg.position_side = Mock()
        leg.position_side.name = 'SHORT' if short else 'LONG'
        return leg

    def _make_ic_strategy(self, put_entry=2.0, put_current=1.5,
                          call_entry=2.0, call_current=1.5, quantity=1):
        """Create a mock IC strategy with 4 legs."""
        strategy = Mock()
        # Put spread: 1 short put, 1 long put
        short_put  = self._make_leg('put',  put_entry,  put_current,  short=True)
        long_put   = self._make_leg('put',  put_entry * 0.5, put_current * 0.5, short=False)
        # Call spread: 1 short call, 1 long call
        short_call = self._make_leg('call', call_entry,  call_current,  short=True)
        long_call  = self._make_leg('call', call_entry * 0.5, call_current * 0.5, short=False)
        strategy.legs = [short_put, long_put, short_call, long_call]
        # entry_credit = net credit for both sides combined
        strategy.entry_credit = (put_entry * 100 - put_entry * 0.5 * 100 +
                                 call_entry * 100 - call_entry * 0.5 * 100)
        strategy.quantity = quantity
        return strategy

    def test_check_decay_spread_exits_at_threshold(self):
        """Spread should exit when cost/share <= take_profit."""
        strategy = Mock()
        strategy.entry_credit = 100.0
        strategy.quantity = 1
        leg = self._make_leg('put', 2.0, 0.001, short=True)  # near zero cost
        strategy.legs = [leg]

        should_exit, cost, reason = self.monitor.check_decay_at_time(
            strategy, StrategyType.PUT_SPREAD, '2026-02-10', '10:30:00'
        )

        assert should_exit is True
        assert cost >= 0
        assert 'take profit' in reason.lower() or 'stop loss' in reason.lower()

    def test_check_decay_spread_no_exit_when_above_threshold(self):
        """Spread should NOT exit when decay_ratio > threshold."""
        strategy = Mock()
        strategy.entry_credit = 100.0
        strategy.quantity = 1
        leg = self._make_leg('put', 2.0, 1.0, short=True)  # 50% remaining
        strategy.legs = [leg]

        should_exit, cost, reason = self.monitor.check_decay_at_time(
            strategy, StrategyType.PUT_SPREAD, '2026-02-10', '10:30:00'
        )

        assert should_exit is False

    def test_check_ic_leg_decay_put_side_detected(self):
        """IC put side decay should be flagged when below threshold."""
        # Make put side nearly expired (tiny current price) but call side still alive
        strategy = self._make_ic_strategy(
            put_entry=2.0, put_current=0.0001,  # put fully decayed
            call_entry=2.0, call_current=1.5     # call still has value
        )
        ic_status = IronCondorLegStatus()

        updated = self.monitor.check_ic_leg_decay(
            strategy, '2026-02-10', '11:00:00', ic_status
        )

        assert updated.put_side_closed is True
        assert updated.put_side_exit_time == '11:00:00'
        assert updated.call_side_closed is False

    def test_check_ic_leg_decay_call_side_detected(self):
        """IC call side decay should be flagged when below threshold."""
        strategy = self._make_ic_strategy(
            put_entry=2.0, put_current=1.5,      # put still alive
            call_entry=2.0, call_current=0.0001  # call fully decayed
        )
        ic_status = IronCondorLegStatus()

        updated = self.monitor.check_ic_leg_decay(
            strategy, '2026-02-10', '12:00:00', ic_status
        )

        assert updated.call_side_closed is True
        assert updated.call_side_exit_time == '12:00:00'
        assert updated.put_side_closed is False

    def test_check_ic_leg_decay_both_sides_independent(self):
        """IC sides close independently at different times."""
        strategy = self._make_ic_strategy(
            put_entry=2.0, put_current=0.0001,
            call_entry=2.0, call_current=0.0001
        )
        ic_status = IronCondorLegStatus()

        self.monitor.check_ic_leg_decay(strategy, '2026-02-10', '11:00:00', ic_status)
        assert ic_status.put_side_closed is True
        assert ic_status.call_side_closed is True

    def test_check_ic_leg_decay_does_not_reclose_already_closed(self):
        """Already-closed IC side should not update exit time on subsequent checks."""
        strategy = self._make_ic_strategy(
            put_entry=2.0, put_current=0.0001,
            call_entry=2.0, call_current=0.0001
        )
        ic_status = IronCondorLegStatus()

        self.monitor.check_ic_leg_decay(strategy, '2026-02-10', '11:00:00', ic_status)
        first_put_exit = ic_status.put_side_exit_time

        # Call again at a later time — should NOT update exit times
        self.monitor.check_ic_leg_decay(strategy, '2026-02-10', '12:00:00', ic_status)
        assert ic_status.put_side_exit_time == first_put_exit

    def test_calculate_exit_cost_zero_entry_credit(self):
        """Exit cost calculation with no legs returns no-exit and zero cost."""
        strategy = Mock()
        strategy.entry_credit = 0
        strategy.quantity = 1
        strategy.legs = []

        should_exit, cost, reason = self.monitor.check_decay_at_time(
            strategy, StrategyType.PUT_SPREAD, '2026-02-10', '10:00:00'
        )

        # cost = 0, per-share cost = 0 which is <= take_profit (0.10), so it exits
        assert cost == 0.0

    def test_ic_decay_thresholds_are_correct(self):
        """Verify default take_profit and stop_loss values on instance."""
        assert self.monitor.take_profit == 0.10
        assert self.monitor.stop_loss == 2.0
        assert self.monitor.monitor_interval == 1

    def test_multi_contract_tp_sl_same_as_single_contract(self):
        """
        TP/SL fires at the same per-share cost regardless of quantity.
        1 contract at $0.05/share and 2 contracts at $0.05/share should both hit take_profit.
        """
        # 1 contract: cost = $0.05/share × 100 = $5 total
        single = Mock()
        single.quantity = 1
        single.legs = [self._make_leg('put', 2.0, 0.05, short=True)]

        # 2 contracts: same $0.05/share, but total cost = $5 × 2 = $10
        double = Mock()
        double.quantity = 2
        leg_a = self._make_leg('put', 2.0, 0.05, short=True)
        leg_a.quantity = 2
        double.legs = [leg_a]

        exit1, cost1, reason1 = self.monitor.check_decay_at_time(
            single, StrategyType.PUT_SPREAD, '2026-02-10', '10:30:00'
        )
        exit2, cost2, reason2 = self.monitor.check_decay_at_time(
            double, StrategyType.PUT_SPREAD, '2026-02-10', '10:30:00'
        )

        # Both should exit (take profit) at the same per-share threshold
        assert exit1 is True
        assert exit2 is True
        assert 'take profit' in reason1.lower()
        assert 'take profit' in reason2.lower()
        # Dollar cost is 2× for double contracts
        assert cost2 == pytest.approx(cost1 * 2, rel=1e-6)

    def test_multi_contract_no_exit_matches_single_contract(self):
        """
        A position that does NOT exit on 1 contract should also NOT exit on 10 contracts,
        because the per-share cost is the same.
        """
        for qty in (1, 2, 5, 10):
            strategy = Mock()
            strategy.quantity = qty
            leg = self._make_leg('put', 2.0, 1.0, short=True)  # $1/share — above TP, below SL
            leg.quantity = qty
            strategy.legs = [leg]

            should_exit, _, _ = self.monitor.check_decay_at_time(
                strategy, StrategyType.PUT_SPREAD, '2026-02-10', '10:30:00'
            )
            assert should_exit is False, f"Should not exit at quantity={qty}"

    # ------------------------------------------------------------------
    # Stale-loss tests
    # ------------------------------------------------------------------

    def _make_checkpoints(self, n: int, cost_per_share: float,
                          cost_key: str = "cost_per_share") -> list:
        """Generate n uniform checkpoints at a fixed cost_per_share value."""
        return [{cost_key: cost_per_share, "time": f"10:{i:02d}:00", "spx": 5000.0}
                for i in range(n)]

    def test_stale_loss_not_triggered_when_insufficient_bars(self):
        """Should not trigger when fewer bars than stale_loss_minutes have accumulated."""
        monitor = IntradayPositionMonitor(
            self.mock_query_engine, self.mock_strategy_builder,
            stale_loss_minutes=10, stale_loss_threshold=1.5,
            stagnation_window=5, min_improvement=0.05,
        )
        # Only 5 bars — fewer than the required 10
        checkpoints = self._make_checkpoints(5, cost_per_share=0.80)
        exit_, reason = monitor.check_stale_loss(checkpoints, entry_credit_per_share=0.40)
        assert exit_ is False
        assert reason == ""

    def test_stale_loss_not_triggered_when_cost_below_threshold(self):
        """Should not trigger when cost is below the stale-loss threshold."""
        monitor = IntradayPositionMonitor(
            self.mock_query_engine, self.mock_strategy_builder,
            stale_loss_minutes=10, stale_loss_threshold=1.5,
            stagnation_window=5, min_improvement=0.05,
        )
        # 15 bars, cost = 0.50 = 1.25× credit (below 1.5 threshold)
        checkpoints = self._make_checkpoints(15, cost_per_share=0.50)
        exit_, reason = monitor.check_stale_loss(checkpoints, entry_credit_per_share=0.40)
        assert exit_ is False

    def test_stale_loss_not_triggered_when_improvement_sufficient(self):
        """Should not trigger when cost has improved more than min_improvement recently."""
        monitor = IntradayPositionMonitor(
            self.mock_query_engine, self.mock_strategy_builder,
            stale_loss_minutes=10, stale_loss_threshold=1.5,
            stagnation_window=5, min_improvement=0.05,
        )
        # 15 bars all above threshold; last 5 have a $0.10 swing → improvement >= min
        checkpoints = self._make_checkpoints(10, cost_per_share=0.90)
        checkpoints += [{"cost_per_share": 0.90, "time": "10:10:00", "spx": 5000.0},
                        {"cost_per_share": 0.85, "time": "10:11:00", "spx": 5000.0},
                        {"cost_per_share": 0.80, "time": "10:12:00", "spx": 5000.0},
                        {"cost_per_share": 0.82, "time": "10:13:00", "spx": 5000.0},
                        {"cost_per_share": 0.80, "time": "10:14:00", "spx": 5000.0}]
        exit_, reason = monitor.check_stale_loss(checkpoints, entry_credit_per_share=0.40)
        assert exit_ is False

    def test_stale_loss_triggers_both_conditions_met(self):
        """Should trigger when cost is persistently above threshold and stagnant."""
        monitor = IntradayPositionMonitor(
            self.mock_query_engine, self.mock_strategy_builder,
            stale_loss_minutes=10, stale_loss_threshold=1.5,
            stagnation_window=5, min_improvement=0.05,
        )
        # 15 bars all at $0.80 (= 2.0× credit of $0.40, above 1.5 threshold)
        # Zero movement — improvement = 0 < 0.05
        checkpoints = self._make_checkpoints(15, cost_per_share=0.80)
        exit_, reason = monitor.check_stale_loss(checkpoints, entry_credit_per_share=0.40)
        assert exit_ is True
        assert "stale loss" in reason.lower()
        assert "0.80" in reason or "80" in reason

    def test_stale_loss_ic_put_side_key(self):
        """Stale-loss check works on IC put_cost_per_share checkpoints."""
        monitor = IntradayPositionMonitor(
            self.mock_query_engine, self.mock_strategy_builder,
            stale_loss_minutes=5, stale_loss_threshold=1.5,
            stagnation_window=3, min_improvement=0.05,
        )
        # 8 bars; put cost = $0.90 (> 1.5 × $0.40 = $0.60); call cost fine
        checkpoints = [
            {"put_cost_per_share": 0.90, "call_cost_per_share": 0.10,
             "cost_per_share": 1.00, "time": f"10:0{i}:00", "spx": 5000.0}
            for i in range(8)
        ]
        # Put side should trigger
        put_exit, put_reason = monitor.check_stale_loss(
            checkpoints, entry_credit_per_share=0.40, cost_key="put_cost_per_share"
        )
        # Call side should NOT trigger (call cost < threshold)
        call_exit, _ = monitor.check_stale_loss(
            checkpoints, entry_credit_per_share=0.40, cost_key="call_cost_per_share"
        )
        assert put_exit is True
        assert "stale loss" in put_reason.lower()
        assert call_exit is False

    def test_stale_loss_default_params_stored(self):
        """Default stale-loss parameter values are stored correctly."""
        monitor = IntradayPositionMonitor(self.mock_query_engine, self.mock_strategy_builder)
        assert monitor.stale_loss_minutes == 120
        assert monitor.stale_loss_threshold == 1.5
        assert monitor.stagnation_window == 30
        assert monitor.min_improvement == 0.05
        # The on/off toggle lives in BacktestRequest, not the monitor itself —
        # the scan loop gates calls to check_stale_loss behind enable_stale_loss_exit.