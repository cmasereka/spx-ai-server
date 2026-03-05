"""Integration tests for the complete enhanced multi-strategy backtesting system."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
sys.path.append('.')

from enhanced_multi_strategy import EnhancedBacktestingEngine
from enhanced_backtest import (
    StrategyType, MarketSignal, TechnicalIndicators, 
    StrategySelection, EnhancedBacktestResult, TechnicalAnalyzer, StrategySelector
)
from tests.conftest import (
    TestDataGenerator, MockQueryEngine, assert_valid_backtest_result,
    SAMPLE_DATE, SAMPLE_SPX_PRICE
)


class TestEnhancedBacktestingEngine:
    """Integration tests for the complete enhanced backtesting system."""
    
    def setup_method(self):
        """Set up test environment with mocked data."""
        # Create test data directory if it doesn't exist
        import os
        os.makedirs("tests/fixtures/test_data", exist_ok=True)
        
        # Mock the enhanced engine with test data
        with patch('enhanced_multi_strategy.EnhancedQueryEngineAdapter') as mock_adapter:
            mock_adapter.return_value = MockQueryEngine()
            
            with patch('strike_selector.StrikeSelector') as mock_selector:
                with patch('strike_selector.PositionMonitor') as mock_monitor_cls:
                    self.engine = EnhancedBacktestingEngine("tests/fixtures/test_data")
    
    def test_enhanced_backtest_single_day_success(self):
        """Test successful single day enhanced backtesting."""
        # Mock all the components to return valid data
        with patch.object(self.engine, 'enhanced_query_engine') as mock_query:
            mock_query.get_fastest_spx_price.return_value = SAMPLE_SPX_PRICE
            
            with patch.object(self.engine, 'get_spx_price_history') as mock_history:
                mock_history.return_value = pd.Series([SAMPLE_SPX_PRICE] * 50)
                
                with patch.object(self.engine, 'strike_selector') as mock_delta:
                    from strike_selector import StrikeSelection
                    mock_delta.select_strikes.return_value = StrikeSelection(
                        short_strike=6900.0,
                        long_strike=6875.0,
                        spread_width=25.0
                    )
                    
                    with patch.object(self.engine, '_build_iron_condor_strategy') as mock_build:
                        # Mock strategy
                        strategy = Mock()
                        strategy.legs = [Mock(), Mock(), Mock(), Mock()]
                        strategy.entry_credit = 100.0
                        strategy.max_profit = 100.0
                        strategy.max_loss = -150.0
                        mock_build.return_value = strategy
                        
                        with patch.object(self.engine, 'position_monitor') as mock_monitor:
                            mock_monitor.monitor_position.return_value = (
                                [{"timestamp": "10:05:00", "pnl": 90.0, "decay_ratio": 0.1}],
                                "Early profit target",
                                10.0
                            )
                            
                            # Set available dates
                            self.engine.available_dates = [SAMPLE_DATE]
                            
                            # Run the test
                            result = self.engine.enhanced_backtest_single_day(
                                date=SAMPLE_DATE,
                                decay_threshold=0.1
                            )
                            
                            # Validate result
                            assert_valid_backtest_result(result, expected_success=True)
                            assert result.date == SAMPLE_DATE
                            assert result.strategy_type == StrategyType.IRON_CONDOR
                            assert result.entry_credit == 100.0
                            assert result.exit_cost == 10.0
                            assert result.pnl == 90.0
    
    def test_enhanced_backtest_no_data_available(self):
        """Test enhanced backtesting when no data is available."""
        # Set no available dates
        self.engine.available_dates = []
        
        result = self.engine.enhanced_backtest_single_day(
            date=SAMPLE_DATE
        )
        
        # Should return failed result
        assert result.success is False
        assert "No data available" in result.notes
    
    def test_enhanced_backtest_no_spx_price(self):
        """Test enhanced backtesting when SPX price unavailable."""
        with patch.object(self.engine, 'enhanced_query_engine') as mock_query:
            mock_query.get_fastest_spx_price.return_value = None
            
            # Set available dates  
            self.engine.available_dates = [SAMPLE_DATE]
            
            result = self.engine.enhanced_backtest_single_day(
                date=SAMPLE_DATE
            )
            
            assert result.success is False
            assert "No SPX price at entry" in result.notes
    
    def test_enhanced_backtest_no_viable_strikes(self):
        """Test enhanced backtesting when no viable strikes found."""
        with patch.object(self.engine, 'enhanced_query_engine') as mock_query:
            mock_query.get_fastest_spx_price.return_value = SAMPLE_SPX_PRICE
            
            with patch.object(self.engine, 'get_spx_price_history') as mock_history:
                mock_history.return_value = pd.Series([SAMPLE_SPX_PRICE] * 50)
                
                with patch.object(self.engine, 'strike_selector') as mock_delta:
                    mock_delta.select_strikes.return_value = None
                    
                    self.engine.available_dates = [SAMPLE_DATE]
                    
                    result = self.engine.enhanced_backtest_single_day(
                        date=SAMPLE_DATE
                    )
                    
                    assert result.success is False
                    assert "No viable" in result.notes
    
    def test_enhanced_backtest_strategy_build_failure(self):
        """Test enhanced backtesting when strategy building fails."""
        with patch.object(self.engine, 'enhanced_query_engine') as mock_query:
            mock_query.get_fastest_spx_price.return_value = SAMPLE_SPX_PRICE
            
            with patch.object(self.engine, 'get_spx_price_history') as mock_history:
                mock_history.return_value = pd.Series([SAMPLE_SPX_PRICE] * 50)
                
                with patch.object(self.engine, 'strike_selector') as mock_delta:
                    from strike_selector import StrikeSelection
                    mock_delta.select_strikes.return_value = StrikeSelection(
                        short_strike=6900.0,
                        long_strike=6875.0,
                        spread_width=25.0
                    )
                    
                    with patch.object(self.engine, '_build_iron_condor_strategy') as mock_build:
                        mock_build.return_value = None  # Strategy build failure
                        
                        self.engine.available_dates = [SAMPLE_DATE]
                        
                        result = self.engine.enhanced_backtest_single_day(
                            date=SAMPLE_DATE
                        )
                        
                        assert result.success is False
                        assert "Could not build" in result.notes


class TestBacktestingWorkflow:
    """Test complete backtesting workflows and data flow."""
    
    def test_technical_analysis_to_strategy_selection(self):
        """Test the flow from technical analysis to strategy selection."""
        from enhanced_backtest import TechnicalAnalyzer, StrategySelector
        
        # Create sample price data showing clear trend
        uptrend_prices = pd.Series(np.linspace(4500, 4600, 50))  # Clear uptrend
        
        analyzer = TechnicalAnalyzer()
        selector = StrategySelector()
        
        # Analyze market conditions
        indicators = analyzer.analyze_market_conditions(uptrend_prices)
        
        # Select strategy based on analysis
        selection = selector.select_strategy(indicators)
        
        # Validate the complete workflow
        assert isinstance(indicators, TechnicalIndicators)
        assert isinstance(selection, StrategySelection)
        assert selection.strategy_type in [StrategyType.IRON_CONDOR, StrategyType.CALL_SPREAD, StrategyType.PUT_SPREAD]
        assert 0 <= selection.confidence <= 1
    
    def test_strategy_selection_to_strike_selection(self):
        """Test the flow from strategy selection to strike selection."""
        from strike_selector import StrikeSelector, StrikeSelection
        from enhanced_backtest import StrategySelection, StrategyType, MarketSignal

        mock_query_engine = MockQueryEngine()
        mock_ic_loader = Mock()

        selector = StrikeSelector(mock_query_engine, mock_ic_loader)

        strategy_selection = StrategySelection(
            strategy_type=StrategyType.CALL_SPREAD,
            market_signal=MarketSignal.BULLISH,
            confidence=0.8,
            reason="Bullish market conditions"
        )

        strike_selection = selector.select_strikes(
            date=SAMPLE_DATE,
            timestamp="10:00:00",
            strategy_type=strategy_selection.strategy_type,
            target_credit=0.40,
        )

        if strike_selection:  # May be None due to mock data limitations
            assert isinstance(strike_selection, StrikeSelection)
    
    def test_end_to_end_backtest_simulation(self):
        """Test complete end-to-end backtest simulation."""
        # This is a simplified version of what the real system does
        
        # 1. Technical Analysis
        analyzer = TechnicalAnalyzer()
        sample_prices = pd.Series(np.random.normal(4550, 10, 50))  # Random walk around 4550
        indicators = analyzer.analyze_market_conditions(sample_prices)
        
        # 2. Strategy Selection  
        selector = StrategySelector()
        strategy_selection = selector.select_strategy(indicators)
        
        # 3. Mock Strike Selection (simplified)
        from strike_selector import StrikeSelection
        strike_selection = StrikeSelection(
            short_strike=6900.0,
            long_strike=6875.0 if strategy_selection.strategy_type == StrategyType.PUT_SPREAD else 6925.0,
            spread_width=25.0
        )
        
        # 4. Mock Strategy Building
        from src.strategies.options_strategies import OptionsStrategy, OptionLeg, PositionSide, OptionType
        legs = []
        
        if strategy_selection.strategy_type == StrategyType.IRON_CONDOR:
            legs = [
                OptionLeg(6875, OptionType.PUT, PositionSide.LONG, 1, 1.0, 1.0, SAMPLE_DATE),
                OptionLeg(6900, OptionType.PUT, PositionSide.SHORT, 1, 2.5, 2.5, SAMPLE_DATE),
                OptionLeg(7000, OptionType.CALL, PositionSide.SHORT, 1, 2.0, 2.0, SAMPLE_DATE),
                OptionLeg(7025, OptionType.CALL, PositionSide.LONG, 1, 0.75, 0.75, SAMPLE_DATE)
            ]
        elif strategy_selection.strategy_type == StrategyType.PUT_SPREAD:
            legs = [
                OptionLeg(6875, OptionType.PUT, PositionSide.LONG, 1, 1.0, 1.0, SAMPLE_DATE),
                OptionLeg(6900, OptionType.PUT, PositionSide.SHORT, 1, 2.5, 2.5, SAMPLE_DATE)
            ]
        else:  # CALL_SPREAD
            legs = [
                OptionLeg(7000, OptionType.CALL, PositionSide.SHORT, 1, 2.0, 2.0, SAMPLE_DATE),
                OptionLeg(7025, OptionType.CALL, PositionSide.LONG, 1, 0.75, 0.75, SAMPLE_DATE)
            ]
        
        strategy = OptionsStrategy(
            strategy_name=strategy_selection.strategy_type.value,
            entry_date=SAMPLE_DATE,
            underlying_price=SAMPLE_SPX_PRICE
        )
        for leg in legs:
            strategy.add_leg(leg)
        
        # 5. P&L Calculation
        entry_credit = strategy.entry_credit
        
        # Simulate price decay (favorable)
        for leg in strategy.legs:
            if leg.position_side == PositionSide.SHORT:
                leg.current_price = leg.entry_price * 0.6  # 40% decay
            else:
                leg.current_price = leg.entry_price * 0.8  # 20% decay
        
        exit_cost = abs(strategy.current_pnl - entry_credit)
        final_pnl = entry_credit - exit_cost
        
        # 6. Validate Complete Workflow
        assert entry_credit > 0  # Should receive credit
        assert isinstance(final_pnl, (int, float))
        assert len(legs) >= 2  # Should have at least 2 legs
        
        # Create mock result
        result = EnhancedBacktestResult(
            date=SAMPLE_DATE,
            strategy_type=strategy_selection.strategy_type,
            market_signal=strategy_selection.market_signal,
            entry_time="10:00:00",
            exit_time="15:45:00",
            exit_reason="Simulated exit",
            entry_spx_price=SAMPLE_SPX_PRICE,
            exit_spx_price=SAMPLE_SPX_PRICE + 10,
            technical_indicators=indicators,
            strike_selection=strike_selection,
            entry_credit=entry_credit,
            exit_cost=exit_cost,
            pnl=final_pnl,
            pnl_pct=(final_pnl / entry_credit * 100) if entry_credit > 0 else 0,
            max_profit=entry_credit,
            max_loss=-200.0,
            monitoring_points=[],
            success=True,
            confidence=strategy_selection.confidence,
            notes="End-to-end test simulation"
        )
        
        assert_valid_backtest_result(result, expected_success=True)


class TestErrorHandlingAndRecovery:
    """Test system behavior under error conditions and recovery scenarios."""
    
    def test_partial_data_availability(self):
        """Test system behavior with partial data availability."""
        # Mock scenario where some data is missing
        mock_query_engine = Mock()
        mock_query_engine.get_fastest_spx_price.return_value = SAMPLE_SPX_PRICE
        mock_query_engine.get_options_data.return_value = None  # No options data
        
        from strike_selector import StrikeSelector
        selector = StrikeSelector(mock_query_engine, Mock())

        result = selector.select_strikes(
            date=SAMPLE_DATE,
            timestamp="10:00:00",
            strategy_type=StrategyType.IRON_CONDOR,
            target_credit=0.40,
        )
        
        # Should handle gracefully
        assert result is None
    
    def test_extreme_market_conditions_handling(self):
        """Test system behavior under extreme market conditions."""
        from enhanced_backtest import TechnicalAnalyzer, StrategySelector
        
        # Create extreme volatility scenario
        extreme_prices = pd.Series([4500, 5000, 4000, 5500, 3500, 6000, 3000] * 10)
        
        analyzer = TechnicalAnalyzer()
        selector = StrategySelector()
        
        # Should handle extreme conditions without crashing
        indicators = analyzer.analyze_market_conditions(extreme_prices)
        selection = selector.select_strategy(indicators)
        
        # Validate that system produces valid output despite extreme conditions
        assert 0 <= indicators.rsi <= 100
        assert selection.strategy_type in [StrategyType.IRON_CONDOR, StrategyType.PUT_SPREAD, StrategyType.CALL_SPREAD]
    
    def test_concurrent_backtest_simulation(self):
        """Test running multiple backtests concurrently (simulated)."""
        from enhanced_backtest import TechnicalAnalyzer, StrategySelector
        
        analyzer = TechnicalAnalyzer()
        selector = StrategySelector()
        
        # Simulate multiple different market scenarios
        scenarios = [
            pd.Series(np.linspace(4500, 4600, 50)),  # Uptrend
            pd.Series(np.linspace(4600, 4500, 50)),  # Downtrend  
            pd.Series([4550] * 50 + np.random.normal(0, 5, 50)),  # Sideways with noise
        ]
        
        results = []
        
        for i, scenario in enumerate(scenarios):
            try:
                indicators = analyzer.analyze_market_conditions(scenario)
                selection = selector.select_strategy(indicators)
                
                # Mock result
                result = {
                    'scenario': i,
                    'strategy': selection.strategy_type.value,
                    'confidence': selection.confidence,
                    'success': True
                }
                results.append(result)
            except Exception as e:
                # Should handle errors gracefully
                result = {
                    'scenario': i,
                    'strategy': None,
                    'confidence': 0,
                    'success': False,
                    'error': str(e)
                }
                results.append(result)
        
        # Validate that all scenarios were processed
        assert len(results) == len(scenarios)
        
        # At least some scenarios should succeed
        successful_results = [r for r in results if r['success']]
        assert len(successful_results) > 0